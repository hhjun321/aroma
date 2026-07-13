# 07 — 스크립트 전체 입출력 매핑

> **Claude 요약:** AROMA 파이프라인을 구성하는 모든 스크립트의 I/O + 핵심 로직을 단일 페이지로 매핑한다. 실행 순서는 `prepare_datasets(step -1) → phase0(profiling) → step1(complexity) → step2(prompts) → step3(roi) → step4(cn_data/train + τ 사전스캔) → step5(generate) → exp*`. 경로 규약은 `_SPEC.md`의 stage-first(`sym_final/{stage}/{ds}`)를 정본으로 하며, `distribution_profiling.py`와 `train_controlnet.py`만 `scripts/`(루트)에 있고 나머지는 `scripts/aroma/`에 있다. 모든 항목은 실제 argparse에 근거한다.

## 전체 스크립트 매핑

| Stage | 스크립트 | 주요 입력 | 주요 출력 | CPU/GPU |
|-------|---------|-----------|-----------|---------|
| step -1 prepare | `aroma/prepare_severstal.py` | Kaggle Severstal (train.csv RLE + train_images) | `Aroma/severstal/` MVTec 레이아웃(train/good·test/class{1..4}·masks) + `severstal_manifest.json` | CPU |
| step -1 prepare | `aroma/prepare_aitex.py` | AITEX (Defect/Mask/NODefect_images, 4096×256) | `aitex_tiled/` 256×256/stride128 타일 레이아웃 + manifest (mask 없는 결함은 skip) | CPU |
| step -1 prepare | `aroma/prepare_mtd.py` | MTD Supervisely(meta.json + ds/img + ds/ann) | `<out>/train/good·test/<class>·ground_truth/<class>/*_mask.png` + `mtd_manifest.json` | CPU |
| step -1 (util) | `aroma/select_context_prototypes.py` | normal 이미지 풀 | K개 context prototype 목록(분포 대표 배경) | GPU(선택, CLIP) |
| phase0 profiling | `distribution_profiling.py` **(루트)** | `--dataset_config` + `--dataset_key` | `S(profiling,ds)`: morphology_features.csv · context_features.csv · distribution_analysis.json · morphology_clusters.json · **compatibility_matrix.json**(matrix_symmetric·P_def_patch·clean_dist) · deficit_analysis.json · threshold_policies.json · recommended_config.yaml · analysis_report.md · figures/ | CPU (SAM 폴백 시 GPU 선택) |
| step1 complexity | `aroma/compute_complexity.py` | `--profiling_dir` | `S(complexity,ds)/complexity_report.json` (MCI·CCI·선택 정책) | CPU |
| step2 prompts | `aroma/prompt_generation.py` | `--profiling_dir` + `--complexity_dir` | `S(prompts,ds)`: prompts.json · prompts_summary.md | CPU |
| step3 roi | `aroma/roi_selection.py` | `--profiling_dir` + `--prompts_dir` | `S(roi,ds)`: roi_candidates.json · roi_selected.json | CPU |
| step3.5 (util) | `aroma/clean_bg_selection.py` | profiling context_features.csv + roi | per-ROI clean-bg 배정 JSON (`--clean_bg_json` 입력용) | CPU |
| step4a cn_data | `aroma/build_train_jsonl.py` | morphology_features.csv · roi_candidates.json · context_features.csv · recommended_config.yaml | `S(cn_data,ds)`: targets/ · hints/ · train.jsonl | CPU |
| step4b train | `train_controlnet.py` **(루트)** | `--data_dir`(=cn_data) | `CN_MODELS/{ds}/best_model/` (+ checkpoints) | GPU |
| step4c 사전스캔 | (compat_gate 진단, `_SPEC §10`) | compatibility_matrix.json | `S(compat_gate,ds)/compat_tau_prescan_{ds}.json` (ds_tau; aitex는 AR_T/TEX_T) | CPU |
| step5 generate(AROMA) | `aroma/generate_defects.py` | `--roi_dir` · `--normal_dir` · `--controlnet_path` · morphology/context/config · `--compat_matrix_json` | `S(synth_aroma,ds)`: 합성 이미지 + annotations.json | GPU(controlnet) / CPU(copy_paste) |
| step5 generate(random) | `aroma/generate_random.py` | `--candidates_json`(roi_candidates.json) · `--normal_dir` | `S(synth_random,ds)`: 합성 이미지 + annotations.json | CPU |
| (비교 arm) | `aroma/generate_casda.py` | CASDA `--metadata_csv` · `--normal_dir` | CASDA arm 합성 + annotations.json | CPU |
| exp3 | `aroma/experiments/exp3_generation_quality.py` | `--aroma_synthetic_dir` · `--random_synthetic_dir` · `--real_data_dir` | `S(exp3)`: FID/PaDiM 생성품질 결과 | CPU (device 기본 cpu) |
| exp4v2 | `aroma/experiments/exp4_v2_supervised_detection.py` | `--aroma/random_synthetic_dir` · `--real_data_dir` | `S(exp4v2)`: exp4v2_results.json · exp4v2_summary.md · baseline best.pt | GPU (YOLOv8) |
| exp5 | `aroma/experiments/exp5_prdc.py` | `--aroma/random_synthetic_dir` · `--real_data_dir` | `S(exp5)`: PRDC 결과 (embed_cache 공유) | GPU (DINOv2) |
| exp6 | `aroma/experiments/exp6_embedding_coverage.py` | `--aroma/random_synthetic_dir` · `--real_data_dir` (+ `--roi_dir_root` rare) | `S(exp6)`: kNN/rare 커버리지 결과 (embed_cache 공유) | GPU (DINOv2, 저비용) |

> 보조 스크립트: `validate_pipeline.py`(루트) — `--config`/`--stage` 기반 dry-run 입출력 점검(처리 없이). `aroma_to_casda_roi.py`·`casda_roi_adapter.py`·`composed_to_exp4v2.py` — CASDA compounding 어댑터(주 파이프라인 외).

## 스크립트별 핵심 계산 로직

### distribution_profiling.py (phase0, 루트)
- `dataset_config.json` 엔트리에서 결함 morphology와 배경 context 분포를 측정하는 9-step Stage 0 오케스트레이터. GT mask는 데이터셋별 `_find_mask_path` 분기로 찾고, 없으면 SAM/Otsu 폴백.
- 결함 인스턴스별 morphology feature와 비-mask 64px 패치별 5개 물리 context descriptor를 추출 → morphology_features.csv / context_features.csv.
- feature별 분포 형태(밸리 개수·GMM BIC 최적 k·P33/P66 bin edge)를 추정해 morph/context 정책과 threshold를 자동 유도(distribution_analysis.json, threshold_policies.json).
- **핵심 novelty**: 패치 공출현에서 clean-grounded 대칭 호환성(SGM, patch granularity) `matrix_symmetric`·`P_def_patch`·`clean_dist`·`symmetric_epsilon`을 계산 → compatibility_matrix.json (step5 symmetric 게이트가 소비).
- cluster×context-cell 별 deficit와 target_synthetic을 산출(deficit_analysis.json) + recommended_config.yaml 생성.

### roi_selection.py (step3)
- 모든 (결함 이미지 × context bin) 후보를 채점. `--score_mode legacy`(기본)는 `0.4·P(M_i)+0.4·P(C_j)+0.2·Deficit(M_i,C_j)`, `--score_mode realism`은 `0.5·ctx_prior+0.3·morph_prior+0.2·quality`(deficit 제거, quality를 게이트에서 등급항으로 승격).
- 4개 sampling strategy: `deficit_aware`(Top-K Deficit Quantile 오버샘플, 기본) · `top_k` · `weighted` · `random`.
- `--class_mode multi` + `--class_floor` + `--per_pair_cap_frac`로 클래스별 바닥 할당·pair 상한을 두는 2-phase 할당, `--img_diversity_cap`으로 이미지 편중 억제, `--rarity_temp`로 희소도 가중.
- 출력은 전체 후보(roi_candidates.json)와 선택본(roi_selected.json) 두 개 — generate_random은 전자, AROMA arm은 `--roi_dir`로 디렉터리 전체 소비.

### build_train_jsonl.py (step4a)
- CASDA 경로를 거치지 않고 **AROMA 내부에서** `train_controlnet.py`가 소비하는 학습 산출물을 직접 생성하는 빌더.
- per-real-defect 루프: morphology_features.csv 한 행(=실 결함 1개) → bbox crop(target PNG) → 3-채널 ControlNet conditioning hint PNG → prompt → train.jsonl 한 줄.
- 모든 category-형태 값은 data-driven(하드코딩 category/매직 임계 없음). defect_subtype은 roi_candidates.json을 (image_id, defect_mask_path)로 join해 morph_label에서 유도.
- 산출: `targets/` · `hints/` · `train.jsonl` (style은 `--style technical` 기본).

### train_controlnet.py (step4b, 루트)
- diffusers 기반 ControlNet fine-tune. SD1.5 backbone은 freeze, ControlNet만 학습(01-Overview 기준). `--data_dir`(cn_data)의 target/hint/train.jsonl 소비.
- 메모리 옵션: `--mixed_precision fp16` · `--gradient_checkpointing` · `--gradient_accumulation_steps`(기본 4) · `--train_batch_size`(기본 1).
- 학습 안정성/재개: `--num_train_epochs` · `--early_stopping_patience` · `--checkpointing_steps`(기본 500) · `--checkpoints_total_limit`(기본 3) · `--resume_from_checkpoint latest` · `--save_optimizer_state`/`--save_fp16`.
- grayscale target 강제 여부는 데이터셋별 차등(`--no_force_grayscale_target`은 컬러 leather 전용), `--augment`로 증강 토글. 산출: `output_dir/best_model/`.

### generate_defects.py (step5, AROMA arm)
- Step 3 ROI 후보를 읽어 결함 mask/crop을 normal 배경에 합성. `--method`: `copy_paste`(alpha/poisson, GPU-free) · `controlnet`(학습본으로 텍스처 생성 후 AROMA-선택 ROI에 paste, GT mask는 실 seed mask 유지) · `inpainting`(stub).
- blend: `--blend_mode alpha|seamless`(seamless=poisson/seamlessClone), `--feather_px`로 경계 페더링.
- **대칭 호환성 게이트**: `--compat_mode symmetric` + `--compat_threshold τ` + `--compat_matrix_json`(compatibility_matrix.json)로 배치 후보를 scan→rank→place, τ 미달 reject.
- **clean-bg 게이트**: `--reject-clean-bg` + `--min-bg-quality` + `--bg-blur-threshold` (+ aitex는 `--texture-dist-threshold`)로 검은/평탄 배경 거부.
- ControlNet 세부: `--cn_steps`(30) · `--cn_cond_scale`(0.7) · `--cn_resolution`(512) · `--cn_no_grayscale`(컬러 대상) · bbox 종횡비 > `--cn_ar_threshold`(2.5)면 copy_paste AR 폴백(`--cn_no_ar_fallback`으로 해제).

### generate_random.py (step5, random arm)
- 통제(baseline) arm. roi_candidates.json에서 top_k ROI를 **균등 무작위** 샘플 → roi_selected.json 기록 → `generate_defects.run()`에 위임.
- 합성 코드는 AROMA arm과 동일 — ROI 선택만 다르게 하여 ROI 모델링 기여를 격리. clean-bg 게이트(`--reject-clean-bg`/`--min-bg-quality`/`--bg-blur-threshold`)는 AROMA arm과 동일하게 적용.

### exp3_generation_quality.py (exp3)
- Cross-domain 생성 품질: random ROI vs AROMA ROI(동일 copy-paste 엔진). `--mode fid|ad|all`로 FID / PaDiM 이상탐지 지표 산출.
- 합성 품질 자체가 아니라 **adaptive ROI 모델링이 합성 학습데이터 품질을 높이는지** 평가가 목적(copy-paste는 신규 morphology 생성 불가라는 한계 명시).
- `--dataset_keys`(nargs) 다중, `--real_data_dir`·`--aroma/random_synthetic_dir` 루트 규약(root/{ds}). `--device`(기본 cpu) · `--image_size`(256).

### exp4_v2_supervised_detection.py (exp4v2, 헤드라인)
- Supervised YOLOv8 검출. 3조건 모두 COCO pretrained에서 real labeled defect를 공통 토대로 **fresh 학습**: baseline(real만) · random(real+random synth) · aroma(real+aroma synth).
- real defect를 seeded로 train/val split(`--val_frac` 기본 0.3) — baseline이 real positive를 학습하고 val은 disjoint real-only 유지. synth는 random/aroma에서만 real 위에 추가.
- `--model`(yolov8n..m/all) · `--condition`(baseline/random/casda/aroma/all) · `--dataset_keys` · `--seeds`(다중) · `--imgsz` · `--baseline_epochs` · `--rect` · `--synth_ratio` 등. `--resume` 지원.
- 산출: exp4v2_results.json · exp4v2_summary.md · baseline best.pt.

### exp5_prdc.py (exp5)
- 외부 임베딩(DINOv2) 좌표계에서 PRDC(Precision/Recall/Density/Coverage) 커버리지 평가. 사전등록 가설: 동일 copy-paste 엔진이므로 Precision/Density는 동등, Recall/Coverage는 aroma > random여야 "ROI 선택의 가치" 입증.
- reference = held-out real 결함 패치(exp4v2와 동일 split 규약 복제, `--val_frac` 0.3 / `--split_seed` 42) → synth 소스가 train split ROI에서만 나와 leakage 차단.
- 조건 간 n을 min-n seeded subsample로 엄격 동일화, `--nearest_k`(기본 3·5·10) sensitivity 전체 보고(주 보고 k=5), `--permutation_reps`(1000) 순열검정. `--embed_cache_dir`로 DINOv2 임베딩 캐시 공유.

### exp6_embedding_coverage.py (exp6)
- CPU급 임베딩 커버리지(exp5 캐시 공유). `--mode knn`: held-out val 결함 crop별 학습 풀(real / real+random / real+aroma)까지 최근접 cosine distance 분포를 이미지 단위 clustered bootstrap으로 검정(+diversity 항으로 중복 합성의 coverage 부풀림 차단).
- `--mode rare`: 후보 소스 crop의 독립 DINOv2 임베딩 k-means에서 rare 모드(빈도 p25 이하 AND val 등장)에 대한 aroma 선택 hit rate를 random 재선택 30-seed null 대비 검정.
- sensitivity 그리드: `--kmeans_k`(8·10·12·15) × `--cluster_seeds`(0..4), `--null_seeds`(30), `--rare_quantile`(0.25). rare 모드는 `--roi_dir_root`(=`S(roi)`) 필요.

## 경로 변수 → 실제 경로 매핑

| 변수 | 실제 경로 |
|------|-----------|
| `$DRIVE` | `/content/drive/MyDrive/data/Aroma` |
| `$AROMA_REF` | `/content/AROMA` (repo 루트) |
| `$AROMA_SCRIPTS` | `/content/AROMA/scripts/aroma` |
| `$AROMA_OUT` | `$DRIVE/aroma_output` |
| `$AROMA_DATA` | `$DRIVE` (exp `--real_data_dir`) |
| `$DATASET_CONFIG` | `/content/AROMA/dataset_config.json` |
| `$SYM_ROOT` | `$AROMA_OUT/sym_final` |
| `$CN_MODELS` | `$SYM_ROOT/controlnet_models` (모델은 `$CN_MODELS/{ds}/best_model`) |
| `distribution_profiling.py` / `train_controlnet.py` | `$AROMA_REF/scripts/` (루트 — `$AROMA_SCRIPTS` 아님) |
| `S(profiling, ds)` | `$SYM_ROOT/profiling/{ds}` |
| `S(complexity, ds)` | `$SYM_ROOT/complexity/{ds}` |
| `S(prompts, ds)` | `$SYM_ROOT/prompts/{ds}` |
| `S(roi, ds)` | `$SYM_ROOT/roi/{ds}` |
| `S(cn_data, ds)` | `$SYM_ROOT/cn_data/{ds}` |
| `S(controlnet_models, ds)` | `$SYM_ROOT/controlnet_models/{ds}` (best_model 하위) |
| `S(compat_gate, ds)` | `$SYM_ROOT/compat_gate/{ds}` |
| `S(synth_aroma, ds)` | `$SYM_ROOT/synth_aroma/{ds}` |
| `S(synth_random, ds)` | `$SYM_ROOT/synth_random/{ds}` |
| `S(exp4v2)` / `S(exp3)` / `S(exp5)` / `S(exp6)` | `$SYM_ROOT/exp4v2` · `/exp3` · `/exp5` · `/exp6` (ds 없이) |
| `S(embed_cache, ds)` | `$SYM_ROOT/embed_cache/{ds}` (exp5·exp6 공유) |

> exp* 루트 인자(`--aroma_synthetic_dir $(S('synth_aroma'))` 등)는 `/{ds}`를 스크립트가 붙이므로, 생성 시 `--output_dir`는 반드시 `S('synth_aroma', DS)`(=`.../synth_aroma/{ds}`)까지 지정한다. 환경변수 참조는 `$VAR`(중괄호 금지).

## 관련 노트

[[00-INDEX]] | [[02-Stage0-Prepare-Profiling]] | [[05-Stage3-ControlNet-Generation]] | [[06-Experiments]]
