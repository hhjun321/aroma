# AROMA 파이프라인 흐름도 (코드 분석 참조용)

> 2026-07-08 정리. 각 스테이지 = 스크립트 1개. 경로는 `scripts/` 기준.
> 입출력 파일명으로 스테이지 간 의존을 따라가면 됨.

---

## 메인 파이프라인 (Stage 0 → Step 4)

```
[Stage 0] distribution_profiling.py
   in : dataset_config.json (--dataset_key)
   out: morphology_features.csv     (결함 1개 = 1행, morph feature)
        context_features.csv        (64px 패치 context feature)
        distribution_analysis.json  (feature 분포 shape + policy)
        morphology_clusters.json    (GMM/percentile 클러스터 정의)
        │
        ▼
[Step 1] compute_complexity.py
   in : profiling_dir (Stage 0 출력)
   out: complexity_report.json      (MCI, CCI 스칼라 + 선택된 morph/context 정책)
        │
        ▼
[Step 2] prompt_generation.py
   in : profiling_dir + complexity_dir
   out: prompts.json                (키="{cluster_id}_{cell_key}" → prompt+prior+deficit)
        prompts_summary.md
        │
        ▼
[Step 3] roi_selection.py                          ★ 점수·선택 로직 핵심
   in : profiling_dir + prompts_dir
   out: roi_candidates.json  (전체 스코어 후보, quality_score 포함)
        roi_selected.json    (전략으로 뽑힌 최종 ROI)
        │
        ▼
[Step 4] generate_defects.py                       ★ 합성 엔진
   in : roi_dir (roi_selected.json) + normal_dir
   out: images/*.jpg + masks/*.png + annotations.json
```

### Step 3 점수 상세 (roi_selection.py)

- **후보 스코어** `score_roi()` (L236): `ROI_score = 0.4·P(M) + 0.4·P(C) + 0.2·Deficit`
  - 상수 `_W_MORPH / _W_CONTEXT / _W_DEFICIT`
- **선택 전략** `--sampling_strategy` (default **`deficit_aware`**, L1616):
  - `deficit_aware` — Top-K deficit quantile 오버샘플
  - `compatibility` — `compat_score = 0.6·ctx_prior + 0.4·morph_prior` 랭킹 (L1198, `_compat_score`). ※ 후보 스코어(0.4/0.4/0.2)와 **다른 공식** — 선택 단계 전용
  - `top_k` / `weighted` / `random`
- 클래스 층화 배분 = `_stratified_*` (L1192~), backfill = compat_score desc cap-aware (L1302)
- (보조) `select_context_prototypes.py` — CLIP→KMeans→medoid로 normal 배경 K개 대표 선택 (배경 풀 distribution 매칭)

### Step 4 합성 방법 (generate_defects.py, `--method`)

```
copy_paste (default)  실 결함 crop → normal에 alpha/seamless blend       [CPU]
controlnet            학습된 ControlNet로 텍스처 생성 → ROI paste          [GPU]
                      GT mask = 실 seed mask (bbox parity 유지)
inpainting            stub (NotImplementedError)
```

- 핵심 함수: `_paste_and_finalize()` (배치+blend+GT mask, 전 method 공용)
  - 배치: `_foreground_mask` → `_foreground_paste_position` → 실패 시 random fallback
  - **clean-bg 게이트** (`_is_clean_background`, fallback 경로만) — 검은/평탄 배경 거부
  - **텍스처 게이트** (`_texture_descriptor`/`_source_bg_descriptor`, 양 경로) — checkerplate 오배치 차단, `--texture-dist-threshold` (2026-07-07 추가)
- controlnet 전용: **AR 게이트** (`controlnet_synthesis` 초입) — bbox 종횡비 > `--cn_ar_threshold` → **copy_paste 폴백**(Option 3, `method="copy_paste_arfallback"`, stats `ar_fallback`)
- 재현성: content-hash seed(image_id, bbox, cell_key, rep) + fingerprint sidecar 캐시

---

## Arm 변형 (Step 4 대체 진입점)

```
generate_random.py    top_k ROI 균등 무작위 → roi_selected.json → generate_defects.run(copy_paste)
                      (random baseline, exp1/exp2 공용)
generate_casda.py     CASDA arm copy_paste
aroma_to_casda_roi.py  roi_selected.json → CASDA roi_metadata.csv (compounding 실험 forward 어댑터)
```

---

## ControlNet 학습 서브트랙 (Step 4 controlnet 전제)

```
build_train_jsonl.py                               (AROMA 내부 학습 데이터 빌드)
   in : morphology_features.csv + roi_candidates.json + context_features.csv + config
   out: targets/*.png (bbox crop) + hints/*.png (3ch conditioning) + train.jsonl
        │  ※ 생성·학습이 동일 hint/prompt 생성기 사용 (분포 일치)
        ▼
train_controlnet.py                                (SD1.5 + ControlNet fine-tune) [GPU]
   out: best_model/  → generate_defects.py --controlnet_path 로 소비
```

---

## 다운스트림 평가 (experiments/)

```
composed_to_exp4v2.py    CASDA composed 출력 → exp4v2 annotations (labelable-only 필터)
                         │
exp4_v2_supervised_detection.py   ★ 헤드라인 지표 (YOLO 검출)
   in : real_data_dir + random_synthetic_dir + aroma_synthetic_dir
   조건: baseline / random / aroma (--condition all)
   cap: n_synth_train = synth_ratio × n_real_train  ← 합성 풀에서 랜덤 subset
   out: exp4v2_results.json (per_seed map50/map50_95 + ci95)

기타 지표:
   exp1_casda_comparison.py    AROMA vs CASDA
   exp2_roi_quality.py         ROI 선택 품질
   exp3_generation_quality.py  생성 품질 (FID 등)
   exp5_prdc.py                Precision/Recall/Density/Coverage
   exp6_embedding_coverage.py  임베딩 커버리지
```

---

## 데이터 준비 (파이프라인 이전 1회)

```
prepare_severstal.py / prepare_mtd.py / prepare_aitex.py
   → dataset_config.json 엔트리용 이미지·마스크 레이아웃 정규화
   (aitex는 256/stride128 tiled + 단일클래스)
```

---

## 핵심 의존 요약 (한 줄)

```
profiling(0) → complexity(1) → prompt(2) → roi_selection(3) → generate_defects(4) → exp4_v2
                                              │                    ↑
                                    roi_candidates.json ──→ build_train_jsonl → train_controlnet
```

- **단일변수 비교 원칙**: arm 간 차이는 ROI 선택(random vs aroma) 또는 생성법(copy_paste vs controlnet)만. 배치·blend·GT mask·normal 풀은 고정.
- **분석 진입 추천 순서**: `roi_selection.py score_roi/_compat_score` (점수 재설계 시) → `generate_defects.py _paste_and_finalize` (합성 품질) → `exp4_v2_supervised_detection.py` cap 로직 (평가 해석).
