# 05 — Stage 3: ControlNet 학습 & 결함 생성

> **Claude 요약:** AROMA 파이프라인에서 가장 큰 스테이지. step4에서 (a) `build_train_jsonl.py`로 실 결함 crop→hint/prompt 학습쌍을 만들고 (b) `train_controlnet.py`로 SD v1.5 + ControlNet(canny init)을 데이터셋별 차등 하이퍼로 fine-tune하며 (c) `compat_gate`의 τ(및 aitex AR/텍스처 임계)를 CPU로 **사전스캔**한다. step5에서 `generate_defects.py --method controlnet`이 이 학습본·τ를 소비해 SGM symmetric 게이트 + clean-bg 게이트 + seamless 블렌딩으로 AROMA arm 합성물을 만들고, `generate_random.py`가 동일 clean-bg 게이트로 통제군(random arm)을 만든다. **τ·seed·epochs는 사후 튜닝 금지**, ControlNet 생성 경로는 `--local_staging` 미사용(sidecar 캐시 Drive 직결 필요).

---

## 목적

step5(생성)가 소비하는 **모든 생성 선결물**을 step4에서 확정하고, step5에서 그 확정값으로 AROMA arm / random arm 합성물을 생성한다.

- **ControlNet 학습본** — 실 결함의 [hint → 결함 이미지] 매핑을 학습해 `controlnet_models/{ds}/best_model` 산출.
- **compat 게이트 τ** — symmetric 스케일·64px 타일링-aware로 데이터셋별 `ds_tau`를 CPU 사전스캔.
- **합성물** — AROMA arm(`synth_aroma/{ds}`)·random arm(`synth_random/{ds}`). 다운스트림 exp4v2/exp3/exp5/exp6가 소비.

실행 순서 체인: `phase0 → step1 → step2 → step3(roi) → **step4(CN 학습 + τ 사전스캔)** → step5(생성) → exp*`. step4는 step0(profiling)·step3(roi_candidates)를 입력으로 쓰므로 step3 뒤·step5 앞.

---

## 입력 / 출력

`S(stage, ds)` = `$AROMA_OUT/sym_final/{stage}/{ds}` (stage-first 규약). `CN_MODELS` = `sym_final/controlnet_models`.

| 구분 | 항목 | 경로 |
|------|------|------|
| 입력 | profiling 산출 | `S('profiling',ds)/morphology_features.csv`, `context_features.csv`, `recommended_config.yaml`, `compatibility_matrix.json`(**`matrix_symmetric` 키 필수**) |
| 입력 | roi 산출 | `S('roi',ds)/roi_candidates.json`, `roi_selected.json`, `clean_bg_selected.json`(step3.5) |
| 출력 (step4a) | CN 학습 데이터 | `S('cn_data',ds)/train.jsonl`, `targets/`, `hints/` |
| 출력 (step4b) | ControlNet 학습본 | `CN_MODELS/{ds}/best_model/` (+ `validation/step_*`, `checkpoint-*`) |
| 출력 (step4c) | compat 게이트 임계 | `S('compat_gate',ds)/compat_tau_prescan_{ds}.json`(키 `ds_tau`); (aitex) `ar_tex_prescan_aitex.json`(`ar_threshold`·`tex_threshold`) |
| 출력 (step5) | AROMA arm 합성물 | `S('synth_aroma',ds)/` + `annotations.json` |
| 출력 (step5) | random arm 합성물 | `S('synth_random',ds)/` + `annotations.json` |

**데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex = tiled(256×256/stride128, single-class).

---

## step4a: build_train_jsonl.py

학습(4b)과 생성(step5)이 **같은 hint/prompt 생성기**(`utils/hint_generator.py`·`utils/prompt_generator.py`)를 쓰도록 학습쌍을 빌드한다. pool = `morphology_features.csv` 각 행(1행 = 실 결함 1개). 결함별로 crop(target) → 3채널 hint → prompt → `train.jsonl` 한 줄. CPU.

```python
for ds in DATASETS:
    os.environ['DS']      = ds
    os.environ['PROF']    = S('profiling', ds)
    os.environ['ROI']     = S('roi', ds)
    os.environ['CN_DATA'] = S('cn_data', ds)
    !python $AROMA_SCRIPTS/build_train_jsonl.py \
        --morphology_csv   $PROF/morphology_features.csv \
        --roi_candidates   $ROI/roi_candidates.json \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --output_dir       $CN_DATA \
        --style            technical
```

| 인자 | 값 | 역할 |
|------|-----|------|
| `--morphology_csv` | `morphology_features.csv` | 결함 pool. 행별 metrics(linearity/solidity/aspect_ratio 등)가 hint R채널 + prompt에 투입 |
| `--roi_candidates` | `roi_candidates.json` | `(image_id, defect_mask_path)`로 join → `defect_subtype`(morph_label), `stability_score`(ctx_prior MAX) |
| `--context_features` | `context_features.csv` | 이미지 defect-patch 평균을 bin_edges로 binning → `background_type`(directional/complex_pattern/smooth) 데이터 유도 |
| `--config` | `recommended_config.yaml` | `context.features` + `context.bin_edges`(임계값을 리터럴 대신 데이터에서 로드) |
| `--output_dir` | `S('cn_data',ds)` | `targets/`(bbox crop) + `hints/`(3채널) + `train.jsonl` |
| `--style` | `technical`(고정) | 결정론적 prompt 생성(random.choice 없음) — 학습·생성 parity |

- `train.jsonl` 스키마: `{"target": <abs PNG>, "hint": <abs PNG>, "prompt": <str>, "negative_prompt": <str>, "source": <abs>}`. 학습 시엔 `target/hint/prompt`만 읽음(negative/source는 legacy 기록 필드).
- **관찰 로그**: `background_type_hist`가 출력됨. `background_type collapsed to a single category` WARN이 뜨면 per-defect 신호가 없다는 뜻 — 그대로 진행하되 기록(학습·생성 동일 분포이므로 parity는 유지). `context join collision` WARN은 하나의 image_id 버킷이 여러 물리 이미지를 평균했다는 관찰 정보(출력엔 영향 없음).
- 확인: 각 `S('cn_data',ds)/train.jsonl` 라인 수를 출력해 0이 아닌지 확인(0이면 `RuntimeError`로 하드-페일).

---

## step4b: train_controlnet.py

**위치는 `scripts/` 루트** — `/content/AROMA/scripts/train_controlnet.py` (`$AROMA_SCRIPTS`(scripts/aroma) 아님). GPU 필수(Colab Pro A100 권장), 데이터셋당 1세션 권장.

### 백본 / 학습 구조 (스크립트 근거)

- **Base**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`). VAE·UNet·CLIP text encoder는 **frozen**(`requires_grad_(False)`), weight_dtype(fp16)로 로드.
- **ControlNet**: `lllyasviel/sd-controlnet-canny`를 초기 가중치로 로드(없으면 `ControlNetModel.from_unet`). **항상 fp32로 로드·학습**(fp16 로드 시 gradient NaN). AMP autocast는 frozen forward만 fp16 — ControlNet forward는 autocast 외부 fp32(수치 안정화, FIX #8).
- **Conditioning**: 3채널 hint(R=defect shape/mask, G=structure, B=texture) → `controlnet_cond`.
- **Loss**: DDPM 노이즈 예측 MSE. `prediction_type`이 `epsilon`이면 target=noise, `v_prediction`이면 velocity. **Min-SNR-γ 가중**(`--snr_gamma` 기본 5.0) 적용. `--gray_loss_lambda`는 기본 0.0(비활성 — VAE latent 4채널은 RGB에 직접 대응하지 않아 채널 일관성 제약이 잘못된 가정).
- **Optimizer**: AdamW, `--learning_rate` 1e-5, `--lr_scheduler cosine`, warmup은 총 step의 5%로 자동 클램프. `train_batch_size` 1 × `gradient_accumulation_steps` 4 = effective batch 4. `--resolution` 512.
- **Sanity**: 학습 전 데이터 검증 + 1-step forward/backward NaN 검증(resume 시 생략). NaN loss/grad는 스킵·adaptive LR 감소(연속 10회 초과 시 abort).
- **저장**: 매 epoch avg loss가 best 갱신 시 `best_model/` 저장. `--checkpointing_steps 500`마다 `checkpoint-{step}/`(controlnet + optimizer/lr_scheduler + global_step.txt). `--validation_steps 200`마다 `validation/step_*/`에 생성 육안 샘플.

### 공통 명령 (플래그 고정)

```python
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs <E> [--augment] [--no_force_grayscale_target] \
    --early_stopping_patience <P> \
    --checkpointing_steps 500 \
    --save_optimizer_state --save_fp16 \
    --resume_from_checkpoint latest
```

| 인자 | 값 | 역할 |
|------|-----|------|
| `--data_dir` | `S('cn_data',ds)` | step4a 산출(train.jsonl + targets/hints) |
| `--output_dir` | `$CN_MODELS/$DS` | `best_model/`·`checkpoint-*`·`validation/` |
| `--mixed_precision` | `fp16` | frozen forward만 AMP fp16(ControlNet은 fp32 유지) |
| `--gradient_checkpointing` | on | ControlNet 활성화 → VRAM 절감 |
| `--num_train_epochs` | ds별 차등 | 아래 표 |
| `--augment` | ds별 차등 | hflip 50% + brightness/contrast jitter 0.9~1.1(target·hint 동일 flip) |
| `--no_force_grayscale_target` | leather만 | grayscale 강제 해제(기본 ON: R==G==B로 컬러 아티팩트 방지) |
| `--early_stopping_patience` | 20 / leather 30 | N epoch 연속 best loss 미개선 시 종료 |
| `--checkpointing_steps` | 500 | 재개용 체크포인트 주기 |
| `--save_optimizer_state` | on | optimizer/lr_scheduler 저장(재개 필수, +~1.4GB/ckpt) |
| `--save_fp16` | on | 가중치 fp16 저장(디스크 ~50% 절감) |
| `--resume_from_checkpoint` | `latest` | 세션 끊김 시 동일 셀 재실행 → 마지막 checkpoint에서 재개 |

### 데이터셋별 차등 (규약 — 사후 변경 금지)

| 데이터셋 | epochs | `--augment` | grayscale target | patience |
|---------|--------|-------------|------------------|----------|
| **severstal** | 100 | OFF | 기본 ON (강판 grayscale) | 20 |
| **mtd** | 150 | **ON** | 기본 ON (자성타일 grayscale) | 20 |
| **aitex** (tiled) | 150 | **ON** | 기본 ON (직물 grayscale — mtd 준용) | 20 |
| **mvtec_leather** | 300 | **ON** | **OFF** (`--no_force_grayscale_target`, 컬러 가죽) | 30 |

> 산출: `$CN_MODELS/{ds}/best_model/`(step5 선결 assert). `ControlNetModel.from_pretrained`로 로드 가능. severstal은 crop 수천 개로 최장 → 단독 세션 권장.

### validation 육안 게이트

전량 생성(step5) 전에 각 데이터셋 `validation/step_*/`의 최신 생성 샘플을 **육안으로 한 번** 본다. 결함 텍스처가 배경과 이질적이지 않고 그럴듯하면 통과. leather가 끝내 비현실적이면 학습 하이퍼(augment/epoch) 재조정 후 재학습이 원칙, 그래도 안 되면 step5에서 "controlnet arm 부적합"으로 정직 보고.

---

## step4c: τ 사전스캔 (compat_gate)

step5의 `generate_defects --compat_mode symmetric --compat_threshold <τ>`가 소비하는 τ를 **여기서 확정**한다(생성 직전 재발명 금지). GPU-의존 게이트 임계를 CPU에서 데이터셋별 발동률로 사전 확정하는 것이 핵심 — 임계를 눈으로 보고 맞추면 사후 튜닝이 된다. CPU.

**왜 사전스캔인가**
- symmetric 스케일은 legacy 확률 스케일과 다르므로 τ=0.5(legacy 중립값)를 그대로 쓰면 안 됨 → **τ=0.5 금지**.
- `matrix_symmetric`은 per-cluster max-norm으로 [0,1]. good 배경에 실 crop을 uniform 배치(rescale-to-fit 재현)하고 **generate_defects 게이트와 동일한 64px 타일링 + mean-compat**(`gd._tile_anchors`·`_COMPAT_TILE=64`·`AGG=mean`)를 재현해, cluster별 tiled mean-compat 분포의 **목표 reject율 R=0.25 percentile**을 τ로 잡는다.
- 채택값 R=0.25는 **수치를 보기 전에 고정**. per-cluster verdict가 OK인 τ들의 median을 `ds_tau`로 저장.

**출력**: `S('compat_gate',ds)/compat_tau_prescan_{ds}.json`(키 `ds_tau`). 전제로 신 profiling(`matrix_symmetric`)이 필요 — 구 profile은 hard-fail.

**verdict 판정** (per-cluster 종합)

| 관측 | 판정 | 조치 |
|------|------|------|
| 다수 OK, `ds_tau` ∈ (agg_min, median), τ≠0.5 | 확정 성공 | step5에서 `--compat_threshold <ds_tau>` |
| 다수 DEGENERATE | tiled-compat 한 점 집중(무판별) | 해당 DS 게이트 무의미 → 착수 보류 |
| 다수 τ≈0.5 | neutral 질량 지배 | patch-gran support·profiling 재점검 |

**aitex 전용 추가 사전스캔** — elongated 결함이 많아 ControlNet 512² squash 왜곡이 생기므로 AR/텍스처 게이트 임계를 CPU로 예측:
- **AR 게이트**: `roi_selected.json`의 `defect_bbox`만으로 완전 결정 — `max(w,h)/min(w,h)` 분포에서 fallback율 ≤ 50%(이상 ≤ 30%) 되는 최소 임계(`AR_T`). AR 초과 ROI는 copy_paste 폴백(개수·bbox parity 유지).
- **텍스처 게이트**: 소스 결함 주변 descriptor와 paste patch descriptor 거리 분포에서 거부율 10~50% 밴드의 임계(`TEX_T`). 0% = 무의미, 80%+ = fallback 폭주.
- 확정값을 `ar_tex_prescan_aitex.json`(키 `ar_threshold`·`tex_threshold`)에 저장 → step5가 파일로 소비. (과거 임계 2.5에서 98% fallback 실측 교훈 — 무리하게 임계를 올려 왜곡물을 허용하는 것은 pilot 육안 통과 없이 금지.)

> 게이트 재현·SGM·patch-granularity 상세는 [[09-Compatibility-Gate]] 참조.

---

## step5: generate_defects.py (AROMA arm)

step4 확정값(CN 학습본·τ·aitex AR/TEX)을 **읽어서 소비만** 한다(재사전스캔 금지). GPU 필수. `--method controlnet` + `--compat_mode symmetric`.

```python
for DS in DATASETS_GEN:
    os.environ['DS']=DS; os.environ['PROF']=S('profiling',DS); os.environ['ROI']=S('roi',DS)
    os.environ['NORMAL']=normal_dir(DS); os.environ['OUT']=S('synth_aroma',DS)   # ← 반드시 이 경로
    os.environ['COMPAT']=f"{S('profiling',DS)}/compatibility_matrix.json"
    os.environ['TAU']=str(TAU_BY_DS[DS])                                          # step4c 확정 τ
    if   DS=="mvtec_leather": os.environ['EXTRA']="--cn_no_grayscale"
    elif DS=="aitex":         os.environ['EXTRA']=f"--cn_ar_threshold {AR_T} --texture-dist-threshold {TEX_T}"
    else:                     os.environ['EXTRA']=""
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir     $ROI \
        --normal_dir  $NORMAL \
        --output_dir  $OUT \
        --method      controlnet \
        --controlnet_path  $CN_MODELS/$DS/best_model \
        --morphology_csv   $PROF/morphology_features.csv \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --n_per_roi 3 --seed 42 --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --compat_mode symmetric --compat_threshold $TAU \
        --compat_matrix_json $COMPAT \
        $EXTRA
```

| 인자 | 값 | 역할 |
|------|-----|------|
| `--roi_dir` | `S('roi',ds)` | `roi_selected.json` + `clean_bg_selected.json` 자동 로드 |
| `--normal_dir` | `normal_dir(ds)` | clean 배경 pool (aitex → `aitex_tiled/train/good`) |
| `--output_dir` | **`S('synth_aroma',ds)`** | exp*가 `--aroma_synthetic_dir`에 `/{ds}`를 붙이므로 이 경로 필수 |
| `--method` | `controlnet` | ControlNet 생성 경로(`--controlnet_path`·morphology·context·config 필수) |
| `--controlnet_path` | `$CN_MODELS/$DS/best_model` | step4b 학습본 |
| `--morphology_csv`·`--context_features`·`--config` | profiling 산출 | hint/prompt conditioning join (CN 경로 필수) |
| `--n_per_roi` | 3 | ROI당 변형 수 |
| `--seed` | 42 | 결정론 |
| `--blend_mode` | `seamless` | Reinhard local-bg 색 이전 + `cv2.seamlessClone`(cv2 없으면 alpha 폴백) |
| `--reject-clean-bg` | on | 검은/평탄(void) 배경 거부 게이트 |
| `--min-bg-quality` | 0.7 | clean-bg 게이트 품질 하한(CASDA 값) |
| `--bg-blur-threshold` | 100.0 | Laplacian-variance 블러 임계(CASDA 값) |
| `--compat_mode` | `symmetric` | SGM `matrix_symmetric` 사용(없으면 hard-fail) |
| `--compat_threshold` | `$TAU`(=`ds_tau`) | placement 게이트 τ(step4c 확정, τ=0.5 금지) |
| `--compat_matrix_json` | `compatibility_matrix.json` | compat row 소스(bin_edges도 포함) |
| `--cn_no_grayscale` | **leather만** | grayscale 강제 해제(모델이 `--no_force_grayscale_target`로 학습됨) |
| `--cn_ar_threshold`·`--texture-dist-threshold` | **aitex만** | step4c 확정 `AR_T`/`TEX_T`(elongated 왜곡 방지) |

- CN 기본값(스크립트): `--cn_steps 30`, `--cn_cond_scale 0.7`, `--cn_resolution 512`(학습과 일치), `--cn_ar_threshold 2.5`(기본), `--cn_grayscale` 기본 ON. AR-gated ROI는 기본적으로 `copy_paste_arfallback`으로 폴백(개수·클래스·bbox parity 유지) — `--cn_no_ar_fallback`으로 순수 skip 가능.
- **활성 확인 로그**: `clean_bg assignment ON`(step3.5 배경 소비 — 안 뜨면 legacy fallback으로 배경 분포가 달라져 공정성 저하, 생성 금지) / `clean_bg resolve used≈total, fallback·mismatch≈0` / `compat gate ON: threshold=… mode=symmetric` / `placement-gate stats: fallback=M%`(>50%면 τ 과대 의심 → step4c 재확인, step5 튜닝 금지) / (aitex) `controlnet stats`(gen_ok/ar_fallback/blank_rate) + `texture-gate stats`.

---

## step5: generate_random.py (random arm / 통제)

무변경 통제군. AROMA arm과 **동일 clean-bg 게이트**·동일 `top_k`/`n_per_roi`/`seed`. positive placement·compat 게이트는 미적용. CPU라 `--local_staging` 사용 가능.

```python
for DS in DATASETS:
    os.environ['DS']=DS; os.environ['ROI']=S('roi',DS)
    os.environ['NORMAL']=normal_dir(DS); os.environ['OUT_R']=S('synth_random',DS)   # ← exp*가 /{ds} 붙임
    !python $AROMA_SCRIPTS/generate_random.py \
        --candidates_json $ROI/roi_candidates.json \
        --normal_dir      $NORMAL \
        --output_dir      $OUT_R \
        --top_k 200 --n_per_roi 3 --seed 42 \
        --local_staging \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

| 인자 | 값 | 역할 |
|------|-----|------|
| `--candidates_json` | `roi_candidates.json` | 공유 후보 pool에서 random 선택(내부에서 `roi_selected.json` staging) |
| `--normal_dir` | `normal_dir(ds)` | 배경 pool |
| `--output_dir` | **`S('synth_random',ds)`** | exp*가 `--random_synthetic_dir`에 `/{ds}` 붙임 |
| `--top_k` | 200 | 선택 ROI 수(aroma와 동일값이어야 공정) |
| `--n_per_roi` | 3 | ROI당 변형 수(aroma와 동일) |
| `--seed` | 42 | 결정론 |
| `--reject-clean-bg`·`--min-bg-quality 0.7`·`--bg-blur-threshold 100.0` | on | **AROMA와 대칭인 clean-bg 게이트**(비교 공정성) |
| `--blend-mode` | 기본 alpha | aroma가 seamless면 `--blend-mode seamless`로 맞춤(대칭) |
| `--min_quality` | 0(OFF) | aroma `roi_selection --min_quality`와 동일값 지정 시 동일 게이트 풀 |

> random은 개선(positive placement·compat)과 무관하지만 clean-bg 게이트·blend_mode는 AROMA와 대칭이어야 exp4v2 대조가 공정하다.

---

## synth pool sizing

합성 pool 크기 = `top_k`(선택 ROI 수) × `n_per_roi`(ROI당 변형). exp4v2가 실제 쓰는 cap = `real_train × synth_ratio`(`real_train = real_defect × (1−val_frac)`, 기본 val_frac 0.3). **pool이 cap보다 작으면 원하는 비율에 못 미침** → 목표 = `real_train × 최대 ratio × 1.2`(20% 헤드룸)로 오버슈트 생성 후 exp4v2에서 seed 결정론 subsample로 비율만 스윕(재생성 불필요).

| 데이터셋 | 후보 수 | real_train@0.3 | ratio 1.0 cap | 권장 top_k | 권장 n_per_roi | 상태 |
|---------|--------:|--------------:|--------------:|-----------:|--------------:|------|
| severstal | 266,624 | 2,534 | 2,534 | **1,000** | 3 | **유일하게 재생성 필요**(현재 400 → 6.3× 부족) |
| mtd | 16,877 | 271 | 271 | 200(유지) | 3 | 충분 |
| aitex | 6,168 | 246 | 246 | 200(유지) | 3 | 충분(1.6×) |
| mvtec_leather | 968 | 64 | 64 | 200(유지) | 3 | 9× 여유(후보 적어 top_k↑ 여지 없음) |

- **top_k↑가 1순위**(서로 다른 crop = 진짜 다양성), n_per_roi↑는 2순위(같은 crop 재배치 → near-duplicate 위험).
- ⚠️ **`--pool_k ≥ n_per_roi` 명시**: aroma 생성 루프가 `_cbg_pairs[rep_idx % len(pool)]`로 배경을 소비하므로 `n_per_roi > pool_k`면 같은 배경 재사용(위치 지터만). leather는 배경 18종 collapse 전례가 있어 특히 중요.
- severstal 재생성은 **cascade** 필요: roi_selection(top_k 1000) → clean_bg_selection(pool_k 6, `--emit_random_arm`) → generate_defects(copy_paste) → generate_random. (2)를 생략하면 `clean_bg resolve mismatch` 급증. arm 총량 parity·출력 수 ≥ 2,534 확인.

---

## 핵심 계산 로직

step5 파이프라인(ROI → 합성 이미지):

1. **clean-bg 게이트** — normal pool에서 검은/평탄(void) 배경 거부. 품질 0..1 ≥ `min_bg_quality`(0.7) AND Laplacian-variance ≥ `bg_blur_threshold`(100.0). step3.5 사전선정 `clean_bg_selected.json` 소비(없으면 legacy 생성-시점 재스캔 fallback → 배경 분포 달라짐, 회피).
2. **compatibility 게이트(placement)** — paste 위치 후보를 64px 타일로 나눠 각 타일 context cell의 `matrix_symmetric` compat를 mean-aggregate. cluster별 mean-compat ≥ τ인 위치만 수락(positive placement: scan·rank·place). unobserved cell = 0.5 neutral. τ 과대 시 fallback 폭주. 상세 [[09-Compatibility-Gate]].
3. **ControlNet 추론** — hint(R=shape/G=structure/B=texture) + technical prompt → SD v1.5 + ControlNet(canny-init 학습본), `cn_steps 30`, `cn_cond_scale 0.7`, 512². grayscale 강제(leather 제외). aitex는 AR 초과 ROI → copy_paste 폴백(`copy_paste_arfallback`).
4. **블렌딩(seamless)** — Reinhard local-bg 색 통계 이전 후 `cv2.seamlessClone`으로 경계 봉합(cv2 없으면 alpha feather 폴백).

random arm은 1(clean-bg 게이트)만 공유하고 2·3·4의 개선은 미적용.

---

## 주의사항

- **`--local_staging`은 ControlNet 생성에 미사용**: sidecar 캐시(`.meta.json`)가 Drive 직결이어야 세션 재개 시 살아남는다. CPU 경로(copy_paste·random·complexity)에는 사용 가능. 학습(4b)도 output_dir을 Drive 직결로 유지.
- **사후 튜닝 금지**: τ·seed·n_per_roi·epochs·augment·grayscale·blend/게이트 설정은 확정값을 그대로 쓰고 **결과 보고 후 변경하지 않는다**. fallback 과다여도 step5에서 임계를 손대지 말고 step4 사전스캔을 재검.
- **τ=0.5 금지**: symmetric 스케일에서 0.5는 legacy 중립값. step2 assert가 `0.0 < ds_tau < 0.5`를 강제.
- **세션 끊김 → 동일 셀 재실행**: 학습은 `--resume_from_checkpoint latest`, 생성은 sidecar 캐시로 이어서(재-GPU 스킵). severstal은 최장이라 단독 세션 권장.
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효. aitex `ar_fallback` 비율을 반드시 기록(ControlNet 생성 비중 = `1 − ar_fallback비율`).
- **step5 재생성 시 exp5/exp6 임베딩 캐시 무효화**: 동일 파일명으로 내용만 덮어쓰면 경로-기반 DINOv2 캐시가 stale 재사용 → 재합성 데이터셋은 exp5/exp6 전 `rm -rf $EMBED_CACHE_DIR/{ds}`.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md) — 검증은 Colab 실행으로. 시간 실측·처리량 벤치 미수행(load-test policy).

---

## 관련 노트

[[00-INDEX]] | [[04-Stage2-ROI-Selection]] | [[06-Experiments]] | [[07-Scripts-Reference]] | [[09-Compatibility-Gate]]
