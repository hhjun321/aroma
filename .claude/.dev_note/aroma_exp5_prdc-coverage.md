# Exp5 — PRDC 커버리지 평가 (외부 임베딩 좌표계, 저비용 L2 증거)

## (사용할 skills: feature-dev)

> 계열: [[aroma_research-core_thesis-and-compounding]], low_compute_validation_plan.md **1순위(PRDC)**. 신규 스크립트 + Colab 가이드 → feature-dev.
> 후속 공유: 2순위 kNN test-coverage·3순위 rare-mode가 **동일 임베딩 캐시**를 재사용하므로 임베딩 추출부는 캐시 우선으로 설계(신규 exp 추가 없이 확장 가능하게).

## 개요

AROMA의 핵심 주장("부족한 형태×배경 조합을 타겟팅해 훈련 분포의 빈 곳을 메운다")을 **AROMA가 선택에 사용하지 않는 외부 feature 좌표계(DINOv2)** 에서 반증 가능한 형태로 검증한다. exp2의 순환성(자체 cluster/cell 라벨 위 지표)과 n=1 약점을 동시에 해소하는 L2 증거.

**사전 등록 가설 (반증 가능)**: aroma/random은 **동일 copy-paste 엔진**이므로 —
- **Precision / Density: 두 조건 동등** (fidelity는 엔진이 결정)
- **Recall / Coverage: aroma > random 유의** (선택 전략만이 분포 커버를 바꿈)

Recall만 오르고 Precision이 동등해야 "선택의 가치"가 입증된다. 둘 다 오르거나 Precision이 깨지면 가설 기각 — 이 비대칭 예측이 사후합리화 반박을 차단한다.

**자원**: feature 추출 T4 데이터셋당 1–3분(캐시 후 재사용), PRDC·permutation은 CPU. 전체 30분 이내.

---

## 설계 (workflow 비판 반영 — load-bearing 3건)

### 1. 기준(reference) 분포 = held-out real 결함 패치 (leakage 차단)

- 합성 crop은 소스 ROI의 근사 복사본 → 소스로 쓰인 결함이 reference에 들어가면 Recall이 자명하게 높아짐(무의미).
- **reference = exp4v2와 동일 규약의 val split** (`_split_defects` 로직·`val_frac=0.3`·seed 동일) — downstream(L3)과 기준 정렬. synth 소스는 train split ROI에서만 나오므로 held-out 성립.
  - 구현: exp4v2의 `_split_defects`를 import 또는 동일 로직 복제(mask 보유 결함의 seeded split). 결과 JSON에 reference 구성(`n_ref`, split seed) 기록.
- real 패치 = GT mask → bbox crop (exp3 `_load_real_defect_patches` 재사용).

### 2. 표본 동일화 + k sensitivity

- 조건 간 **n 엄격 동일화**: `n = min(n_aroma, n_random)` 로 seeded subsample (PRDC는 표본 수에 민감).
- `--nearest_k 3 5 10` — k별 전 지표 보고(단일 k cherry-pick 금지). 주 보고 k=5.

### 3. 유의성 = permutation test (bootstrap-Wilcoxon 금지)

- 두 synth 임베딩 풀을 합쳐 **무작위 재분할 1000회** → Δ(Recall), Δ(Coverage)의 null 분포 → empirical p = (r+1)/(K+1).
- Precision/Density는 **동등성**을 주장하므로 TOST(또는 최소한 permutation CI가 사전 정의한 동등 마진 ±δ 내) — δ는 구현 중 확정(TODO), 우선 |Δ|의 permutation 95% CI 보고.

---

## 수정 내용

### 1. 신규 `scripts/aroma/experiments/exp5_prdc.py`

구성 (exp3_generation_quality.py를 템플릿으로):

- **입력 로딩**: exp3의 `_get_image_lists`/`_resolve_masks_generic`/`_load_real_defect_patches` 재사용(import 가능하면 import, 경로 문제 시 최소 복제 — exp3·exp4v2 간 기존 관례 확인 후 결정). synth 패치 = `{dir}/{ds}/annotations.json`의 `image_path`(+`mask_path` 있으면 bbox crop, 없으면 `bbox` 필드 — severstal annotation에 bbox 존재 확인됨).
- **임베딩**: DINOv2 ViT-S/14 (`torch.hub` facebookresearch/dinov2, GPU 권장·CPU 가능). 실패 폴백 InceptionV3(torchvision) — 사용 백본을 결과 JSON에 기록.
  - **Drive 캐시**: `--embed_cache_dir` → `{cache}/{ds}/{split_tag}_{backbone}.npy` (+ 파일 목록 manifest로 무효화 판정). 2·3순위 실험이 그대로 재사용.
- **PRDC 계산**: `prdc` 패키지(`pip install prdc`, Kynkäänniemi P&R + Naeem D&C 구현) — 자체 구현 금지.
- **permutation**: numpy 기반 1000회(`--permutation_reps`), seed 고정.
- **출력**: `exp5_prdc_results.json` — `{ds: {k: {aroma: {precision,recall,density,coverage}, random: {...}, delta: {...}, p_perm: {...}, ci95: {...}}, meta: {n, n_ref, backbone, split_seed}}}` + `exp5_prdc_summary.md`(k=5 주표 + k-sensitivity 부표 + 가설 판정 행).

CLI (exp3 스타일):
```
--real_data_dir --aroma_synthetic_dir --random_synthetic_dir
--dataset_keys severstal mvtec_leather aitex mtd
--val_frac 0.3 --split_seed 42        # reference split (exp4v2 규약 정렬)
--nearest_k 3 5 10 --permutation_reps 1000
--backbone dinov2_vits14              # 폴백 inception_v3
--embed_cache_dir $AROMA_OUT/embed_cache
--output_dir $AROMA_OUT/exp5_prdc
--seed 42 --device cuda
```

### 2. 신규 `AROMA연구분석/colab_execute/exp5_execute.md`

house workflow 형식(STEP 0 env → 1 설치(`pip install prdc`) → 2 실행 → 3 결과 확인/판정 기준). 판정 기준 명시: Recall·Coverage p_perm<0.05(4종 방향 일치) AND Precision·Density CI가 동등 마진 내.

---

## 수정 대상 파일

- **신규** `scripts/aroma/experiments/exp5_prdc.py`
- **신규** `AROMA연구분석/colab_execute/exp5_execute.md`
- (기존 무수정 — exp3는 import 소스로만)

---

## 암묵적 요구사항 (엣지)

- **mvtec_leather aroma 합성 부재**(현재): 해당 조건 skip + 명시 로그(silent 금지). 합성 생성 후 재실행.
- **n_ref 부족**(aitex val 17장 수준): `prdc`는 소표본에서 불안정 → `n_ref < 30`이면 `unstable: true` 플래그 + 해석 경고(수치는 보고).
- **synth mask 없는 annotation**: `bbox` 필드 폴백, 둘 다 없으면 전체 이미지 crop 대신 **skip + 카운트**(전체 이미지는 배경이 지배해 임베딩 오염).
- **패치 크기 이질성**: 백본 입력 리사이즈(DINOv2 224/InceptionV3 299) 고정 — exp3 `_resize_to_tensor` 관례.
- **재현성**: subsample·permutation·split 전부 seed 고정, 결과 JSON에 기록.
- **fp16/normalize**: DINOv2 임베딩 L2-normalize 여부는 PRDC 표준(비정규 유클리드)을 따름 — 문헌 기본값 유지, 변경 시 명시.

---

## 테스트 (Colab, pytest 금지)

1. 로컬 `py_compile`.
2. Colab 스모크: severstal 단독 + `--permutation_reps 100` → JSON 스키마·`n == n_aroma == n_random`·k 3종 존재 확인.
3. **sanity(음성 대조)**: 동일 synth 풀을 반으로 갈라 aroma/random 자리에 넣으면 Δ≈0, p 균등분포 근처 — permutation 구현 검증.
4. 전체 4종(leather는 합성 후) 실행 → 가설 판정표.

---

## 미확정 (TODO — 구현 중/실측 후 확정)

1. **동등 마진 δ**(Precision/Density TOST): 문헌 관례 부재 → 1차 실측 CI 폭을 보고 후 확정(우선 CI 보고 방식).
2. **exp3 재사용 방식**: import vs 복제 — exp3가 스크립트 직접실행 구조라 import 부작용(로깅/argparse) 확인 후 결정.
3. **DINOv2 Colab 캐시**: torch.hub 다운로드가 세션마다 반복되면 Drive에 weight 캐시 추가 여부.
