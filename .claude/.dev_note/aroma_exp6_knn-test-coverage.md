# Exp6-knn — 임베딩 kNN test-coverage (기제 증거, CPU급)

## (사용할 skills: feature-dev)

> 계열: [[aroma_exp5_prdc-coverage]](임베딩 캐시 공유), low_compute_validation_plan.md **2순위**. [[aroma_exp6_rare-mode-coverage]]와 **단일 스크립트 `exp6_embedding_coverage.py`(`--mode knn|rare`)** 공유 — exp3의 `--mode fid/ad` 선례.

## 개요

copy-paste 증강이 검출기를 돕는 **기제**("테스트 분포에 가까운 결함 밀도 증가")를 지표와 1:1 대응으로 측정한다. held-out val의 각 real 결함 crop에 대해, 세 학습 풀까지의 **최근접 cosine distance**를 비교:

| 풀 | 구성 |
|----|------|
| `real` | real train 결함 crop만 (baseline 상당) |
| `real+random` | + random synth crop |
| `real+aroma` | + aroma synth crop |

**가설**: 1-NN distance 분포가 `real+aroma`에서 유의하게 좌측 이동(min over pool이므로 단조 감소는 자명 — **비교 대상은 random 대비 aroma의 추가 감소분** Δ(1-NN dist)). coverage curve(반경 r 이내 val 비율, r 스윕)도 병기.

**자원**: exp5 임베딩 캐시 재사용 시 **CPU only 수 분**. 신규 임베딩은 real **train** crop뿐(exp5는 val만 캐시 — train tag 추가).

## 설계 (workflow 비판 반영 — load-bearing 3건)

1. **검정 = 이미지 단위 clustered bootstrap** (인스턴스 단위 Mann-Whitney 금지): 같은 이미지의 crop들은 독립이 아니므로, val **이미지**를 재표집 단위로 bootstrap(기본 2000회) → Δmean(1-NN dist)의 CI·p. severstal처럼 이미지당 다중 결함인 경우 필수.
2. **diversity 항 병기**: 중복/노이즈 합성이 coverage를 부풀리는 것 차단 — synth 풀 내 pairwise distance 요약(mean/median) + distinct-source 수를 같은 표에 보고. aroma가 coverage↑ + diversity↓(뭉침)면 정직하게 드러남.
3. **정보 분리 명문화**: "AROMA 선택은 val/test를 보지 않는다"(선택 입력 = train 결함 프로파일링) — leakage 오해 차단 문구를 결과 md와 논문에 기술.

### 지표 정의
- 주 지표: `d1(v, P) = min_{p∈P} cosine_dist(v, p)` — val crop v, 풀 P. 요약 = mean/median over val.
- Δ 보고: `Δ = mean_d1(real+random) − mean_d1(real+aroma)` (>0 = aroma가 더 가깝게 커버).
- coverage curve: `C_P(r) = |{v: d1(v,P) ≤ r}| / n_val`, r = real 풀 d1 분위수 grid → AUC 요약.
- **cosine distance** (PRDC의 유클리드와 다름 — retrieval 관례) — L2-normalize 후 1−cos. 결과 meta에 명시.
- **n parity**: synth 풀 크기 동일화(min-n seeded subsample — exp5 `_equalize_n` 재사용). real 풀은 세 조건 공통이라 동일화 불요.

## 수정 내용

### 신규 `scripts/aroma/experiments/exp6_embedding_coverage.py` (knn 모드 부분)

- **로더/임베딩/캐시 = exp5 재사용**: `_load_real_defect_crops`·`_load_synth_crops`·`_build_backbone`·`_embed_crops`·`_cached_embed`·`_split_defects`를 exp5_prdc에서 **import**(exp5는 __main__ 가드 有, top-level 부작용은 logging뿐 — exp3 import와 동일 패턴). 캐시 tag 신규: `real_train{val_frac}s{seed}`.
- crop→이미지 매핑 보존: `_load_real_defect_crops`가 반환하는 crop_ids(`{img_path}|bbox`)에서 img_path 추출 → clustered bootstrap 단위.
- 함수: `_knn_dist(val_feat, pool_feat)`(cosine, 청크 처리), `_clustered_bootstrap(d1_a, d1_r, img_ids, reps, seed)`, `_coverage_curve(...)`, `_diversity(feat)`(pairwise mean/median + distinct source count from crop_ids).
- 출력: `exp6_results.json`의 `knn` 노드 — `{ds: {d1_mean: {real, random, aroma}, delta_AR, ci95, p_boot, coverage_auc: {...}, diversity: {...}, meta}}` + summary md 표.

### CLI (공통 + knn)
```
--mode knn --real_data_dir --aroma_synthetic_dir --random_synthetic_dir
--dataset_keys ... --val_frac 0.3 --split_seed 42 --seed 42
--bootstrap_reps 2000 --backbone dinov2_vits14
--embed_cache_dir $AROMA_OUT/embed_cache --output_dir $AROMA_OUT/exp6
```

### 신규 `AROMA연구분석/colab_execute/exp6_execute.md`
workflow 형식, knn·rare 두 모드 통합 가이드(rare는 [[aroma_exp6_rare-mode-coverage]] 설계 반영).

## 수정 대상 파일
- **신규** `scripts/aroma/experiments/exp6_embedding_coverage.py`
- **신규** `AROMA연구분석/colab_execute/exp6_execute.md`
- (exp5_prdc.py는 import 소스 — 무수정. 단 import 가능성 확인 후 필요시 exp5의 함수를 공용 헬퍼로 승격하는 대신 **exp5에서 직접 import** 유지가 원칙)

## 암묵적 요구사항 (엣지)
- real train crop 0(mask 전멸) → 데이터셋 skip+로그. leather aroma 부재 → 조건 skip(exp5 규약).
- val 이미지 1~2장(극소) → bootstrap 무의미 → `unstable` 플래그.
- d1 동률/0 거리(합성이 소스 crop과 사실상 동일) 가능 — 정상(그게 커버) — 단 **val과 0 거리**면 leakage 신호 → 경고 로그(소스는 train에서만 와야 함).
- 임베딩 재사용 시 backbone 혼용 금지(meta 대조).

## 테스트 (Colab, pytest 금지)
1. `py_compile`.
2. severstal 스모크(`--bootstrap_reps 200`) → JSON 스키마·Δ/CI 존재.
3. **음성 대조**: random 풀 반분을 aroma/random 자리에 → Δ≈0, p 균등.
4. leakage sanity: `min d1(val, synth)`가 0에 근접한 pair 수 로그 확인(0이어야 정상).

## 미확정 (TODO)
1. coverage curve의 r-grid 정의(real 풀 d1의 {10..90} 분위수 제안) — 구현 중 확정.
2. exp5 함수 import 시 순환/부작용 실측 — 문제 시 공용 모듈(`experiments/_embed_common.py`) 분리로 전환(그 경우 exp5도 리팩터 — 별도 결정).
