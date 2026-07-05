# Exp6-rare — 독립 클러스터 rare-mode hit rate (조합 타겟팅 검증, CPU급)

## (사용할 skills: feature-dev)

> 계열: [[aroma_exp5_prdc-coverage]](임베딩 캐시 공유), [[aroma_exp6_knn-test-coverage]](**동일 스크립트** `exp6_embedding_coverage.py --mode rare`), low_compute_validation_plan.md **3순위**.

## 개요

논문의 "부족한 (형태×배경) 조합 타겟팅" 문구와 직결되는 **이산 모드 증거**. AROMA 자체 라벨(cluster/cell)이 아닌 **독립 좌표계(DINOv2 임베딩 k-means)** 로 결함 모드를 정의해 순환성을 제거하고, rare 모드에 대한 선택 hit rate를 **random 재선택 30-seed null 분포** 대비로 검정한다 — exp2의 n=1 약점을 비순환 좌표계에서 해소.

**핵심 비용 이점**: random "재선택"은 `roi_candidates.json` 메타데이터에서 seed만 바꿔 뽑는 CPU 밀리초 연산(재합성·재임베딩 불필요 — 모드 배정은 **소스 real 결함 crop** 임베딩에 이미 있음). synth crop은 소스의 근사 복사이므로 소스 crop으로 모드를 판정하는 것이 타당하고, 30-seed 반복이 사실상 공짜가 되는 근거.

**자원**: 임베딩은 real 결함 crop 전체(train split) — exp5/exp6-knn 캐시와 공유. 클러스터링·null 분포는 CPU 수 분.

## 설계 (workflow 비판 반영 — load-bearing 3건)

1. **JSD-to-real 항 삭제** (원안 기각 사유): "real 빈도 분포에 가깝게"를 성공으로 놓으면 rare **과대표집**이라는 AROMA 목적과 내부 모순. **rare-mode hit rate만** 주 지표로.
2. **sensitivity 필수**: k-means **k ∈ {8,10,12,15} × clustering seed 5개** 전 조합 보고 — 단일 (k,seed) cherry-pick 금지. 판정은 "전 조합에서 방향 일치 + 다수 조합 p<0.05" 수준으로 겸손하게.
3. **test-presence 필터**: rare 모드가 held-out **val 결함에도 등장**하는 모드만 hit-rate 집계에 포함 — "train에서 rare"가 test에서 무의미한 모드일 가능성 차단(downstream 관련성 보강). 필터 전/후 rare 모드 수를 meta로 보고.

### 정의·절차 (데이터셋별, (k, cseed) 조합별)
1. **모드 정의**: train-split real 결함 crop 임베딩에 k-means(k, cseed) → 모드 배정. val crop은 학습된 centroid에 최근접 배정(test-presence 판정용).
2. **rare 모드** = train 모드 빈도 ≤ p25 (하위 사분위) AND val에 ≥1 등장.
3. **hit rate(선택셋 S)** = S의 소스 crop 중 rare 모드 소속 비율. S_aroma = `roi_selected.json`의 (image, defect_bbox) → 해당 crop 임베딩 → 모드.
4. **null 분포**: random 전략을 seed 30개로 재실행(= `roi_candidates.json`에서 균일 n(S_aroma)개 추출 — generate_random.select_random 로직 재사용, budget 동일) → hit rate 30개 → empirical p = (count(null ≥ observed)+1)/31, null 95% interval 병기. **z-score/Stouffer 결합 금지**(정규성 위반) — empirical p만.
5. 참고 지표(부차): 모드 커버리지 수(선택셋이 건드린 distinct 모드 수) — hit rate와 함께 표기(주장에는 미사용).

### 공정성 불변식
- budget parity: null 추출 수 = |S_aroma| 정확 일치.
- 후보 풀 동일: aroma·random-null 모두 동일 `roi_candidates.json`(quality gate를 켠 경우 gated 풀 — [[aroma_exp4v2_quality-gate-fairness]] 대칭 규약과 정합).
- crop 임베딩은 (image_path, defect_bbox) 키로 1회 계산해 aroma/null 공유 — 좌표계 완전 동일.

## 수정 내용

### `scripts/aroma/experiments/exp6_embedding_coverage.py` (rare 모드 부분 — knn 과 공유 골격)

- **입력**: `--roi_dir`(→ `roi_selected.json`+`roi_candidates.json`; 데이터셋별 `$AROMA_OUT/roi/{ds}`) 추가. candidate 엔트리의 `image_path`+`defect_bbox`로 소스 crop 절단(신규 `_load_candidate_crops` — bbox는 roi_selection 산출 스키마 확인 후 규약 고정: TODO 1).
- k-means: sklearn `KMeans(n_clusters=k, random_state=cseed, n_init=10)`.
- 함수: `_assign_modes`, `_rare_modes(train_freq, val_presence)`, `_hit_rate(S, modes)`, `_null_hit_rates(candidates, n, seeds=30)`(균일 무중복 추출 — generate_random.select_random과 동일 로직), `_sensitivity_grid(k_list, cseed_list)`.
- 출력: `exp6_results.json`의 `rare` 노드 — `{ds: {grid: {"k10_s0": {observed, null_mean, null_ci95, p_emp, n_rare_modes, n_rare_after_filter}, ...}, verdict: {direction_consistent, n_sig}, meta}}` + summary md(그리드 히트맵 표).

### CLI (rare 전용 추가)
```
--mode rare --roi_dir_root $AROMA_OUT/roi
--kmeans_k 8 10 12 15 --cluster_seeds 0 1 2 3 4
--null_seeds 30 --rare_quantile 0.25
```

## 수정 대상 파일
- `scripts/aroma/experiments/exp6_embedding_coverage.py` (knn 노트와 동일 파일 — 함께 구현)
- `AROMA연구분석/colab_execute/exp6_execute.md` (통합 가이드)

## 암묵적 요구사항 (엣지)
- rare 모드 0개(필터 후) → 해당 (k,cseed) 셀 `no_rare_modes` 표기(강제 판정 금지).
- `roi_selected.json`의 (image,bbox)가 candidates에 없거나 crop 실패 → skip+카운트(`n_unmatched`) — 크면 스키마 불일치 신호.
- k ≥ train crop 수(소형 데이터셋) → 해당 k skip. aitex처럼 crop 수 적으면(≈70) k=15가 과분할 — sensitivity 표에서 자연 노출.
- KMeans 결정성: `random_state=cseed` 고정, n_init 명시.
- observed가 null 최대값 초과 시 p=1/31이 하한 — "p<0.033 이하 불가"를 해석에 명시(30-seed 한계).

## 테스트 (Colab, pytest 금지)
1. `py_compile`.
2. severstal 스모크(k=10, cseed=0, null 10) → JSON 스키마·p 범위(0<p≤1).
3. **양성 대조**: null 추출을 "rare 모드만 편향 추출"로 바꾼 가짜 S → p≈1/31 근접(검정 방향 확인).
4. 전 그리드(4k×5seed) 실행 시간 확인(데이터셋당 CPU 수 분 목표).

## 미확정 (TODO)
1. **roi_candidates/selected의 bbox 필드명·형식**: `defect_bbox`로 추정 — roi_selection.py 산출 스키마(355~372 candidates dict)에서 `defect_bbox` 존재 확인됨. 형식([x,y,w,h] vs [x1,y1,x2,y2])은 구현 시 실측 확정.
2. **rare_quantile 0.25 vs 절대 빈도 floor**: 모드 빈도가 균등에 가까우면 p25가 무의미 — 실측 분포 보고 후 필요시 보조 기준(빈도 ≤ 전체/2k 등) 추가.
3. exp6 스크립트 골격(knn과 공유 부분)은 [[aroma_exp6_knn-test-coverage]] 설계를 따름 — 두 dev_note를 한 feature-dev 사이클로 구현.
