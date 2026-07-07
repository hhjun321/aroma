I've verified all the load-bearing code claims. The verification set's code assertions hold: morph_prior is a cluster-level frequency prior (line 307-310, 338), score is image-blind with jitter tiebreak (486-505), random samples uniformly without replacement over the same candidate pool (1396-1399), quality gate is a pre-filter feeding select_rois (378-431, 1442-1452), and cross-class stratification floor dominates when top_k is a multiple of K (929-951, docstring 873-888). Now I'll write the report.

# AROMA ROI 선택 전략 개선 — 최종 우선순위 로드맵 및 적대 검증 종합

## 0. 한 문장 요약

현행 `roi_score = 0.4·morph_prior + 0.4·ctx_prior + 0.2·deficit`는 두 개의 0.4 항이 모두 **빈도 prior**(흔한 형태·전형 배치 우대)라 downstream 커버리지 목표와 정반대이며, 나아가 **image-blind**(같은 bin 내 모든 crop 동점 → 해시 jitter로 임의 선택)라서 "어떤 crop이 더 좋은 교보재인가"라는 신호가 전무하다. 개선의 정도(正道)는 이 두 결함(빈도 편향 + image-blind)을 **선택 전략 내부에서만** 교정하되, random 팔의 후보 풀·budget·엔진은 절대 건드리지 않는 것이다.

코드 확인으로 검증된 사실:
- `morph_prior = P(cluster)` (line 307-310, 338) — c2 같은 희소 형태에 **낮은** 점수를 준다. 진단 정확.
- score는 image-blind, tie는 `_img_jitter` blake2b 해시로만 분리 (line 486-511, 코드 주석 464-472가 직접 인정).
- random 분기는 동일 candidates 풀에서 `rng.choice(..., replace=False)` 균일추출 (line 1396-1399) — score 정의를 어떻게 바꿔도 random은 불변.
- quality gate(`apply_quality_gate`, 378-431)는 `select_rois` **이전** 단계 pre-filter라 두 팔에 동일 `--min_quality`를 주면 이미 구조적으로 대칭.
- cross-class floor = `top_k // K` (929)가 지배 → top_k가 K 배수면 deficit-mass 분기가 inert, 순수 균등 배분 (docstring 873-888 명시).

---

## 1. 생존 제안 우선순위 로드맵

**전제(모든 선택-로직 제안의 공통 선행조건):** severstal이 `(b1)측정붕괴`가 아니라 `(b2)선택약함`임이 P0로 먼저 확정되어야 한다. b1이면 어떤 ROI를 골라도 mAP가 안 움직이므로 아래 P1~P4는 severstal에 무효다.

---

### P0 — 실패의 선택-가능성 분해 (진단 확정, 코드 변경 없음) · verdict: **adopt**

- **무엇을:** 각 null을 "선택으로 해결 가능/불가"로 확정 분류. mtd(base 0.92)·leather(base 0.83 일관) = random 포화 천장 → **선택 무관**. severstal(base 0.497, synth inert, c2 0.327→0.323) = 측정(b1) vs 선택(b2) 미확정. aitex_single = 측정정상+헤드룸, 유일 positive.
- **판별 절차:** (1) severstal에 aitex식 tiling(6.25:1 → ~256² 타일) 적용 후 aroma/random 재측정 — random조차 baseline 초과하면 **b1(측정붕괴)**, 여전히 평탄하면 **b2/inert**. (2) exp6 rare-mode hit-rate로 "선택은 됐으나 엔진이 c2를 못 살림(선택무관)" vs "선택약함(P1~P4로 개선가능)" 추가 분리.
- **공정성:** 양팔 동일 tiling·동일 평가. 비대칭 없음. tiling 재측정은 기능검증이므로 load-test 정책 무관.
- **비용:** 기존 downstream 파이프라인(CPU). exp6 rare는 임베딩 캐시 재사용 시 CPU 수 분.
- **기대:** 모든 데이터셋에 정직한 규정 부여. **이것 없이 P1~P4를 실행하는 것은 rigging에 가깝다.**

---

### P1 — Deficit-dominant rescore + morphology rarity 반전 · verdict: **adopt_with_fix**

- **무엇을:** `score = 0.2·morph_rarity + 0.3·ctx_prior + 0.5·deficit`. morph_rarity = `1 − min(1, P(cluster))`로 빈도 prior를 rarity로 뒤집고, deficit(유일한 갭 신호)을 지배항으로. ctx_prior만 "호환/그럴듯함" 앵커로 남겨 inert 합성 방지.
- **어디를:** `_W_MORPH/_W_CONTEXT/_W_DEFICIT`(171-175)를 CLI 인자(`--w_morph/--w_ctx/--w_deficit`, 합=1 assert)로 노출. `score_roi`(236-250)에서 morph_prior 대신 morph_rarity 사용 옵션(`--morph_rarity` 플래그). `build_candidates`는 morph_prior/ctx_prior/deficit 원값 그대로 저장(provenance 유지), 재가중은 score_roi에서만.
- **기제:** 검출기가 못 배운 것은 전형 다수 샘플이 아니라 학습분포가 비운 (morphology×context) 조합. severstal이 다수 클러스터로 쏠려 c2 회복 0이던 것과 정합.
- **공정성:** random(1396-1399)은 score를 전혀 안 보므로 정의 변경이 random 풀·budget·엔진에 무영향. weighted 분기(1401-1409)도 raw roi_score 유지(provenance).
- **fix:** 가중치를 **결과 보고 전** train-only 원리로 사전 고정(seed 전 동일, cherry-pick 금지). P0로 severstal b2 확정 선행.
- **비용:** CPU only. **기대:** aitex(헤드룸) 이득 확대, severstal b2 경로 직접 겨냥. mtd/leather 무효.

---

### P2 — Diversity-first 선택: morphology-space 커버리지 (coreset / facility-location, **CPU-only**) · verdict: **adopt_with_fix**

- **무엇을:** image-blind tie를 명시적 coverage 목적함수로 대체. `pick_key = roi_score + λ·min_dist_to_selected`. 특징벡터는 이미 계산된 morphology_features.csv 컬럼(linearity/solidity/extent/aspect_ratio/eccentricity/circularity) + context feature를 minmax 정규화·concat. greedy facility-location으로 형태공간을 고르게 덮음. **임베딩·GPU 불필요.**
- **어디를:** `build_candidates`(355-372)에 현재 안 실리는 형태/context feature 컬럼을 candidate dict에 추가. `_pair_aware_allocation`의 pair 내부/Phase-2 backfill 정렬키 `_order_key`(508-511)를 coverage-greedy로 교체하되 `λ=0`이면 byte-identical no-op.
- **기제:** Ghiasi 2021(Simple Copy-Paste), Sener&Savarese 2018(k-center), Wei&Bilmes 2015(facility location). img_diversity_cap은 "다른 픽셀"만 보장할 뿐 "형태공간 균등 커버"는 보장 못 함 — 그 gap을 직접 메움.
- **공정성:** random은 이 함수 미호출(uniform 그대로). 후보 풀·budget(top_k)·엔진 불변, 선택 순서만 변경. λ는 train 전 고정·전 seed 동일.
- **fix:** λ 사전 고정. build_candidates에 feature 컬럼 싣는 변경 확인. P0로 severstal b2 확정 선행.
- **비용:** CPU only, O(N·k) naive 거리 <수초. **기대:** aitex 이득, severstal c2 diversity-collapse 잔여분. **임베딩 버전(P4)보다 먼저 시도할 것 — 가장 저위험·저비용.**

---

### P3 — morphology suitability 소프트 가중 (대칭, gate 아님) · verdict: **adopt_with_fix**

- **무엇을:** OFF인 hard gate(`--min_quality`) 대신 이미 계산·저장된 `quality_score`(326-362, ∈{0.3~1.0})를 **곱셈 소프트 factor**로: `order_key = base_score · (0.7 + 0.3·quality_score)`. directional 강판에 compact_blob 같은 부적합 crop을 **배제가 아니라 후순위**로만 강등.
- **어디를:** `moderated_score`/`_order_key`(438, 508)에 곱셈 인자 삽입. `apply_quality_gate`(378)는 손대지 않고 OFF 유지. `--background_type` 인자 이미 존재.
- **기제:** severstal directional인데 형태 부적합 crop을 copy-paste하면 비현실 패턴이 c2 신호를 희석. suitability는 도메인 불변 구조관계라 test 누수 없음.
- **공정성:** ★핵심 — 후보를 **제거하지 않으므로** aroma/random 풀·budget 완전 동일. 어떤 클래스도 0으로 굶기지 않아 starvation·parity 붕괴를 **원천 회피**(quality-gate-fairness dev_note가 지적한 confound 회피). random은 order_key 미사용.
- **fix:** 0.7/0.3 계수·background_type 매핑을 test 무관 도메인 지식으로 사전 고정. P0로 b2 확정 선행. **hard gate(P3-alt)보다 이 soft 버전을 우선.**
- **비용:** CPU only(quality_score 재사용). **기대:** severstal b2 '부적합 형태 희석' 가설. mtd/leather 무효.

---

### P3-alt — 대칭 quality gate (양팔 공통 pre-filter) · verdict: **adopt_with_fix** (P3의 열등 대안)

- **무엇을:** hard gate를 "전략"이 아니라 "공유 후보 풀 정의"로 위치. 양팔에 동일 `--min_quality` 전달 → 동일 passing 풀에서 각자 전략 적용.
- **어디를:** 구조는 이미 대칭(apply_quality_gate → select_rois passing 전달, random도 passing 위에서 동작, 1396-1399). **필요 변경은 실험 러너가 random·aroma 두 호출에 동일 `--min_quality`·`--background_type`을 전달하도록 계약/assert로 고정**하는 것.
- **공정성:** memory 계획의 공정성 보정(선택지1)과 정확히 일치. aroma-only gate confound를 구조적으로 제거.
- **fix:** 임계값 사전 고정(스윕 후 best cherry-pick 금지), 클래스 전멸 시 그 클래스 게이트 무효화 규칙 명문화(425-430은 warning만), 데이터셋별 background_type 확정(severstal=directional, leather=organic).
- **주의:** hard 제거라 임계값이 결과를 좌우(rigging medium), 클래스 전멸 시 budget 재배분으로 합성 성격 변화. **동일 목표를 starvation 없이 달성하는 P3(soft)가 우월.**

---

### P4 — 임베딩-공간 rare-mode 커버리지 (exp5/6 캐시 재사용, greedy farthest-point) · verdict: **adopt_with_fix** [제안 4·6·11(mechanism)·13 통합]

- **무엇을:** deficit의 저차원 context-grid 근사를 DINOv2 임베딩 실측으로 보완. real-**train** 결함 임베딩 대비 후보의 novelty(min-dist)를 coverage_gain으로, greedy facility-location 마진으로 계산해 정렬키에 소량 가산. **P2의 임베딩 버전.**
- **어디를:** `build_candidates`(271)에서 crop 임베딩을 EMBED_CACHE_DIR(exp5/6와 동일 sha256 키)에서 로드. `_order_key`/`moderated_score`(508, 438)에 cov_gain 항(가중은 morph/ctx에서 각 0.05 차감, 합=1). deficit_aware pair 정렬키에만 주입, floor/img_cap 구조 불변. 캐시 미존재 시 jitter fallback으로 byte-identical no-op.
- **기제:** severstal c2 회복 0의 유력 원인 = 선택된 c2가 형태클러스터상 c2지만 임베딩상 mode collapse. exp6가 커버리지 측정에 쓰는 좌표계를 선택에 되먹임.
- **공정성:** cov_gain은 aroma 내부 정렬키. random 미진입(1396-1399). ★**real-train 임베딩만 사용, test/val 절대 미사용(assert로 코드 강제)** — 정보누수 금지.
- **fix:** **① 이름 'held-out coverage'는 오도적 → 'train-coverage'로 정정, test/val 사용 코드 차단 assert.** ② λ·k(mode 수)·임베딩 모델 seed 전 사전 고정. ③ 순환논증 금지 — **selection objective로 쓴 임베딩/exp5·6 coverage 지표를 성공 판정 지표로 재사용하지 말 것**(별도 held-out downstream mAP로 평가). ④ farthest-point는 outlier(라벨노이즈)를 우선할 수 있어 **P1의 outlier 감쇠와 결합 필수**(순수 farthest 금지). ⑤ 캐시가 source crop 임베딩을 포함하는지 먼저 검증.
- **비용:** 임베딩 필요(캐시 재사용 시 증분 0, 부재 시 T4 데이터셋당 수 분). greedy는 CPU. **P2(CPU-only)가 무효일 때만 격상 권장.**
- **기대:** severstal b2 정면 겨냥(몇 안 되는 원리적 후보), aitex 우위의 재현 기제 명시.

---

### P5 — Image-aware crop 품질 신호 (degenerate 마스크 페널티만) · verdict: **adopt_with_fix**

- **무엇을:** 같은 bin 내 crop 구분 불가 문제. **단, "대표성(prototypicality) 보너스"는 제외** — 클러스터 중심 대표 crop 우대는 rarity 목표(P1/P4)와 상충하므로 순효과 불명. **degenerate/경계잘림 마스크 페널티 + 면적 하한 페널티만** 채택(순이롭다).
- **어디를:** `build_candidates`(326-372)에서 crop-level 면적/마스크 무결성 페널티를 'crop_quality' 필드로. `score_roi`에 옵션 항(`--w_crop`, 기본 0=무영향).
- **공정성:** crop_quality는 후보 풀 부착 필드, random은 정렬에 미사용. 풀 불변.
- **fix:** prototypicality 보너스 삭제, 면적 하한/마스크 무결성만. centroid 부재 시 fallback 검증.
- **비용:** CPU only. **기대:** aitex(세장형, crop 품질 편차 큼) 유효, severstal degenerate crop 배제로 b2 소폭 기여.

---

### P6 — cross-class deficit 비례화 (severstal multi-class 한정 ablation) · verdict: **adopt_with_fix**

- **무엇을:** cross-class floor(`top_k//K`)가 지배해 top_k=K배수면 deficit-mass가 inert → c2에 rare-class budget이 균등하게만 감. floor를 축소(`floor_frac·top_k//K`)해 remainder를 키우고 남는 budget을 클래스 deficit-mass 비례로 c2에 배분. starvation 방지 하한 유지.
- **어디를:** `_stratified_pair_aware`(929) floor 계산에 `floor_frac` 인자: `floor = max(1, int(floor_frac·(top_k//K)))`. `--cross_class_floor_frac`(기본 1.0=byte-identical). class_floor=True·K>1일 때만.
- **기제:** docstring(873-888)이 이 ablation 여지를 직접 명시. AROMA 원 thesis(deficit-proportional)가 cross-class 층에서만 꺼져 있음.
- **공정성:** random은 stratification 미진입. aroma 내부 배분만, top_k 불변, train deficit 기준(test 무관). floor 하한으로 starvation 방지.
- **fix:** floor_frac 사전 고정(스윕 best cherry-pick 금지). severstal multi-class에서 b2 확정 후 ablation으로만. **c2만 보지 말고 다른 클래스 mAP 동반 확인**(과배분 시 타 클래스 굶어 전체 하락 가능).
- **비용:** CPU only. **기대:** severstal multi 'rare-class 배분 부족' 가설. 단일클래스 무영향.

---

### P7 — EL2N/GraNd-style learnability 가중 (heuristic, aitex 한정) · verdict: **adopt_with_fix**

- **무엇을:** 극단(너무 애매/검출불가능하게 작은) 결함 감점, 중간대 learnable 가점(U-shape). `learn = w_q·quality_score + w_a·size_band(area)`, size_band는 area가 [p10,p90]이면 1, 밖이면 감쇠. **gate 아니라 연속 순서 가중.**
- **어디를:** `moderated_score`/`_order_key`(438, 508)에 항 추가, `--learn_weight`(기본 0=byte-identical). quality_score·area 재사용(area는 CSV에서 1컬럼 추가 로드).
- **공정성:** gate가 아니라 순서 가중 → random byte-identical, 풀·budget 불변. size_band 임계는 train split에서만 사전 고정.
- **fix:** ★**severstal 이득 주장 삭제**(inert는 생성 단계 문제, 선택 재배열로 해결 불가) → aitex 조건부로 한정. ★**matching_score↔EL2N gradient-difficulty 등가 주장 금지** — heuristic learnability 대용임만 명시. 임계 결과 후 튜닝 금지.
- **비용:** CPU only. **기대:** aitex만. mtd/leather/severstal 무효.

---

### P8 — Responsiveness gate (해석 프레임워크로만, 성능 주장 금지) · verdict: **adopt_with_fix (조건부, rigging risk high)**

- **무엇을:** "어느 데이터셋에서 aroma>random 가능한가"를 선택 실행 전 데이터로 판정. headroom(intra-class kNN 밀도)·slack(임베딩 분포 엔트로피) 계산, 미통과(천장 의심) 시 selection을 random으로 수렴(가중치 uniform 평탄화). gate가 aroma를 random쪽으로 미는 **보수적** 장치라 aroma-편향 아님.
- **어디를:** `diagnose_responsiveness`를 roi_summary.md/JSON에 기록. gate_pass=False면 `select_rois`(1341) 유효 가중치를 morph/ctx로 평탄화.
- **★rigging risk high 핵심:** 임계를 "4종 데이터셋에서 사전 관측한 분포로 1회 설정"하는 것은 결과를 아는 상태에서 curve-fitting = tautology. aitex만 positive라는 결과를 알고 경계에 임계를 놓으면 "사전 예측"이 사실상 결과 강제. N=4로 임계 설정+같은 4개 검증 = 자유도 없음.
- **fix:** 임계를 **downstream 결과를 보기 전** 데이터셋-독립 원리(절대 밀도/엔트로피 이론값)로 pre-register하거나, leave-one-dataset-out(N=4라도 신규 데이터셋 추가 권장). 이를 못하면 기각. **성능 장치가 아닌 "언제·왜 통하는가" 해석 프레임워크로만 논문에 제시, 성능 주장 금지.**
- **비용:** 임베딩(캐시 재사용). **기대:** 직접 이득 없음. 산출물 가치는 조건부 주장을 코드로 강제하는 것.

---

## 2. 기각 제안

- **Hard-positive difficulty weighting (morphology 이상치=under-represented 우선)** — **reject.** 목적함수 방향이 downstream에 역효과. "저밀도(outlier) 무조건 가점"은 severstal 검은배경 라벨노이즈 같은 진짜 아티팩트를 우선 선택해 학습을 악화(memory의 severstal 라벨노이즈 사례와 역행). EL2N 후속연구는 "라벨노이즈 있으면 극단 고득점은 해롭다"인데 이 제안은 그 극단을 가점. severstal c2 겨냥 주장도 오인(inert는 조합 희소가 아니라 생성 단계 문제). → **방향 반전(U-shape, P7)으로 흡수됨.** 순수 rarity 가점 형태는 기각.

*(중복 통합으로 흡수: 제안4·6·11(mechanism coverage-gap)·13(quality-score manifold) → 모두 P4로 통합. 제안9(EL2N)→P7. 제안10(exp6 appearance)→P4의 임베딩 경로. 이들은 기각이 아니라 단일 구현으로 병합.)*

---

## 3. 구조적 한계 — "무조건 우위"는 과학적으로 불가능할 수 있음

정직성 제약에 따라 명시한다. **아래는 선택 로직으로 못 이길 수 있으며, 그 경우 조건부 주장이 정답이다.**

1. **천장 데이터셋(mtd base 0.92, leather base 0.83 single·multi 일관):** random 합성만으로 이미 성능 포화. random이 특징공간을 사실상 다 덮어 **선택 여지가 없다.** P1~P8 어느 것도 여기서 random을 유의하게 못 이길 가능성이 높다. → **"선택 무관"으로 정직히 규정**하는 것이 산출물의 일부. "aroma는 천장 데이터셋에서 random과 통계적으로 동등하다"가 정답이며, 이것을 실패가 아니라 **경계 조건의 발견**으로 보고한다.

2. **severstal inert(생성 단계 무효):** P0에서 (b1)측정붕괴 또는 (엔진이 c2를 못 살림)으로 확정되면, **어떤 선택 개선도 mAP를 못 움직인다.** 선택은 "무엇을 붙일지"만 정할 뿐, 붙인 것이 학습신호가 안 되면(대비 약함/측정 6.25:1 종횡비 병리) 무효. 이 경우 severstal은 "선택으로 해결 불가"로 규정.

3. **결론적 주장 형태:** AROMA의 downstream 가치는 **무조건적이지 않고 조건부다** — "측정이 정상이고(non-degenerate 종횡비/tiling), baseline 헤드룸이 있으며(천장 아님), 합성이 학습가능한 positive를 만드는(non-inert)" 데이터셋에서만 data-driven 선택이 random을 이긴다. 현 증거로 그 조건을 만족하는 유일한 사례는 aitex_single(Δ+0.097, t=4.51). **P0의 판별과 P8의 responsiveness gate는 이 조건부성을 rigging 없이 명문화하는 장치다.**

---

## 4. 가장 결정적인 단일 실험 (최저 비용으로 "선택 개선이 통하는지" 판정)

### 실험: severstal tiling 재측정 (P0의 (b1) vs (b2) 게이트)

**왜 이것이 결정적인가:** 모든 선택-로직 제안(P1~P7)의 효과는 severstal이 `(b2)선택약함`일 때만 발현한다. severstal은 유일하게 "헤드룸 있음(base 0.497) + null"인 데이터셋이라, 여기서 선택이 통하는지가 AROMA 선택 로직 전체의 가치를 가른다. aitex는 이미 positive(추가 확인 불필요), mtd/leather는 천장(어떤 실험도 못 이김). **단 하나를 고른다면 severstal의 실패 원인을 측정 vs 선택으로 가르는 것.**

**절차 (코드 변경 없음, CPU/기존 파이프라인):**
1. severstal(1600×256, 6.25:1)에 aitex와 **동일한 tiling**(→ ~256² 타일) 적용.
2. tiled severstal에서 aroma/random **양팔 동일 조건**으로 재측정(3-seed).

**판정:**
- random조차 baseline을 넘으면 → **(b1) 측정붕괴가 원인.** 선택 무관. severstal을 "측정 문제"로 규정하고 선택 로직 노력에서 제외.
- random은 평탄한데 aroma가 넘으면 → **(b2) 선택약함 확정.** P1~P4가 실제로 통할 수 있는 대상. 여기서 P2(CPU-only coverage-greedy)를 먼저 붙여 이득 확대 확인.
- 둘 다 평탄하면 → **inert(생성/엔진 문제).** 선택으로 해결 불가로 정직히 규정(§3.2).

**비용:** 기존 downstream 파이프라인 재실행(기능검증, load-test 정책 무관). 임베딩·GPU 불필요. **가장 싼 단일 실험으로 선택 로직 투자의 방향 전체를 결정한다.**

**공정성:** 양팔 동일 tiling·동일 평가·동일 budget. 선택 전략 비대칭 없음. tiling은 test 정보 누수 아님(전처리).

---

## 5. 실행 순서 요약

| 순위 | 항목 | 비용 | 선행조건 | 위험 |
|------|------|------|----------|------|
| **P0** | 실패 분해 진단 + §4 severstal tiling 실험 | CPU | 없음 | none |
| P1 | Deficit-dominant rescore + rarity 반전 | CPU | P0(b2) | low |
| P2 | morphology-space coverage-greedy | CPU | P0(b2) | low |
| P3 | suitability 소프트 가중 | CPU | P0(b2) | low |
| P5 | degenerate 마스크 페널티 | CPU | 없음 | low |
| P6 | cross-class deficit 비례화(severstal multi) | CPU | P0(b2) | medium |
| P7 | learnability U-shape 가중(aitex) | CPU | 없음 | low |
| P4 | 임베딩 rare-mode coverage | GPU(캐시 재사용) | P2 무효 시 격상 | medium |
| P3-alt | hard 대칭 gate | CPU | P3 대신 | medium |
| P8 | responsiveness gate(해석용) | GPU(캐시) | pre-register | high |
| — | Hard-positive rarity 가점 | — | **기각** | — |

**공통 불변식(전 항목):** random은 동일 candidates 풀에서 uniform 추출(1396-1399) 불변, top_k(budget) 불변, copy-paste 엔진 불변. 오직 aroma의 **선택 전략(정렬키/재가중)만** 변경. 모든 하이퍼파라미터는 결과 보고 전 train-only 원리로 사전 고정(cherry-pick 금지). test/val 임베딩 사용 금지(P4/P8은 assert로 코드 강제).