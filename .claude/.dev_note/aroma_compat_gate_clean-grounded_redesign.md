# AROMA compat 게이트 clean-grounded 재설계 — SGM + patch-granularity (게이트 무력화 진단 통합)

---

## (사용할 skills: feature-dev)

> **설계 출처**: (1) 다중-에이전트 워크플로우 `clean-grounded-compat-design` (Ground→Design 3안→적대 Verify→Synthesize, 3안 모두 `survives=false` 동일 사유), (2) `compat_gate_placement_investigation_20260709`(게이트 메커니즘 코드 확정 + leather 무력화 조사). 두 문서를 본 devnote로 통합.
> **선행 진단**: [[aroma_placement_aware_score_redesign]](compat 게이트 1차), [[aroma_step4_h1-recombination-no-info]](재조합 무정보), [[aroma_step1_deficit-global-good-only]].
> **상태**: **구현 전 계획.** 착수 게이트(§5 CPU 진단)를 통과한 데이터셋에만 착수. 무조건 구현 금지.

## 개요

compat 배치 게이트·선택의 기준분포 `P(cell|cluster)`가 **결함 이미지의 image-mean context**에서 학습된다(`distribution_profiling.py` step6 :872-916). 결함 존재가 배경 통계를 오염 + granularity가 게이트 질의와 불일치 → 순수 normal 배경에 paste할 때 domain-shift + 무력화. clean(good) 분포는 `context_features.csv`에 이미 존재하나 realism 모드에서 `deficit` weight=0이라 **기능적으로 사장**.

**확정 설계**: compat 기준을 순수 normal 분포로 재정의(clean-grounded)하되, 워크플로우 반증이 확정한 no-op의 두 구조적 원인을 정면으로 고치는 **SGM(대칭 기하평균) + clean_dist patch-granularity화** 병합안. selection·gate 양쪽을 동일 clean-grounded 기준으로 재정렬.

---

## 1. 게이트 메커니즘 (코드 확정)

### 재료 3개 (Phase0 산물, 동일 택소노미 공유)

| 재료 | 출처 | 정의 |
| --- | --- | --- |
| `compatibility_matrix.json` `matrix[cluster][cell]` | `distribution_profiling.py` step6 (:856-928) | **`image_type=='defect'` 행의 image-mean context**로 계수(:872-907) = P(cell \| cluster). 이미지당 1 cell |
| cell_key 택소노미 | `_extract_context_features`(:207) + `_context_cell_key`(:471) | patch → 5 feature → P33/P66 3-bin 이산화 → `"2_2_0_1_0"` |
| bin_edges | `recommended_config.yaml`(`_compute_bin_edges` :488) | feature별 [P33,P66], matrix 구축·게이트 평가 공용 |

5 context feature: `local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency`(:242-247).

### 판정 (`generate_defects.py:911-925`)
```python
patch = nrgb[py:py+crop_h, px:px+crop_w]                    # 후보 paste 위치의 clean-bg patch (crop 크기)
cell  = _cell_key_fn(_ctx_feat_fn(gray(patch)), bin_edges)  # 동일 5feature/3bin cell
return float(compat_row.get(cell, 0.5)) >= compat_threshold # 호환도 ≥ τ면 수락
```
`compat_row = matrix[str(roi_entry['cluster_id'])]`(:2376). 배치 루프 `_place_on`(:890-989): foreground/random 경로 게이트 실패 시 `max_bg_tries` 재샘플 → Stage2(:1000-1021) `normal_pool`에서 다른 배경 교체 → 전부 실패 시 마지막 후보 강제 paste(silent drop 없음, `gate_stats.fallback=True`). rng: `compat_on=False`(τ=0) → 신규 draw 0 → legacy byte-identical.

### soft 매칭 (:913-916)
`get(cell, 0.5)`: 미관측 cell = 중립 0.5. exact-match 아니라 soft `>=τ`.

---

## 2. 무력화 진단 (root cause)

### 발견 A — granularity 불일치 (matrix=image-mean vs 게이트=local crop-patch) **[진짜 root cause]**
matrix cell = 결함 이미지의 **image-mean** context(이미지당 1 cell, :888-907) → 관측 cell이 극소. 게이트는 **local crop-size patch** 1개의 cell을 질의(:906,919). 이미지평균 cell 집합에 국소 patch cell이 거의 없음 → 대부분 미관측 → 0.5 중립. 이것이 "leather 4.7% coverage / ~95% 중립통과"의 실제 기제.

> ⚠️ 조사 원문(investigation_20260709 발견 A/H1)은 이를 "matrix=64px per-patch, defect-majority skip 계수"로 기술했으나 **코드 오기**. step6은 per-patch가 아니라 **per-image-mean** 계수(:900-907 morph_row당 1 count). 정정하여 통합.

### 발견 B — matrix = 결함측 분포, clean 매칭은 미반영
matrix는 결함 이미지 배경의 P(cell|cluster). 게이트는 이걸 clean(good) 배경 patch에 조회. good context는 `context_features.csv`에 `image_type=='good'`로 이미 존재하나(:610) matrix는 **안 씀**(:877 defect만). clean-side 분포 미반영이 구조적 약점(domain-shift).

### 발견 C — coverage-변조 단일 메커니즘 (net 결과만 스펙트럼)
소프트 게이트의 무력화는 **하나의 메커니즘**이다: τ=0.5에서 관측 cell(compat<0.5)은 거부, 미관측 cell(0.5 fallback)은 수락. compat_max<0.5인 데이터셋(§5 실측: aitex 0.456·severstal 0.194·mtd 0.167 전 cluster <0.5)에서는 **관측 cell이 전멸거부**되고 통과분은 미관측뿐이다. 이 단일 기제가 coverage에 따라 net 결과만 다르게 나타난다:
- **leather**(관측 4.7%): 대다수가 미관측→0.5→수락 → net **over-accept**(τ0.5 accept≈neutral≈95%). *기존 §2 발견 C의 over-accept 서술은 이 leather 케이스만 기술했음* — 일반화 아님. (단 cluster2는 compat_max=0.766≥0.5라 관측 1.4%가 통과, 순수 neutral-only는 아님.)
- **severstal**(관측 다수): 관측 cell이 관측→거부되어 net **over-reject**(cluster별 57~77% 거부).
- **aitex**(중간): 관측 cell은 전멸거부하나 neutral fallback으로 **4/5 cluster net 수락**(accept 34/68/59/67/71, cluster0만 다수거부) — severstal과 한 범주(over-reject)로 묶지 말 것.
- **mtd**(near-ceiling): TV 0.128 경계 + 관측 다수, 저우선.
- **τ>0.5**: 4종 모두 관측·미관측 대부분 거부 → exact-match 근접 → reject-heavy fallback.

### 발견 D — selection↔placement 직교 (워크플로우 반증) **[결정적]**
placement 사실성을 건드리는 유일 경로 = compat 게이트뿐. SELECTION 재랭킹(`build_candidates` ctx_prior)은 실제로 바뀌나 **placement와 직교** — 게이트는 `cluster_id`로 row 선택(:2376), 선택된 cell_key가 위치를 제약 안 함. 위치는 여전히 random+reject(**Scenario B**). +H1(recombination no-info, severstal flat 실측).
→ **"기준분포만 교체"는 목표(배치 사실성)를 못 바꾼다.** 3 설계안(bayes-invert/clean-conditional/SGM) 모두 이 사유로 `survives=false`. 진짜 병목 = compat 공식이 아니라 **(A) granularity 불일치 + soft-fallback 구조**.

### 발견 E — **build(64px) vs query(crop-size) 스케일 불일치 [미해결, SGM 미포함]**
게이트 질의(`generate_defects._compat_ok` :919)는 clean-bg patch를 **결함 crop 크기 그대로**(`nrgb[py:py+crop_h, px:px+crop_w]`) 잘라 cell화한다. 그러나 matrix·bin_edges·clean_dist·P_def_patch는 **전부 64px patch**(`_context_worker` GRID_SIZE=64)로 학습된다. `_extract_context_features`(:207)는 patch 크기를 **정규화하지 않음**.
- 5 feature 중 `frequency_energy`(`r=min(h,w)//4`+FFT 차원 의존)는 **강한 size 의존**; 극소 crop(예 8×8)은 LBP/FFT/variance 통계 무의미. → **같은 배경이라도 crop 크기가 64px에서 멀면 다른 cell로 분류** → 64px 학습 matrix 조회가 어긋남. 결함 크기 범위 넓은 셋(leather/mtd/aitex: ~8px~600px)에서 "적절한 위치" 판단 부정확.
- ⚠️ **SGM/patch-gran(§3) 개선은 이 축을 안 고침** — build 측을 image-mean→64px-patch로만 바꿨고, build(64px) vs query(crop) 불일치는 그대로. 별개 갭.
- **fix 방향(§7 TODO)**: footprint를 64px 타일로 덮어 각 타일 compat 집계(mean/min) = build·query 동일 스케일 + 실제 면적 커버(정합안). 대안: 위치중심 64px window / query resize→64(고주파 왜곡).

---

## 3. 확정 설계 — SGM + patch-granularity (clean_dist AND P_def)

반증이 확정한 no-op를 정면 회피하는 두 축:

### 3-1. clean_dist AND P_def(matrix) 둘 다 patch granularity로 (핵심 no-op 회피)
**게이트 무력화의 root = build(image-mean) vs query(local-patch) granularity 비대칭** (§2 발견 A/C, §5-1 †). 게이트 관측 cell 집합(support KEY)은 `matrix` KEY에서 오는데, 현행 `matrix`는 결함이미지 image-mean 계수(이미지당 1 cell, :888-907)라 support가 극소(leather 5 cell). **clean_dist만 patch-gran화하면 이미 support에 있는 cell의 VALUE만 바뀔 뿐 KEY(관측 cell)를 안 늘려** leather는 여전히 over-accept(§5-1 † 실측 증거).

→ **P_def(matrix)도 patch granularity로 재구축**한다: step6에서 `defect_mean_ctx`(이미지당 1 cell) 대신 `image_type=="defect"`의 **전 64px patch**를 cell화해 `P_def_patch(cell|cluster)` 산출(patch는 속한 image의 `cluster_assignments` 상속). clean_dist(good 전 patch)와 **동일 granularity**. 그러면:
- 게이트 support KEY = patch-gran defect cell(leather 5→~191) → good local query patch가 관측 cell에 걸릴 확률↑ → 미관측 0.5 fallback 구조적 하락 → **게이트 실제 물림**(leather 4.7%→~99%대, §5-1 headroom 실현).
- clean_dist·P_def 둘 다 proper local-patch context 분포 → SGM `sqrt(P_def·P_clean)`가 동일 cell 공간에서 대칭 정의.

**⚠️ 검증 필요(단정 금지)** — C3 교훈, 아래 둘은 착수 전 §5로 확인:
- **cluster 신호 washout** → **local 검증 완료(2026-07-09), 실 cluster 기준: 데이터셋별로 갈림.** 실 profiling(context_features.csv + bin_edges + cluster_assignments)로 재현:
  - **leather: WASHOUT 실재** — 비어있지 않은 cluster가 0·2 둘뿐이고 **둘 다 `compact_blob`**(cluster1 elongated은 matrix EMPTY), per-cluster `P_def_patch` pairwise TV=**0.130**(<0.15). 형태 clustering이 leather를 거의 못 나눠 배경 context도 동일. → patch-gran으로 support는 확장되나(5→191) **per-cluster 판별은 붕괴** → 게이트가 cluster별 compat이 아니라 **cluster-무관 clean-plausibility 필터**로 degrade(여전히 배경-매칭엔 유효하나 "결함cluster별 호환" 전제는 leather서 실패).
  - **mtd: 판별 유지** — 5 cluster pairwise TV 평균 0.390(0.23~0.52). per-cluster compat 유의미.
  - ⚠️ **정정 기록**: 초기엔 group=defect-type(cluster proxy)로 "washout 없음"(leather 0.42)이라 판정했으나, **실 cluster는 defect-type과 달라**(leather 5 type이 morphology cluster 0·2로 collapse) proxy가 오도했음. 실 cluster_assignments 필수였다.
- **selection 결합**: patch-gran matrix_symmetric은 cluster당 cell ~191로 폭증 → `build_candidates`가 이걸 ctx_prior로 쓰면 candidate 폭발 → **게이트 전용으로 분리**(§4-2).

### 3-2. SGM 공식 (기준분포 재정렬)
`compat_sym(k,c) = norm_{S_c}( sqrt( (P_def_patch(k|c)+ε)·(P_clean(c)+ε) ) )`, ε=1e-3, S_c={c: P_def_patch(k|c)>0} (patch-gran defect support, §3-1 — 게이트 support KEY 확장의 핵심).
- 대칭 기하평균 AND-시맨틱: 결함군집 **AND** 순수 normal 둘 다 흔한 cell만 고득점.
- **per-cluster max 정규화**: 값역 [0,1] → 압축값 vs 0.5 fallback 불일치 제거.
- 채택 이유: 3안 중 유일하게 `deficit_conflict=false`·`realism_break=false`. bayes-invert 기각(압축 compat_clean vs 0.5 fallback → 게이트 반전버그), clean-conditional 기각(object-centric서 폐기된 clean-marginal을 폐기 regime에서 부활 = deficit-conflict).

### 3-3. 미관측 cell 처리
0.5 유지(**hard gate 아님** → leather over-reject 회피). threshold를 사전스캔으로 <0.5에 두어 **관측-저지지 cell만 표적기각**. patch-granularity(3-1)로 관측 cell 비중을 늘려 게이트가 자연히 물리게 하는 것이 정답(hard gate 대신).

### 3-4. 대안 (기각·후속)
- **patch-size 정규화**(발견 A 대응): 3-1 clean_dist patch-gran으로 흡수.
- **positive steering**(compat 내림차순 위치 랭킹, "고호환 유도"): 직교성(발견 D)을 더 정면으로 치나 **양의 배치기하 = Scenario A 재설계**로 본 devnote 범위 밖. → 후속.
- **leather placement 게이트 제외**: §5 진단서 coverage 극저 확정 시 τ=0 유지(newpipe 가이드).
- **τ data-driven**(데이터셋별): 사전스캔에 포함.
- **cell 택소노미 coarse화**(feature 5→↓ 또는 `N_CONTEXT_BINS` 3→2, distribution_profiling.py:80-85): **기각** (CASDA 대조 통찰, workflow 반증). 관찰(게이트 near-noop)은 실재하나 기전 귀속이 틀림 — matrix는 image-mean 계수(:888-907, 이미지당 1 cell)라 관측 distinct cell 상한이 택소노미 크기가 아니라 **이미지 수**이고, coarse화는 image-mean들을 같은 cell로 collapse시켜 distinct cell을 오히려 **줄인다**("coverage 증가" 인과 역방향). coarse화가 실제 하는 건 hit율↑(0.5 fallback↓)뿐인데 이는 발견 A의 granularity 비대칭(build=image-mean vs query=local-patch, :906/919)에 **직교** — root는 §3-1(patch-gran)이 직접 겨냥. bin 축소는 §3-1과 경쟁+판별력 손실(net-wash~negative). SGM(§3-2)과 무관. one-liner 이식 가능하나 lever 아님. → 착수 안 함.
- **CASDA식 배경 IMAGE compat-가중 선택**(generate_defects.py:2375 `rng.choice(normal_images)`→`rng.choices(weights=compat_row.get(cell,0.5))`): **기각** (workflow 반증). hook·데이터 존재(compat_row 이미 :2376, normal→cell 사전계산은 step7 img_mean_ctx :954-973 재사용)로 이식은 가능하나 3중 반증. (1) **배경선택 ≠ placement**: 배경 IMAGE 바꿔도 그 안 위치는 여전히 uniform(:792-793, 실패 시 :309-310) → Scenario B 유지. (2) **SGM no-op의 자리이동**: matrix sparsity로 대다수 normal이 미관측 cell→0.5→가중 uniform 붕괴 = SGM 죽인 그 no-op을 초기 sampling으로 옮김. 의미 가지려면 위 coarse화 선행 필요(비독립). (3) **arm confound**: :2375는 AROMA·random arm 공유 → AROMA arm만 넣으면 논지 오염, 양 arm이면 AROMA-specific 아님. +H1 벽. CASDA placement 사실성의 실제 출처는 배경선택 아니라 **소스좌표 보존**(§7 이관).

---

## 4. 영향도 분석 / 수정 내용

### 변경 상태
- `compatibility_matrix.json`에 신규 키(`matrix_symmetric`,`clean_dist`,`symmetric_epsilon`) **추가**, 기존 `matrix` 보존.
- compat_mode=symmetric: `roi_selected.json` ctx_prior→roi_score 정렬·선택 집합 변경 / 게이트 paste 위치 변경(개수·클래스·bbox parity).

### 수정 파일
1. **`scripts/distribution_profiling.py`** step6(:856-928): `matrix`(image-mean, 유지) + 신규 `P_def_patch`(defect 전 patch cell 분포)·`clean_dist`(good 전 patch)·`matrix_symmetric`(=SGM(P_def_patch, clean_dist), **patch-gran support**)·`symmetric_epsilon` 추가(additive). patch-gran cell 분포 산출을 `_context_dist(context_rows, image_type, bin_edges)` 헬퍼로 통일(good→clean_dist, defect→P_def_patch; step7 deficit 경로 동작 유지). `bin_edges`·`_context_cell_key` 불변. matrix_symmetric 크기↑(cluster당 ~5→~191 cell) — JSON 저장만 증가.
2. **`scripts/aroma/roi_selection.py`** build_candidates: **무변경** — 여전히 image-mean `matrix`를 ctx_prior로 사용, **matrix_symmetric(patch-gran)으로 전환 안 함**. 근거: (a) cluster당 ~191 cell로 candidate 폭증, (b) 발견 D에서 selection↔placement 직교 확인 → selection clean-grounding은 placement 목표에 무가치. → **clean-grounding은 게이트 전용**. `score_roi` 무변경.
3. **`scripts/aroma/generate_defects.py`** compat_row loader(:2318 부근): `.get("matrix_symmetric")` fallback `.get("matrix")`. `_compat_ok`(:911-925) **무변경**(row 값만 symmetric).
4. **config**: `compat_mode`(symmetric|defect, ablation/reversible), `compat_threshold`=데이터셋별 사전스캔값.

---

## 5. 착수 게이트 — CPU 진단 (개요, Colab)

재프로파일 예산 전 **필수 선행**. 기존 산출물만 로드(`context_features.csv`·`compatibility_matrix.json`·`morphology_clusters.json`·`recommended_config.yaml`), 재프로파일 불필요. GPU 불요.
**실행 스크립트는 `AROMA연구분석/colab_execute/`에 별도 가이드로 작성(TODO)** — 본 devnote는 지표·판정만.

지표:
1. **분포 발산**: `TV(P(cell|good), P(cell|defect))`, cosine.
2. **fallback률 (granularity별)**: good 패치 cell이 defect support S_c 밖으로 떨어지는 비율을 **image-mean vs local-patch** 두 기준으로.
3. **accept/reject flip**: 기준 defect→clean(SGM) 교체 시 관측 cell 중 threshold 넘나드는 개수.
4. (선택) H4 steering: `--compat_threshold τ` vs `0` 소규모 생성 → `placement-gate stats` fallback률·위치 차이.

판정:
- `TV<0.10`(texture, leather 예상) → **확정 no-op → 해당 데이터셋 착수 중단** (제거할 domain-shift 없음).
- fallback률 image-mean>80%인데 **local-patch서 크게 하락** → granularity가 lever임을 실증 → 진행.
- TV 큼 + patch fallback 수용 + flip 유의 → **정식 착수**.
- leather coverage 극저 재현(리포트 4.7% 대조) → leather placement 게이트 제외 확정.

### 5-1. 4종 CPU 진단 실측 (2026-07-09, `.claude/.etc/roi_approve/roi_box/compat_gate_diag_*.json`)

| DS | clusters | 관측cell | cov∈matrix | cov∈defect-patch † | TV | cosine | flip(τ0.5) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| leather | 3 (c1 EMPTY) | 5 | 4.67% | 99.56% (+95%p) | 0.6955 | 0.2675 | 3 |
| aitex | 5 | 40 | 77.82% | 92.45% (+15%p) | 0.4327 | 0.6614 | 21 |
| severstal | 5 | 107 | 87.48% | 99.99% (+12%p) | 0.2637 | 0.8319 | 21 |
| mtd | 5 | 94 | 67.24% | 99.81% (+33%p) | 0.1281 | 0.9656 | 48 |

† **`cov∈defect-patch` 헤드룸 = P_def를 patch-gran 재구축했을 때의 상한.** 초기 설계는 clean_dist만 patch-gran화해 게이트 support KEY(관측 cell)를 안 늘려 leather no-op이 재발했다(H3 leather τ0.5 accept≈neutral≈95.5%가 직접 증거). **개정(§3-1): P_def(matrix)도 patch-gran으로 재구축**해 support KEY를 확장(leather 5→~191 cell) → 이 헤드룸이 게이트에 실제 반영된다. 단 실현 여부는 §5 CPU 진단의 **cluster별 P_def_patch 발산(washout)**·fallback률 재측정으로 확인 후 확정(단정 금지).

**H3 τ분해 핵심**:
- 전 데이터셋 **compat_max<0.5**(leather c2=0.766·aitex c1=0.456 제외 대부분 <0.2, severstal ≤0.194·mtd ≤0.167). P(cell|cluster) 확률값이 여러 cell에 분산돼 구조적으로 작음.
- τ0.5: 관측 cell 전멸거부(compat<τ) → 통과=미관측뿐. τ0.7: 4종 모두 거의 accept=0(전량거부→last-candidate 폴백).
- **threshold 사전스캔 제약**: τ는 반드시 **compat_max 미만(~0.05)** 에서 스캔해야 한다. τ≥0.3이면 관측 cell 전멸(H3 실측) — 이것이 "전멸거부"의 진짜 원인이자 lever다. 정규화(§3-2)는 이진 게이트 목적에는 threshold 재보정과 **중복**이므로 '필수'로 격상하지 않는다; 정규화의 비중복 가치(미관측 0.5가 관측-high 위에 서는 순위역전 교정)는 positive steering(§7, 범위 밖)에서만 발현한다. compat_min<τ<compat_max 구간의 표적기각 실효는 소규모 생성(H4)으로 확증 필요.

### 5-2. SGM fix-simulation (local 검증, 2026-07-09 — leather/mtd, defect-type proxy)

patch-gran P_def + SGM + per-cluster max-norm 게이트를 good patch 조회로 시뮬(scratchpad):

| DS | 평균 neutral(미관측)률 OLD→NEW | support/row OLD→NEW | τ0.5 거동(NEW) |
| --- | --- | --- | --- |
| leather | **95.0% → 17.1%** | 3~9 → 93~155 | accept 9~47% / reject 53~91% |
| mtd | **69.9% → 5.3%** | 21~52 → 125~185 | accept 24~37% / reject 63~76% |

- **over-accept 치료 확인**: 미관측 중립통과 급락 → 게이트가 실제 물림(discrimination 회복). §3-1 방향(P_def patch-gran으로 support KEY 확장) 기계적으로 작동.
- **max-norm이 τ 유의미화**: raw compat_max가 cluster별 **0.08~0.67 (3~8× 편차)**(leather color 0.53 vs cut 0.21, mtd uneven 0.08) → 단일 raw τ 불가. max-norm 후 전 cluster max=1.0 → 단일 τ로 cluster 전반 표적기각(τ sweep 단조). **→ 이전 workflow C2("정규화 중복") 판정 부분 정정**: max-norm은 ranking뿐 아니라 **이진 게이트 cross-cluster τ 정합**에 실질 기여(근거는 C2가 반박한 "확률값이 τ 못 넘어"[오답]가 아니라 cluster별 compat_max 3~8× 편차).
- **per-row nuance(중요)**: §5-1 aggregate coverage 99.6%는 union 기준 과대. 게이트는 cluster row **1개**를 조회하므로 row별 support가 좁고, narrow type(leather color support 93 → neutral **42.8%**)은 부분 over-accept 잔존. 게이트 실효는 cluster별로 다르다.
- **정직(H1/§6 유지)**: '치료'=기계적 discrimination(τ0.5 reject 53~91%)이지 downstream 이득 아님. reject-only라 H1 조합다양성 축소 긴장 그대로. τ는 사전스캔으로 낮게(τ0.1 accept 53~93%) 잡아 최악만 기각 권장.
- **washout(§3-1 ⚠️#1) — 실 cluster로 재검증 후 갈림**: leather **WASHOUT**(실 cluster 0·2 pairwise TV 0.130, 둘 다 compact_blob) / mtd 판별 유지(0.390). → leather 게이트는 cluster-무관 clean-plausibility 필터로 degrade. (초기 defect-type proxy "0.42 통과"는 오도, 정정됨.)
- **support 15-vs-5 불일치 해소**: 실 bin_edges로 재현 시 image-mean support = **5(leather)/94(mtd) = 실 matrix 정확 일치**. 로컬 15는 재계산 bin_edges 아티팩트였음. fix-sim 위 수치도 실 bin_edges/cluster 기준 재산출(leather neutral 96.8→2.9%, mtd 69.5→6.9%).

### 5-3. profiling 재검증 완료 (Colab 4종, 2026-07-09 — commit 6c8658f 실측)

`profiling_symmetric_rebuild_verify_execute.md`로 4종 재실행 → 신규 4키 emit·정규화·support 확장 전부 확인:

| DS | legacy matrix | matrix_symmetric | 확장 | 정규화 max | drift |
| --- | --- | --- | --- | --- | --- |
| leather | 5 | 191 | 38× | 1.0 (비어있지 않은 2 cluster) | 3→4 cluster (구 코드 profiling) |
| mtd | 94 | 205 | 2.2× | 1.0 (5 cluster) | 없음 |
| severstal | 107 | 203 | 1.9× | 1.0 (5 cluster) | 없음 |
| aitex | 43 | 87 | 2.0× | 1.0 (5 cluster) | 미세(diag 40→43, 버전차) |

→ **SGM 구현 실 Colab 확정 검증.** GMM 시드 고정(`random_state=42`)이라 run-to-run 재현; leather drift는 업로드 profiling이 구 코드 산물이었던 탓(비결정성 아님). matrix_symmetric union은 patch 기반이라 cluster 재라벨과 무관하게 동일. **defect-mode·symmetric 모두 신 profiling으로 통일 권장(구 것 혼용 금지).** ⚠️ CPU 진단(§5-1)·washout(§3-1)은 구 클러스터 기준 — 신 profiling(특히 leather 신 4-cluster)으로 재측정 필요.

### 5-4. τ 사전스캔 (local, 2026-07-09 — 타일링-aware, symmetric, R=0.25)

신 profiling(profiling_tobe) + 타일링 게이트 재현(gd._tile_anchors, mean) + crop 크기=morphology_features.defect_bbox로 good 배경 tiled mean-compat 분포 → R=0.25 percentile:

| DS | ds_tau | cluster τ | neutral(미관측) | OK |
| --- | --- | --- | --- | --- |
| leather | **0.1148** | 2:0.102 / 3:0.128 | 0.8% | 2/2 |
| mtd | **0.2348** | 0.20~0.25 (5 cluster) | 0.2~10% | 5/5 |

- **τ=0.5 금지 실증**: 데이터 τ가 0.10~0.25로 훨씬 낮음(확률/max-norm 스케일). 전 cluster τ ∈ (agg_min, median)·non-degenerate·≠0.5 → OK.
- **over-accept end-to-end 치료**: neutral leather 0.8%/mtd 0.2~10% — 개선 전(image-mean matrix+crop query) 95%/70% over-accept가 (patch-gran P_def → SGM → max-norm → 타일링) 체인으로 거의 소멸. 게이트 실제 물림 확정.
- leather washout에도 τ 산출(cluster 2·3 유사값=cluster-무관 필터). caveat: crop 크기 morphology_features(pre-selection 전체) — Colab은 roi_selected 사용 권장. severstal/aitex는 Colab(§10).

착수 후: threshold 사전스캔([[feedback_prescan_thresholds]]) → profiling 1회 재실행(신규 키 emit 확인) → Colab 소규모 생성으로 fallback률 하락·표적기각 로그 확인. 테스트 코드 신규 작성·pytest 금지(CLAUDE.md).

---

## 6. 무결성 / scope

- JSON 키 삭제 없음(`matrix`/`bin_edges`/`deficit` 보존) → 구 프로파일·구 실행 파싱 OK.
- `compat_mode=defect` → 바이트동일 legacy(ablation/reversible).
- `score_roi`·`_compat_ok` 제어흐름 불변 → realism 가중 0.5/0.3/0.2 정합.
- π(k) double-count 없음(SGM은 posterior 미사용).
- ε=1e-3·threshold 사전스캔값 **test 수치 보기 전 고정**(사후 튜닝 금지).
- **정직한 scope**: detector headline lever 아님. "올바른 배경에만 결함이 붙게 하는 배경-매칭 무결성 개선". 진짜 lever(양의 배치기하 = Scenario A, positive steering)는 범위 밖.
- **downstream(detector metric) 미실증 경고**: 게이트는 코드상(`_place_on` :890-989) **reject-and-resample 전용** = 랜덤 분포에서 저compat 조합을 쳐낼 뿐, 위치 정보를 더하거나 조합 다양성을 늘리지 못한다(positive steering 범위 밖). H1은 소데이터(leather/mtd)의 유일 정보원이 (결함,배경) **조합 다양성**이라 못박는다(E8: leather 배경 217종). 따라서 게이트를 물리게 고치면 **leather에서 H1의 가치원인 조합 다양성을 가장 크게 축소** — leather는 기제(TV 0.70·headroom) 최대수혜이자 downstream 역방향 긴장이 가장 큰 케이스이지 downstream 최강 사례가 아니다("leather 최우선"으로 격상 금지). **어느 데이터셋에도 metric lift 경로가 실증돼 있지 않다**: severstal=H1 no-op 확정(E3–E5 multi-run flat), leather/mtd=조합-다양성 축소 긴장으로 미검증, aitex=placement metric 데이터 없음.
- **구현 가치 조건**: patch-granularity로 fallback률이 실제 하락함을 §5로 실증 못 하면 구현 가치 없음. CPU 진단이 실측 게이트. §5 CPU 진단은 게이트가 물리는지(fallback률 하락)만 답하며 metric 이동을 함의하지 않는다 — **이는 metric 가치가 아니라 조사기록 정확성 가치**. 실제 구현 착수는 §5 fallback-drop 실증 **+** H4 placement A/B에서 유의미 delta 확인 후로 게이팅.

## 7. TODO / 후속

- [x] `AROMA연구분석/colab_execute/`에 §5 CPU 진단 실행 가이드(.md) 작성 — `compat_gate_cpu_diagnosis_execute.md`
- [x] profiling 재검증 가이드 `profiling_symmetric_rebuild_verify_execute.md` + 4종 Colab 재실행 완료(§5-3, support 확장 확인)
- [ ] 4종(leather/aitex/mtd/severstal) 진단 → coverage/accept 분해 표 → 데이터셋별 착수 판정 (**신 profiling으로 재측정** — 5-1은 구 클러스터 기준)
- [x] **발견 E 해소 — footprint 64px 타일링** (commit 6b027f7): `_tile_anchors` + `_compat_ok` 타일링(symmetric 한정, mean 집계). local 검증(leather/mtd): crop=64서 legacy와 0% 변화(통제), off-64서 cell 불일치 94~100%·out-of-support 21%→≤1% → 스케일 불일치 실측 치료. defect 모드 byte-identical.
- [ ] **AR-fallback(고종횡비 ControlNet 미생성) 분석 (workflow, 2026-07-10)** — `.claude/.etc/roi_approve/ar_filter_check.md` 검토. 근원=생성 stretch-squash(`generate_defects.py:2145 resize((512,512))`·un-squash :2169) vs 학습 `Resize(512)+CenterCrop(512)`(train_controlnet :120-131, 세장형 **중앙 정사각 창만 학습 = 형상 미학습**). AR 게이트 :2030-2032(τ2.5). **정정**: `test_controlnet.py` 리포지토리에 없음(CASDA 유물); 유일 규격지점=controlnet_synthesis. 고AR 유병률 severstal 58.9%(class2 98%)·mtd 36.3%·leather 20.7%. **판정**: 문서 제안(랭킹매칭)·"통계 규격"은 AR 근본(기하) 미겨냥. 2층 — 스미어제거=추론전용 **타일분할**(재학습X, 학습 CenterCrop창과 정합)/letterbox; 세장형 형상생성=**재학습 필수**(aspect보존 전처리). downstream: fallback rate↓ 확실하나 mAP=H1 반증(전환 집중지 severstal 0pp 예측·flat 실측). **권고=파일럿게이팅**: Phase1 추론 타일분할(FID/시각 판정)→Phase2 severstal class2 downstream 게이트→신호時만 재학습. severstal 검증 없이 재학습 금지.
- [ ] **MEDIUM (리뷰 발견, 미해소)**: (a) **ControlNet AR-fallback copy_paste**(generate_defects.py:~1726) 호출이 compat_row/threshold/tile 미전달 → symmetric 모드서 AR-fallback 샘플은 게이트 완전 우회(기존 누락, 타일링과 무관). symmetric 게이트 균일 커버 원하면 배선 추가. (b) **대형 footprint perf**: ~970px footprint(1024 이미지)면 ~256 tile × max_bg_tries(20) ≈ 5k feature 추출/이미지. bounded·cheap(64×64)이나 곱셈적 — 대형 crop 많은 셋서 per-image 비용↑. tile cap 또는 anchor별 cell 캐시 고려(load-test 미측정).
- [ ] τ data-driven 재보정
- [ ] leather 제외 정책 확정(newpipe 가이드 반영)
- [x] **positive placement (Scenario A, 3요건)** 구현 완료 (commit 362e7dd, symmetric 한정). `_positive_place`: fit-only 후보(요건1) + void-tile straddle 배제(요건2) + footprint mean-compat top-K rng 샘플(요건3-b, 슬라이딩 best-mean). local smoke PASS(fit/void 0위반, positive mean-compat≫random +0.4~0.69) + PNG 시각검증(`.claude/.etc/positive_place_viz/`). **잔여 MEDIUM**(우리 4셋 무영향): M2 texture+symmetric 동시 시 texture가 위치 steer 안 함, M3 foreground 제약 생략(full-frame이라 no-op). **H1 downstream 판정 여전 미실행** — positive placement가 mAP 올리는지는 exp4v2 A/B 필요.
- [x] **이미지-레벨 배경 선택** 2회 진화: (1) commit 23bd373 cluster-aggregate compat 랭킹 → probe로 불일치·이질감 원인 확정(cluster=형태·mean 동점·aggregate≠D_v, mtd c0 cell 교집합 40%·top10 local_var cv 0.68); (2) commit **8056130** D_v-specific 배경 분포 히스토그램 교집합으로 교체(`_dv_bg_hist`/`_cell_hist`/`_hist_intersection`, 결함영역+void 제외). local 검증: D_v↔top10 교집합 0.023→0.327(mtd c0)·선택bg variance가 D_v수준 정합·viz 이질감 해소. 위치레벨(positive placement) 불변. **realism/정합 개선, mAP 미입증(H1).**
- [ ] (별도 devnote) positive steering = Scenario A 배치기하 재설계 — **CASDA 기전 정정**(workflow 확인): CASDA placement 사실성은 배경선택(get_compatible_background compose_casda_images.py:562)이 아니라 **소스좌표 보존**(_compose_single_task: x1_j=원본 x1+jitter±100 :842, y1=원본 소스 y 유지 :867, scale 0.875~1.0)에서 나옴. "결함을 원본이 있던 좌표대로(소폭 jitter) 재배치"가 양의 배치기하의 원천 — AROMA random+reject(Scenario B) 대비 실제 lever. Scenario A 재설계는 배경선택이 아니라 **ROI별 원본 좌표 메타 보존 → paste 위치 prior** 방향으로 착안.
