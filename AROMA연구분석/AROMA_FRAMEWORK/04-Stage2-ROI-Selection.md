# 04 — Stage 2: ROI 선택 (deficit-aware)

> **Claude 요약:** phase0(profiling) + step2(prompts) 산출을 읽어 `(결함 crop × 컨텍스트 빈)` 후보를 스코어링하고, `deficit_aware` 샘플링 + `realism` 스코어로 top-K ROI를 선별하는 단계다. multi 3종(severstal·mvtec_leather·mtd)은 class-stratified 게이트(`--class_floor --per_pair_cap_frac 0.05`)를 추가하고, aitex(single, tiled)는 이 게이트를 제거해 single로 축퇴한다. 소스-crop 다양성 붕괴를 막는 `--img_diversity_cap 1`이 기본. 산출 `roi_candidates.json`/`roi_selected.json`은 step3.5(clean-bg 선정)·step4(ControlNet 학습)·step5(생성)의 공통 선결 입력이다.

---

## 목적

결함 이미지의 morphology 클러스터와 컨텍스트 빈(compatibility matrix)을 교차하여 "어떤 결함 crop을 어떤 컨텍스트에 합성할지"의 ROI 후보 풀을 만들고, 그중 top-K를 선별한다. AROMA arm 생성(step5)과 ControlNet 학습(step4)이 소비하는 공통 선결 산출이며, random arm은 동일 후보 풀에서 무작위 샘플하므로 통제 대조가 성립한다.

- 실행 환경: **CPU**
- 체인 위치: `phase0 → step1(complexity) → step2(prompts) → [step3(roi_selection)] → step3.5(clean_bg_selection) → step4(CN 학습+τ) → step5(생성) → exp*`

---

## 입력 / 출력

| 구분 | 경로 `S(stage, ds)` | 내용 |
|------|--------------------|------|
| 입력 | `S('profiling', ds)` | `morphology_features.csv`, `morphology_clusters.json`, `compatibility_matrix.json`, `deficit_analysis.json` |
| 입력 | `S('prompts', ds)` | `prompts.json` (step2 산출) |
| 출력 | `S('roi', ds)/roi_candidates.json` | 전체 스코어링 후보 풀 (대용량, **삭제/미생성 불가** — random arm·CN train jsonl·exp 메트릭이 재소비) |
| 출력 | `S('roi', ds)/roi_selected.json` | 선택된 top-K ROI (step4·step5 입력) |
| 출력 | `S('roi', ds)/roi_summary.md` | 사람이 읽는 요약 테이블 + 소스-crop 다양성 통계 |

> step3.5(`clean_bg_selection.py`)가 같은 `S('roi', ds)` 디렉터리에 `clean_bg_selected.json` / `clean_bg_random_arm.json` / `clean_bg_summary.md`를 추가로 쓴다(아래 §step3.5).

---

## step3: roi_selection.py

`is_multi(ds)`로 분기하여 v2-1 4종 전부 실행한다.

- **공통**: `--sampling_strategy deficit_aware --score_mode realism --top_k 200 --img_diversity_cap 1`
- **multi 3종 추가**: `--class_mode multi --class_floor --per_pair_cap_frac 0.05` (class-stratified 할당 + class floor)
- **aitex(single)**: 위 3개 플래그 **제거** → single 기본값으로 축퇴
- ⚠️ **`--rarity_temp` 미전달**(기본 1.0 유지) — realism 정합. rarity 온도 스케일은 legacy(deficit 스코어) 전용이라 realism selection에 섞지 않는다.

```python
# multi 3종 (severstal / mvtec_leather / mtd)
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROF \
    --prompts_dir       $PROMPTS \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 200 --img_diversity_cap 1 \
    --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
    --output_dir        $ROI

# aitex (single, tiled) — multi 게이트 3개 플래그 제거
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROF \
    --prompts_dir       $PROMPTS \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 200 --img_diversity_cap 1 \
    --output_dir        $ROI
```

### 파라미터 표

| 플래그 | 값 | 의미 (스크립트 근거) |
|--------|-----|---------------------|
| `--sampling_strategy` | `deficit_aware` | pair-aware 2-phase 할당(coverage-first + deficit-second). `(cluster_id, cell_key)` 쌍마다 base quota 1을 주고, 남은 quota를 PairDeficit 비례로 Hamilton(largest-remainder) 분배 후 각 쌍 내부는 roi_score 내림차순 선택. |
| `--score_mode` | `realism` | ROI_score = `0.5·P(C) + 0.3·P(M) + 0.2·quality`. deficit는 스코어에서 제외(weight 0, provenance로 JSON엔 유지), quality를 hard-gate에서 graded term으로 승격. (legacy: `0.4·P(M)+0.4·P(C)+0.2·Deficit`) |
| `--top_k` | `200` | 선택할 ROI 수. 후보 수보다 크면 클램프. |
| `--img_diversity_cap` | `1` | (Fix4) 동일 소스 crop `(image_path, defect_bbox)`을 최대 1회만 선택. 소수 crop 수십 회 반복(다양성 붕괴 confound) 제거. distinct source < top_k인 풀에는 bounded repetition 허용 + 로그. deficit_aware에만 적용. |
| `--class_mode` | `multi` (aitex 제외) | 정보성 플래그. 실제 stratification은 `--class_floor`가 게이트. |
| `--class_floor` | on (aitex 제외) | (Fix2) class_key 버킷별 symmetric floor(`top_k // K`) 부여 → 다른 class가 공유 `(cluster,cell)` 경쟁에서 이겨도 특정 class가 0으로 굶지 않음. K≤1(단일 class)이면 no-op(byte-identical). |
| `--per_pair_cap_frac` | `0.05` (aitex 제외) | (Fix1) 단일 `(cluster_id, cell_key)` 쌍이 차지할 수 있는 slot 상한 = `ceil(0.05 × top_k)` = 10. 초과분은 under-saturated 쌍에 PairDeficit 순 라운드로빈 재분배(monoculture 방지). |

> multi/single 차이는 위 class 게이트 3개뿐이며, 이는 exp4v2의 `--class_mode multi` per-class 측정(3종) / aitex single-class 측정과 정합한다.

### 로그 확인 포인트

- multi 3종: `stratified_pair_aware: K classes, ...` allocation + class별 floor 로그(특정 class가 floor 미달이면 pair diversity 부족 경고).
- aitex: class 관련 로그 없이 single 경로.
- 공통: `Saved roi_candidates.json (N), roi_selected.json (M)` + `Source-defect diversity: distinct (image,bbox)=..., max repetition per source=...`.

---

## step3.5: clean_bg_selection.py

step3와 대칭인 **clean-background 사전 선정** 단계(`clean_bg_selection.py`, `scripts/aroma/`). 원본 good을 생성 시점에 재스캔하지 않고, **프로파일링 파생 파일**(`context_features.csv` · `compatibility_matrix.json`)과 `roi_selected.json`만으로 각 ROI에 배정할 clean 배경을 오프라인으로 랭킹한다. 실행 환경 **CPU**(픽셀 재스캔·GPU 불요).

```python
!python $AROMA_SCRIPTS/clean_bg_selection.py \
    --profiling_dir  $PROF \
    --roi_dir        $ROI \
    --output_dir     $ROI \
    --emit_random_arm
```

- **랭킹 신호(데이터-유도 가중)**: `w_src`(per-source hist∩) + `w_class`(class-조건부 hist∩) + `w_size`(bbox size-fit). 각 신호의 관측 lift(best − median)에 비례해 정규화 → 변별 불가 신호는 자동 downweight(예: 크기 균일 셋은 `w_size≈0`).
- **void/품질 사전필터**: `local_variance`·`edge_density`의 P1을 void floor로, per-image void_frac의 P90을 컷으로 데이터-유도(하드코딩 없음). ALL-reject 시 전체 풀로 fallback(무음 0-출력 방지).
- **`--emit_random_arm`**: 동일 ROI 집합에 무작위 배경을 배정한 `clean_bg_random_arm.json`(대칭 대조군) 생성 → AROMA arm vs random arm이 **배경 정체성만 다르고 배치/블렌딩은 동일**한 계측기.
- **E1 재현 하드 게이트**: `src_fit_ceiling_mean`이 E1 sim_best(aitex~0.89 / mtd~0.50 / severstal~0.62, ±0.05)와 근접해야 배포. `src_match_frac < 1.0`이면 구 `roi_selected.json` × 신 `context_features.csv` 혼용 → **step3 재실행** 필요(즉시 assert).
- 선택 인자: `--geometry_prior`(Phase 3, 클래스별 edge/surface prior로 paste 위치 precompute, 기본 OFF), `--pool_k`(per-ROI 배경 풀 상한, 기본 P95), `--void_frac_max`(기본 P90), `--no_reject_clean_bg`.

> 정직성: 히스토그램 매칭 변별력은 **도메인-조건부**(aitex 강함, severstal/mtd는 랜덤과 사실상 구분 불가). 본 단계의 확실한 가치는 **재현성 + 대칭 대조군 + per-seed 배치 분산 제거**이며, 일반적 mAP 향상은 주장하지 않는다(GPU 검증 별도).

---

## 핵심 계산 로직

### realism 스코어 (score_roi, `--score_mode realism`)

```
ROI_score = 0.5·P(C) + 0.3·P(M) + 0.2·quality      # 모든 항 [0,1] 클램프
```

- `P(C) = ctx_prior` — compatibility_matrix[cluster][cell] (배경↔결함 호환도). realism에서 최대 가중(0.5)으로 승격.
- `P(M) = morph_prior` — deficit_analysis의 클러스터 prior.
- `quality` — subtype-matching 스코어(아래). legacy에서 hard-gate였던 것을 graded term(0.2)으로.
- **deficit는 스코어에서 제외**(weight 0). 다만 provenance로 각 후보 JSON에 `deficit` 필드는 유지 → 아래 deficit_aware 할당의 **quota 분배**에 여전히 사용된다.

quality 계산: `quality = SuitabilityEvaluator.matching_score(subtype, background_type)`, `subtype = DefectCharacterizer.classify_defect_subtype({linearity, solidity, aspect_ratio})`. 의존성 없거나 morphology 메트릭 결측이면 `('general', 1.0)`으로 no-op(게이트 통과, 오분류 금지).

### deficit_aware 할당 (`_pair_aware_allocation`, 2-phase)

1. **Phase 1 — pair-aware coverage + deficit**: 후보를 `(cluster_id, cell_key)` 쌍으로 그룹화. 각 쌍에 base quota 1(커버리지 보장), 남은 `top_k - n_pairs`를 PairDeficit(`mean(deficit)`) 비례로 Hamilton 분배. deficit=0 쌍은 base만 받음. 쌍 내부는 `roi_score`(realism) 내림차순 선택.
2. **Phase 1b — per-pair cap**: `eff_cap = ceil(per_pair_cap_frac × top_k)` 초과 quota를 잘라 under-saturated 쌍에 PairDeficit 순 재분배.
3. **Phase 2 — quality backfill**: Phase 1이 top_k 미만이면 미선택 풀에서 global roi_score 순 채움(cap-aware: 초과 쌍·초과 소스 skip).
4. **img_cap 처리**: `_source_key = (image_path, defect_bbox)`별 선택 횟수를 Counter로 추적, cap 초과 소스 skip. 점수 동점은 `_img_jitter`(blake2b digest, salt 없음 → 재현 가능)로 서로 다른 소스로 분산.
5. `top_k`가 distinct source 수를 초과하면 bounded repetition으로 cap을 1씩 완화하며 채우고 로그(풀에 distinct source 부족을 명시).

### class floor 로직 (`_stratified_pair_aware`, multi K>1)

- class_key 버킷별 **symmetric floor `top_k // K`** 부여, 잔여 `top_k % K` slot만 per-class deficit-mass largest-remainder로 분배.
- ⚠️ **주의(정직성)**: `top_k=200, K=4`처럼 나누어떨어지면 잔여=0 → deficit-mass 분기는 inert하고 cross-class 분할은 **순수 균등(anti-starvation)**이다. class 수준의 deficit-proportional oversampling이 **아니다**. AROMA의 deficit-proportional 논지는 **각 class 내부**의 PairDeficit Hamilton 할당이 담당한다. 논문에서 "class 간 deficit-aware"라 주장하려면 floor를 줄여 잔여가 커지는 ablation이 필요.
- cap 강제는 per-bucket이 아니라 **post-concat 전역 pass**로 이동(같은 쌍이 여러 class 버킷에 걸쳐 cap 초과 가능). 초과 쌍의 최저-스코어 멤버부터 eviction → class-aware refill로 각 class를 floor까지 복구 → 전역 backfill.

---

## clean_bg_selection / select_context_prototypes

| 스크립트 | 역할 |
|----------|------|
| `clean_bg_selection.py` | (step3.5, 위 참조) profiling 파생 파일로 ROI별 clean 배경을 오프라인 랭킹·배정. `clean_bg_selected.json`(+ random arm)을 산출, step5가 자동 로드. 원본 good 픽셀 재스캔 없음. |
| `select_context_prototypes.py` | 수천 장 normal 풀에서 CLIP(ViT-B-32) 임베딩 → L2 정규화 → PCA(opt 64) → MiniBatchKMeans(K) → cluster medoid로 **K개 대표 컨텍스트 프로토타입** 선정. 파일 순서 편향 없이 normal 분포를 근사(CASDA-comparable by distribution). 결정론적(KMeans random_state=seed, medoid는 sorted path tie-break). 출력 `context_prototypes.json` + `context/`(symlink/copy). severstal처럼 normal이 방대해 1:1 head-slice가 편향될 때 배경/baseline negative 풀을 분포-매칭으로 축약하는 보조 도구. |

---

## 주의사항

- **사후 튜닝 금지**: `top_k` · `img_diversity_cap` · selection 전략을 결과 보고 후 변경하지 않는다.
- **selection 규격 고정**: 4종 모두 `deficit_aware + realism`. multi/single 차이는 class 게이트뿐(dataset_config 자동 분기).
- **`--rarity_temp` 미전달**(1.0): realism 정합. `score_mode='realism'`인데 `rarity_temp≠1.0`이면 deficit-온도 재정렬이 legacy 가중으로 recompute하며 realism 스코어를 무시 → 경고 로그.
- **상류 삼중 종속**: `image_id`(phase0 고유키)·`cluster_id`(step1 GMM)·`prompts`(step2)에 종속. phase0 재실행 시 step1·step2를 모두 재실행한 뒤 step3를 돌린다. 하나라도 구버전이면 step3.5의 `src_match_frac < 1.0`으로 발각.
- **`roi_candidates.json`은 대용량**(로컬 mtd 실측 ~13MB vs `roi_selected.json` ~15KB)이지만 **삭제/미생성 불가** — `generate_random.py`는 selected가 아니라 candidates 풀에서 무작위 샘플하고, `build_train_jsonl.py`·exp1/2/6 메트릭이 재소비한다.
- **aitex는 tile-level·single-class** → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로. `--local_staging`은 CPU selection에 사용 가능(선택)이나 산출이 소형 JSON이라 Drive 직결로 충분.

---

## 관련 노트

[[00-INDEX]] | [[03-Stage1-Complexity-Prompts]] | [[05-Stage3-ControlNet-Generation]] | [[07-Scripts-Reference]] | [[09-Compatibility-Gate]]
