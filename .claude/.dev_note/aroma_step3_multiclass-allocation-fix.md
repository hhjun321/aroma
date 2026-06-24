# Step 3 ROI 선택 — Multi-Class 할당 결함 수정 (generic class-floor + diversity-cap)

> 효과 크기(per-class map50 회복)는 multi-seed 재실행 전까지 미확정. 메커니즘은 코드 + roi_selected.json 산출물로 확정.

---

## (사용할 skills: feature-dev)

## 개요

exp4v2 severstal(현재 유일한 multi-class run, nc=4)에서 c2 클래스만 AROMA가 baseline·random 둘 다에 회귀(c2 map50: baseline 0.2617 / random 0.2821 / aroma 0.2304). c1/c3/c4는 개선.

Drive 산출물 `roi_selected.json` / `roi_candidates.json` 직접 분석으로 원인 확정: `_pair_aware_allocation`의 **class-blind ROI 할당**이 multi-class에서 드러나는 일반 결함. single-class(MVTec 전체)는 클래스가 1개(K=1)라 클래스 간 경쟁 자체가 없어 **구조적으로 면역** → 다른 데이터셋 무탈. severstal이 첫 nc>1 run이라 처음 노출.

**핵심 원칙: 본 수정은 severstal 특이 대응이 아니라 multi-class(nc>1) 일반 대응.** dataset 분기·class id 하드코딩·severstal magic number 금지. 임의 multi-class 도메인(임의 클래스 수·임의 imbalance)에 일반화 = AROMA 연구 철학. 모든 신규 파라미터 default = 현행 동작과 byte-identical(single-class 경로 불변).

---

## 근본 원인 (확정)

`scripts/aroma/roi_selection.py:280-365 _pair_aware_allocation`:
- candidates를 `(cluster_id, cell_key)` **class-blind** 그룹핑(L306-308), base quota=1/pair(L321), intra-pair는 roi_score 내림차순(L346-347).
- severstal top_k=200 선택 결과: **c1 131(65.5%) / c2 11(5.5%) / c3 58(29%) / c4 0(0%)**. 자연 풀 분포 14 / 2.7 / 74.6 / 8.6%. → c1 flood, c4 완전 starvation, c3 under.
- per-class deficit은 평탄(~0.005 전 클래스) → skew는 deficit 아닌 roi_score+pair 경쟁의 부산물.

두 증상:
1. **starvation-to-0** (c4): c4 ROI가 공유 pair마다 roi_score로 밀려 base 슬롯을 한 번도 못 이김 → 0개.
2. **monoculture** (c2): 단일 pair `(cluster, cell_key=1_0_0_1_0)`가 deficit 분배로 quota 독점 → 동일 morphology('elongated', 동일 prompt, roi_score 0.173≈풀평균 2배) 11개 → detector가 한 archetype에 과적합 → c2 precision 0.3571(전체 최저) 붕괴. random의 c2는 uniform=다양 → 일반화 → 개선.

---

## 아키텍처 (중요 — 게이트 위치)

`select_rois` / `_pair_aware_allocation`는 **exp4_v2가 직접 호출하지 않음**. 단계형 파이프라인:

```
roi_selection.py run()  ──writes──▶  roi_selected.json
        ▲ (CLI: --sampling_strategy, --top_k, --seed; 현재 class_mode 없음)
        │
generate_defects.py run()  ──reads roi_selected.json──▶  images + annotations.json
        │  (annotations[i].source_roi = roi_entry["image_path"])
        ▼
exp4_v2_supervised_detection.py  ──reads annotations.json──▶  YOLO train
        (_parse_severstal_class 는 label-write 시점에만, 선택과 무관)
```

결론:
- 게이트는 **`roi_selection.py` 자체 CLI/run + step3 orchestration**에 둔다. **exp4_v2 수정 안 함**.
- class 신호는 `morphology_features.csv`의 **`defect_type` 컬럼**에서 generic 취득 (이미 존재, `compute_complexity.py:414 _class_diversity_neff`가 `r["defect_type"]` 읽음). `build_candidates`(roi_selection.py:227-269)가 이 행들을 순회.
- exp4_v2의 `_parse_severstal_class`(severstal `class{N}` path regex)를 선택 레이어에서 **재사용 금지** — generic 레이어에 severstal 가정 밀반입. `defect_type`이 dataset-agnostic이라 더 우수.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `roi_selected.json` 선택 결과 (multi-class 게이트 활성 시 클래스별 분포 변화). single-class·non-opt-in run은 불변.
- 하류 `generate_defects.py` → synth 이미지/annotations 변화 (multi run 재실행 필요).

### 그 상태를 전제로 동작하는 기존 로직
- `generate_defects.py`: `roi_selected.json` 입력 → multi run 재생성 필요.
- baseline/random 조건: `select_rois`가 `random`/`top_k`/`weighted`를 `_pair_aware_allocation` 밖으로 라우팅(L388-404) → **미진입, 완전 불변**. 비교 공정성 유지.
- val/test set·라벨·메트릭: 학습측 synth 선택만 변경 → val 불변(고정 yardstick).

### starvation/flood 계열 추가 확인
- 본 수정이 "0개" 상태를 **없애는** 방향(per-class floor). 기존에 "0개"를 가정한 하류 로직 없음(synth는 있는 클래스만 라벨링).

---

## 수정 내용

모든 신규 파라미터 default = no-op → single-class·미opt-in run byte-identical.

### 신규 파라미터

| 레이어 | 파라미터 | default | default 시 동작 |
|--------|---------|---------|----------------|
| `build_candidates` | (내부) `candidate["class_key"] = str(row.get("defect_type") or "_")` | 항상 부착 | single-class면 단일 값 → K=1 → inert |
| `select_rois`/`_pair_aware_allocation` | `class_floor: bool = False` | False | Fix2 stratification skip |
| `select_rois`/`_pair_aware_allocation` | `per_pair_cap_frac: Optional[float] = None` | None | Fix1 cap+redistribute skip |
| `select_rois`/`_pair_aware_allocation` | `rarity_temp: float = 1.0` | 1.0 | `moderated_score == roi_score` |
| `run()` + CLI | `--class_mode`(또는 `--nc`), `--class_floor`, `--per_pair_cap_frac`, `--rarity_temp` | single / off / None / 1.0 | 전달 안 하면 현행 |

### 1. `scripts/aroma/roi_selection.py` — `build_candidates` (L255-269)

각 candidate dict에 `"class_key": str(row.get("defect_type") or "_")` 추가. generic(컬럼 이미 존재), single-class는 단일 값 → 무해.

### 2. `scripts/aroma/roi_selection.py` — `moderated_score(c, rarity_temp)` 헬퍼 (Fix3, DEFER)

`rarity_temp == 1.0`이면 `c["roi_score"]` 그대로 반환(완전 no-op). `>1`이면 deficit 항만 `d ** rarity_temp`로 극단 압축(prior anchor=morph/ctx_prior는 유지). raw `roi_score`는 dict·JSON에 그대로 보존(provenance). **ablation knob으로만 탑재, first-ship 아님.**

### 3. `scripts/aroma/roi_selection.py` — `_pair_aware_allocation` 시그니처 (Fix1)

`per_pair_cap_frac=None, rarity_temp=1.0` 추가.
- 정렬 키 2곳(L346 intra-pair, L353-357 backfill)을 `key=lambda c: moderated_score(c, rarity_temp)`로 교체 (단일 ranking source of truth). `rarity_temp=1.0`이면 키 동일 → byte-identical.
- L342~L344 사이에 cap 블록: `cap = ceil(per_pair_cap_frac * top_k)` (budget-scaled, class 무관). `quotas[p] > cap`이고 후보가 cap 초과인 pair만 캡핑, freed 합산. freed를 `pair_deficit` desc 라운드로빈으로 미포화 distinct pair에 1슬롯씩 재분배(각 cap 한도). 잔여는 기존 Phase-2 backfill로 흡수 → 총 선택 ≤ top_k 보존. `per_pair_cap_frac is None`이면 블록 전체 skip.

### 4. `scripts/aroma/roi_selection.py` — `_stratified_pair_aware(...)` 신규 (Fix2, 최우선)

`select_rois`의 `deficit_aware` 분기에서 `class_floor`이고 K>1일 때 호출:
- `class_key`로 버킷팅. `K = len(distinct keys)`. **`class_of is None or K <= 1` → `_pair_aware_allocation(...)` 그대로 반환(byte-identical).**
- else: `top_k // K` 대칭 floor. 잔여(`top_k - K*floor`)는 클래스별 deficit-mass 가중 largest-remainder(**기존 Hamilton 헬퍼 재사용**). 각 클래스 quota를 후보 수로 clamp, 초과분 미포화 클래스로 재분배.
- 버킷별로 **기존 `_pair_aware_allocation`(Fix1 cap/rarity 인자 전달) 실행** → concat → 글로벌 Phase-2 backfill로 정확히 top_k. → c4=0/c1 flood 치료, 클래스 내 deficit-awareness 보존.

### 5. `scripts/aroma/roi_selection.py` — `select_rois` (L368, L406-407)

신규 파라미터 추가. `deficit_aware`를 `class_floor and K>1`이면 `_stratified_pair_aware(...)`, 아니면 `_pair_aware_allocation(...)`으로 dispatch. `random`/`top_k`/`weighted` 분기 불변.

### 6. `scripts/aroma/roi_selection.py` — `run()` + CLI(`_parse_args`)

`--class_mode`(또는 `--nc`), `--class_floor`, `--per_pair_cap_frac`, `--rarity_temp` 추가 → `select_rois`로 전달.

### 7. `AROMA연구분석/colab_execute/step3_execute.md` — orchestration 게이트

데이터셋 `class_mode == 'multi'`일 때만 `--class_mode multi --class_floor --per_pair_cap_frac 0.05` 부여 (시작값, 튜닝 대상 — dataset별 값 베이킹 금지). 그 외(single)는 아무것도 전달 안 함 → byte-identical. **`ds == 'severstal'` 분기 금지.**

### Ship 순서
- **SHIP-FIRST (한 묶음, multi 게이팅)**: Fix2(class_floor) + Fix1(per_pair_cap_frac). starvation/flood/monoculture 전부 구조적 해소.
- **DEFER**: Fix3(rarity_temp) — default 1.0 inert 탑재만, 추후 ablation sweep용.

---

## 암묵적 요구사항 / 엣지

| 상황 | 처리 |
|------|------|
| `class_of is None` 또는 K<=1 | `_pair_aware_allocation` 그대로 (single-class byte-identical) |
| `per_pair_cap_frac is None` | cap 블록 전체 skip |
| `rarity_temp == 1.0` | `moderated_score` = raw roi_score (정렬 불변) |
| 클래스 후보 수 < quota | clamp 후 미포화 클래스로 surplus 재분배, 총 top_k 유지 |
| cap 재분배 잔여 미배치 | 기존 Phase-2 global roi_score backfill로 흡수 |
| `defect_type` 컬럼 비어있는 미래 multi 데이터셋 | 단일 키 `"_"` → K=1 → fix silently inert. **데이터 prep 계약**(코드 분기 아님): multi 프로파일링은 반드시 `defect_type` 채워야 함 |
| baseline/random | `_pair_aware_allocation` 미진입 → 불변 |

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` (전 로직: `build_candidates`, `moderated_score` 신규, `_pair_aware_allocation`, `_stratified_pair_aware` 신규, `select_rois`, `run`, CLI)
- `AROMA연구분석/colab_execute/step3_execute.md` (orchestration 게이트)
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — **수정 안 함** (선택 하류)

---

## 테스트 (Colab, pytest 금지)

1. **single-class 불변**: MVTec single 데이터셋 step3 재실행(multi 플래그 없이), `roi_selected.json`을 pre-change 산출물과 **byte-identical diff**.
2. **multi 구조 치료**: multi run `roi_selected.json`을 `class_key`별 버킷팅 — **전 클래스 count > 0**(starvation 없음), 단일 `(cluster_id, cell_key)` pair가 `ceil(per_pair_cap_frac * top_k)` 초과 안 함(monoculture 없음).
3. **효과 (multi-seed)**: exp4v2 multi를 **≥3 seed**(42,1,2) 재실행 — 클래스별 map50가 해당 클래스 `random` 미만으로 회귀 없음 + 전체 map50 ≥ baseline. 현재 n=1 증거 업그레이드, publication 주장 전 필수.
4. **공정성**: baseline/random `roi_selected.json`·메트릭 불변 확인.

검증 셀 예시:
```python
import json, collections, math
sel = json.load(open(f"{AROMA_OUT}/roi/severstal/roi_selected.json"))
by_cls = collections.Counter(s.get("class_key") for s in sel)
by_pair = collections.Counter((s["cluster_id"], s["cell_key"]) for s in sel)
top_k = len(sel); cap = math.ceil(0.05 * top_k)
print("per-class:", dict(by_cls))            # 전 클래스 > 0
print("max pair pick:", by_pair.most_common(1), "cap:", cap)  # <= cap
```

---

## 미확정 사항 (TODO)

- `per_pair_cap_frac` 시작값 0.05는 튜닝 seed — multi run에서 sweep 후 확정. dataset별 값 베이킹 금지.
- Fix3 `rarity_temp` 적용 범위(전 모드 vs nc>1 한정) — default 1.0이라 inert, ablation 시 결정.
- 효과 크기는 단일 시드 → multi-seed 재실행 전 확정 보류.
- **Cross-class 층은 deficit-proportional 아님 (thesis 프레이밍 결정 필요)**: 클래스 간 예산은 대칭 floor `top_k // K`가 지배하고, deficit-mass largest-remainder를 타는 슬롯은 `top_k % K`(< K)뿐. `top_k`가 `K`의 배수면(severstal top_k=200, K=4 → remainder 0) deficit-mass 분기는 inert → 클래스 간 균등 분할. 이는 의도된 anti-starvation 설계(클래스 0개 방지)이며 **클래스 단위 deficit-proportional oversampling이 아님**. AROMA의 deficit 비례 thesis는 클래스 *내부*(`_pair_aware_allocation`의 PairDeficit Hamilton, 그대로 유지)에서만 성립. publication에서 "deficit-aware across classes"로 주장하려면 (a) 설계 결정으로 명시하거나 (b) floor를 줄여 non-trivial remainder가 deficit-mass 가중을 타게 하는 ablation을 비교할 것. (코드 docstring `_stratified_pair_aware`에 동일 caveat 기록함.)
