# step3 — `roi_selection.py` (aroma-sym selection) Colab 실행

> **정본**: `_SPEC.md` §3 step3. env·output 루트는 `_SPEC.md`에서 그대로 가져온다. 문서마다 재발명 금지.
> **목적**: phase0(profiling) + step2(prompts) 산출을 읽어 결함 crop × 컨텍스트 빈 후보를 스코어링하고,
> `deficit_aware + realism`으로 ROI 목록을 선별한다 (AROMA arm 생성·CN 학습의 공통 선결).
> **실행 환경**: CPU
> **입력**: `S('profiling', ds)` + `S('prompts', ds)`  →  **출력**: `S('roi', ds)`

---

## 실행 순서 (체인)

```
phase0(profiling) → step1(complexity) → step2(prompts) → [step3(roi_selection)] → step4(CN 학습) → step5(생성) → exp4v2/exp3/exp5/exp6
```

step3은 step2 완료 후 실행한다. step3 산출(`roi_candidates.json` / `roi_selected.json`)은
step4(ControlNet 학습·`build_train_jsonl.py`)와 step5(생성·random arm)의 공통 입력이다.

## 전제 (실행 전 확인)

- phase0 완료 — `S('profiling', ds)`에 `compatibility_matrix.json`(matrix_symmetric 포함), `morphology_features.csv`, `context_features.csv` 존재.
- step2 완료 — `S('prompts', ds)`에 `prompts.json` 존재.
- ⚠️ **phase0를 재실행했다면 step1·step2를 모두 재실행한 뒤 step3를 돌린다.** step3의 `image_id`(phase0 고유키 종속)·`cluster_id`(step1 GMM 종속)·`prompts`(step2 종속)가 삼중으로 상류에 종속된다. 하나라도 구버전이면 roi_selected.json이 신 profiling과 어긋나 **step3.5에서 `src_match_frac<1.0`으로 발각**된다(그 전에 예방할 것).
- 데이터셋 v2-1 5종: `severstal · mvtec_leather · mtd · aitex · kolektor`. **aitex = tiled(256×256/stride128, single-class)**. kolektor는 domain=mvtec(마스크 리졸버 공유)·class_mode=single.

---

## STEP 0 — 공통 환경 셀 (`_SPEC.md` §1 정본 그대로 — 수정 금지)

```python
import os, json

# ===== 공통 환경 (sym_final 전 문서 동일 — 수정 금지) =====
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
# ===== 단일 버전 루트 (stage-first: {stage}/{ds}) =====
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step4 산출, step5 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex", "kolektor"]   # v2-1 5종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex/kolektor=single (자동)
```

---

## STEP 1 — 선행 확인 (profiling·prompts 존재)

```python
import pathlib

print(f"{'ds':<16} class_mode  {'profiling':<10}{'prompts':<10}")
for DS in DATASETS:
    prof = pathlib.Path(f"{S('profiling', DS)}/prompts.json")  # placeholder; 아래에서 개별 확인
    have_prof = pathlib.Path(f"{S('profiling', DS)}/compatibility_matrix.json").exists()
    have_prom = pathlib.Path(f"{S('prompts', DS)}/prompts.json").exists()
    cm = "multi" if is_multi(DS) else "single"
    print(f"{DS:<16} {cm:<11} {'OK' if have_prof else 'MISSING':<10}{'OK' if have_prom else 'MISSING':<10}")
```

> aitex·kolektor는 `class_mode=single`로 떠야 정상(자동 분기 기준). `severstal·mvtec_leather·mtd`는 `multi`.

---

## STEP 2 — selection (multi 3종 / single aitex·kolektor 분기 루프)

`roi_selection.py`를 `is_multi(ds)`로 분기하여 5종 전부 실행한다.

- **공통 (_SPEC §3 step3)**: `--sampling_strategy deficit_aware --score_mode realism --subtype_mode percentile --img_diversity_cap 1`
- **`--top_k`는 데이터셋별로 다르다 (`synth_pool_sizing.md` §2 정본)**: **severstal 1000**, 나머지 4종 200. severstal은 `real_train=2534`라 ratio 1.0 증강에 2,534장이 필요한데 `top_k=200 × n_per_roi=3 = 600`으로는 6.3× 부족하다. 후보가 266,624개이므로 `top_k=1000`은 상위 0.4%로 품질 꼬리 유입이 미미하다. ⚠️ **전 데이터셋 200으로 돌리면 severstal이 exp4v2에서 ratio 1.0에 도달하지 못해 기존 결과와 비교 불가**가 된다.
- **3종(multi) 추가**: `--class_mode multi --class_floor --per_pair_cap_frac 0.05` (stratified pair-aware allocation + class floor)
- **aitex·kolektor(single)**: 위 3개 플래그를 **제거** (single 기본값으로 축퇴)
- ⚠️ **`--rarity_temp` 미전달** — realism 정합(deficit_aware가 rarity를 온도 스케일하지 않도록 기본값 1.0 유지).
- ⚠️ **`--subtype_mode percentile` 필수** — 미전달 시 기본값 `fixed`(전 데이터셋 공통 하드코딩 상수)로 조용히 실행되고, 구 산출물과 byte-identical이 되어 **재실행한 의미가 사라진다**. 논문 Table 4/4b가 percentile을 기술하므로 정본은 percentile이다.

`!python` 매직은 IPython 전용이라 스레드에서 동작하지 않으므로, 루프는 셀 안에서 순차 `!python`으로 실행한다.

```python
# deficit_aware + realism 공통.
#   --img_diversity_cap 1 : 동일 소스 결함 crop((image_path, defect_bbox))을 최대 1회만 선택
#     → 소수 crop 수십 회 반복(다양성 붕괴) 제거. distinct source < top_k인 클래스에만
#       bounded repetition 허용 + 로그. deficit_aware allocation 에만 적용.
#   multi 3종만 --class_mode multi --class_floor --per_pair_cap_frac 0.05 (stratified allocation).
#   aitex·kolektor(single)는 3개 플래그 제거 → single 축퇴(byte-identical to single 기본).
#   --rarity_temp 미전달 (realism 정합).
# 데이터셋별 top_k — synth_pool_sizing.md §2 정본. severstal만 1000(real_train 2534 대응).
TOP_K = {"severstal": 1000, "mvtec_leather": 200, "mtd": 200, "aitex": 200, "kolektor": 200}

for DS in DATASETS:
    os.environ['DS']     = DS
    os.environ['PROF']   = S('profiling', DS)
    os.environ['PROMPTS']= S('prompts', DS)
    os.environ['ROI']    = S('roi', DS)
    os.environ['TOPK']   = str(TOP_K[DS])

    if is_multi(DS):   # severstal / mvtec_leather / mtd
        print(f"\n===== {DS}  (multi: class-gated allocation, top_k={TOP_K[DS]}) =====")
        !python $AROMA_SCRIPTS/roi_selection.py \
            --profiling_dir     $PROF \
            --prompts_dir       $PROMPTS \
            --sampling_strategy deficit_aware --score_mode realism \
            --subtype_mode      percentile \
            --top_k $TOPK --img_diversity_cap 1 \
            --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
            --output_dir        $ROI
    else:              # aitex (single, tiled) / kolektor (single)
        print(f"\n===== {DS}  (single: multi 플래그 제거, top_k={TOP_K[DS]}) =====")
        !python $AROMA_SCRIPTS/roi_selection.py \
            --profiling_dir     $PROF \
            --prompts_dir       $PROMPTS \
            --sampling_strategy deficit_aware --score_mode realism \
            --subtype_mode      percentile \
            --top_k $TOPK --img_diversity_cap 1 \
            --output_dir        $ROI
```

> 로그 확인 포인트:
> - multi 3종: `stratified_pair_aware` allocation + class별 floor 로그(특정 class가 floor 미달이면 과소 주의).
> - aitex·kolektor: `--class_mode multi` 관련 로그 없이 single로 진행.
> - 공통: `Saved roi_candidates.json (N), roi_selected.json (M)`.
> - **공통: `subtype_mode=percentile — thresholds in use: aspect_ratio A / B | solidity C / D`** — 이 줄이 없으면 `--subtype_mode`가 누락되어 fixed로 돌았다는 뜻이니 즉시 재실행.
> - **kolektor에서만** `Fixed fallback: solidity` 경고가 뜬다(동질성 가드). 다른 4종에서 뜨면 profiling이 다른 것이므로 조사.

---

## STEP 2-1 — percentile 임계 대조 (논문 Table 4b 검증)

`--subtype_mode percentile`이 유도한 임계가 논문 Table 4b와 일치하는지 자동 대조한다. 아래 기준값은 **로컬 미러 실측**(`aroma_dataset`, 2026-07-29)이며, 동일 profiling이면 소수 2자리까지 재현되어야 한다.

```python
import numpy as np, csv, io

# 논문 Table 4b 기준값 (AR P33, AR P66, Sol P33, Sol P66)
# kolektor solidity는 동질성 가드(중간 tertile < sd의 15%)로 fixed 폴백 → 0.700 / 0.900
TABLE4B = {
    "aitex":        (3.13, 16.43, 0.648, 0.900),
    "kolektor":     (4.39,  5.73, 0.700, 0.900),   # solidity = fixed fallback
    "severstal":    (2.80,  7.80, 0.900, 0.965),
    "mtd":          (1.70,  3.62, 0.846, 0.934),
    "mvtec_leather":(1.58,  4.03, 0.874, 0.954),
}
DEGEN_RATIO = 0.15                                  # roi_selection._TERTILE_DEGENERATE_RATIO
FIXED_EQUIV = {"aspect_ratio": (2.0, 5.0), "solidity": (0.7, 0.9)}

for DS in DATASETS:
    rows = list(csv.DictReader(io.open(f"{S('profiling', DS)}/morphology_features.csv", encoding='utf-8')))
    got, fb = [], []
    for f in ("aspect_ratio", "solidity"):
        x = np.array([float(r[f]) for r in rows if r.get(f) not in (None, '', 'nan')])
        p33, p66 = np.percentile(x, 33), np.percentile(x, 66)
        if x.std() > 0 and (p66 - p33) / x.std() < DEGEN_RATIO:
            p33, p66 = FIXED_EQUIV[f]; fb.append(f)
        got += [p33, p66]
    ref = TABLE4B.get(DS)
    ok = ref is not None and all(abs(g - r) < 0.02 for g, r in zip(got, ref))
    print(f"{DS:14s} AR {got[0]:7.3f}/{got[1]:7.3f}  Sol {got[2]:.4f}/{got[3]:.4f}"
          f"  {'PASS' if ok else 'DRIFT(조사)'}"
          f"{'  [fixed fallback: ' + ','.join(fb) + ']' if fb else ''}")
    assert ref is None or ok, (
        f"[{DS}] 유도 임계가 Table 4b와 불일치 — profiling이 논문 산출과 다르다. "
        f"got={[round(g,4) for g in got]} expected={ref}. phase0를 재확인할 것.")
```

> **DRIFT가 나면** 임계 자체가 아니라 **profiling(`morphology_features.csv`)이 논문 기준과 다른 것**이다. `image_id` 고유키(`31ee0aa`)·`image_w/image_h`(`b1bb497`) 반영 여부를 먼저 확인한다. 임계는 profiling의 결정론적 함수이므로 동일 입력이면 항상 동일하다.
>
### 산출 개수 대조 (top_k 누락 검출)

`--top_k`가 데이터셋별로 다르므로, 실제 선택 수가 기대값과 맞는지 확인한다. 후보 수는 subtype/top_k와 무관하게 profiling에만 의존하므로 함께 대조한다.

```python
import json

# 로컬 미러 실측(2026-07-29) = 논문 산출 기준. candidates는 profiling 결정론적 함수.
EXPECT = {   # ds: (candidates, selected)
    "severstal":     (266624, 1000),
    "mvtec_leather": (   968,  200),
    "mtd":           ( 16877,  200),
    "aitex":         (  5887,  200),
    "kolektor":      (   511,  200),
}
for DS in DATASETS:
    roi = S('roi', DS)
    nc = len(json.load(open(f"{roi}/roi_candidates.json")))
    ns = len(json.load(open(f"{roi}/roi_selected.json")))
    ec, es = EXPECT[DS]
    print(f"{DS:14s} cand {nc:>7} (기대 {ec:>7}) {'OK' if nc==ec else 'MISMATCH'}"
          f"   sel {ns:>5} (기대 {es:>5}) {'OK' if ns==es else 'MISMATCH'}")
    assert nc == ec, f"[{DS}] 후보 수 불일치 — profiling이 논문 산출과 다르다 ({nc} vs {ec})."
    assert ns == es, f"[{DS}] 선택 수 불일치 — --top_k 확인 ({ns} vs {es}). TOP_K 딕셔너리 누락 가능."
```

> **선택 수 MISMATCH의 최빈 원인 = `--top_k` 데이터셋별 값 누락.** severstal을 200으로 돌리면 여기서 `1000 vs 200`으로 잡힌다(실제 발생 사례, 2026-07-29). 후보 수 MISMATCH는 profiling 버전 차이이므로 phase0를 확인한다.

> **fixed fallback은 kolektor solidity 1건만 정상.** kolektor의 실측 solidity tertile(0.9810 / 0.9839)은 폭이 sd의 3.4%로, 52건 중 17건이 그 0.003 폭 안에서 compact_blob / irregular / general로 갈린다 — 측정 노이즈를 subtype으로 승격시키는 상황이라 해당 특징만 fixed로 되돌린다. 다른 4종은 58~118%로 건전하다.

---

## STEP 3 — 결과 확인 (roi_candidates.json + roi_selected.json)

```python
import json
from collections import Counter

for DS in DATASETS:
    roi = S('roi', DS)
    cand_p, sel_p = f"{roi}/roi_candidates.json", f"{roi}/roi_selected.json"
    if not (os.path.exists(cand_p) and os.path.exists(sel_p)):
        print(f"[{DS}] ✗ 산출 없음 — STEP 2 재확인 ({roi})"); continue
    cand = json.load(open(cand_p)); sel = json.load(open(sel_p))
    print(f"\n=== {DS} ({'multi' if is_multi(DS) else 'single'}) ===")
    print(f"  후보 {len(cand)}  →  선택 {len(sel)}")
    # 클러스터 분포
    cl = Counter(r.get('cluster_id') for r in sel)
    print("  cluster 분포:", {k: cl[k] for k in sorted(cl)})
    # subtype 분포 (single/multi 공통 참고)
    st = Counter(r.get('defect_subtype', 'general') for r in sel)
    print("  subtype 분포:", dict(st))
    # deficit 상위 3
    for r in sorted(sel, key=lambda x: x.get('deficit', 0), reverse=True)[:3]:
        print(f"    [{r.get('cluster_id')}|{r.get('cell_key')}] "
              f"score={r.get('roi_score',0):.3f} deficit={r.get('deficit',0):.3f}")
```

**출력 파일** (`S('roi', ds)` 아래):

| 파일 | 내용 |
|------|------|
| `roi_candidates.json` | 전체 스코어링 결과 (image_id, cluster_id, cell_key, roi_score, deficit, prompt, quality_score) |
| `roi_selected.json` | 선택된 top_k ROI 목록 (step5 생성·step4 CN 학습 입력) |
| `roi_summary.md` | 마크다운 테이블 |

> `roi_candidates.json`은 step4 `build_train_jsonl.py --roi_candidates`가, `roi_selected.json`(및 후보)은
> step5 `generate_defects.py --roi_dir $(S('roi',ds))` / `generate_random.py --candidates_json .../roi_candidates.json`이 소비한다.
>
> ⚠️ **`roi_candidates.json`은 전체 후보 풀이라 대용량**(로컬 mtd 실측 ~13MB vs `roi_selected.json` ~15KB, 약 889×). 그러나 **삭제/미생성 불가** — random arm(`generate_random.py`는 selected가 아니라 **candidates 풀에서 무작위 샘플**), ControlNet train jsonl(`build_train_jsonl`), exp1/2/6 품질·커버리지 메트릭이 재소비한다. copy_paste 전용 경로만 쓰는 경우엔 미참조지만, 5종 파이프라인 전체에서는 필수. (슬림 스키마화는 별도 최적화 과제.)

---

## 무결성 / 정직 (`_SPEC.md` §5)

- **사후 튜닝 금지**: `top_k`·`img_diversity_cap`·selection 전략을 결과 보고 후 변경하지 않는다.
- **selection 규격 고정**: 5종 모두 `deficit_aware + realism`. multi/single 차이는 `class_mode`(dataset_config 자동 분기)뿐이며, 이는 exp4v2의 `--class_mode multi` per-class 측정(3종) / aitex·kolektor single-class 측정과 정합한다.
- **`--rarity_temp` 미전달**: realism 정합. rarity 온도 스케일은 legacy(rarity 스코어) 전용이므로 realism selection에 섞지 않는다.
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- `--local_staging`은 CPU selection 단계에 사용 가능(선택). 본 단계 산출은 소형 JSON이라 Drive 직결로 충분.
