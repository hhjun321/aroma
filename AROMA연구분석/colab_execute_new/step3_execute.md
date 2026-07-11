# step3 — `roi_selection.py` (aroma-sym selection) Colab 실행

> **정본**: `_SPEC.md` §3 step3. env·output 루트는 `_SPEC.md`에서 그대로 가져온다. 문서마다 재발명 금지.
> **목적**: phase0(profiling) + step2(prompts) 산출을 읽어 결함 crop × 컨텍스트 빈 후보를 스코어링하고,
> `deficit_aware + realism`으로 ROI 목록을 선별한다 (AROMA arm 생성·CN 학습의 공통 선결).
> **실행 환경**: CPU
> **입력**: `S('profiling', ds)` + `S('prompts', ds)`  →  **출력**: `S('roi', ds)`

---

## 실행 순서 (체인)

```
phase0(profiling) → step1(complexity) → step2(prompts) → [step3(roi_selection)] → step5(CN 학습) → step4(생성) → exp4v2/exp3/exp5/exp6
```

step3은 step2 완료 후 실행한다. step3 산출(`roi_candidates.json` / `roi_selected.json`)은
step5(ControlNet 학습·`build_train_jsonl.py`)와 step4(생성·random arm)의 공통 입력이다.

## 전제 (실행 전 확인)

- phase0 완료 — `S('profiling', ds)`에 `compatibility_matrix.json`(matrix_symmetric 포함), `morphology_features.csv`, `context_features.csv` 존재.
- step2 완료 — `S('prompts', ds)`에 `prompts.json` 존재.
- 데이터셋 v2-1 4종: `severstal · mvtec_leather · mtd · aitex`. **aitex = tiled(256×256/stride128, single-class)**.

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
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step5 산출, step4 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
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

> aitex는 `class_mode=single`로 떠야 정상(자동 분기 기준). `severstal·mvtec_leather·mtd`는 `multi`.

---

## STEP 2 — selection (multi 3종 / single aitex 분기 루프)

`roi_selection.py`를 `is_multi(ds)`로 분기하여 4종 전부 실행한다.

- **공통 (_SPEC §3 step3)**: `--sampling_strategy deficit_aware --score_mode realism --top_k 200 --img_diversity_cap 1`
- **3종(multi) 추가**: `--class_mode multi --class_floor --per_pair_cap_frac 0.05` (stratified pair-aware allocation + class floor)
- **aitex(single)**: 위 3개 플래그를 **제거** (single 기본값으로 축퇴)
- ⚠️ **`--rarity_temp` 미전달** — realism 정합(deficit_aware가 rarity를 온도 스케일하지 않도록 기본값 1.0 유지).

`!python` 매직은 IPython 전용이라 스레드에서 동작하지 않으므로, 루프는 셀 안에서 순차 `!python`으로 실행한다.

```python
# deficit_aware + realism 공통.
#   --img_diversity_cap 1 : 동일 소스 결함 crop((image_path, defect_bbox))을 최대 1회만 선택
#     → 소수 crop 수십 회 반복(다양성 붕괴) 제거. distinct source < top_k인 클래스에만
#       bounded repetition 허용 + 로그. deficit_aware allocation 에만 적용.
#   multi 3종만 --class_mode multi --class_floor --per_pair_cap_frac 0.05 (stratified allocation).
#   aitex(single)는 3개 플래그 제거 → single 축퇴(byte-identical to single 기본).
#   --rarity_temp 미전달 (realism 정합).
for DS in DATASETS:
    os.environ['DS']     = DS
    os.environ['PROF']   = S('profiling', DS)
    os.environ['PROMPTS']= S('prompts', DS)
    os.environ['ROI']    = S('roi', DS)

    if is_multi(DS):   # severstal / mvtec_leather / mtd
        print(f"\n===== {DS}  (multi: class-gated allocation) =====")
        !python $AROMA_SCRIPTS/roi_selection.py \
            --profiling_dir     $PROF \
            --prompts_dir       $PROMPTS \
            --sampling_strategy deficit_aware --score_mode realism \
            --top_k 200 --img_diversity_cap 1 \
            --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
            --output_dir        $ROI
    else:              # aitex (single, tiled)
        print(f"\n===== {DS}  (single: multi 플래그 제거) =====")
        !python $AROMA_SCRIPTS/roi_selection.py \
            --profiling_dir     $PROF \
            --prompts_dir       $PROMPTS \
            --sampling_strategy deficit_aware --score_mode realism \
            --top_k 200 --img_diversity_cap 1 \
            --output_dir        $ROI
```

> 로그 확인 포인트:
> - multi 3종: `stratified_pair_aware` allocation + class별 floor 로그(특정 class가 floor 미달이면 과소 주의).
> - aitex: `--class_mode multi` 관련 로그 없이 single로 진행.
> - 공통: `Saved roi_candidates.json (N), roi_selected.json (M)`.

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
| `roi_selected.json` | 선택된 top_k ROI 목록 (step4 생성·step5 CN 학습 입력) |
| `roi_summary.md` | 마크다운 테이블 |

> `roi_candidates.json`은 step5 `build_train_jsonl.py --roi_candidates`가, `roi_selected.json`(및 후보)은
> step4 `generate_defects.py --roi_dir $(S('roi',ds))` / `generate_random.py --candidates_json .../roi_candidates.json`이 소비한다.
>
> ⚠️ **`roi_candidates.json`은 전체 후보 풀이라 대용량**(로컬 mtd 실측 ~13MB vs `roi_selected.json` ~15KB, 약 889×). 그러나 **삭제/미생성 불가** — random arm(`generate_random.py`는 selected가 아니라 **candidates 풀에서 무작위 샘플**), ControlNet train jsonl(`build_train_jsonl`), exp1/2/6 품질·커버리지 메트릭이 재소비한다. copy_paste 전용 경로만 쓰는 경우엔 미참조지만, 4종 파이프라인 전체에서는 필수. (슬림 스키마화는 별도 최적화 과제.)

---

## 무결성 / 정직 (`_SPEC.md` §5)

- **사후 튜닝 금지**: `top_k`·`img_diversity_cap`·selection 전략을 결과 보고 후 변경하지 않는다.
- **selection 규격 고정**: 4종 모두 `deficit_aware + realism`. multi/single 차이는 `class_mode`(dataset_config 자동 분기)뿐이며, 이는 exp4v2의 `--class_mode multi` per-class 측정(3종) / aitex single-class 측정과 정합한다.
- **`--rarity_temp` 미전달**: realism 정합. rarity 온도 스케일은 legacy(rarity 스코어) 전용이므로 realism selection에 섞지 않는다.
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- `--local_staging`은 CPU selection 단계에 사용 가능(선택). 본 단계 산출은 소형 JSON이라 Drive 직결로 충분.
