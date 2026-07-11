# phase0 — `distribution_profiling.py` (sym_final 정본)

> **목적**: v2-1 4종(`severstal · mvtec_leather · mtd · aitex`)의 분포 프로파일링을 stage-first 루트(`sym_final/profiling/{ds}`)로 재생성한다. symmetric 게이트(clean-grounded SGM)를 쓰려면 `compatibility_matrix.json`에 신규 키(`matrix_symmetric`·`P_def_patch`·`clean_dist`·`symmetric_epsilon`)가 있어야 하므로, 본 단계에서 그 키의 존재를 **hard assert** 한다(없으면 `--compat_mode symmetric` 자체가 hard-fail).
> **실행 환경**: **CPU**. MVTec/aitex/mtd/severstal은 ground_truth 마스크 사용 → SAM 불요(마스크 없으면 Otsu fallback 자동).
> **전제**: 저장소는 commit `6c8658f` 이상(신규 키를 emit하는 코드)이어야 한다. **추가로 `b1bb497` 이상**이면 `context_features.csv`가 이미지 실제 dim(`image_w`/`image_h`) 컬럼을 방출하고(step3.5 정밀 배치·스케일의 전제), `31ee0aa` 이상이면 `image_id`가 **클래스-고유키**(`{defect_type}_{stem}`, good=`_{stem}`)로 생성된다(MVTec leather stem 충돌 해소). aitex는 tiled 데이터셋(`aitex_tiled/train/good`, single-class)이 이미 준비되어 있어야 하며 `dataset_config.json`의 `image_dir`가 이를 가리키므로 자동 해소된다.
>
> **로컬 재검증(2026-07-11, mtd 20-ROI, `AROMA연구분석/local_revalidation_mtd20_20260711.md`)**: 신 코드로 4-stage end-to-end 무결 동작 확인 — `image_w/image_h`로 patch-격자 dim 과소추정(mean 63.2px)이 **0px으로 완전 폐쇄**(956/956), step3.5 위치 소비 clamp-free 40/40.
> **실행 순서 체인**: **phase0 → step1 → step2 → step3 → step4(ControlNet 학습) → step5(생성) → exp3/exp4v2/exp5/exp6**. 본 문서는 체인의 최상류.
> **경로 주의**: `distribution_profiling.py`는 `scripts/`(루트)에 있다 — `$AROMA_REF/scripts/...`로 호출한다. (`compute_complexity.py`/`prompt_generation.py`만 `scripts/aroma/`.)

---

## STEP 0 — 공통 환경 셀 (sym_final 전 문서 동일 — 그대로 복사)

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

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
```

---

## STEP 1 — 기존 profiling 백업 (drift 대비 — 필수)

재실행은 GMM 클러스터링·bin_edges를 **재계산**한다. GMM은 시드 고정(`random_state=42`)이라 현 코드 run-to-run은 재현되지만, **업로드된 구 profiling은 구 코드 산물**이라 cluster 수·라벨이 다를 수 있다. drift 시 defect-mode·symmetric 모두 신 profiling으로 통일해야 하므로(구 것 혼용 금지), 먼저 백업한다.

```python
import shutil

for DS in DATASETS:
    src = S('profiling', DS)
    if os.path.exists(src):
        bak = f"{src}_bak_pre_symmetric"
        if not os.path.exists(bak):
            shutil.copytree(src, bak); print("백업 생성:", bak)
        else:
            print("백업 이미 존재(재사용):", bak)
    else:
        print("기존 profiling 없음 — 신규 생성:", DS)
```

---

## STEP 2 — profiling 실행 (DATASETS 4종 루프)

```python
for DS in DATASETS:
    os.environ['DS'] = DS
    os.environ['PROF'] = S('profiling', DS)
    print(f"\n===== profiling: {DS} → {os.environ['PROF']} =====")
    # distribution_profiling.py 는 scripts/ 루트에 있음 → $AROMA_REF/scripts/
    !python $AROMA_REF/scripts/distribution_profiling.py \
        --dataset_config $DATASET_CONFIG \
        --dataset_key    $DS \
        --output_dir     $PROF \
        --num_workers    -1
```

> `--num_workers -1` = 전 코어 사용(단일 데이터셋 순차 실행이므로 중첩 없음). SAM이 필요한 도메인이면 `--sam_checkpoint /content/sam_vit_h.pth` 추가. `--n_gmm_components`(기본 5)·`--max_images`(디버그용)는 그대로.
> aitex는 `dataset_config.json`의 `image_dir`가 `aitex_tiled/train/good`(single-class)이므로 별도 인자 없이 tiled 규약이 자동 적용된다.

---

## STEP 3 — 출력 확인 + symmetric 신규 키 assert + drift 체크

### 3-1. 산출 파일 존재 확인

```python
from pathlib import Path
for DS in DATASETS:
    p = Path(S('profiling', DS))
    print(f"\n[{DS}]")
    for fname in ['morphology_features.csv', 'context_features.csv',
                  'distribution_analysis.json', 'morphology_clusters.json',
                  'compatibility_matrix.json', 'deficit_analysis.json']:
        print(f"  {'OK  ' if (p / fname).exists() else '누락 '} {fname}")
```

### 3-2. symmetric 신규 키 hard assert (없으면 --compat_mode symmetric hard-fail)

```python
EXPECT = {'mvtec_leather': dict(matrix=5, sym=191), 'mtd': dict(matrix=94, sym=205)}  # 로컬 실측 기대치

for DS in DATASETS:
    c = json.load(open(f"{S('profiling', DS)}/compatibility_matrix.json"))
    print(f"\n===== [{DS}] symmetric 키 검증 =====")

    # (a) 키 존재 — 하나라도 없으면 코드가 6c8658f 미반영이거나 재실행 실패
    for k in ('matrix', 'matrix_symmetric', 'P_def_patch', 'clean_dist', 'symmetric_epsilon'):
        print(f"  {k:18s} {'OK' if k in c else 'MISSING'}")
    assert 'matrix_symmetric' in c and 'P_def_patch' in c \
        and 'clean_dist' in c and 'symmetric_epsilon' in c, \
        f"[{DS}] symmetric 키 누락 — --compat_mode symmetric 사용 불가. 코드 버전(>=6c8658f) 확인 후 재실행."

    ms, pdp, cd = c['matrix_symmetric'], c['P_def_patch'], c['clean_dist']
    sym_union = set().union(*[set(r) for r in ms.values()]) if ms else set()
    mat_union = set().union(*[set(r) for r in c['matrix'].values()]) if c['matrix'] else set()
    print(f"  [support] matrix union={len(mat_union)}  matrix_symmetric union={len(sym_union)}  "
          f"clean_dist cells={len(cd)}  epsilon={c['symmetric_epsilon']}")

    # (b) 정규화: 비어있지 않은 cluster row max=1.0 (max-norm)
    for k, r in ms.items():
        mx = max(r.values()) if r else None
        print(f"    cluster {k}: {'EMPTY' if not r else f'|cells|={len(r)} max={mx:.6f}'}")

    # (c) 분포합=1
    print(f"  [분포합] clean_dist sum={sum(cd.values()):.4f} (1.0 기대)")
    for k, r in pdp.items():
        if r: print(f"    P_def_patch cluster {k} sum={sum(r.values()):.4f} (1.0 기대)")

    # (d) 로컬 실측 대조 (leather/mtd만)
    if DS in EXPECT:
        e = EXPECT[DS]
        print(f"  [로컬 대조] matrix {len(mat_union)} vs 기대 {e['matrix']}  |  "
              f"symmetric {len(sym_union)} vs 기대 {e['sym']}  "
              f"→ {'재현' if len(sym_union)==e['sym'] else 'drift/코드버전 확인'}")
    else:
        print(f"  [{DS}] 로컬 기대치 없음 — symmetric union > matrix union 이면 support 확장 성립.")
```

### 3-2b. context_features 신규 스키마 확인 (image_id 고유키 + 실제 dim 컬럼)

`b1bb497`/`31ee0aa` 이상에서 재실행했는지 확인. `image_w`/`image_h` 컬럼이 있으면 step3.5가 patch-격자 추정 대신 **실제 dim**을 써 위치·스케일이 정밀해진다(없으면 자동 grid fallback — 무오류이나 edge-flush가 실제 가장자리보다 최대 ~63px 안쪽).

```python
import csv
for DS in DATASETS:
    p = f"{S('profiling', DS)}/context_features.csv"
    with open(p, encoding='utf-8') as f:
        rd = csv.DictReader(f); cols = rd.fieldnames
        first = next(rd, {})
    has_dim = ('image_w' in cols) and ('image_h' in cols)
    # image_id 고유키: good=_{stem}, defect={class}_{stem}
    sample_ids = [first.get('image_id','')]
    print(f"[{DS}] image_w/image_h 컬럼: {'OK' if has_dim else '누락(구 코드 — grid fallback)'}"
          f"  | image_id 예: {sample_ids}")
    assert has_dim, f"[{DS}] image_w/image_h 없음 — b1bb497 이상으로 재실행해야 step3.5 정밀 배치 성립."
```

> DRIFT 주의: image_id 고유키(`31ee0aa`)로 **전 데이터셋 image_id 문자열이 변경**된다(mtd/severstal/aitex는 stem이 이미 고유라 로직 무영향, leather는 stem 충돌 해소로 클러스터 정상화). 구 profiling과 문자열 불일치 → step3(roi_selection)·step3.5(clean_bg) 조인은 반드시 **동일 재실행 산출물**끼리 사용(구/신 혼용 금지).

### 3-3. drift 체크 — 백업 대비 legacy matrix 재현 여부

```python
for DS in DATASETS:
    cur_path = f"{S('profiling', DS)}/compatibility_matrix.json"
    bak_path = f"{S('profiling', DS)}_bak_pre_symmetric/compatibility_matrix.json"
    if not os.path.exists(bak_path):
        print(f"[{DS}] 백업 없음 — drift 비교 불가 (신규 생성)"); continue
    cur, bak = json.load(open(cur_path)), json.load(open(bak_path))
    same_bins   = cur.get('bin_edges') == bak.get('bin_edges')
    same_matrix = cur.get('matrix') == bak.get('matrix')
    print(f"[{DS}] bin_edges 동일={same_bins}  legacy matrix 동일={same_matrix}", end='  ')
    if same_bins and same_matrix:
        print("→ drift 없음(신규 키만 additive 추가)")
    else:
        print("→ ⚠️ DRIFT: 재실행이 legacy matrix/clusters를 바꿈. defect-mode 실험·로컬 검증치가 영향받음. "
              "defect-mode·symmetric 모두 신 profiling으로 통일(구 것 혼용 금지).")
```

---

## 판정 / 다음 단계

- [ ] 3-1 산출 파일 6종 존재 (4 데이터셋 전부)
- [ ] 3-2 신규 키 4종 존재 (assert 통과), 비어있지 않은 cluster row max=1.0, clean_dist·P_def_patch 합=1
- [ ] 3-2 leather 5→191 / mtd 94→205 재현 (또는 support 확장)
- [ ] 3-2b `image_w/image_h` 컬럼 존재(assert 통과), image_id 고유키 포맷 확인
- [ ] 3-3 drift 없음 (있으면 신 profiling으로 통일, 혼용 금지)

통과 시 → **step1**(`compute_complexity.py`, 입력 `S('profiling',ds)`).

---

## 무결성 / 정직 (_SPEC §5)

- 신규 키는 **additive** — 기존 `matrix`/`bin_edges`/`deficit` 보존. 단 재실행은 이들을 **재계산**하므로 STEP 1 백업 + 3-3 drift 체크 필수.
- output 경로는 반드시 stage-first `S('profiling', ds)`(=`sym_final/profiling/{ds}`). ds-first 금지 — exp3/5/6 루트 규약(root/{ds})과 깨진다.
- leather는 cluster washout(실측 TV 0.13) 경향 → symmetric이 물려도 cluster-무관 clean-plausibility 필터로 degrade. 헤드라인 금지.
- aitex는 tile-level·single-class → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- 사후 튜닝 금지(결과 보고 후 파라미터 변경 금지). 테스트 코드 신규 작성·pytest 금지(CLAUDE.md) — 검증은 본 Colab 셀 실행으로.
