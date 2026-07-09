# profiling 재검증 — matrix_symmetric emit + 신규 키 검증 (symmetric 트랙 선결 1)

> **목적**: commit `6c8658f`의 clean-grounded SGM 게이트를 쓰려면 `compatibility_matrix.json`에 신규 키(`matrix_symmetric`·`P_def_patch`·`clean_dist`·`symmetric_epsilon`)가 있어야 한다(없으면 `--compat_mode symmetric` **hard-fail**). 본 가이드는 profiling을 재실행해 이 키를 emit하고, 산출이 로컬 검증치와 일치하는지 확인한다.
> **실행 환경**: Colab, **CPU**(SAM fallback 미사용 시 GPU 불요). MVTec/aitex/mtd/severstal ground_truth 마스크 사용.
> **위치**: symmetric 트랙 3선결 중 (1). 이후 (2) `compat_gate_cpu_diagnosis_execute.md` CPU 진단 → (3) τ 사전스캔 → symmetric 활성.
> **⚠️ drift 경고**: 재실행은 GMM 클러스터링·bin_edges를 **재계산**한다. GMM init 확률성으로 legacy `matrix`·`morphology_clusters`가 기존과 **미세하게 달라질 수 있고**, 이는 defect-mode(현행 3-way) 실험 baseline과 로컬 검증치(leather 5-cell 등)에도 영향. → **반드시 기존 profiling 백업 후 재실행하고 §4 drift 체크**.

---

## 1. 환경변수

```python
import os, json
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = f"{os.environ['AROMA_REF']}/scripts"
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', f"{os.environ['AROMA_REF']}/dataset_config.json")

DS = 'mvtec_leather'   # ← mvtec_leather / mtd / aitex / severstal
os.environ['DS'] = DS
os.environ['PROFILING_DIR'] = f"{os.environ['AROMA_OUT']}/profiling/{DS}"
print("DS =", DS, "| PROFILING_DIR =", os.environ['PROFILING_DIR'])
```

## 2. 기존 profiling 백업 (drift 대비 — 필수)

```python
import os, shutil, time
src = os.environ['PROFILING_DIR']
if os.path.exists(src):
    bak = f"{src}_bak_pre_symmetric"
    if not os.path.exists(bak):
        shutil.copytree(src, bak)
        print("백업 생성:", bak)
    else:
        print("백업 이미 존재(재사용):", bak)
else:
    print("기존 profiling 없음 — 신규 생성")
```

## 3. profiling 재실행 (matrix_symmetric emit)

```python
# 코드는 commit 6c8658f 반영본이어야 함 (git clone 최신). 명령·인자는 기존과 동일 — 신규 키는 step6가 자동 추가.
!python $AROMA_SCRIPTS/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    $DS \
    --output_dir     $PROFILING_DIR \
    --num_workers    -1
```

> `--n_gmm_components 5`(기본)·`--max_images`(디버그용, 미지정=전체) 그대로. SAM 미사용(ground_truth 마스크 존재). 컨텍스트 추출(64px patch)·GMM·compat 산출 순으로 진행.

## 4. drift 체크 — legacy matrix가 백업과 일치하나

```python
import json, os
cur = json.load(open(f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"))
bak_path = f"{os.environ['PROFILING_DIR']}_bak_pre_symmetric/compatibility_matrix.json"
if os.path.exists(bak_path):
    bak = json.load(open(bak_path))
    same_bins = cur.get('bin_edges') == bak.get('bin_edges')
    same_matrix = cur.get('matrix') == bak.get('matrix')
    print(f"bin_edges 동일: {same_bins}")
    print(f"legacy matrix 동일: {same_matrix}")
    if not (same_bins and same_matrix):
        print("⚠️ DRIFT — 재실행이 legacy matrix/clusters를 바꿈. defect-mode 실험·로컬 검증치가 영향받음.")
        print("   → defect-mode(현행 3-way)는 백업 profiling으로 유지하거나, 새 baseline으로 재실행 판단 필요.")
    else:
        print("drift 없음 — legacy 재현. 신규 키만 additive 추가됨.")
else:
    print("백업 없음 — drift 비교 불가")
```

> drift가 있으면 GMM 결정론 부재 탓. symmetric 산출 자체는 유효하나, **defect-mode 비교는 백업본 사용** 권장(혼용 금지).

## 5. 신규 키 검증 (로컬 실측치 대조)

```python
import json, os, collections
c = json.load(open(f"{os.environ['PROFILING_DIR']}/compatibility_matrix.json"))
DS = os.environ['DS']

# 5-1. 키 존재
for k in ('matrix','matrix_symmetric','P_def_patch','clean_dist','symmetric_epsilon'):
    print(f"  {k:18s} {'OK' if k in c else 'MISSING'}")
assert 'matrix_symmetric' in c, "matrix_symmetric 없음 — 코드가 6c8658f 미반영이거나 재실행 실패"

ms = c['matrix_symmetric']; pdp = c['P_def_patch']; cd = c['clean_dist']
sym_union = set().union(*[set(r) for r in ms.values()]) if ms else set()
mat_union = set().union(*[set(r) for r in c['matrix'].values()]) if c['matrix'] else set()

print(f"\n[support] legacy matrix union={len(mat_union)}  matrix_symmetric union={len(sym_union)}  clean_dist cells={len(cd)}")
print(f"  symmetric_epsilon={c['symmetric_epsilon']}")

# 5-2. per-cluster max=1.0 (정규화 확인), 빈 cluster는 {}
print("[정규화] cluster별 row max (비어있지 않으면 1.0 기대):")
for k, r in ms.items():
    mx = max(r.values()) if r else None
    print(f"  cluster {k}: {'EMPTY' if not r else f'|cells|={len(r)} max={mx:.6f}'}")

# 5-3. clean_dist·P_def_patch 합=1 확인
print(f"[분포합] clean_dist sum={sum(cd.values()):.4f} (1.0 기대)")
for k, r in pdp.items():
    if r: print(f"  P_def_patch cluster {k} sum={sum(r.values()):.4f} (1.0 기대)")

# 5-4. 로컬 실측 기대치 대조 (leather/mtd만)
EXPECT = {'mvtec_leather': dict(matrix=5, sym=191), 'mtd': dict(matrix=94, sym=205)}
if DS in EXPECT:
    e = EXPECT[DS]
    print(f"\n[로컬 대조 {DS}] matrix {len(mat_union)} vs 기대 {e['matrix']}  |  "
          f"matrix_symmetric {len(sym_union)} vs 기대 {e['sym']}")
    print("  일치하면 재현 성공. 불일치 시 drift(§4)·코드버전 확인.")
else:
    print(f"\n[{DS}] 로컬 기대치 없음 — matrix_symmetric union > legacy matrix union 이면 support 확장 성립.")
```

## 6. 판정 / 다음 단계

- [ ] 5-1 신규 키 4종 존재
- [ ] 5-2 비어있지 않은 cluster row max=1.0 (max-norm)
- [ ] 5-3 clean_dist·P_def_patch 합=1
- [ ] 5-4 leather 5→191 / mtd 94→205 재현 (또는 support 확장)
- [ ] §4 drift 없음 (있으면 defect-mode는 백업본 사용)

통과 시 → **(2)** `compat_gate_cpu_diagnosis_execute.md`로 TV·fallback·flip 진단 → **(3)** symmetric 스케일 τ 사전스캔 → `--compat_mode symmetric --compat_threshold <τ>`로 소규모 생성 검증.

## 무결성

- 신규 키는 **additive** — 기존 `matrix`/`bin_edges`/`deficit` 보존(코드 6c8658f). 단 재실행은 이들을 **재계산**하므로 §4 drift 체크 필수.
- 4종 전부 재실행 후 §5 표 기록. symmetric 활성(τ>0)은 CPU 진단·사전스캔 통과 데이터셋에만(devnote `aroma_compat_gate_clean-grounded_redesign` §5·§6 게이팅).
- leather는 cluster washout(실측 TV 0.13, cluster 0·2 둘 다 compact_blob) → symmetric이 물려도 cluster-무관 clean-plausibility 필터로 degrade. 헤드라인 금지.
- 테스트 코드 신규 작성·pytest 금지(CLAUDE.md).
