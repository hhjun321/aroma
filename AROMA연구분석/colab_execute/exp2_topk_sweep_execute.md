# AROMA Exp 2 — top_k Sweep (coverage-vs-budget) 실행 가이드

**목적**: exp2 ROI 품질을 **여러 top_k 예산 × Random ≥5 seed**로 측정 → AROMA vs Random 예산 효율 + **significance**(단일 seed 소표본 노이즈 제거).
**배경**: top_k=200 단일 + random 단일 seed는 rare-pair 적은 셋서 포화·소표본 노이즈(예: aitex rare-pair ~8개 → 비단조 1.0→0.875는 ±1 아티팩트). AROMA는 결정론이라 seed 무관, Random만 42~46 재선택해 mean±std로 노이즈 제거·유의성 판정. v2-1 4셋(severstal/mvtec_leather/aitex/mtd) 기준.
**런타임**: CPU. **전제**: step3 profiling/prompts가 4셋에 존재. **최신 roi_selection.py**(class_floor stratified 반영) 클론 필수.

> ⚠️ 본 sweep은 AROMA(`deficit_aware`)와 Random 선택을 **각 top_k에서 새로 생성**한다. step3 본 산출(`$AROMA_OUT/roi/`)을 덮어쓰지 않도록 별도 `roi_sweep/` 디렉터리에 저장한다.

---

## 환경변수 설정

```python
import os, json

os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

# sweep 출력 루트
os.environ['SWEEP_ROOT'] = f"{os.environ['AROMA_OUT']}/roi_sweep"
os.environ['EXP2_SWEEP'] = f"{os.environ['AROMA_OUT']}/exp2_sweep"

print("AROMA_OUT  :", os.environ['AROMA_OUT'])
print("SWEEP_ROOT :", os.environ['SWEEP_ROOT'])
print("EXP2_SWEEP :", os.environ['EXP2_SWEEP'])
```

---

## 설정 — top_k 목록 + 데이터셋별 플래그

```python
TOP_K_SWEEP = [20, 30, 50, 100, 200]
# AROMA v2-1 데이터셋 4종 (aitex/mtd 선행: multidomain_integration_verify_execute.md로 prepare+Stage1-3 완료)
DATASETS    = ["severstal", "mvtec_leather", "aitex", "mtd"]
SEED        = 42                        # AROMA deterministic seed (deficit_aware는 RNG 미사용 → 값 불변)
RANDOM_SEEDS = [42, 43, 44, 45, 46]     # random arm ≥5 seed (significance용 — 단일 seed 소표본 노이즈 제거)

# dataset_config에서 class_mode 조회 → multi면 stratified 플래그 추가
with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)

def aroma_class_flags(ds):
    """severstal 등 class_mode=multi 데이터셋은 per-class floor 활성."""
    if CFG.get(ds, {}).get("class_mode") == "multi":
        return ["--class_mode", "multi", "--class_floor"]
    return []

for ds in DATASETS:
    print(f"{ds:18s} class_mode={CFG.get(ds, {}).get('class_mode', 'single')}  "
          f"flags={aroma_class_flags(ds)}")
```

---

## STEP 1 — sweep 실행 (AROMA + Random 재선택)

각 (top_k, dataset)에 대해 AROMA(`deficit_aware`, B1)와 Random 선택을 생성.
`!python` 매직은 스레드/루프 내 신뢰성 위해 의도적으로 `subprocess.run` 사용 (step4 배치와 동일 이유).

```python
import sys, subprocess
from pathlib import Path

AROMA_OUT     = os.environ['AROMA_OUT']
AROMA_SCRIPTS = os.environ['AROMA_SCRIPTS']
SWEEP_ROOT    = os.environ['SWEEP_ROOT']
SCRIPT        = f"{AROMA_SCRIPTS}/roi_selection.py"

def select(ds, strategy, top_k, out_dir, seed=SEED):
    prof = f"{AROMA_OUT}/profiling/{ds}"
    prom = f"{AROMA_OUT}/prompts/{ds}"
    if not Path(f"{prof}").exists() or not Path(f"{prom}").exists():
        return 'skip', f"{ds}: profiling/prompts 없음 (step3 미완)"
    cmd = [sys.executable, SCRIPT,
           '--profiling_dir', prof,
           '--prompts_dir',   prom,
           '--output_dir',    out_dir,
           '--sampling_strategy', strategy,
           '--top_k', str(top_k),
           '--seed',  str(seed)]
    if strategy == 'deficit_aware':
        cmd += ['--img_diversity_cap', '1'] + aroma_class_flags(ds)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = '\n'.join((r.stderr or '').strip().splitlines()[-3:])
        return 'fail', f"{ds}: {tail}"
    return 'ok', ''

# AROMA(deficit_aware)는 결정론 → (k,ds)당 1회. random은 uniform RNG → RANDOM_SEEDS별 재선택.
for K in TOP_K_SWEEP:
    print(f"\n===== top_k = {K} =====")
    for ds in DATASETS:
        a_out = f"{SWEEP_ROOT}/aroma/k{K}/{ds}"
        sa, ma = select(ds, 'deficit_aware', K, a_out)          # 결정론, 1회
        rstats = []
        for S in RANDOM_SEEDS:
            r_out = f"{SWEEP_ROOT}/random/k{K}/seed{S}/{ds}"     # seed별 별도 디렉터리
            sr, mr = select(ds, 'random', K, r_out, seed=S)
            rstats.append(sr)
            if mr: print(f"      seed{S}: {mr}")
        ok_r = all(s == 'ok' for s in rstats)
        icon = '✓' if (sa == 'ok' and ok_r) else ('↷' if ('skip' in [sa]+rstats) else '✗')
        print(f"  {icon} {ds:18s} aroma={sa}  random={len([s for s in rstats if s=='ok'])}/{len(RANDOM_SEEDS)} ok")
        if ma: print(f"      {ma}")
```

---

## STEP 2 — 각 top_k에서 exp2 품질 평가

```python
EXP2_SWEEP = os.environ['EXP2_SWEEP']
EXP2_SCRIPT = f"{AROMA_SCRIPTS}/experiments/exp2_roi_quality.py"

# (K, seed)별 exp2: aroma_root는 결정론(동일), random_root만 seed별. → aroma 지표는
# 매 run 동일값, random 지표만 seed별 변동 → STEP3서 random을 mean±std로 집계.
for K in TOP_K_SWEEP:
    a_root = f"{SWEEP_ROOT}/aroma/k{K}"
    for S in RANDOM_SEEDS:
        r_root = f"{SWEEP_ROOT}/random/k{K}/seed{S}"
        out    = f"{EXP2_SWEEP}/k{K}/seed{S}"
        cmd = [sys.executable, EXP2_SCRIPT,
               '--aroma_roi_dir',  a_root,
               '--random_roi_dir', r_root,
               '--dataset_keys',   *DATASETS,
               '--output_dir',     out,
               '--seed', str(S)]   # exp2 --seed는 equalize 전용(선택엔 무관)
        r = subprocess.run(cmd, capture_output=True, text=True)
        tag = f"k{K}/seed{S}"
        print(f"{tag}: {'ok' if r.returncode == 0 else 'FAIL'}")
        if r.returncode != 0:
            print('\n'.join((r.stderr or '').strip().splitlines()[-5:]))
```

---

## STEP 3 — coverage-vs-budget 집계 + 곡선

```python
import json, pathlib, statistics as st

METRICS = ["context_coverage", "rare_pair_coverage", "entropy", "gini"]
# agg[ds][metric][K] = {'aroma': v(결정론 상수), 'random': [seed별 v ...]}
agg = {}
for K in TOP_K_SWEEP:
    for S in RANDOM_SEEDS:
        p = pathlib.Path(f"{EXP2_SWEEP}/k{K}/seed{S}/exp2_results.json")
        if not p.exists():
            continue
        res = json.load(open(p))
        for ds, d in res.items():
            for m in METRICS:
                node = agg.setdefault(ds, {}).setdefault(m, {}).setdefault(
                    K, {'aroma': None, 'random': []})
                if 'aroma' in d:  node['aroma'] = d['aroma'][m]          # 매 seed run 동일
                if 'random' in d: node['random'].append(d['random'][m])  # seed별 변동

def _mstd(xs):
    if not xs: return (float('nan'), 0.0)
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)

# 지표별 표: AROMA(상수) vs Random mean±std
for m in METRICS:
    print(f"\n=== {m}  (AROMA  vs  Random mean±std, n={len(RANDOM_SEEDS)} seeds) ===")
    print("dataset".ljust(15) + "".join(f"k{K}".rjust(22) for K in TOP_K_SWEEP))
    for ds in DATASETS:
        if ds not in agg or m not in agg[ds]:
            continue
        row = ds.ljust(15)
        for K in TOP_K_SWEEP:
            node = agg[ds][m].get(K)
            if not node or node['aroma'] is None:
                row += "-".rjust(22); continue
            rm, rs = _mstd(node['random'])
            row += f"{node['aroma']:.3f} v {rm:.3f}±{rs:.3f}".rjust(22)
        print(row)
```

### STEP 3b — significance (AROMA vs Random seed 분포)

AROMA는 결정론적 단일값, Random은 n-seed 분포 → **one-sample 관점**: AROMA가 random 분포를 벗어나는가(z-score + 초과 seed 수). + **cross-domain 방향 부호검정**(spec의 paired 대체): 각 지표서 AROMA가 random_mean을 이기는 도메인 수 → sign test.

```python
import statistics as st
KEY_K = 200   # 운영예산 (원하면 다른 K로)
print(f"=== significance @ k{KEY_K}: AROMA vs Random(n={len(RANDOM_SEEDS)}) ===")
print("metric".ljust(20)+"dataset".ljust(15)+"aroma".rjust(8)+"rand(m±s)".rjust(16)+"z".rjust(7)+"beats".rjust(7))
sign = {}   # metric -> [win/lose/tie per domain]
for m in METRICS:
    hi = (m != "gini")   # gini만 낮을수록 우위
    for ds in DATASETS:
        node = agg.get(ds, {}).get(m, {}).get(KEY_K)
        if not node or not node['random'] or node['aroma'] is None:
            continue
        a = node['aroma']; rm, rs = _mstd(node['random'])
        z = (a - rm)/rs if rs > 1e-9 else (0.0 if a == rm else float('inf'))
        beats = sum((a > r) if hi else (a < r) for r in node['random'])
        print(m.ljust(20)+ds.ljust(15)+f"{a:.3f}".rjust(8)+f"{rm:.3f}±{rs:.3f}".rjust(16)
              +f"{z:+.1f}".rjust(7)+f"{beats}/{len(node['random'])}".rjust(7))
        win = (a > rm) if hi else (a < rm)
        sign.setdefault(m, []).append(1 if win else (0 if a == rm else -1))

# cross-domain 방향 부호검정 (leather 등 포화 도메인은 tie로 자동 약화)
print(f"\n=== cross-domain sign @ k{KEY_K} (AROMA가 random_mean 이긴 도메인 수) ===")
from math import comb
for m in METRICS:
    s = sign.get(m, [])
    w = sum(v == 1 for v in s); l = sum(v == -1 for v in s); t = sum(v == 0 for v in s)
    nb = w + l
    # 양측 부호검정 p (동점 제외): P(X>=max(w,l)) 2배
    p = min(1.0, 2*sum(comb(nb, k) for k in range(max(w, l), nb+1))/(2**nb)) if nb else 1.0
    print(f"  {m.ljust(20)} win={w} lose={l} tie={t}  sign p={p:.3f}"
          + ("  (n작아 참고용)" if nb < 6 else ""))
```

> **해석**: `beats=5/5` + `z` 큰 양수(gini는 큰 음수) = AROMA가 random 분포를 확실히 벗어남(도메인 단위 유의). cross-domain sign은 4셋(포화 leather는 tie)이라 n 작음 → **참고용**; 완전 유의는 도메인 수↑ 또는 per-domain seed 분포로 판단. context_coverage는 random 우위 예상(tradeoff, 정직 병기).

```python
# matplotlib 곡선: 데이터셋 × metric 그리드 (AROMA red, Random mean±std band)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
dss = [d for d in DATASETS if d in agg]
fig, axes = plt.subplots(len(dss), len(METRICS),
                         figsize=(4*len(METRICS), 3*len(dss)), squeeze=False)
for i, ds in enumerate(dss):
    for j, m in enumerate(METRICS):
        ax = axes[i][j]
        Ks = [K for K in TOP_K_SWEEP if agg[ds].get(m, {}).get(K)]
        if Ks:
            # AROMA (결정론 line)
            ay = [agg[ds][m][K]['aroma'] for K in Ks]
            ax.plot(Ks, ay, marker="o", color="#d62728", label="aroma")
            # Random mean ± std band
            rm = [_mstd(agg[ds][m][K]['random'])[0] for K in Ks]
            rs = [_mstd(agg[ds][m][K]['random'])[1] for K in Ks]
            ax.plot(Ks, rm, marker="s", color="#1f77b4", label="random(mean)")
            ax.fill_between(Ks, np.array(rm)-np.array(rs), np.array(rm)+np.array(rs),
                            color="#1f77b4", alpha=0.2)
        if i == 0: ax.set_title(m, fontsize=10)
        if j == 0: ax.set_ylabel(ds, fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0 and j == 0: ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("top_k")
plt.tight_layout()
out_png = f"{os.environ['AROMA_OUT']}/exp2_sweep/coverage_vs_budget.png"
plt.savefig(out_png, bbox_inches="tight", dpi=130)
print("saved:", out_png)
```

---

## 해석 가이드

| 관찰 | 의미 |
|------|------|
| 작은 top_k서 AROMA rare_pair > Random | **deficit-aware 효율** — 적은 예산으로 rare pair 더 많이 충전 (핵심 주장) |
| 큰 top_k서 양쪽 1.0 수렴 | 예산 충분 → 포화 (변별 사라짐). 작은 예산 영역이 결정적 |
| context_coverage: 작은 top_k서 AROMA가 Random과 동등↑ | B1 효과 (coverage-first) — breadth도 안 짐 |
| severstal: B1 후 ctx_cov AROMA ≥ Random | 이전 0.710<0.785 역전 확인 |
| rare<top_k 셋(aitex ~8, leather 포화): top_k≥rare서 1.0 포화 | 작은 top_k 구간만 변별. aitex는 소표본이라 seed 분산 큼(band 넓음) |

**핵심 서사 (논문 §4.1)**: endpoint 단일값 대신 **coverage-vs-budget 곡선 + Random seed band(±std)**로 "AROMA가 더 적은 예산으로 rare-pair 커버" 효율을 입증. AROMA 우위 지표 = **rare_pair·entropy·gini(균형)**; **context_coverage는 random 우위(breadth-vs-depth tradeoff) 정직 병기**. significance는 per-domain z-score/beats(5/5)로, cross-domain sign은 참고용(4셋 n작음).

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$AROMA_OUT/roi_sweep/aroma/k{K}/{ds}/roi_selected.json` | AROMA top_k별 선택(결정론) |
| `$AROMA_OUT/roi_sweep/random/k{K}/seed{S}/{ds}/roi_selected.json` | Random top_k×seed별 선택 |
| `$AROMA_OUT/exp2_sweep/k{K}/seed{S}/exp2_results.json` | top_k×seed별 지표 |
| `$AROMA_OUT/exp2_sweep/coverage_vs_budget.png` | 데이터셋×지표 곡선(random mean±std band) |

---

## 주의

- **B1 반영 확인**: `roi_selection.py:592` 분기가 coverage-first(pair당 1개, deficit 순)인 최신 코드여야 함. 구버전이면 severstal ctx_cov 역전 안 보임.
- **rare-pair 수별 변별 구간(v2-1)**: severstal(rare 多)/mtd는 전 top_k 변별, aitex(~8)/leather(포화)는 작은 K 구간·소표본만 변별 — 곡선의 포화 평탄 + 넓은 seed band로 확인.
- **멀티클래스**: 4셋 전부 `class_mode='multi'` → AROMA에 `--class_mode multi --class_floor` 자동(STEP1 `aroma_class_flags`). Random은 floor 없이 uniform(control).
- **공정성**: 각 top_k에서 AROMA/Random 동일 K로 재선택 → n_selected 동일 → equalize no-op.
- **multi-seed**: AROMA(`deficit_aware`)는 RNG 미사용 → seed 무관 상수(1회 생성). Random만 `RANDOM_SEEDS`(42~46)로 재생성 → mean±std. 이로써 단일 seed 소표본 노이즈(aitex rare_pair 비단조 등) 제거 + significance(z-score/beats/sign) 산출.
- **significance 한계**: cross-domain sign은 4셋(포화 leather tie)이라 n 작음 → 참고용. 도메인 단위 유의는 per-domain z-score/beats(5/5)로 판단. context_coverage는 random 우위(tradeoff) 정직 병기.
