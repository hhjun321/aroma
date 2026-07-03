# AROMA Exp 2 — top_k Sweep (coverage-vs-budget) 실행 가이드

**목적**: exp2 ROI 품질을 **여러 top_k 예산**에서 측정해 AROMA vs Random의 *예산 효율* 비교.
**배경**: top_k=200 단일 측정은 rare-pair 수가 적은 셋서 커버리지 1.0 포화 → 변별 소실. rare-pair > top_k 인 셋(severstal 358, carpet 1161)만 변별. 작은 top_k가 포화 해소 + B1(coverage-first deficit) 효과 노출. deficit-rich 5셋(carpet/leather/macaroni/fryum/severstal) 기준.
**런타임**: CPU. **전제**: step3 profiling/prompts가 5셋에 존재 (exp2_execute STEP1과 동일 입력). **B1 반영된 최신 roi_selection.py** 클론 필수.

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
import json, pathlib

METRICS = ["context_coverage", "rare_pair_coverage", "entropy", "gini"]
# 데이터셋별로 metric × top_k × strategy 수집
data = {}  # data[ds][metric][strategy] = [(K, value), ...]
for K in TOP_K_SWEEP:
    p = pathlib.Path(f"{EXP2_SWEEP}/k{K}/exp2_results.json")
    if not p.exists():
        print(f"[MISS] k{K}")
        continue
    res = json.load(open(p))
    for ds, d in res.items():
        for strat in ("aroma", "random"):
            if strat not in d:
                continue
            for m in METRICS:
                data.setdefault(ds, {}).setdefault(m, {}).setdefault(strat, []).append((K, d[strat][m]))

# 텍스트 표: rare_pair_coverage vs top_k (핵심 지표)
print(f"\n=== rare_pair_coverage (AROMA / Random) vs top_k ===")
print(f"{'dataset':18}" + "".join(f"{('k'+str(K)):>16}" for K in TOP_K_SWEEP))
for ds in DATASETS:
    if ds not in data or "rare_pair_coverage" not in data[ds]:
        continue
    a = dict(data[ds]["rare_pair_coverage"].get("aroma", []))
    r = dict(data[ds]["rare_pair_coverage"].get("random", []))
    row = f"{ds:18}"
    for K in TOP_K_SWEEP:
        av = a.get(K); rv = r.get(K)
        cell = f"{av:.3f}/{rv:.3f}" if av is not None and rv is not None else "-"
        row += f"{cell:>16}"
    print(row)
```

```python
# matplotlib 곡선: 데이터셋 × metric 그리드 (AROMA red, Random blue)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dss = [d for d in DATASETS if d in data]
fig, axes = plt.subplots(len(dss), len(METRICS),
                         figsize=(4*len(METRICS), 3*len(dss)), squeeze=False)
for i, ds in enumerate(dss):
    for j, m in enumerate(METRICS):
        ax = axes[i][j]
        for strat, color in (("aroma", "#d62728"), ("random", "#1f77b4")):
            pts = sorted(data[ds].get(m, {}).get(strat, []))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=color, label=strat)
        if i == 0:
            ax.set_title(m, fontsize=10)
        if j == 0:
            ax.set_ylabel(ds, fontsize=9)
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
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
| rare<top_k 셋(fryum 45, macaroni 76, leather 134): top_k≥rare서 1.0 포화 | 작은 top_k 구간만 변별 |

**핵심 서사 (논문 §4.1)**: endpoint(top_k=200) 단일값 대신 **coverage-vs-budget 곡선**으로 "AROMA가 더 적은 예산으로 rare-pair/context 커버" 효율을 입증. entropy/gini는 균등성 보상(random 강점)이라 보조 지표로만.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$AROMA_OUT/roi_sweep/{aroma,random}/k{K}/{ds}/roi_selected.json` | top_k별 선택 |
| `$AROMA_OUT/exp2_sweep/k{K}/exp2_results.json` | top_k별 지표 |
| `$AROMA_OUT/exp2_sweep/coverage_vs_budget.png` | 데이터셋×지표 곡선 |

---

## 주의

- **B1 반영 확인**: `roi_selection.py:592` 분기가 coverage-first(pair당 1개, deficit 순)인 최신 코드여야 함. 구버전이면 severstal ctx_cov 역전 안 보임.
- **rare-pair 수별 변별 구간**: severstal(358)/carpet(1161)은 전 top_k 변별, fryum(45)/macaroni(76)/leather(134)는 top_k < rare 구간(작은 K)서만 변별 — 곡선서 포화 평탄 구간으로 확인.
- **severstal 멀티클래스**: AROMA에 `--class_mode multi --class_floor` 자동 적용(STEP1 `aroma_class_flags`). cross-class deficit은 여전히 floor-uniform(점검 #4) — 본 sweep은 그 한계와 독립.
- **공정성**: 각 top_k에서 AROMA/Random 동일 K로 재선택 → n_selected 동일 → equalize no-op. Random은 uniform `rng.choice`(seed 42).
