# Severstal exp4v2 — Multi-Class 할당 fix 재실행 가이드

> **목적**: `roi_selection.py` multi-class 할당 fix(class_floor + diversity-cap) 적용 후 Severstal exp4v2 재평가.
> c2 회귀(class monoculture·c4 starvation) 구조적 해소 여부 + per-class 비회귀 확인.
> **런타임**: step3/step4 = CPU, exp4v2 = GPU(A100).
> **전제**: prep + phase0 + step1 + step2 완료(불변). fix는 **step3만** 건드림 → 그 상류 재실행 불필요.

---

## 0. 재실행 범위 (왜 step3부터인가)

fix는 `roi_selection.py`(step3 ROI 선택)에만 적용. 따라서:

```
phase0 / step1 / step2        ← 불변, 재실행 불필요
  │
  ▼
step3  roi_selection.py (AROMA, --class_floor 신규)   ← ★재실행 (새 ROI)
  │     random arm = 전략 'random', 새 플래그 미사용 → 결과 불변(1회만 있으면 됨)
  ▼
step4  generate_defects.py (AROMA)                    ← ★재합성 (새 ROI 기반)
  │     generate_random.py → random synth 불변(있으면 skip)
  ▼
exp4v2 (--class_mode multi, --seeds 42 1 2)           ← ★재평가
```

**핵심**: exp4v2 평가에 fix를 반영하려면 반드시 **새 AROMA synth**가 있어야 한다. exp4v2만 단독 재실행하면 옛 synth를 읽어 fix 무반영.

---

## 1. 코드 최신화 + 환경변수

```python
# 최신 코드 (fix 포함) 반영
!cd /content/AROMA && git pull
```

```python
import os, json
from pathlib import Path

# ── 공통 (00_setup에서 설정됨, 누락 대비) ──────────────────────
os.environ['DRIVE']          = os.environ.get('DRIVE', '/content/drive/MyDrive/data/Aroma')
os.environ['AROMA_SCRIPTS']  = os.environ.get('AROMA_SCRIPTS', '/content/AROMA/scripts/aroma')
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}/Aroma"

DATASET_KEY = 'severstal'
os.environ['DATASET_KEY']    = DATASET_KEY

# step3/step4 경로
os.environ['PROFILING_DIR']  = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
os.environ['PROMPTS_DIR']    = f"{os.environ['AROMA_OUT']}/prompts/{DATASET_KEY}"
os.environ['ROI_DIR']        = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}"
os.environ['ROI_DIR_RANDOM'] = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}_random"

# step4 synth 출력 — exp4v2가 읽는 디렉토리와 정합해야 함
#   exp4v2: aroma_synthetic_dir/{ds}/annotations.json, random_synthetic_dir/{ds}/...
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"           # → /severstal 하위
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"    # → /severstal 하위
os.environ['SYNTHETIC_DIR']    = f"{os.environ['AROMA_SYNTH_DIR']}/{DATASET_KEY}"
os.environ['RANDOM_DIR']       = f"{os.environ['RANDOM_SYNTH_DIR']}/{DATASET_KEY}"

# exp4v2 출력 (fix 결과를 옛 결과와 분리 → fresh 디렉토리)
os.environ['EXP4V2_OUT']       = f"{os.environ['AROMA_OUT']}/exp4v2/severstal_multi_fixed"

# step4 배경 = CLIP context prototypes 하위 context 폴더
os.environ['NORMAL_DIR'] = f"{os.environ['DRIVE']}/severstal/context_select/context"

# severstal이 class_mode=multi인지 확인 (dataset_config.json — fix에서 추가됨)
with open(os.environ['DATASET_CONFIG']) as f:
    _cfg = json.load(f)
print("class_mode :", _cfg.get(DATASET_KEY, {}).get('class_mode'))   # 'multi' 기대
print("AROMA_SYNTH_DIR :", os.environ['AROMA_SYNTH_DIR'])
print("EXP4V2_OUT      :", os.environ['EXP4V2_OUT'])
```

> `class_mode`가 `multi`가 아니면 git pull 누락 — step3 multi 플래그가 안 붙는다. 반드시 확인.

---

## 2. Step 3 — ROI 재선택 (AROMA, multi-class 할당)

AROMA arm만 multi 플래그로 재실행. `--class_floor`(per-class 균등 floor) + `--per_pair_cap_frac 0.05`(monoculture cap).

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware \
    --top_k             200 \
    --seed              42 \
    --output_dir        $ROI_DIR \
    --class_mode        multi \
    --class_floor \
    --per_pair_cap_frac 0.05
```

> random arm(`roi/severstal_random`)은 fix 무관(전략 'random'은 새 플래그 미사용). 이미 있으면 재실행 불필요. 없으면 1회:
> ```python
> !python $AROMA_SCRIPTS/roi_selection.py \
>     --profiling_dir $PROFILING_DIR --prompts_dir $PROMPTS_DIR \
>     --sampling_strategy random --top_k 200 --seed 42 \
>     --output_dir $ROI_DIR_RANDOM
> ```

### 2.1 fix 효과 즉시 확인 (★ 학습 전 게이트)

```python
import json, os, collections, math

sel  = json.load(open(f"{os.environ['ROI_DIR']}/roi_selected.json"))
def cls(e):
    import re
    m = re.search(r'class(\d+)', e.get('image_path','')); return m.group(1) if m else 'NA'

by_cls  = collections.Counter(cls(s) for s in sel)
by_pair = collections.Counter((s['cluster_id'], s['cell_key']) for s in sel)
top_k = len(sel); cap = math.ceil(0.05 * top_k)

print("선택 ROI:", top_k)
print("class별 선택:", dict(sorted(by_cls.items())))     # ★ 전 클래스(1~4) > 0 = starvation 해소
print("최다 pair:", by_pair.most_common(3), "| cap:", cap)  # ★ 최다 <= cap = monoculture 해소
```

**합격 기준** (이게 안 되면 학습 의미 없음):
- class 1~4 **전부 count > 0** (이전: c4=0). → starvation 해소.
- 단일 (cluster, cell) pair 최다 픽 **≤ cap(=10)** (이전: c2 단일 pair 11개). → monoculture 해소.
- class 분포가 이전(c1 65.5%) 대비 균등화 (floor=top_k//4=50 부근).

> 불합격 시: git pull 확인 / `class_mode multi`·`--class_floor` 누락 확인 / dataset_config class_mode 확인.

---

## 3. Step 4 — AROMA synth 재합성

새 ROI로 copy-paste 재합성. random synth는 불변(있으면 skip).

```python
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $ROI_DIR \
    --normal_dir  $NORMAL_DIR \
    --output_dir  $SYNTHETIC_DIR \
    --method      copy_paste \
    --n_per_roi   3 \
    --blend_mode  alpha \
    --feather_px  4 \
    --seed        42 \
    --local_staging
```

> random baseline synth가 `$RANDOM_DIR`에 없으면 1회만:
> ```python
> !python $AROMA_SCRIPTS/generate_random.py \
>     --candidates_json $ROI_DIR/roi_candidates.json \
>     --normal_dir $NORMAL_DIR --output_dir $RANDOM_DIR \
>     --top_k 200 --n_per_roi 3 --seed 42 --local_staging
> ```

**합성 클래스 분포 확인** (새 AROMA synth가 다양해졌는지):

```python
import json, os, re, collections
ann = json.load(open(f"{os.environ['SYNTHETIC_DIR']}/annotations.json"))
hist = collections.Counter(
    (re.search(r'class(\d+)', a.get('source_roi') or '') or [None,'NA'])[1] for a in ann)
print("AROMA synth source class 분포:", dict(sorted(hist.items())))   # 전 클래스 분포, c2 단일 archetype 탈피
```

---

## 4. exp4v2 — multi-class 재평가 (3 seed)

새 AROMA synth로 baseline/random/aroma 3조건 학습. 단일 시드 노이즈 배제 위해 `--seeds 42 1 2`. epoch 100(이전 분석 run과 정합), synth_ratio 1.0.

```python
!pip install ultralytics opencv-python-headless -q
!nvidia-smi
```

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal \
    --class_mode multi \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seeds 42 1 2 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --imgsz 640 \
    --baseline_epochs 100 \
    --patience 20 \
    --batch 64 \
    --cache ram \
    --rect \
    --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --resume
```

| 인자 | 값 | 이유 |
|------|----|------|
| `--seeds 42 1 2` | 3 seed | 단일시드(n=1) 노이즈 배제 — fix 효과 확정 필수 |
| `--synth_ratio 1.0` | 1.0 | 이전 분석 run(ratio 1.0)과 정합 비교 |
| `--baseline_epochs 100` | 100 | 이전 run과 동일 |
| `--output_dir` | `..._fixed` | 옛 결과와 분리(혼동 방지). resume도 이 폴더 기준 |

> **소요**: A100 기준 3 seed × 3조건 ≈ 수 시간. `--resume`으로 중단 재개. 세션 한도 시 seed/조건 단위 재개됨.
> **부하 측정 아님**(효과 재현) — load-test 정책 비해당. 단 GPU 장시간이므로 사용자 판단 하 실행.

---

## 5. 결과 확인 — per-class 회복 판정

```python
import json, os
res = json.load(open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json"))
arms = res["severstal"]["yolov8n"]
CONDS = ["baseline", "random", "aroma"]
n_seeds = next((arms[c].get("n_seeds") for c in CONDS if c in arms), "?")
print(f"n_seeds={n_seeds}\n")

classes = sorted({c for cond in CONDS for c in (arms.get(cond,{}).get("per_class") or {})})
print(f"{'class':<6}  " + "  ".join(f"{c:>10}" for c in CONDS) + f"  {'Δ(A-R)':>10}")
print("-"*56)
for cl in classes:
    vals = {}
    row = []
    for cond in CONDS:
        v = ((arms.get(cond,{}).get("per_class") or {}).get(cl) or {}).get("map50")
        vals[cond] = v if isinstance(v,float) else None
        row.append(f"{v:.4f}" if isinstance(v,float) else "N/A")
    a, r = vals.get("aroma"), vals.get("random")
    d = f"{a-r:+.4f}" if isinstance(a,float) and isinstance(r,float) else "N/A"
    print(f"{cl:<6}  " + "  ".join(f"{c:>10}" for c in row) + f"  {d:>10}")
```

**성공 기준**:
- **c2 회귀 해소**: aroma c2 map50 ≥ random c2 (이전: aroma 0.2304 < random 0.2821).
- **다른 클래스 비회귀**: c1/c3/c4 aroma가 random 미만으로 떨어지지 않음.
- **전체 map50**: aroma ≥ baseline.
- 3 seed라 std/CI로 판정 — 단일 seed swing 아님 확인.

---

## 6. 주의 / 비교 무결성

- **single-class 불변 검증**(별도): MVTec single ds를 fix 코드로 step3 재실행 → `roi_selected.json` sha256이 pre-change와 동일해야 함(byte-identical 보장).
- **공정성**: baseline/random은 `_pair_aware_allocation` 미진입 → fix 무영향. random synth 재합성 불필요(전략 'random' 불변).
- **경로**: exp4v2는 `{aroma_synthetic_dir}/severstal/annotations.json` + `images/` 필요. step4 출력 경로(`SYNTHETIC_DIR`)와 정합 확인. n_synth=0 뜨면 경로/annotations image_path 점검.
- **결과 분리**: `EXP4V2_OUT`을 `severstal_multi_fixed`로 둬 옛 `severstal_multi` 결과와 비교 보존. 같은 폴더 재사용 시 `--resume`이 옛 결과를 skip하므로 fresh 폴더 권장.
- **cross-class 균등 caveat**: top_k=200, K=4 → 각 클래스 floor=50 균등. 클래스 **간**은 starvation-first 균등, 클래스 **내**에서만 deficit-aware. per-pair_cap_frac=0.05는 튜닝 seed — 결과 보고 조정 가능.

---

## 부록 — per_pair_cap_frac 스윕 (선택)

monoculture 강도 조절. cap이 작을수록 다양성↑(단 너무 작으면 deficit 신호 약화).

```python
# step3만 cap 바꿔 재실행 → 2.1 게이트로 분포만 빠르게 확인 후 학습 결정
for frac in [0.03, 0.05, 0.10]:
    !python $AROMA_SCRIPTS/roi_selection.py \
        --profiling_dir $PROFILING_DIR --prompts_dir $PROMPTS_DIR \
        --sampling_strategy deficit_aware --top_k 200 --seed 42 \
        --output_dir "$ROI_DIR"_cap{frac} \
        --class_mode multi --class_floor --per_pair_cap_frac {frac}
```
