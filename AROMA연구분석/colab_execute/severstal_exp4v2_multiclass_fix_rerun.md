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

AROMA arm만 multi 플래그로 재실행. `--class_floor`(per-class 균등 floor) + `--per_pair_cap_frac 0.05`(monoculture cap) + `--img_diversity_cap 1`(Fix4: source-defect 다양성 cap).

> ### ★ Fix4 — `--img_diversity_cap 1` (소스 결함 다양성 붕괴 제거)
>
> 이전 run의 치명적 confound: `roi_selected.json` entries=1690 인데 **distinct (image_path, defect_bbox)=88** (소수 crop을 최대 87회, 평균 ~19회 반복). 후보 pool은 3620 distinct source 였으나 3532개가 미사용. 원인은 image-blind roi_score 동점 + stable-sort head 슬라이싱(pool ceiling 아님). 그 결과 AROMA가 CASDA(distinct 1692) 대비 결함 외형 다양성에서 ~19배 불리 → AROMA 저성능의 1차 교란요인.
>
> `--img_diversity_cap 1` 은 동일 (image,bbox) crop 을 **최대 1회만** 선택하게 강제하고, 동점 해소용 deterministic per-source jitter(hashlib 기반, 재현 가능)를 켜서 서로 다른 pair 가 서로 다른 이미지를 surface 하도록 한다. 결과: distinct (image,bbox) ≈ min(top_k, distinct_sources). class_floor 와 per_pair_cap 은 그대로 유지된다. distinct source 가 그 class 의 floor 보다 적은 class 에 한해서만 bounded repetition 을 허용하고 로그를 남긴다(아래 c2 참고). 생략(=None) 시 기존 동작과 byte-identical(single-class headline 경로 불변).
>
> **c2(병목, distinct=117) 상호작용**: floor=422 > distinct 117 이므로 c2는 distinct 117 전량 선택 후 부족분(~305 슬롯)을 **bounded repetition** 으로 채운다(로그: `class 'c2' below floor` 또는 `bounded repetition`). 이는 데이터 부족의 정직한 노출이며(c2 diversity ceiling=117 고정), AROMA 실패가 아니다. c1/c3/c4는 distinct 가 floor 보다 많으므로 전부 distinct(반복 0)로 채워진다.

> ### ★ top_k 상향 (synth pool 확대) — 공정성 + per-class 가용성(c2-full) 노트
>
> **왜 200 → 1690 인가**: 이전 run은 `--top_k 200 × --n_per_roi 3 = 600` synth pool이었다. n_real_train≈2534(val_frac=0.3) 대비 synth:real ≈ 0.24:1 로, copy-paste augmentation의 유효 band(synth:real 1:1~3:1)에 한참 못 미쳐 augmentation이 baseline을 못 이겼다. 이번엔 **band 안으로** 끌어올린다.
> - **pool = top_k × n_per_roi**. n_per_roi=3 고정 → pool의 유일한 lever는 top_k.
> - synth_ratio=1.0(primary) cap = max(1, int(2534×1.0)) = **2534**. 이 한 번의 generation으로 1.0/1.5/2.0 스윕을 모두 먹이기 위해, **가장 큰 ratio(2.0, cap=5068)** 기준으로 한 번만 생성한다: `top_k=1690 → pool=1690×3=5070 ≥ 5068`. 학습 cap 트리밍은 무료(생성이 비싼 단계)이므로 1.0/1.5는 이 pool에서 trim down 된다.
> - **per-class 가용성(c2-full)**: Severstal distinct ROI = c1≈2881, **c2≈117(병목)**, c3≈10539, c4≈1843. top_k=1690, K=4 → floor=1690//4=**422 ROIs/class**. c1/c3/c4는 422를 distinct로 채우지만, c2는 distinct가 117뿐 → roi_selection의 availability clamp(L584-589)가 c2 quota를 117로 줄이고, 남는 ~305 슬롯을 c1/c3/c4에 deficit-mass 순으로 재분배(L590-606)한다. 결과: **{c1/c3/c4 상향, c2=117 distinct 전량}** = 사용자 directive(희소 class 전량 사용 + 풍부 class 비례 확대) 그대로. **추가 코드 불필요**(emergent). c2 image 수 = 117×3 = 351로 고정(모든 조건 동일); c2 diversity ceiling은 117로 영구 고정이며 어떤 ratio도 이를 못 올린다 → c2 map50이 낮으면 data-scarcity 발견이지 AROMA 실패가 아니다.
> - **only 1:1만 돌릴 거면** top_k=845(pool 2535 ≥ 2534)로 충분(floor=211, c2는 동일하게 117 clamp). 스윕 계획이면 1690 권장.

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware \
    --top_k             1690 \
    --seed              42 \
    --output_dir        $ROI_DIR \
    --class_mode        multi \
    --class_floor \
    --per_pair_cap_frac 0.05 \
    --img_diversity_cap 1
```

> 실행 후 stdout 로그에서 `Source-defect diversity: distinct (image,bbox)=...` 줄을 확인한다. Fix 전(88)과 달리 **1500~1690 근처**(c2 117 반복분만큼만 1690 미만)여야 한다. `roi_summary.md` 의 "Source-defect diversity" 섹션에도 동일 수치가 기록된다.

> random arm(`roi/severstal_random`)은 fix 무관(전략 'random'은 새 플래그 미사용). **단 pool parity를 위해 top_k는 AROMA와 동일하게 1690** 으로 맞춰 재생성한다(이전 200 캐시가 있으면 삭제/덮어쓰기 필요 — pool이 600→5070으로 달라짐):
> ```python
> !python $AROMA_SCRIPTS/roi_selection.py \
>     --profiling_dir $PROFILING_DIR --prompts_dir $PROMPTS_DIR \
>     --sampling_strategy random --top_k 1690 --seed 42 \
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

**합격 기준** (이게 안 되면 학습 의미 없음) — top_k=1690 기준:
- class 1~4 **전부 count > 0** (이전: c4=0). → starvation 해소.
- 단일 (cluster, cell) pair 최다 픽 **≤ cap = ceil(0.05×1690) = 85** (이전 top_k=200 시 cap=10). → monoculture 해소. (1:1만 돌릴 때 top_k=845면 cap=ceil(0.05×845)=43.)
- class 분포: floor = top_k//4 = **422**. c1/c3/c4 ≈ 422+ (surplus 재분배로 일부 더 큼), **c2 ≈ 117** (= c2 distinct 가용 전량, floor 미달이 정상 — availability clamp 작동 증거). c2가 422가 아니라 ~117로 나와야 c2-full directive가 코드에서 제대로 작동한 것.
- 큰 top_k에서 per_pair_cap이 c2(distinct pair 적음)에 거의 안 걸릴 수 있음 — c2 최다 pair가 cap보다 작아도 정상.

> 불합격 시: git pull 확인 / `class_mode multi`·`--class_floor` 누락 확인 / dataset_config class_mode 확인.
> **공정성 노트**: AROMA·random·CASDA 모두 동일 pool(≥ 학습 cap)로 맞춘다. AROMA의 c2-full+abundant-scaled 형태는 CASDA에도 `--per_class_cap`으로 동일 적용(Cell B) → 어느 arm도 특혜 없음. 선택 **전략**만 다르고(random vs CASDA-suitability vs AROMA-deficit) pool 규모·n_per_roi·학습 budget은 동일.

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

> random baseline synth: pool parity를 위해 **top_k=1690**(AROMA와 동일), n_per_roi=3 유지. 이전 top_k=200(pool 600) 캐시가 있으면 반드시 재생성(pool이 달라짐):
> ```python
> !python $AROMA_SCRIPTS/generate_random.py \
>     --candidates_json $ROI_DIR/roi_candidates.json \
>     --normal_dir $NORMAL_DIR --output_dir $RANDOM_DIR \
>     --top_k 1690 --n_per_roi 3 --seed 42 --local_staging
> ```
> **n_per_roi=3 은 AROMA·random·CASDA 세 생성기 전부 동일**(생성 단계 parity 불변식; exp4v2 학습에는 --n_per_roi 인자 없음). pool의 lever는 top_k(AROMA/random)·per_class_cap(CASDA) 뿐이다.

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
| `--synth_ratio 1.0` | 1.0 (primary) | cap=max(1,int(2534×1.0))=**2534**. pool(5070)≥cap → trim to 2534 = 등가 budget. 스윕: 1.5(cap=3801)·2.0(cap=5068) — pool 5070이 2.0까지 커버 |
| `--baseline_epochs 100` | 100 | 이전 run과 동일 |
| `--output_dir` | `..._fixed` | 옛 결과와 분리(혼동 방지). resume도 이 폴더 기준 |

> **공정성 (load-bearing)**: cap = max(1, int(n_real_train × synth_ratio))이 random/aroma(그리고 4조건 doc의 casda)에 **동일** 적용된다. 모든 pool이 cap 이상이므로 세 synth arm 전부 정확히 cap 만큼만 공급 → `n_synth_train` 3조건 동일(이게 fairness 게이트). baseline은 synth=0(설계상). 어떤 arm도 더 많은 synth를 받지 않는다. 보고는 결과 그대로(baseline이 이겨도, AROMA가 random/CASDA에 져도) — rig 금지.
> **synth:real 해석 주의**: cap=2534는 n_real_train(2534) 기준 1:1. YOLO는 background negative(~2534장)도 보므로 effective real≈5068이지만, 문헌의 band(1:1~3:1)는 **synth:real-positives** 기준이므로 1:1 framing이 맞다. 3:1(cap≈7600)은 c2 117 ROI를 ~65× 반복해야 해 diversity-dishonest → primary로 쓰지 않는다.
> **스윕 시 ⚠️ resume**: synth_ratio만 바꾸고 pool/top_k는 그대로 → `--resume`이 기존 severstal 결과를 skip한다. 다른 ratio로 재실행하려면 `$EXP4V2_OUT/exp4v2_results.json`(+ `_seeds/seed*/exp4v2_results.json`)에서 severstal 항목을 삭제하거나 fresh `--output_dir`(예: `..._fixed_r1.5`)을 쓸 것.

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
- **cross-class 가용성-aware caveat**: top_k=1690, K=4 → floor=422. 단 c2는 distinct 가용 117뿐이라 clamp되어 ~117만 선택되고 surplus는 c1/c3/c4로 재분배(availability-aware balance). 즉 클래스 floor는 **강제 균등이 아니다** — 희소 class(c2)는 전량 사용하고 풍부 class가 그 잉여를 흡수한다. 이는 옛 equal-floor down-cap(c1/c3/c4를 c2 수준으로 깎아 풍부 class diversity를 낭비)보다 deficit-aware에 **더** 충실하다. per-pair_cap_frac=0.05는 튜닝 seed — 결과 보고 조정 가능.
- **c2 diversity ceiling = 117 (영구)**: 어떤 synth_ratio도 c2의 distinct ROI 수를 117 이상 못 올린다. image 수는 117×3=351로 모든 조건 동일. image-parity로 c2를 ~422~525장 만들려면 117 crop을 ~11~13× 반복해야 하는데 generate_defects는 global n_per_roi뿐(per-class n_per_roi 없음)이라 코드 변경 필요 + c2 overfitting 위험 → diversity-honest(현 zero-code 기본)를 채택. c2 map50이 낮으면 data-scarcity 발견으로 보고하고 c2를 인위로 반복해 parity를 위장하지 않는다.

---

## 부록 — per_pair_cap_frac 스윕 (선택)

monoculture 강도 조절. cap이 작을수록 다양성↑(단 너무 작으면 deficit 신호 약화).

```python
# step3만 cap 바꿔 재실행 → 2.1 게이트로 분포만 빠르게 확인 후 학습 결정
for frac in [0.03, 0.05, 0.10]:
    !python $AROMA_SCRIPTS/roi_selection.py \
        --profiling_dir $PROFILING_DIR --prompts_dir $PROMPTS_DIR \
        --sampling_strategy deficit_aware --top_k 1690 --seed 42 \
        --output_dir "$ROI_DIR"_cap{frac} \
        --class_mode multi --class_floor --per_pair_cap_frac {frac} --img_diversity_cap 1
```
