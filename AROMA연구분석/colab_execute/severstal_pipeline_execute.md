# Severstal — Phase 0 ~ Step 4 Colab 실행 가이드

> **대상 데이터셋**: `severstal` (단일 데이터셋 기준)
> **실행 환경**: CPU (전 단계 `copy_paste`/Otsu 기준. SAM 미지정 시 fallback 마스크 자동)
> **전제**: prep 단계(`prepare_severstal.py` + `select_context_prototypes.py`)가 **이미 완료**되어 있어야 한다.
> 이 가이드는 prep 완료를 전제로 phase0 → step4 만 다룬다.

---

## 0. 개요 + 전제

### 0.1 파이프라인 의존 사슬

```
prep (prepare_severstal + select_context_prototypes)   ← 본 가이드 전제(완료 가정)
  │
  ▼
phase0  distribution_profiling.py   → profiling/severstal/
  │
  ▼
step1   compute_complexity.py       → complexity/severstal/   (MCI/CCI)
  │
  ▼
step2   prompt_generation.py        → prompts/severstal/      (profiling + complexity 입력)
  │
  ▼
step3   roi_selection.py            → roi/severstal/          (profiling + prompts 입력, complexity 직접 미사용)
  │
  ├─ AROMA  : --sampling_strategy deficit_aware
  └─ random : --sampling_strategy random  (별도 roi/severstal_random/)
  │
  ▼
step4   generate_defects.py         → synthetic/severstal/    (roi + normal_dir 입력)
        generate_random.py          → baseline_random/severstal/  (Step3 roi_candidates.json 공유)
```

- `profiling/severstal/morphology_features.csv` 가 phase0 완료 판정 sentinel.
- step1~step3 는 phase0 산출물(CSV/JSON)만 읽으므로 도메인/경로 결합이 없다(안전).
- **모든 합성 방법은 동일한 copy-paste** 다. AROMA vs random vs CASDA-ROI 차이는 **Step3 ROI selection 전략(`--sampling_strategy`)** 에서만 발생한다.

### 0.2 전제 — prep 산출물 확인

prep가 만든 다음 레이아웃이 존재해야 한다 (`$DRIVE = /content/drive/MyDrive/data/Aroma`):

```
$DRIVE/severstal/
  train/good/                       # 정상 강판 이미지 (image_dir)
  test/class1/  test/class2/  test/class3/  test/class4/   # 결함 이미지 (seed_dirs → defect_type=class1..4)
  masks/{stem}.png                  # merged GT 마스크 (prepare_severstal RLE→PNG)
  masks/class{1..4}/{stem}.png      # per-class GT 마스크
  context_select/context/           # CLIP 1000 context prototypes 이미지 (select_context_prototypes 산출)
  context_select/context_prototypes.json   # 파일명 리스트 (이미지 폴더 아님)
```

> **중요(GAP-1, 선결)**: 아래 §1.0 "선결 조치" 박스를 먼저 읽어라. `distribution_profiling.py`에 severstal mask 분기가 없으면 GT 마스크를 읽지 못하고 전 결함이 Otsu fallback으로 처리되어 phase0~step4 결과가 전부 무효가 된다.

---

## 0.5 선결 조치 (phase0 실행 전 필수)

> ✅ **[GAP-1 / 해결됨] GT 마스크 인식 — severstal 분기 코드 적용 완료**
>
> `scripts/distribution_profiling.py`의 `_find_mask_path()`에 **severstal 분기 추가됨**(per-class `masks/class{N}/{stem}.png` 우선, merged `masks/{stem}.png` fallback). prep가 만든 GT 마스크를 profiler가 정상 인식한다. 아래 설명은 미적용 시 위험 기록(이미 해결). git pull로 최신 코드 반영 확인.
>
> (이전 상태) severstal 분기가 없어 `_find_mask_path()`가 항상 `None` 반환 → SAM/Otsu fallback(`stage1b_seed_characterization.py`의 `THRESH_BINARY_INV+OTSU`)으로 마스크 추정.
>
> - prep가 만든 GT 마스크(`masks/{stem}.png`, `masks/class{c}/{stem}.png`)를 **profiler가 절대 읽지 못한다.**
> - 1600×256 저대비 강판은 객체/배경 분리가 없어 Otsu가 결함이 아닌 명암 영역을 잡는다 → `morphology_features.csv`(linearity / aspect_ratio / solidity 등) 전부 오염 → cluster / deficit / compatibility 분석 전부 신뢰 불가.
> - 연쇄 영향: Step4 `copy_paste`의 `use_real_mask` 경로가 의존하는 `defect_bbox` / `defect_mask_path` 도 Otsu 기반이 되어 **합성 결함 모양까지 가짜**가 된다(실패 시 타원 폴백).
>
> **적용된 매핑** (mvtec과 동일 3단계 parent 패턴):
> - 결함 이미지 `.../severstal/test/class{N}/{stem}.png`
> - → per-class(우선): `image_path.parent.parent.parent / "masks" / "class{N}" / f"{stem}.png"` (defect_type subfolder)
> - → merged(fallback): `image_path.parent.parent.parent / "masks" / f"{stem}.png"`
>
> **검증**: phase0 실행 후 §3 출력 확인 셀에서 `mask_source`에 `fallback_otsu`가 결함 수만큼 찍히지 **않는지**(= GT 마스크 정상 인식) 확인. `fallback_count`로도 확인. fallback이 대량이면 prep 마스크 경로/도메인 설정 재점검.

---

## 1. 환경변수 셀

> `00_setup.md`의 공통 셀(Drive 마운트 / AROMA 클론 / 의존성 설치 / `DRIVE`·`AROMA_SCRIPTS`·`DATASET_CONFIG`·`AROMA_OUT` 설정)을 먼저 실행한다.
> 그 위에 severstal 전용 변수를 아래 셀로 추가한다.

```python
import os, json
from pathlib import Path

# ── 공통(00_setup에서 이미 설정됨, 누락 대비 기본값) ──────────────
os.environ['DRIVE']          = os.environ.get('DRIVE', '/content/drive/MyDrive/data/Aroma')
os.environ['AROMA_SCRIPTS']  = os.environ.get('AROMA_SCRIPTS', '/content/AROMA/scripts/aroma')
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"

# distribution_profiling.py 만은 AROMA_SCRIPTS(aroma 하위)가 아니라
# AROMA_REF(/content/AROMA) 아래의 scripts/distribution_profiling.py를 가리킨다.
os.environ['AROMA_REF']      = '/content/AROMA'

# ── Severstal 단일 데이터셋 ─────────────────────────────────────
DATASET_KEY = 'severstal'   # ← 단일 ds 기준. 다른 ds로 바꿀 때만 여기 수정

os.environ['DATASET_KEY']    = DATASET_KEY
os.environ['PROFILING_DIR']  = f"{os.environ['AROMA_OUT']}/profiling/{DATASET_KEY}"
os.environ['COMPLEXITY_DIR'] = f"{os.environ['AROMA_OUT']}/complexity/{DATASET_KEY}"
os.environ['PROMPTS_DIR']    = f"{os.environ['AROMA_OUT']}/prompts/{DATASET_KEY}"
os.environ['ROI_DIR']        = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}"
os.environ['ROI_DIR_RANDOM'] = f"{os.environ['AROMA_OUT']}/roi/{DATASET_KEY}_random"
os.environ['SYNTHETIC_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic/{DATASET_KEY}"
os.environ['RANDOM_DIR']     = f"{os.environ['AROMA_OUT']}/baseline_random/{DATASET_KEY}"

# ── Severstal 경로 ──────────────────────────────────────────────
SEVERSTAL_ROOT = f"{os.environ['DRIVE']}/severstal"
os.environ['SEVERSTAL_ROOT'] = SEVERSTAL_ROOT

# Step4/random 의 --normal_dir 는 dataset_config 에 normal_dir 필드가 없으므로 수동 지정.
# CLIP context prototypes 의 "하위 context 폴더" 를 가리킨다 (루트 context_select 아님, GAP-4).
os.environ['NORMAL_DIR'] = f"{SEVERSTAL_ROOT}/context_select/context"

print("DATASET_KEY   :", DATASET_KEY)
print("PROFILING_DIR :", os.environ['PROFILING_DIR'])
print("NORMAL_DIR    :", os.environ['NORMAL_DIR'])

# dataset_config 의 severstal 항목 정합 확인 (image_dir / seed_dirs=class1..4)
with open(os.environ['DATASET_CONFIG']) as f:
    _cfg = json.load(f)
_entry = _cfg.get(DATASET_KEY, {})
print("\nconfig.image_dir :", _entry.get('image_dir'))
print("config.seed_dirs :", [Path(d).name for d in _entry.get('seed_dirs', [])])  # ['class1','class2','class3','class4'] 기대
```

| 변수 | 값 | 설명 |
|------|----|------|
| `AROMA_REF` | `/content/AROMA` | **phase0 전용** — `distribution_profiling.py`가 여기 `scripts/` 아래에 있음 |
| `AROMA_SCRIPTS` | `/content/AROMA/scripts/aroma` | step1~step4 스크립트 |
| `PROFILING_DIR` | `$AROMA_OUT/profiling/severstal` | phase0 출력 |
| `COMPLEXITY_DIR` | `$AROMA_OUT/complexity/severstal` | step1 출력 |
| `PROMPTS_DIR` | `$AROMA_OUT/prompts/severstal` | step2 출력 |
| `ROI_DIR` | `$AROMA_OUT/roi/severstal` | step3 AROMA 출력 |
| `ROI_DIR_RANDOM` | `$AROMA_OUT/roi/severstal_random` | step3 random 전략 출력 |
| `SYNTHETIC_DIR` | `$AROMA_OUT/synthetic/severstal` | step4 AROMA 합성 출력 |
| `RANDOM_DIR` | `$AROMA_OUT/baseline_random/severstal` | generate_random 베이스라인 출력 |
| `NORMAL_DIR` | `$DRIVE/severstal/context_select/context` | step4/random 배경 = context prototypes 폴더 |

---

## 2. prep 선행 명령 (참조 — 이미 완료 전제)

이미 완료되어 있다면 건너뛴다. 재현/검증용 참조 명령:

```python
# (참조) prepare_severstal — RLE→PNG 마스크 생성 + train/good, test/class{1..4} 레이아웃 구성
!python $AROMA_SCRIPTS/prepare_severstal.py \
    --train_csv    <Severstal train.csv 경로> \
    --train_images <Severstal train_images 경로> \
    --output_dir   $SEVERSTAL_ROOT

# (참조) select_context_prototypes — CLIP 임베딩으로 정상 배경 1000장 선별
#   산출: $SEVERSTAL_ROOT/context_select/context/        (이미지 K장 → step4 normal_dir)
#         $SEVERSTAL_ROOT/context_select/context_prototypes.json  (파일명 리스트)
!python $AROMA_SCRIPTS/select_context_prototypes.py \
    --image_dir $SEVERSTAL_ROOT/train/good \
    --k         1000 \
    --output    $SEVERSTAL_ROOT/context_select
```

> prep 인자(`--train_csv` 등)는 prep 스크립트가 별도 구현 중인 항목이라 정확한 플래그명은 prep 스크립트 자체 확인 필요(본 가이드 검증 범위 밖). 위는 형태 참조용.

**prep 완료 확인 셀:**

```python
import os
from pathlib import Path

root = Path(os.environ['SEVERSTAL_ROOT'])
checks = {
    'train/good':                 root / 'train' / 'good',
    'test/class1':                root / 'test' / 'class1',
    'test/class4':                root / 'test' / 'class4',
    'masks/ (merged)':            root / 'masks',
    'masks/class1':               root / 'masks' / 'class1',
    'context_select/context':     root / 'context_select' / 'context',
}
for name, p in checks.items():
    n = len(list(p.glob('*'))) if p.exists() else 0
    status = '✅' if p.exists() and (name.startswith('masks/ ') or n > 0 or p.is_dir()) else '❌ 누락'
    print(f"  {status}  {name:28s}  ({n} entries)")
```

---

## 3. Phase 0 — `distribution_profiling.py`

> **선결**: §0.5 GAP-1 (severstal mask 분기) 처리 후 실행. 미처리 시 전 결함 Otsu fallback.
> Severstal은 **GT 마스크가 있는 도메인**이다(ISP처럼 마스크 없는 도메인이 아님).
> `--domain` 플래그는 없다 — domain은 config의 `"domain": "severstal"` 키에서 읽는다.

```python
# smoke test (이미지 5장으로 경로/마스크 인식 빠른 확인 — phase0에만 권장)
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    $DATASET_KEY \
    --output_dir     $PROFILING_DIR \
    --num_workers    8 \
    --max_images     5

# 전체 실행
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    $DATASET_KEY \
    --output_dir     $PROFILING_DIR \
    --num_workers    8 \
    --n_gmm_components 5
```

> SAM fallback을 쓸 경우에만 `--sam_checkpoint /content/sam_vit_h.pth` 추가. (GT 마스크 분기를 추가했다면 SAM 불필요.)

**확정 인자**: `--dataset_config`(default `/content/AROMA/dataset_config.json`), `--dataset_key`(필수), `--output_dir`(필수), `--num_workers`(default -1=auto), `--max_images`(default None), `--n_gmm_components`(default 5), `--sam_checkpoint`(default None).

**phase0 출력 확인 + 마스크 소스 점검:**

```python
import os, json
from pathlib import Path

profiling_dir = Path(os.environ['PROFILING_DIR'])
for fname in [
    'morphology_features.csv', 'context_features.csv',
    'distribution_analysis.json', 'morphology_clusters.json',
    'compatibility_matrix.json', 'deficit_analysis.json',
]:
    status = '✅' if (profiling_dir / fname).exists() else '❌ 누락'
    print(f'  {status}  {fname}')

# GAP-1 검증: mask_source 분포 (GT 마스크가 실제로 읽혔는지)
import csv
csv_path = profiling_dir / 'morphology_features.csv'
if csv_path.exists():
    from collections import Counter
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    # defect_type 분포 (class1..4, Class2 희소 여부 확인)
    print("\ndefect_type 분포:", dict(Counter(r.get('defect_type','?') for r in rows)))
    # mask_source 컬럼이 있으면 fallback 비율 확인
    if rows and 'mask_source' in rows[0]:
        ms = Counter(r['mask_source'] for r in rows)
        print("mask_source 분포:", dict(ms))
        if any('fallback' in k or 'otsu' in k.lower() for k in ms):
            print("⚠️  fallback 마스크 검출 — GAP-1 severstal 분기 미반영 가능. §0.5 확인.")
```

---

## 4. Step 1 — `compute_complexity.py` (MCI / CCI)

> phase0 산출물만 읽는다. 도메인/경로 결합 없음(안전). 준비조건: `profiling/severstal/morphology_features.csv` 존재.

```python
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $PROFILING_DIR \
    --output_dir    $COMPLEXITY_DIR \
    --weight_mode   equal \
    --local_staging
```

**확정 인자**: 필수 `--profiling_dir`, `--output_dir`. 선택 `--config`(default None → `scripts/aroma/config/aroma_step1.yaml`), `--weight_mode`(ablation 프리셋), `--local_staging`(flag), `--staging_root`(default None).

**출력 확인:**

```python
import json, os
with open(f"{os.environ['COMPLEXITY_DIR']}/complexity_report.json") as f:
    rep = json.load(f)
print("MCI:", rep.get('MCI'), " CCI:", rep.get('CCI'))
print("morphology_policy:", rep.get('morphology_policy'))
print("context_policy   :", rep.get('context_policy'))
```

---

## 5. Step 2 — `prompt_generation.py`

> phase0 + step1 출력을 읽는다. 준비조건: `complexity/severstal/` 디렉터리 존재.

```python
!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir  $PROFILING_DIR \
    --complexity_dir $COMPLEXITY_DIR \
    --output_dir     $PROMPTS_DIR
```

**확정 인자**: 필수 `--profiling_dir`, `--complexity_dir`, `--output_dir`. 선택 인자 없음.

**출력 확인:**

```python
import json, os
with open(f"{os.environ['PROMPTS_DIR']}/prompts.json") as f:
    prompts = json.load(f)
print(f"총 프롬프트 수: {len(prompts)}")
```

---

## 6. Step 3 — `roi_selection.py`

> phase0 + step2 출력을 읽는다(complexity 직접 미사용). 준비조건: `prompts/severstal/prompts.json` 존재.
> **AROMA / random / CASDA-ROI 의 차이는 전부 여기 `--sampling_strategy` 에서만 발생한다.**

### 6.1 AROMA (deficit_aware)

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy deficit_aware \
    --top_k             200 \
    --img_diversity_cap 1 \
    --seed              42 \
    --output_dir        $ROI_DIR
```

### 6.2 random 베이스라인 (동일 후보 풀, 무작위 선택)

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_DIR \
    --prompts_dir       $PROMPTS_DIR \
    --sampling_strategy random \
    --top_k             200 \
    --seed              42 \
    --output_dir        $ROI_DIR_RANDOM
```

**확정 인자**: 필수 `--profiling_dir`, `--prompts_dir`, `--output_dir`. 선택 `--sampling_strategy`(default `deficit_aware`, choices `[deficit_aware, top_k, weighted, random]`), `--top_k`(default 200), `--seed`(default 42).

> CASDA-ROI 비교는 별도 ROI selection 구현/전략 사용 — 본 가이드의 확인된 인자(`random`/`deficit_aware`/`top_k`/`weighted`)에 CASDA 전용 값은 없으므로 미확정으로 둔다.

**출력 확인 (Class2 희소 영향 점검):**

```python
import json, os
from collections import Counter

with open(f"{os.environ['ROI_DIR']}/roi_selected.json") as f:
    selected = json.load(f)
with open(f"{os.environ['ROI_DIR']}/roi_candidates.json") as f:
    candidates = json.load(f)

print(f"전체 후보: {len(candidates)}  선택된 ROI: {len(selected)}")
# Class2 희소 → defect_type/cluster 별 분포가 한쪽으로 치우치는지 확인
print("클러스터별 선택 수:", dict(Counter(r.get('cluster_id') for r in selected)))
```

---

## 7. Step 4 — `generate_defects.py` + `generate_random.py`

> roi + normal_dir 입력. 준비조건: `roi/severstal/roi_selected.json` 존재.
> **`--normal_dir` 는 context prototypes 의 하위 `context/` 폴더** (`$SEVERSTAL_ROOT/context_select/context`).
> dataset_config 에는 `normal_dir` 필드가 없으므로 §1에서 수동 지정한 `$NORMAL_DIR` 를 쓴다.

### 7.1 AROMA copy-paste 합성

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

### 7.2 random 베이스라인 — `generate_random.py`

> 동일 copy-paste, ROI selection 만 random. **Step3 AROMA 의 `roi_candidates.json`(공유 후보 풀)** 을 입력으로 받는다.

```python
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI_DIR/roi_candidates.json \
    --normal_dir      $NORMAL_DIR \
    --output_dir      $RANDOM_DIR \
    --top_k           200 \
    --n_per_roi       3 \
    --seed            42 \
    --local_staging
```

**확정 인자**
- `generate_defects.py`: 필수 `--roi_dir`, `--normal_dir`, `--output_dir`. 선택 `--method`(default `copy_paste`), `--n_per_roi`(3), `--blend_mode`(default `alpha`, choices `[alpha]`), `--feather_px`(4), `--seed`(42), `--local_staging`(flag).
- `generate_random.py`: 필수 `--candidates_json`, `--normal_dir`, `--output_dir`. 선택 `--random_roi_dir`(default `{output_dir}/_random_roi`), `--top_k`(200), `--n_per_roi`(3), `--seed`(42), `--local_staging`(flag).

**출력 확인:**

```python
import json, os
from pathlib import Path
from collections import Counter

for tag, d in [('AROMA', os.environ['SYNTHETIC_DIR']), ('random', os.environ['RANDOM_DIR'])]:
    ann_path = Path(d) / 'annotations.json'
    if not ann_path.exists():
        print(f"[{tag}] annotations.json 없음: {d}")
        continue
    with open(ann_path) as f:
        ann = json.load(f)
    n_dry = sum(1 for a in ann if a.get('dry_run'))
    img_dir = Path(d) / 'images'
    n_files = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))) if img_dir.exists() else 0
    print(f"[{tag}] 생성 {len(ann)-n_dry}개 (dry_run {n_dry})  images/={n_files}")
```

---

## 8. Severstal 특수 주의 (요약 박스)

> **[비정사각 1600×256]** 강판 이미지는 6.25:1 종횡비.
> - phase0 context 패치 그리드(`_context_worker`, gs=64)는 256//64=4행 × 1600//64=25열 = 이미지당 100패치. 잔여 픽셀 drop만 발생, 크래시 없음.
> - GT mask resize 는 INTER_NEAREST. prep masks 가 이미 256×1600 이라 일치 → resize 미발생.
> - FFT/LBP context feature 는 64×64 패치 단위라 종횡비 무관.
> - 단, mvtec(정사각) 가정의 aspect_ratio/ROI 좌표계 해석과 다를 수 있으니 결과 해석 시 유의.

> **[defect_type = ClassId / Class2 희소]**
> - `seed_dirs` = `test/class1..class4` → `defect_type = class1..class4` (basename). dataset_config 에 이미 올바르게 설정됨(`_resolve_seed_entries` 정상 동작, 문제 아님).
> - **Class2 결함 수가 적음(Neff<4 불균형)**. GMM(`--n_gmm_components 5`)·morphology 클러스터링에서 class2 표본 부족으로 silhouette 불안정 / cluster·deficit 왜곡 가능. Step1 MCI/CCI, Step3 `deficit_aware`(p75 우선) 선택이 소수 클래스로 치우칠 수 있다 → §6 출력 확인으로 분포 점검.

> **[context prototypes 를 normal_dir 로]**
> - Step4/random `--normal_dir` = `$SEVERSTAL_ROOT/context_select/context` (**하위 context 폴더**).
> - 루트 `context_select/` 를 주면 `context_prototypes.json`(비이미지)은 무시되지만, `train/good` 전체를 주는 것과 혼동 금지.
> - 강판 full-frame 특성상 `_foreground_mask` 가 면적비 <2% 또는 >90% 로 대부분 None 반환 → random placement 폴백. **즉 Severstal 결함은 foreground 제약 없이 무작위 배치된다**(의도된 동작, 정상).

> **[mask PNG 인식 = GAP-1]** §0.5 선결 박스 참조. severstal 분기 미반영 시 GT 마스크 미사용 → 전 단계 무효.

---

## 9. 선결 / 주의 종합 (gap)

### 치명적 (선결 필요)
- **[GAP-1 / phase0 / CRITICAL]** `_find_mask_path()`에 severstal 분기 없음 → 전 결함 Otsu fallback → morphology 전부 오염 → phase0~step4 전체 무효. **phase0 실행 전 분기 추가 필수**(§0.5). 매핑: `.../test/class{N}/{stem}.jpg` → `.../masks/{stem}.png` (또는 `masks/class{N}/{stem}.png`).

### 동작하지만 주의
- **[GAP-3 / phase0]** 1600×256 패치 그리드/mask resize 는 크래시 없이 동작(확인됨). aspect_ratio 해석만 유의.
- **[GAP-4 / step4]** `--normal_dir` 는 반드시 `context_select/context` 하위 폴더. 루트/`train/good` 혼동 금지. 강판은 foreground 제약 없이 random 배치됨(정상).
- **[GAP-5 / 연쇄]** GAP-1 미해결 시 step4 `use_real_mask` 의 `defect_bbox`/`defect_mask_path` 도 Otsu 기반 → 합성 결함 모양까지 가짜(타원 폴백).
- **[Class2 희소]** Step1/Step3 에서 소수 클래스 과소/과대 표집 가능 — 출력 분포 점검.

### 미확정 (확인 후 사용)
- prep 스크립트(`prepare_severstal.py` / `select_context_prototypes.py`)의 정확한 인자명은 prep 구현 확정 후 검증 필요(§2 는 형태 참조).
- CASDA-ROI 비교 전략은 본 가이드 확인 인자(`deficit_aware`/`random`/`top_k`/`weighted`)에 없으므로 별도 구현 확인 필요.

### 비-gap 확인 (문제 아님)
- defect_type 도출 정상(`class1..class4`), dataset_config severstal 항목(domain/image_dir/seed_dirs) 이미 올바름.
- step1~step3 는 phase0 산출물만 읽어 도메인 결합 없음(안전).
