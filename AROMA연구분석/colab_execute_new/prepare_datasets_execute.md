# AROMA 데이터셋 준비 (v2-1 4종) — sym_final 파이프라인 선결

> **목적**: sym_final 파이프라인(`phase0 → … → exp*`)이 소비하는 **v2-1 4종**을 AROMA 호환 레이아웃으로 준비하고 `dataset_config.json`에 등록한다.
> **대상**: `severstal`(강판) · `mvtec_leather`(가죽) · `aitex`(직물, **tiled**) · `mtd`(자성타일).
> **실행 환경**: CPU (severstal context prototypes만 CLIP → T4 권장).
> **위치**: 이 문서는 파이프라인 **step -1**(phase0 앞). 4종 모두 ✓여야 `phase0_execute.md`로 진입.

파이프라인 체인: **prepare_datasets(이 문서) → phase0 → step1 → step2 → step3 → step4 → step5 → exp\***

---

## 데이터셋별 준비 방식 요약

| 데이터셋 | 원본 | 준비 스크립트 | class_mode | 출력(=image_dir 기준) |
|---------|------|--------------|-----------|----------------------|
| severstal | Kaggle RLE CSV | `prepare_severstal.py` (+ context prototypes) | multi | `severstal/train/good`, `test/class1..4` |
| mvtec_leather | **Drive 기존**(표준 레이아웃) | **준비 불요** — 경로 확인만 | multi | `mvtec/leather/train/good`, `test/{color,cut,fold,glue,poke}` |
| aitex | AITEX.zip (Kaggle) | `prepare_aitex.py --tile 256 --stride 128` | **single** | `aitex_tiled/train/good`, `test/defect` |
| mtd | Dataset Ninja Supervisely | `prepare_mtd.py` | multi | `mtd/train/good`, `test/{5 class}` |

> ⚠️ **스크립트 경로**: `prepare_severstal.py`·`prepare_aitex.py`·`prepare_mtd.py`·`select_context_prototypes.py`는 `scripts/aroma/`(=`$AROMA_SCRIPTS`)에 있다. (mvtec_leather는 스크립트 없음.)
> ⚠️ **aitex는 tiled 전용**: `aitex_tiled`(256×256/stride128)만 사용한다. 구 비타일 `aitex` 레이아웃과 **혼용 금지**(baseline mAP50 0.066로 학습 불가).

---

## STEP 0 — 환경변수

```python
import os

# ── 공통 (sym_final _SPEC과 정합) ──
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'          # prepare_*.py (severstal/aitex/mtd), select_context_prototypes
os.environ['DATASET_CONFIG'] = '/content/AROMA/dataset_config.json'

# ── 원본(raw) 경로 ──
os.environ['AITEX_RAW']   = f"{os.environ['DRIVE']}/aitex_raw"         # AITEX.zip 해제 위치 (Defect_images/Mask_images/NODefect_images)
os.environ['AITEX_TILED'] = f"{os.environ['DRIVE']}/aitex_tiled"       # tiled 출력 (구 aitex와 별도!)
os.environ['MTD_RAW']     = f"{os.environ['DRIVE']}/mtd_raw"           # Supervisely 루트 (meta.json, ds/img, ds/ann)
os.environ['MTD_OUT']     = f"{os.environ['DRIVE']}/mtd"
os.environ['SEV_SRC']     = '/content/drive/MyDrive/data/Severstal'    # Severstal 원본 (실제 위치에 맞게 수정)
os.environ['SEV_TRAIN_CSV'] = f"{os.environ['SEV_SRC']}/train.csv"
os.environ['SEV_TRAIN_IMG'] = f"{os.environ['SEV_SRC']}/train_images"
os.environ['SEV_OUT']     = f"{os.environ['DRIVE']}/severstal"
os.environ['MVTEC_OUT']   = f"{os.environ['DRIVE']}/mvtec"             # mvtec/leather/... 추출 위치

for k in ('AITEX_RAW','AITEX_TILED','MTD_RAW','MTD_OUT','SEV_OUT','MVTEC_OUT'):
    print(f"{k:12s} {os.environ[k]}")
```

---

## 1. severstal — RLE→PNG + 레이아웃 + context prototypes

원본: Kaggle Severstal (`train.csv` RLE + `train_images/`). 4 결함 클래스(ClassId 1–4).

```python
# 원본 확인
import os
print("csv :", os.path.exists(os.environ['SEV_TRAIN_CSV']), os.environ['SEV_TRAIN_CSV'])
print("imgs:", os.path.isdir(os.environ['SEV_TRAIN_IMG']), os.environ['SEV_TRAIN_IMG'])
```

```python
# RLE→PNG mask + MVTec-style 레이아웃 (train/good, test/class1..4, masks/)
!python $AROMA_SCRIPTS/prepare_severstal.py \
    --train_csv     $SEV_TRAIN_CSV \
    --train_images  $SEV_TRAIN_IMG \
    --output_dir    $SEV_OUT
```

```python
# context prototypes (CLIP 1000, PCA 64) — profiling/생성이 참조
!python $AROMA_SCRIPTS/select_context_prototypes.py \
    --image_dir   $SEV_OUT/train/good \
    --k           1000 --pca 64 --seed 42 \
    --model       ViT-B-32 \
    --output      $SEV_OUT/context_select
```

> severstal mask는 `ground_truth/`가 아니라 `masks/`(+`masks/class{c}/`)에서 해소된다(도메인 규약). class2가 희소(Neff<4)라 multi-class 불균형 존재.

---

## 2. mvtec_leather — Drive 기존 (준비 제외, 확인만)

MVTec-AD leather는 이미 **Google Drive에 표준 AROMA 호환 레이아웃**(`train/good`, `test/{defect}`, `ground_truth/{defect}`)으로 존재한다. **다운로드·변환·추출이 모두 불필요** — 경로만 확인한다.

```python
# 레이아웃 확인 (준비 불요 — 이미 Drive에 존재)
import os
from pathlib import Path
le = Path(os.environ['MVTEC_OUT']) / "leather"      # $DRIVE/mvtec/leather
for sub in ['train/good', 'test/color', 'test/cut', 'test/fold', 'test/glue', 'test/poke',
            'ground_truth/color']:
    p = le / sub
    n = len(list(p.iterdir())) if p.exists() else 0
    print(f"{'✓' if p.exists() else '✗'} leather/{sub:<20} n={n}")
```

> `$MVTEC_OUT/leather` 경로가 dataset_config `mvtec_leather.image_dir`(`.../mvtec/leather/train/good`)과 일치하는지 확인. 없으면 Drive 실제 위치에 맞게 config 경로만 수정. mask는 `ground_truth/{defect}/{stem}_mask.png`에서 자동 해소(domain=mvtec).

---

## 3. aitex — tiled 정규화 (256×256 / stride 128)

원본: AITEX.zip(Kaggle) → `Defect_images/`, `Mask_images/`, `NODefect_images/`. 원본은 4096×256(16:1)이라 **반드시 타일링**한다.

```python
# AITEX.zip 해제
!unzip -o $DRIVE/AITEX.zip -d $AITEX_RAW
```

```python
# tiled 정규화 (256/stride128, 50% overlap, 단일클래스)
!python $AROMA_SCRIPTS/prepare_aitex.py \
    --defect_images   $AITEX_RAW/Defect_images \
    --mask_images     $AITEX_RAW/Mask_images \
    --nodefect_images $AITEX_RAW/NODefect_images \
    --output_dir      $AITEX_TILED \
    --tile 256 --stride 128 --min_tile_area 50
```

```python
# 산출 확인
import json, os
mani = f"{os.environ['AITEX_TILED']}/aitex_manifest.json"
if os.path.exists(mani):
    m = json.load(open(mani))
    print("tiled:", m.get('tiled'), "| counts:", m.get('counts'))
else:
    print("MISSING —", mani)
from pathlib import Path
for sub in ['train/good', 'test/defect', 'ground_truth/defect']:
    p = Path(os.environ['AITEX_TILED']) / sub
    print(f"{'✓' if p.exists() else '✗'} aitex_tiled/{sub}  n={len(list(p.iterdir())) if p.exists() else 0}")
```

> 단일클래스: 모든 결함 코드를 `test/defect`로 병합. 타일 stem에 `__tile_r{r}_c{c}`가 붙어 exp4v2가 동일 원본의 타일을 같은 train/val 측으로 묶는다(50% overlap leak 가드). **`aitex_tiled` 신규 dir 사용, 구 aitex와 혼용 금지.**

---

## 4. mtd — Supervisely bitmap → full-frame mask

원본: Dataset Ninja Supervisely(`meta.json`, `ds/img/*.jpg`, `ds/ann/*.jpg.json`). 5 클래스(blowhole/break/crack/fray/uneven), 전부 bitmap.

```python
# 원본 확인
import os, glob, json, collections
R = os.environ['MTD_RAW']
print("meta:", os.path.exists(f"{R}/meta.json"),
      "| ann:", len(glob.glob(f"{R}/ds/ann/*.json")),
      "| img:", len(glob.glob(f"{R}/ds/img/*")))
c = collections.Counter(); emp = 0
for f in glob.glob(f"{R}/ds/ann/*.json"):
    o = json.load(open(f))['objects']; emp += (not o)
    for x in o: c[x.get('classTitle')] += 1
print("good≈", emp, "| per class=", dict(c))   # 기대: good≈956, 5 class 합≈459
```

```python
# Supervisely → MVTec-style (train/good, test/{class}, ground_truth/{class}/{stem}_mask.png)
!python $AROMA_SCRIPTS/prepare_mtd.py \
    --supervisely_root $MTD_RAW \
    --output_dir       $MTD_OUT
```

```python
# 산출 확인
import json, os
mani = f"{os.environ['MTD_OUT']}/mtd_manifest.json"
if os.path.exists(mani):
    c = json.load(open(mani))['counts']
    print("good:", c['good'], "| defect_entries:", c['defect_entries'], "| per_class:", c['per_class'])
    print("skipped_empty:", c['skipped_empty_mask'], " no_img:", c['skipped_no_img'])
else:
    print("MISSING —", mani)
```

---

## 5. dataset_config.json 등록 / 검증

4종 엔트리는 저장소 `dataset_config.json`에 이미 커밋되어 있다. **경로가 실제 `DRIVE`와 일치하는지 검증**한다(불일치 시 config의 해당 경로만 수정). 표준 스키마:

```json
"severstal":     { "domain": "severstal", "class_mode": "multi",  "image_dir": ".../severstal/train/good",      "seed_dirs": [".../severstal/test/class1", "...class2", "...class3", "...class4"] },
"mvtec_leather": { "domain": "mvtec",     "class_mode": "multi",  "image_dir": ".../mvtec/leather/train/good",   "seed_dirs": [".../test/color", ".../cut", ".../fold", ".../glue", ".../poke"] },
"aitex":         { "domain": "aitex",     "class_mode": "single", "image_dir": ".../aitex_tiled/train/good",     "seed_dirs": [".../aitex_tiled/test/defect"] },
"mtd":           { "domain": "mtd",       "class_mode": "multi",  "image_dir": ".../mtd/train/good",             "seed_dirs": [".../test/blowhole", ".../break", ".../crack", ".../fray", ".../uneven"] }
```

```python
import json, os
cfg = json.load(open(os.environ['DATASET_CONFIG'], encoding='utf-8'))
ok_all = True
for ds in ('severstal', 'mvtec_leather', 'aitex', 'mtd'):
    e = cfg.get(ds)
    if not e:
        print(f"✗ {ds}: config에 없음"); ok_all = False; continue
    img_ok = os.path.isdir(e['image_dir'])
    seeds = e.get('seed_dirs', [])
    seed_ok = sum(os.path.isdir(d) for d in seeds)
    print(f"{'✓' if img_ok and seed_ok == len(seeds) else '✗'} {ds:14s} "
          f"class_mode={e['class_mode']:6s} image_dir={img_ok} seed_dirs={seed_ok}/{len(seeds)}")
    if not (img_ok and seed_ok == len(seeds)): ok_all = False
print("\nALL OK — phase0 진입 가능" if ok_all else "\n✗ 경로 불일치 — config 또는 준비 재확인")
```

---

## 6. 다음 단계

4종 모두 ✓ → **`phase0_execute.md`**(distribution_profiling, sym_final)로 진입. phase0가 `dataset_config.json`의 `image_dir`/`seed_dirs`를 읽어 프로파일링한다.

---

## 무결성 / 주의

- **aitex tiled 전용**: `aitex_tiled` 신규 dir만 사용, 구 비타일 `aitex` 혼용 금지. `class_mode=single`(전 결함코드 병합).
- **severstal mask**: `masks/`에서 해소(ground_truth 아님). class2 희소 → multi 불균형은 데이터 특성.
- **mvtec_leather**: **Drive에 기존** — 다운로드·변환 없음, 경로 확인만. mask=`ground_truth/{defect}/{stem}_mask.png`.
- **severstal 원본**: Drive `SEV_SRC=/content/drive/MyDrive/data/Severstal`(`train.csv`+`train_images/`, 원시 Kaggle 포맷) → `prepare_severstal.py`로 AROMA 레이아웃 변환 필요(§1).
- **재실행 안전**: prepare_* 스크립트·추출은 멱등(기존 산출 위에 재생성). aitex/severstal는 원본 dir을 실제 위치에 맞게 수정.
- **경로 일관성**: 전 경로가 `DRIVE=/content/drive/MyDrive/data/Aroma` 기준. dataset_config의 절대경로가 DRIVE와 어긋나면 config를 수정(코드·스크립트 수정 아님).
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 위 확인 셀로.
