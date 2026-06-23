# Severstal Prep — `prepare_severstal.py` + `select_context_prototypes.py` Colab 실행

> **실행 환경**: prepare = CPU | select_context_prototypes = GPU 권장(CLIP 임베딩)
> **목적**: Kaggle Severstal 원본(train.csv RLE + train_images)을 AROMA 파이프라인 레이아웃으로 변환 + 정상 배경 프로토타입 1000장 선택.
> **위치**: 이 단계는 **phase0~step4 (severstal_pipeline_execute.md) 및 exp4v2 의 선행**.

---

## 0. 개요

```
[1] prepare_severstal.py           RLE→PNG, test/class{1-4} 조직, normal 도출
        ↓ Aroma/severstal/{train/good, test/class{1-4}, masks/, severstal_manifest.json}
[2] select_context_prototypes.py   CLIP→KMeans(1000)→medoid → context_select/
        ↓ Aroma/severstal/context_select/{context_prototypes.json, context/}
   → 이후 phase0~step4 + exp4v2 가 이 레이아웃을 소비
```

원본 Severstal은 Google Drive에 업로드되어 있다고 전제(예: `/content/drive/MyDrive/data/Severstal/`).

---

## 1. 환경변수 설정

```python
import os

# AROMA 베이스
os.environ['DRIVE']         = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_SCRIPTS'] = '/content/AROMA/scripts/aroma'

# Severstal 원본 (CASDA가 쓰던 Drive 경로 — 실제 위치에 맞게 수정)
os.environ['SEV_SRC']       = '/content/drive/MyDrive/data/Severstal'
os.environ['SEV_TRAIN_CSV'] = f"{os.environ['SEV_SRC']}/train.csv"
os.environ['SEV_TRAIN_IMG'] = f"{os.environ['SEV_SRC']}/train_images"

# AROMA severstal 출력 루트 (prepare 산출)
os.environ['SEV_OUT']       = f"{os.environ['DRIVE']}/severstal"

print("SEV_TRAIN_CSV :", os.environ['SEV_TRAIN_CSV'])
print("SEV_TRAIN_IMG :", os.environ['SEV_TRAIN_IMG'])
print("SEV_OUT       :", os.environ['SEV_OUT'])
```

> 환경변수 참조는 `$VAR` (중괄호형 금지), 실행은 `!python` 접두사 (colab-execution.md).

---

## 2. 패키지 설치 (최초 1회)

```python
!pip install opencv-python-headless pillow numpy -q   # prepare_severstal
!pip install open_clip_torch scikit-learn -q          # select_context_prototypes (CLIP+KMeans)
# open_clip 미설치 시 transformers fallback (단 --model 무시, ViT-B/32 고정):
# !pip install transformers -q
```

---

## 3. prepare_severstal.py — RLE→PNG + 레이아웃

```python
!python $AROMA_SCRIPTS/prepare_severstal.py \
    --train_csv     $SEV_TRAIN_CSV \
    --train_images  $SEV_TRAIN_IMG \
    --output_dir    $SEV_OUT
```

생성물:
```
Aroma/severstal/
  train/good/                  normal 이미지 (train_images ∖ train.csv ImageId)
  test/class1/ .. class4/      결함 이미지 (ClassId별 → defect_type=class{N})
  masks/{ImageId}.png          merged binary (전 class OR — single 모드)
  masks/class1/ .. class4/     per-class binary (multi 모드)
  severstal_manifest.json
```

**검증**:

```python
import json, os
from pathlib import Path

sev = Path(os.environ['SEV_OUT'])
mani = json.load(open(sev / 'severstal_manifest.json'))
print("normal:", len(list((sev/'train'/'good').glob('*'))))
for c in range(1, 5):
    nd = len(list((sev/'test'/f'class{c}').glob('*')))
    nm = len(list((sev/'masks'/f'class{c}').glob('*.png')))
    print(f"  class{c}: defect={nd}  per-class mask={nm}")
print("merged masks:", len(list((sev/'masks').glob('*.png'))))
```

> **마스크 정합 육안 확인 권장**: `masks/{id}.png`를 동일 `test/class{N}/{id}` 위에 오버레이 → 결함 위치 일치(RLE column-major 디코드 검증).
> Class 2는 희귀 → class2 결함 수가 매우 적음(정상). 이것이 **Neff<4 불균형 → DEFICIT 우위 발현 후보**.

---

## 4. select_context_prototypes.py — CLIP 1000 프로토타입

```python
!python $AROMA_SCRIPTS/select_context_prototypes.py \
    --image_dir   $SEV_OUT/train/good \
    --k           1000 \
    --pca         64 \
    --seed        42 \
    --model       ViT-B-32 \
    --output      $SEV_OUT/context_select
```

> **`--output`는 반드시 `$SEV_OUT/context_select`** — exp4v2 severstal 분기가 `severstal/context_select/context_prototypes.json`을 정확히 이 경로에서 찾는다. 다른 경로면 프로토타입이 **조용히 무시되고 전체 normal 사용**(exp4v2 로그에 경고 출력됨).
> `--pca 0` 이면 PCA 생략(512차원 그대로 KMeans).
> open_clip 없으면 transformers fallback이 `--model` 무시하고 ViT-B/32 고정(경고 출력) — 재현성 위해 open_clip 권장.

생성물: `context_select/context_prototypes.json` (prototypes 파일목록) + `context_select/context/` (선택 1000장).

**검증**:

```python
import json, os
p = json.load(open(f"{os.environ['SEV_OUT']}/context_select/context_prototypes.json"))
print("prototypes:", len(p.get('prototypes', [])))   # 1000 기대 (normal<1000이면 그 수)
```

---

## 5. 다음 단계

prep 완료 후:
- **phase0~step4**: `severstal_pipeline_execute.md` (distribution_profiling → compute_complexity → prompt → roi → generate) — 합성 조건(random/aroma)용
- **exp4v2 baseline**: `exp4v2_execute.md` 참조, `--dataset_keys severstal --condition baseline --class_mode {single,multi} --imgsz 640`

> phase0의 `distribution_profiling.py _find_mask_path()`에 severstal 분기 적용됨(prep masks/ GT 인식). phase0 출력 `mask_source`가 `fallback_otsu` 대량이 아니면 GT 정상 인식.

---

## 주의

- **prepare_severstal 선행 필수**: 미실행 시 `train/good`·`test/class{N}` 부재 → phase0/exp4v2 경로 부재 실패.
- **dataset_config severstal 등록 확인**: `domain="severstal"`, `image_dir=.../severstal/train/good`, `seed_dirs=test/class{1-4}`.
- **이미지 1600×256 비정사각**: exp4v2는 `--imgsz 640~1280` 권장(letterbox).
- **CLAUDE.md 정책**: pytest 금지, 검증은 Colab 직접 (위 검증 셀).
