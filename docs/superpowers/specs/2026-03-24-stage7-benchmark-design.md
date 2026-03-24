# Stage 7 Benchmark — Design Spec

**Date:** 2026-03-24
**Status:** Draft

---

## Overview

Stage 6 에서 구성된 증강 데이터셋으로 3개 모델을 벤치마크하여
원본셋(baseline) 대비 증강셋(aroma_full / aroma_pruned) 의 성능 우위를 검증한다.

**핵심 가설:** background_type 기반 ROI 합성 증강셋을 학습에 포함시키면
원본셋만 사용할 때보다 이상 탐지 성능이 향상된다.

**실험 규모:** 3 models × 3 dataset groups × 30 categories = **270 runs**

---

## 데이터셋 카테고리 구성 (30개 cat_dir)

| 도메인 | cat_dir 수 | 예시 |
|--------|-----------|------|
| ISP-AD | 3 | isp_ASM, isp_LSM_1, isp_LSM_2 |
| MVTec AD | 15 | bottle, cable, capsule, … (15개) |
| VisA | 12 | candle, cashew, capsules, … (12개) |
| **합계** | **30** | |

`dataset_config.json` 의 `seed_dir` 경로에서 `parents[1]` 로 cat_dir 추출 후 중복 제거.

---

## Design Decisions

| 항목 | 결정 | 이유 |
|------|------|------|
| 모델 로딩 | `timm` (EfficientNet-B4/ResNet50) + `anomalib` (DRAEM) | ultralytics 패턴 동일 — 라이브러리 제공 사전학습 모델, 커스텀 아키텍처 없음 |
| 학습 방식 | Supervised (good + defect 학습) | 합성 결함을 학습에 직접 활용 → 가설 검증 가능 |
| inject/clean | 미적용 | Stage 6 이 이미 group 별 디렉터리 구성 완료; 추가 파일 이동 불필요 |
| 테스트셋 | `baseline/test/` 공통 사용 | 모든 group 동일 조건 평가 |
| Resume | `experiment_meta.json` 존재 시 스킵 | CASDA 패턴; 중단 후 재개 가능 |
| 설정 관리 | `configs/benchmark_experiment.yaml` | CASDA 패턴; 실험 재현성 확보 |
| 결과 저장 | `outputs/benchmark_results/{cat}/{model}/{group}/` | 실험별 독립 디렉터리 |

---

## 벤치마크 모델

### 모델 1: EfficientNet-B4 (timm)

```python
import timm

# aroma_full / aroma_pruned: 이진 분류 (good=0, defect=1)
model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)
# train/good/ + train/defect/ 를 레이블과 함께 fine-tuning

# baseline: train/defect/ 없음 → 분류 헤드 학습 불가
# → ImageNet pretrained feature + cosine distance 기반 anomaly score
#   (num_classes=1000 그대로 유지, feature vector 추출 후 train/good/ 평균과 거리 측정)
model = timm.create_model("efficientnet_b4", pretrained=True, features_only=True)
```

- **baseline**: 사전학습 feature distance 기반 one-class scoring (fine-tuning 없음)
- **aroma_full / aroma_pruned**: 이진 분류 fine-tuning (defect 라벨 학습)
- Image-AUROC, Image-F1 평가

### 모델 2: ResNet50 (timm)

```python
# baseline: feature distance (pretrained, no fine-tuning)
model = timm.create_model("resnet50", pretrained=True, features_only=True)

# aroma_full / aroma_pruned: 이진 분류 fine-tuning
model = timm.create_model("resnet50", pretrained=True, num_classes=2)
```

- EfficientNet-B4 와 동일 방식
- baseline 비교 기준: pretrained feature distance vs. supervised fine-tuned classifier

### 모델 3: DRAEM (anomalib)

```python
from anomalib.models import Draem
from anomalib.engine import Engine
engine = Engine()

# group 별 분기 (run_benchmark() 내부)
if group == "baseline":
    # train/defect/ 없음 → DTD fallback (DRAEM 원논문 설정)
    model = Draem()
else:
    # aroma_full / aroma_pruned: 합성 결함 텍스처 직접 사용
    model = Draem(anomaly_source_path=str(train_defect_dir))
```

- **baseline group**: `anomaly_source_path` 미지정 → anomalib 내장 DTD 랜덤 텍스처 (원논문 설정)
- **aroma_full / aroma_pruned**: `train/defect/` → `anomaly_source_path` 로 전달
- `train/good/` 로 정상 분포 학습
- Image-AUROC, Pixel-AUROC (마스크 있는 도메인), Image-F1 평가

---

## File Structure

```
configs/benchmark_experiment.yaml   — 모델·그룹·메트릭 정의
stage7_benchmark.py                 — 실험 오케스트레이터
utils/ad_metrics.py                 — 메트릭 추출 유틸리티
scripts/analyze_results.py          — 결과 집계 + 비교표 생성
tests/test_stage7.py                — 단위 테스트
```

---

## configs/benchmark_experiment.yaml

```yaml
experiment:
  name: "aroma_benchmark_v1"
  seed: 42
  output_dir: "outputs/benchmark_results"

dataset:
  image_size: 256
  pruning_threshold: 0.6
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 4
  task_by_domain:             # domain 별 task 모드 — 오케스트레이터가 동적 선택
    isp: classification       # ISP-AD: 픽셀 마스크 없음
    mvtec: segmentation       # MVTec AD: 픽셀 마스크 있음 → Pixel-AUROC 활성화
    visa: segmentation        # VisA: 픽셀 마스크 있음 → Pixel-AUROC 활성화

dataset_groups:
  baseline:
    description: "원본 train/good 만"
  aroma_full:
    description: "원본 + 전체 Stage 4 합성"
  aroma_pruned:
    description: "원본 + quality_score 필터링 합성"

models:
  efficientnet_b4:
    backbone: "efficientnet_b4"
    pretrained: true
    epochs: 30
    lr: 0.0001
    optimizer: "adam"
  resnet50:
    backbone: "resnet50"
    pretrained: true
    epochs: 30
    lr: 0.0001
    optimizer: "adam"
  draem:
    enable_sspcab: false
    epochs: 700
    lr: 0.0001

evaluation:
  metrics: [image_auroc, image_f1, pixel_auroc]
  pixel_auroc_domains: ["mvtec", "visa"]  # ISP-AD 는 마스크 없어 제외
  save_images: false
```

---

## stage7_benchmark.py

### `run_benchmark()`

```python
def run_benchmark(
    config_path: str,
    cat_dir: str,
    groups: list[str] | None = None,    # None = 전체 3개
    models: list[str] | None = None,    # None = 전체 3개
    resume: bool = True,
) -> dict
```

- 반환: `{model: {group: {"image_auroc": float, "image_f1": float, "pixel_auroc": float | None}}}`
- 각 실험 완료 시 `experiment_meta.json` 저장

### Resume 로직

```python
meta_path = output_dir / model / group / "experiment_meta.json"
if resume and meta_path.exists():
    # 저장된 결과 로드하여 반환, 재학습 없음
    return json.loads(meta_path.read_text())
```

### 학습 데이터 경로 구성

```python
# 모든 그룹 공통 테스트셋
test_dir = cat_dir / "augmented_dataset" / "baseline" / "test"

# 그룹별 학습셋
train_good_dir   = cat_dir / "augmented_dataset" / group / "train" / "good"
defect_path      = cat_dir / "augmented_dataset" / group / "train" / "defect"
train_defect_dir = defect_path if defect_path.exists() else None
# baseline 의 경우 train_defect_dir = None

# domain 별 task 결정 (dataset_config.json 의 domain 필드 참조)
task = config["dataset"]["task_by_domain"][domain]   # "classification" | "segmentation"
```

### CLI

```bash
python stage7_benchmark.py \
    --config configs/benchmark_experiment.yaml \
    --cat_dir {path}/{category} \
    [--groups baseline aroma_full aroma_pruned] \
    [--models efficientnet_b4 resnet50 draem] \
    [--no-resume]
```

---

## utils/ad_metrics.py

```python
def extract_metrics(
    y_true: list[int],
    y_score: list[float],
    pixel_true: list | None = None,
    pixel_score: list | None = None,
) -> dict:
    """
    Returns:
        {
          "image_auroc": float,
          "image_f1": float,
          "pixel_auroc": float | None   # pixel 데이터 없으면 None
        }
    """
```

- `sklearn.metrics` 사용 (roc_auc_score, f1_score)
- pixel_auroc: `pixel_true` / `pixel_score` 없으면 `None` 반환

---

## scripts/analyze_results.py

`outputs/benchmark_results/` 순회 → 집계 → 출력물 생성.

### 출력물

| 파일 | 내용 |
|------|------|
| `benchmark_summary.json` | 전체 결과 집계 |
| `comparison_table.md` | 도메인 × 모델 × group 비교표 |
| `comparison_table.csv` | 동일 내용 CSV |
| 콘솔 출력 | baseline 대비 aroma_full / aroma_pruned 개선율 (%) |

### 비교표 형식 (예시)

```
| Domain   | Category | Model         | baseline | aroma_full | aroma_pruned |
|----------|----------|---------------|----------|------------|--------------|
| MVTec AD | bottle   | efficientnet  | 0.921    | 0.934      | 0.941        |
| MVTec AD | bottle   | resnet50      | 0.897    | 0.908      | 0.915        |
| MVTec AD | bottle   | draem         | 0.883    | 0.896      | 0.902        |
```

---

## 작업일지 셀 패턴 (Step 13-3a/b/c)

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import run_benchmark

REPO        = Path("/content/drive/MyDrive/project/aroma")
CONFIG      = json.loads((REPO / "dataset_config.json").read_text())
CONFIG_PATH = str(REPO / "configs/benchmark_experiment.yaml")

DOMAIN_FILTER = "visa"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

cat_map: dict[str, str] = {}
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir = str(Path(entry["seed_dir"]).parents[1])
    cat_map[cat_dir] = entry["image_dir"]

MODELS = ["efficientnet_b4", "resnet50", "draem"]
GROUPS = ["baseline", "aroma_full", "aroma_pruned"]

all_cats, skip = [], 0
for cat_dir in cat_map:
    out_root = REPO / "outputs/benchmark_results" / Path(cat_dir).name
    all_done = all(
        (out_root / m / g / "experiment_meta.json").exists()
        for m in MODELS for g in GROUPS
    )
    if all_done:
        skip += 1
    else:
        all_cats.append(cat_dir)

if not all_cats:
    print(f"✓ {LABEL} 모든 실험 완료")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []
    for cat_dir in tqdm(all_cats, desc=f"Stage7 {LABEL}"):
        try:
            run_benchmark(config_path=CONFIG_PATH, cat_dir=cat_dir)
        except Exception as e:
            failed.append({"category": Path(cat_dir).name,
                           "error": str(e), "type": type(e).__name__})
    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:100]}")
```

---

## tests/test_stage7.py 커버리지

| 테스트 케이스 | 설명 |
|---|---|
| `test_config_loads` | YAML 파싱 정상 — 필수 키 존재 확인 |
| `test_resume_skips_completed` | `experiment_meta.json` 존재 시 재학습 없이 스킵 |
| `test_metrics_extraction_basic` | y_true/y_score → image_auroc/image_f1 정상 반환 |
| `test_pixel_auroc_none_without_mask` | pixel 데이터 없으면 pixel_auroc=None 반환 |
| `test_train_path_construction` | group 별 train_good_dir / train_defect_dir 경로; baseline 은 train_defect_dir=None |
| `test_draem_baseline_no_anomaly_source` | baseline → DRAEM anomaly_source_path 미지정(DTD fallback) 확인 |
| `test_draem_aroma_anomaly_source` | aroma_full → DRAEM anomaly_source_path=train_defect_dir 확인 |
| `test_task_selection_by_domain` | isp → classification, mvtec → segmentation, visa → segmentation |
| `test_classifier_baseline_feature_mode` | baseline group → EfficientNet-B4/ResNet50 feature distance 모드 (num_classes 미설정) |
