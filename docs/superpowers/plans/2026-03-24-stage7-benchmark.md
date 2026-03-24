# Stage 7 Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `utils/ad_metrics.py` + `stage7_benchmark.py` + `configs/benchmark_experiment.yaml` + `scripts/analyze_results.py` 구현 — EfficientNet-B4/ResNet50(timm) + DRAEM(anomalib) 으로 baseline/aroma_full/aroma_pruned 3 group 을 비교한다.

**Architecture:** `utils/ad_metrics.py` 는 순수 sklearn 메트릭 라이브러리; `stage7_benchmark.py` 는 오케스트레이터(load_config, build_train_paths, build_draem_model, build_classifier_model, run_benchmark); `scripts/analyze_results.py` 는 결과 집계 스크립트. 단위 테스트는 timm/anomalib 를 mock — 실제 GPU 학습은 Colab 에서 수행.

**Tech Stack:** Python, PyYAML, scikit-learn, timm, anomalib, torch

**Spec:** `docs/superpowers/specs/2026-03-24-stage7-benchmark-design.md`

---

## File Structure

| 파일 | 역할 |
|------|------|
| Create: `configs/benchmark_experiment.yaml` | 모델·그룹·메트릭·하이퍼파라미터 설정 |
| Create: `utils/ad_metrics.py` | `extract_metrics()` — sklearn 기반 AUROC/F1 |
| Create: `stage7_benchmark.py` | `run_benchmark()` 오케스트레이터 + CLI |
| Create: `scripts/analyze_results.py` | 결과 집계 + 비교표 생성 |
| Create: `tests/test_stage7.py` | 9 test cases |
| Modify: `docs/작업일지/2026-03-23.md` | Step 13-1/2/3a/3b/3c/4 셀 추가 |

---

## Task 1: 실패 테스트 먼저 작성

**Files:**
- Create: `tests/test_stage7.py`

- [ ] `tests/test_stage7.py` 생성:

```python
# tests/test_stage7.py
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: YAML 파싱 — 필수 키 존재 확인
# ---------------------------------------------------------------------------

def test_config_loads(tmp_path):
    from stage7_benchmark import load_config

    yaml_text = textwrap.dedent("""\
        experiment:
          name: test
          seed: 42
          output_dir: outputs/benchmark_results
        dataset:
          image_size: 256
          pruning_threshold: 0.6
          train_batch_size: 32
          eval_batch_size: 32
          num_workers: 4
          task_by_domain:
            isp: classification
            mvtec: segmentation
            visa: segmentation
        dataset_groups:
          baseline:
            description: "원본"
          aroma_full:
            description: "전체 합성"
          aroma_pruned:
            description: "필터링 합성"
        models:
          efficientnet_b4:
            backbone: efficientnet_b4
            pretrained: true
            epochs: 30
            lr: 0.0001
            optimizer: adam
          resnet50:
            backbone: resnet50
            pretrained: true
            epochs: 30
            lr: 0.0001
            optimizer: adam
          draem:
            enable_sspcab: false
            epochs: 700
            lr: 0.0001
        evaluation:
          metrics: [image_auroc, image_f1, pixel_auroc]
          pixel_auroc_domains: [mvtec, visa]
    """)
    cfg_file = tmp_path / "benchmark_experiment.yaml"
    cfg_file.write_text(yaml_text)

    cfg = load_config(str(cfg_file))

    assert "experiment" in cfg
    assert "dataset" in cfg
    assert "models" in cfg
    assert "evaluation" in cfg
    assert "task_by_domain" in cfg["dataset"]
    assert cfg["dataset"]["task_by_domain"]["mvtec"] == "segmentation"


# ---------------------------------------------------------------------------
# Test 2: resume — experiment_meta.json 존재 시 스킵
# ---------------------------------------------------------------------------

def test_resume_skips_completed(tmp_path):
    from stage7_benchmark import _should_skip

    output_dir = tmp_path / "outputs" / "benchmark_results" / "bottle"
    model, group = "efficientnet_b4", "baseline"
    meta_path = output_dir / model / group / "experiment_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"image_auroc": 0.92}))

    assert _should_skip(output_dir, model, group) is True


def test_resume_runs_when_missing(tmp_path):
    from stage7_benchmark import _should_skip

    output_dir = tmp_path / "outputs" / "benchmark_results" / "bottle"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert _should_skip(output_dir, "efficientnet_b4", "baseline") is False


# ---------------------------------------------------------------------------
# Test 3: metrics — image_auroc / image_f1 정상 반환
# ---------------------------------------------------------------------------

def test_metrics_extraction_basic():
    from utils.ad_metrics import extract_metrics

    y_true  = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    result  = extract_metrics(y_true, y_score)

    assert "image_auroc" in result
    assert "image_f1" in result
    assert "pixel_auroc" in result
    assert abs(result["image_auroc"] - 1.0) < 1e-6
    assert result["image_f1"] > 0.0


# ---------------------------------------------------------------------------
# Test 4: pixel_auroc=None (픽셀 데이터 없음)
# ---------------------------------------------------------------------------

def test_pixel_auroc_none_without_mask():
    from utils.ad_metrics import extract_metrics

    y_true  = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    result  = extract_metrics(y_true, y_score)

    assert result["pixel_auroc"] is None


def test_pixel_auroc_computed_with_mask():
    from utils.ad_metrics import extract_metrics

    y_true      = [0, 1]
    y_score     = [0.1, 0.9]
    pixel_true  = [0, 0, 1, 1]
    pixel_score = [0.1, 0.2, 0.8, 0.9]
    result = extract_metrics(y_true, y_score, pixel_true=pixel_true,
                             pixel_score=pixel_score)

    assert result["pixel_auroc"] is not None
    assert 0.0 <= result["pixel_auroc"] <= 1.0


# ---------------------------------------------------------------------------
# Test 5: group 별 train 경로 구성
# ---------------------------------------------------------------------------

def test_train_path_construction(tmp_path):
    from stage7_benchmark import build_train_paths

    # aroma_full: defect 디렉터리 존재
    defect_dir = tmp_path / "augmented_dataset" / "aroma_full" / "train" / "defect"
    defect_dir.mkdir(parents=True, exist_ok=True)

    good_dir, defect_path = build_train_paths(str(tmp_path), "aroma_full")

    assert good_dir == tmp_path / "augmented_dataset" / "aroma_full" / "train" / "good"
    assert defect_path == defect_dir


def test_train_path_baseline_no_defect(tmp_path):
    from stage7_benchmark import build_train_paths

    # baseline: defect 디렉터리 없음
    good_dir, defect_path = build_train_paths(str(tmp_path), "baseline")

    assert good_dir == tmp_path / "augmented_dataset" / "baseline" / "train" / "good"
    assert defect_path is None


# ---------------------------------------------------------------------------
# Test 6: DRAEM baseline — anomaly_source_path 미지정 (DTD fallback)
# ---------------------------------------------------------------------------

def test_draem_baseline_no_anomaly_source():
    from stage7_benchmark import build_draem_model

    with patch("stage7_benchmark.Draem") as MockDraem:
        build_draem_model(train_defect_dir=None)
        MockDraem.assert_called_once_with()


# ---------------------------------------------------------------------------
# Test 7: DRAEM aroma — anomaly_source_path=train_defect_dir
# ---------------------------------------------------------------------------

def test_draem_aroma_anomaly_source(tmp_path):
    from stage7_benchmark import build_draem_model

    defect_dir = tmp_path / "train" / "defect"
    defect_dir.mkdir(parents=True, exist_ok=True)

    with patch("stage7_benchmark.Draem") as MockDraem:
        build_draem_model(train_defect_dir=defect_dir)
        MockDraem.assert_called_once_with(anomaly_source_path=str(defect_dir))


# ---------------------------------------------------------------------------
# Test 8: domain 별 task 선택
# ---------------------------------------------------------------------------

def test_task_selection_by_domain():
    from stage7_benchmark import get_task_for_domain

    cfg = {
        "dataset": {
            "task_by_domain": {
                "isp": "classification",
                "mvtec": "segmentation",
                "visa": "segmentation",
            }
        }
    }

    assert get_task_for_domain("isp", cfg) == "classification"
    assert get_task_for_domain("mvtec", cfg) == "segmentation"
    assert get_task_for_domain("visa", cfg) == "segmentation"


# ---------------------------------------------------------------------------
# Test 9: baseline → features_only 모드 (EfficientNet-B4/ResNet50)
# ---------------------------------------------------------------------------

def test_classifier_baseline_feature_mode():
    from stage7_benchmark import build_classifier_model

    with patch("stage7_benchmark.timm") as mock_timm:
        mock_timm.create_model.return_value = MagicMock()
        build_classifier_model("efficientnet_b4", group="baseline")
        mock_timm.create_model.assert_called_once_with(
            "efficientnet_b4", pretrained=True, features_only=True
        )


def test_classifier_aroma_num_classes():
    from stage7_benchmark import build_classifier_model

    with patch("stage7_benchmark.timm") as mock_timm:
        mock_timm.create_model.return_value = MagicMock()
        build_classifier_model("resnet50", group="aroma_full")
        mock_timm.create_model.assert_called_once_with(
            "resnet50", pretrained=True, num_classes=2
        )
```

- [ ] 실패 확인: `pytest tests/test_stage7.py -v`
  - Expected: `ImportError` 또는 `ModuleNotFoundError` (구현 없음)

---

## Task 2: `utils/ad_metrics.py` 구현

**Files:**
- Create: `utils/ad_metrics.py`

- [ ] `utils/ad_metrics.py` 생성:

```python
"""utils/ad_metrics.py

Stage 7 이상 탐지 메트릭 추출 유틸리티.

sklearn.metrics 기반 image_auroc / image_f1 / pixel_auroc 계산.
"""
from __future__ import annotations


def extract_metrics(
    y_true: list[int],
    y_score: list[float],
    pixel_true: list | None = None,
    pixel_score: list | None = None,
) -> dict:
    """이미지·픽셀 레벨 이상 탐지 메트릭을 계산한다.

    Args:
        y_true: 이미지 레벨 레이블 (0=정상, 1=결함).
        y_score: 이미지 레벨 anomaly score (클수록 결함에 가까움).
        pixel_true: 픽셀 레벨 레이블 플랫 리스트 (옵션).
        pixel_score: 픽셀 레벨 anomaly score 플랫 리스트 (옵션).

    Returns:
        {
          "image_auroc": float,
          "image_f1":    float,
          "pixel_auroc": float | None   # pixel 데이터 없으면 None
        }
    """
    from sklearn.metrics import f1_score, roc_auc_score

    image_auroc = float(roc_auc_score(y_true, y_score))

    # F1: threshold = 0.5 (score > 0.5 → 결함 예측)
    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    image_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    pixel_auroc = None
    if pixel_true is not None and pixel_score is not None:
        pixel_auroc = float(roc_auc_score(pixel_true, pixel_score))

    return {
        "image_auroc": image_auroc,
        "image_f1": image_f1,
        "pixel_auroc": pixel_auroc,
    }
```

- [ ] 메트릭 테스트 통과 확인:
  ```
  pytest tests/test_stage7.py::test_metrics_extraction_basic tests/test_stage7.py::test_pixel_auroc_none_without_mask tests/test_stage7.py::test_pixel_auroc_computed_with_mask -v
  ```
  - Expected: 3 passed

- [ ] 커밋:
  ```bash
  git add utils/ad_metrics.py
  git commit -m "feat: add utils/ad_metrics.py with sklearn-based AUROC/F1 extraction"
  ```

---

## Task 3: `configs/benchmark_experiment.yaml` 생성

**Files:**
- Create: `configs/benchmark_experiment.yaml`

- [ ] `configs/` 디렉터리가 없으면 생성, `configs/benchmark_experiment.yaml` 작성:

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
  task_by_domain:
    isp: classification
    mvtec: segmentation
    visa: segmentation

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
  pixel_auroc_domains: ["mvtec", "visa"]
```

- [ ] YAML 파싱 테스트 통과 확인:
  ```
  pytest tests/test_stage7.py::test_config_loads -v
  ```
  - Expected: FAIL (load_config 미구현) → Task 4 에서 통과

- [ ] 커밋:
  ```bash
  git add configs/benchmark_experiment.yaml
  git commit -m "feat: add configs/benchmark_experiment.yaml for Stage 7 benchmark"
  ```

---

## Task 4: `stage7_benchmark.py` 구현

**Files:**
- Create: `stage7_benchmark.py`

- [ ] `stage7_benchmark.py` 생성:

```python
"""stage7_benchmark.py

Stage 7 of the AROMA pipeline: anomaly detection benchmark.

3 models × 3 dataset groups × 30 categories = 270 runs.
Models: EfficientNet-B4 / ResNet50 (timm) + DRAEM (anomalib).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

# anomalib / timm 은 런타임 import — 테스트 시 mock 가능
try:
    import timm
except ImportError:
    timm = None  # type: ignore

try:
    from anomalib.models import Draem
except ImportError:
    Draem = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """benchmark_experiment.yaml 을 로드한다."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_train_paths(cat_dir: str, group: str) -> tuple[Path, Optional[Path]]:
    """group 별 train_good_dir / train_defect_dir 경로를 반환한다.

    baseline 의 경우 train/defect/ 없음 → train_defect_dir = None.
    """
    aug = Path(cat_dir) / "augmented_dataset"
    train_good_dir = aug / group / "train" / "good"
    defect_path = aug / group / "train" / "defect"
    train_defect_dir = defect_path if defect_path.exists() else None
    return train_good_dir, train_defect_dir


def get_task_for_domain(domain: str, config: dict) -> str:
    """domain → 'classification' | 'segmentation' 반환."""
    return config["dataset"]["task_by_domain"].get(domain, "classification")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _should_skip(output_dir: Path, model: str, group: str) -> bool:
    """experiment_meta.json 존재 시 True (재학습 불필요)."""
    meta_path = output_dir / model / group / "experiment_meta.json"
    return meta_path.exists()


def _save_meta(output_dir: Path, model: str, group: str, metrics: dict) -> None:
    meta_dir = output_dir / model / group
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "experiment_meta.json").write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_draem_model(train_defect_dir: Optional[Path]):
    """DRAEM 모델을 생성한다.

    baseline: anomaly_source_path 미지정 → Perlin noise fallback (원논문 설정).
    aroma_full / aroma_pruned: train_defect_dir → anomaly_source_path 전달.
    """
    if train_defect_dir is None:
        return Draem()
    return Draem(anomaly_source_path=str(train_defect_dir))


def build_classifier_model(backbone: str, group: str, pretrained: bool = True):
    """EfficientNet-B4 / ResNet50 모델을 생성한다.

    baseline: features_only=True — pretrained feature distance 기반 one-class scoring.
    aroma_full / aroma_pruned: num_classes=2 — 이진 분류 fine-tuning.
    """
    if group == "baseline":
        return timm.create_model(backbone, pretrained=pretrained, features_only=True)
    return timm.create_model(backbone, pretrained=pretrained, num_classes=2)


# ---------------------------------------------------------------------------
# Training / evaluation stubs (Colab 에서 실제 GPU 학습)
# ---------------------------------------------------------------------------

def _train_classifier(model, train_good_dir: Path, train_defect_dir: Optional[Path],
                      group: str, config: dict, seed: int) -> None:
    """timm 분류 모델 학습.

    baseline: 학습 없음 (pretrained feature distance 사용).
    aroma_full / aroma_pruned: good + defect 이진 분류 fine-tuning.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    if group == "baseline":
        return  # baseline 은 fine-tuning 없이 feature distance 사용

    model_cfg = config["models"]
    # backbone 이름 추출 (timm model 에서)
    for name, mcfg in model_cfg.items():
        if name in ("efficientnet_b4", "resnet50"):
            epochs = mcfg.get("epochs", 30)
            lr = mcfg.get("lr", 1e-4)
            break

    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["image_size"],) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # train/good (label 0) + train/defect (label 1) 구성
    train_dir = train_good_dir.parent
    dataset = datasets.ImageFolder(str(train_dir), transform=transform)
    loader = DataLoader(dataset,
                        batch_size=config["dataset"]["train_batch_size"],
                        shuffle=True,
                        num_workers=config["dataset"]["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()


def _evaluate_classifier(model, test_dir: Path, group: str,
                          config: dict) -> tuple[list, list]:
    """timm 분류 모델 평가 → (y_true, y_score) 반환."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["image_size"],) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    loader = DataLoader(dataset,
                        batch_size=config["dataset"]["eval_batch_size"],
                        shuffle=False,
                        num_workers=config["dataset"]["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    y_true, y_score = [], []

    with torch.no_grad():
        if group == "baseline":
            # feature distance: train/good 평균 feature 와의 코사인 거리
            good_dir = _get_baseline_good_dir(test_dir)
            good_dataset = datasets.ImageFolder(str(good_dir.parent.parent / "train"),
                                                transform=transform)
            good_loader = DataLoader(good_dataset, batch_size=32, shuffle=False)
            feats = []
            for imgs, _ in good_loader:
                f = model(imgs.to(device))
                if isinstance(f, (list, tuple)):
                    f = f[-1]
                feats.append(F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1))
            mean_feat = torch.cat(feats).mean(0, keepdim=True)

            for imgs, labels in loader:
                f = model(imgs.to(device))
                if isinstance(f, (list, tuple)):
                    f = f[-1]
                f = F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
                scores = 1.0 - F.cosine_similarity(f, mean_feat)
                y_true.extend(labels.tolist())
                y_score.extend(scores.cpu().tolist())
        else:
            for imgs, labels in loader:
                out = model(imgs.to(device))
                probs = F.softmax(out, dim=1)[:, 1]  # defect 확률
                y_true.extend(labels.tolist())
                y_score.extend(probs.cpu().tolist())

    return y_true, y_score


def _get_baseline_good_dir(test_dir: Path) -> Path:
    """test_dir 에서 augmented_dataset/baseline/train/good 경로를 역산한다."""
    # test_dir = .../augmented_dataset/baseline/test
    return test_dir.parent.parent / "baseline" / "train" / "good"


def _run_draem(model, train_good_dir: Path, test_dir: Path,
               config: dict) -> dict:
    """anomalib Engine 으로 DRAEM 학습 + 평가 → metrics dict 반환."""
    from anomalib.data import Folder
    from anomalib.engine import Engine

    engine = Engine()
    datamodule = Folder(
        name="aroma",
        root=str(train_good_dir.parent.parent),
        normal_dir=str(train_good_dir),
        test_dir=str(test_dir),
        image_size=config["dataset"]["image_size"],
        train_batch_size=config["dataset"]["train_batch_size"],
        eval_batch_size=config["dataset"]["eval_batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )
    engine.fit(model=model, datamodule=datamodule)
    test_result = engine.test(model=model, datamodule=datamodule)

    # anomalib 반환값에서 메트릭 추출
    metrics = {}
    if test_result:
        r = test_result[0] if isinstance(test_result, list) else test_result
        metrics["image_auroc"] = float(r.get("image_AUROC", 0.0))
        metrics["image_f1"] = float(r.get("image_F1Score", 0.0))
        metrics["pixel_auroc"] = (
            float(r["pixel_AUROC"]) if "pixel_AUROC" in r else None
        )
    return metrics


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    config_path: str,
    cat_dir: str,
    groups: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    resume: bool = True,
) -> dict:
    """cat_dir 에 대해 모든 model × group 실험을 수행한다.

    Args:
        config_path: benchmark_experiment.yaml 경로.
        cat_dir: 카테고리 루트 (e.g. .../mvtec/bottle).
        groups: 실행할 그룹 목록 (None = 전체 3개).
        models: 실행할 모델 목록 (None = 전체 3개).
        resume: True 면 experiment_meta.json 존재 시 재학습 스킵.

    Returns:
        {model: {group: {"image_auroc": float, "image_f1": float,
                         "pixel_auroc": float | None}}}
    """
    config = load_config(config_path)
    seed = config["experiment"]["seed"]
    output_root = Path(config["experiment"]["output_dir"])
    cat_name = Path(cat_dir).name
    output_dir = output_root / cat_name

    # domain 결정 — cat_dir 이름에서 추론 (dataset_config.json 없이)
    # cat_dir 구조: .../mvtec/bottle → parents[0].name = "mvtec"
    domain = Path(cat_dir).parent.name

    all_groups = groups or list(config["dataset_groups"].keys())
    all_models = models or list(config["models"].keys())

    test_dir = Path(cat_dir) / "augmented_dataset" / "baseline" / "test"
    task = get_task_for_domain(domain, config)
    pixel_auroc_domains = config["evaluation"].get("pixel_auroc_domains", [])
    use_pixel = domain in pixel_auroc_domains

    results: dict = {}

    for model_name in all_models:
        results[model_name] = {}
        model_cfg = config["models"][model_name]

        for group in all_groups:
            if resume and _should_skip(output_dir, model_name, group):
                meta = json.loads(
                    (output_dir / model_name / group / "experiment_meta.json").read_text()
                )
                results[model_name][group] = meta
                continue

            train_good_dir, train_defect_dir = build_train_paths(cat_dir, group)

            if model_name == "draem":
                model = build_draem_model(train_defect_dir)
                metrics = _run_draem(model, train_good_dir, test_dir, config)

            else:
                backbone = model_cfg["backbone"]
                model = build_classifier_model(backbone, group,
                                               pretrained=model_cfg.get("pretrained", True))
                _train_classifier(model, train_good_dir, train_defect_dir,
                                  group, config, seed)

                y_true, y_score = _evaluate_classifier(model, test_dir, group, config)

                from utils.ad_metrics import extract_metrics
                metrics = extract_metrics(y_true, y_score)
                if not use_pixel:
                    metrics["pixel_auroc"] = None

            _save_meta(output_dir, model_name, group, metrics)
            results[model_name][group] = metrics

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 7 — AD benchmark")
    p.add_argument("--config", default="configs/benchmark_experiment.yaml",
                   help="benchmark_experiment.yaml 경로.")
    p.add_argument("--cat_dir", required=True,
                   help="카테고리 루트 디렉터리.")
    p.add_argument("--groups", nargs="*",
                   help="실행할 그룹 (미지정=전체).")
    p.add_argument("--models", nargs="*",
                   help="실행할 모델 (미지정=전체).")
    p.add_argument("--no-resume", action="store_true",
                   help="이미 완료된 실험도 재실행.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_benchmark(
        config_path=args.config,
        cat_dir=args.cat_dir,
        groups=args.groups,
        models=args.models,
        resume=not args.no_resume,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] 전체 테스트 통과 확인: `pytest tests/test_stage7.py -v`
  - Expected: 11 passed (위에 작성된 모든 테스트)

- [ ] 커밋:
  ```bash
  git add stage7_benchmark.py tests/test_stage7.py
  git commit -m "feat: add stage7_benchmark.py orchestrator and test suite"
  ```

---

## Task 5: `scripts/analyze_results.py` 생성

**Files:**
- Create: `scripts/analyze_results.py`

- [ ] `scripts/` 디렉터리가 없으면 생성, `scripts/analyze_results.py` 작성:

```python
"""scripts/analyze_results.py

Stage 7 벤치마크 결과 집계 + 비교표 생성.

outputs/benchmark_results/ 를 순회하여:
  - benchmark_summary.json
  - comparison_table.md
  - comparison_table.csv
를 생성하고, baseline 대비 aroma_full / aroma_pruned 개선율을 출력한다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


GROUPS  = ["baseline", "aroma_full", "aroma_pruned"]
MODELS  = ["efficientnet_b4", "resnet50", "draem"]
METRICS = ["image_auroc", "image_f1", "pixel_auroc"]


def collect_results(output_root: Path) -> dict:
    """outputs/benchmark_results/{cat}/{model}/{group}/experiment_meta.json 수집."""
    summary: dict = {}
    for cat_dir in sorted(output_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        summary[cat] = {}
        for model in MODELS:
            summary[cat][model] = {}
            for group in GROUPS:
                meta = cat_dir / model / group / "experiment_meta.json"
                if meta.exists():
                    summary[cat][model][group] = json.loads(meta.read_text())
    return summary


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def build_comparison_table(summary: dict) -> list[dict]:
    """비교표 행 리스트 반환."""
    rows = []
    for cat, cat_data in sorted(summary.items()):
        for model, model_data in cat_data.items():
            row = {"category": cat, "model": model}
            for group in GROUPS:
                g_data = model_data.get(group, {})
                row[f"{group}_auroc"]   = g_data.get("image_auroc")
                row[f"{group}_f1"]      = g_data.get("image_f1")
                row[f"{group}_px_auroc"] = g_data.get("pixel_auroc")
            rows.append(row)
    return rows


def save_summary_json(summary: dict, output_root: Path) -> None:
    out = output_root / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out}")


def save_comparison_markdown(rows: list[dict], output_root: Path) -> None:
    lines = [
        "| Category | Model | baseline AUROC | aroma_full AUROC | aroma_pruned AUROC |",
        "|----------|-------|---------------|------------------|--------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['model']} "
            f"| {_fmt(r['baseline_auroc'])} "
            f"| {_fmt(r['aroma_full_auroc'])} "
            f"| {_fmt(r['aroma_pruned_auroc'])} |"
        )
    out = output_root / "comparison_table.md"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")


def save_comparison_csv(rows: list[dict], output_root: Path) -> None:
    fieldnames = ["category", "model"] + [
        f"{g}_{m}"
        for g in GROUPS
        for m in ["auroc", "f1", "px_auroc"]
    ]
    out = output_root / "comparison_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")


def print_improvement(summary: dict) -> None:
    """baseline 대비 aroma_full / aroma_pruned 평균 개선율 출력."""
    improvements: dict[str, list[float]] = {
        "aroma_full": [], "aroma_pruned": []
    }
    for cat_data in summary.values():
        for model_data in cat_data.values():
            base = model_data.get("baseline", {}).get("image_auroc")
            if base is None:
                continue
            for aug in ("aroma_full", "aroma_pruned"):
                aug_val = model_data.get(aug, {}).get("image_auroc")
                if aug_val is not None:
                    improvements[aug].append((aug_val - base) / (base + 1e-9) * 100)

    print("\n── 평균 Image-AUROC 개선율 (baseline 대비) ──")
    for aug, deltas in improvements.items():
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"  {aug:15s}: {avg:+.2f}%  (n={len(deltas)})")
        else:
            print(f"  {aug:15s}: 데이터 없음")


def main(output_root: str = "outputs/benchmark_results") -> None:
    root = Path(output_root)
    if not root.exists():
        print(f"결과 디렉터리 없음: {root}")
        return

    summary = collect_results(root)
    rows = build_comparison_table(summary)

    save_summary_json(summary, root)
    save_comparison_markdown(rows, root)
    save_comparison_csv(rows, root)
    print_improvement(summary)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="outputs/benchmark_results")
    args = p.parse_args()
    main(args.output_root)
```

- [ ] import 확인:
  ```
  python -c "from scripts.analyze_results import collect_results; print('OK')"
  ```
  - Expected: `OK`

- [ ] 커밋:
  ```bash
  git add scripts/analyze_results.py
  git commit -m "feat: add scripts/analyze_results.py for benchmark result aggregation"
  ```

---

## Task 6: 작업일지 Step 13 셀 추가

**Files:**
- Modify: `docs/작업일지/2026-03-23.md`

- [ ] `docs/작업일지/2026-03-23.md` 의 `## 다음 액션` 섹션 앞에 다음 내용 삽입:

````markdown
---

## Stage 7 Colab 실행 셀

### Step 13-1: 벤치마크 환경 설치 확인

```python
import sys
sys.path.insert(0, "/content/aroma")

try:
    import timm
    print(f"timm {timm.__version__} ✓")
except ImportError:
    print("timm 미설치: !pip install timm")

try:
    import anomalib
    print(f"anomalib {anomalib.__version__} ✓")
except ImportError:
    print("anomalib 미설치: !pip install anomalib")
```

---

### Step 13-2: 단건 벤치마크 테스트 (MVTec bottle)

```python
import json, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import run_benchmark

REPO        = Path("/content/drive/MyDrive/project/aroma")
CONFIG_PATH = str(REPO / "configs/benchmark_experiment.yaml")
CONFIG      = json.loads((REPO / "dataset_config.json").read_text())

# MVTec bottle cat_dir 추출
for key, entry in CONFIG.items():
    if key.startswith("_"):
        continue
    cat_dir = str(Path(entry["seed_dir"]).parents[1])
    if Path(cat_dir).name == "bottle":
        break

result = run_benchmark(
    config_path=CONFIG_PATH,
    cat_dir=cat_dir,
    groups=["baseline", "aroma_full"],
    models=["efficientnet_b4"],
)
print(json.dumps(result, indent=2))
```

---

### Step 13-3a/b/c: 도메인별 배치 벤치마크

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

MODELS = ["efficientnet_b4", "resnet50", "draem"]
GROUPS = ["baseline", "aroma_full", "aroma_pruned"]

cat_map: dict[str, str] = {}
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir = str(Path(entry["seed_dir"]).parents[1])
    cat_map[cat_dir] = entry["image_dir"]

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

*(Step 13-3a: `DOMAIN_FILTER = "isp"`, Step 13-3b: `"mvtec"`, Step 13-3c: `"visa"`)*

---

### Step 13-4: 결과 분석 + 비교표 생성

```python
import sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
from scripts.analyze_results import main as analyze

REPO = Path("/content/drive/MyDrive/project/aroma")
analyze(output_root=str(REPO / "outputs/benchmark_results"))
```

````

- [ ] 커밋:
  ```bash
  git add docs/작업일지/2026-03-23.md
  git commit -m "docs: add Stage 7 Colab cells (Step 13-1/2/3/4) to work journal"
  ```

---

## 검증

- [ ] 전체 테스트: `pytest tests/test_stage7.py -v`
  - Expected: 11 passed
- [ ] 회귀 없음: `pytest tests/ -q --ignore=tests/test_parallel.py`
  - Expected: 모든 기존 테스트 통과
