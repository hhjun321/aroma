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
# Test 6: YOLO11 모델 빌더
# ---------------------------------------------------------------------------

def test_build_yolo_model():
    from stage7_benchmark import build_yolo_model

    mock_yolo_cls = MagicMock()
    with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_cls)}):
        build_yolo_model()
    mock_yolo_cls.assert_called_once_with("yolo11n-cls.pt")


# ---------------------------------------------------------------------------
# Test 7: EfficientDet 분류기 빌더
# ---------------------------------------------------------------------------

def test_build_effdet_classifier():
    from stage7_benchmark import build_effdet_classifier

    mock_feat_info = MagicMock()
    mock_feat_info.channels.return_value = [16, 32, 64, 128]
    mock_backbone = MagicMock()
    mock_backbone.feature_info = mock_feat_info
    mock_det = MagicMock()
    mock_det.backbone = mock_backbone

    mock_create = MagicMock(return_value=mock_det)
    mock_nn = MagicMock()
    mock_nn.Module = object

    with patch.dict("sys.modules", {
        "effdet": MagicMock(create_model=mock_create),
        "torch": MagicMock(),
        "torch.nn": mock_nn,
    }):
        build_effdet_classifier(pretrained=False)
    mock_create.assert_called_once_with("efficientdet_d0", pretrained=False, num_classes=90)


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
# Test 9: YOLO11 baseline — fine-tuning 없음
# ---------------------------------------------------------------------------

def test_yolo_baseline_skips_training():
    from stage7_benchmark import _train_yolo

    mock_model = MagicMock()
    cfg = {"models": {"yolo11": {"epochs": 30}}, "dataset": {"image_size": 256, "train_batch_size": 32}}
    _train_yolo(mock_model, Path("/fake/good"), group="baseline", config=cfg, seed=42)
    mock_model.train.assert_not_called()


def test_yolo_aroma_calls_train():
    from stage7_benchmark import _train_yolo

    mock_model = MagicMock()
    cfg = {"models": {"yolo11": {"epochs": 10, "lr": 0.01}},
           "dataset": {"image_size": 256, "train_batch_size": 8}}
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        _train_yolo(mock_model, Path("/fake/aug/aroma_full/train/good"),
                    group="aroma_full", config=cfg, seed=42)
    mock_model.train.assert_called_once()
