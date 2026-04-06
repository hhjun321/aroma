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
# Test 2-b: _ensure_test_dir — CASDA 방식 test set 복사
# ---------------------------------------------------------------------------

def test_ensure_test_dir_already_exists(tmp_path):
    """group/test/ 이미 존재 → 복사 없이 그대로 반환."""
    from stage7_benchmark import _ensure_test_dir

    group_test = tmp_path / "augmented_dataset" / "aroma_full" / "test"
    group_test.mkdir(parents=True)
    (group_test / "good").mkdir()

    result = _ensure_test_dir(str(tmp_path), "aroma_full")
    assert result == group_test


def test_ensure_test_dir_symlink_from_baseline(tmp_path):
    """group/test/ 없고 baseline/test 존재 → symlink 또는 복사 후 반환."""
    from stage7_benchmark import _ensure_test_dir

    baseline_test = tmp_path / "augmented_dataset" / "baseline" / "test"
    (baseline_test / "good").mkdir(parents=True)
    (baseline_test / "good" / "img.png").write_bytes(b"")

    result = _ensure_test_dir(str(tmp_path), "aroma_full")

    expected = tmp_path / "augmented_dataset" / "aroma_full" / "test"
    assert result == expected
    # symlink 또는 copytree 어느 쪽이든 내용 접근 가능
    assert (result / "good" / "img.png").exists()


def test_ensure_test_dir_raises_when_baseline_missing(tmp_path):
    """baseline/test 없으면 FileNotFoundError."""
    import pytest
    from stage7_benchmark import _ensure_test_dir

    (tmp_path / "augmented_dataset" / "aroma_full").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        _ensure_test_dir(str(tmp_path), "aroma_full")


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


# ---------------------------------------------------------------------------
# Test 10: defect 이미지 0건 → skip + error 결과 반환
# ---------------------------------------------------------------------------

def _make_minimal_config(tmp_path):
    """run_benchmark 에 필요한 최소 config + 디렉터리 구조 생성."""
    cfg = {
        "experiment": {"name": "test", "seed": 42, "output_dir": str(tmp_path / "out")},
        "dataset": {
            "image_size": 256,
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "num_workers": 0,
            "pruning_threshold": 0.6,
            "task_by_domain": {"isp": "classification"},
        },
        "dataset_groups": {
            "baseline": {"description": "baseline"},
            "aroma_full": {"description": "full"},
        },
        "models": {
            "yolo11": {"model": "yolo11n-cls.pt", "epochs": 1, "lr": 0.01},
        },
        "evaluation": {
            "metrics": ["image_auroc", "image_f1", "pixel_auroc"],
            "pixel_auroc_domains": ["mvtec"],
        },
    }
    return cfg


def test_skip_when_no_defect_images(tmp_path):
    """aroma 그룹에서 defect/ 미존재 시 NoDefectImages 로 skip."""
    from stage7_benchmark import run_benchmark

    cfg = _make_minimal_config(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml as _yaml
    cfg_path.write_text(_yaml.dump(cfg))

    # 카테고리 디렉터리 구성: good만 있고 defect 없음
    cat_dir = tmp_path / "isp" / "LSM_1"
    aug = cat_dir / "augmented_dataset"
    (aug / "baseline" / "train" / "good").mkdir(parents=True)
    (aug / "baseline" / "test" / "good").mkdir(parents=True)
    (aug / "aroma_full" / "train" / "good").mkdir(parents=True)
    # defect/ 없음 → NoDefectImages 기대

    results = run_benchmark(
        config_path=str(cfg_path),
        cat_dir=str(cat_dir),
        groups=["aroma_full"],
        models=["yolo11"],
        resume=False,
    )

    assert "yolo11" in results
    aroma_result = results["yolo11"]["aroma_full"]
    assert aroma_result is not None
    assert aroma_result["error"] == "NoDefectImages"


def test_skip_when_defect_dir_empty(tmp_path):
    """aroma 그룹에서 defect/ 존재하나 비어있으면 NoDefectImages 로 skip."""
    from stage7_benchmark import run_benchmark

    cfg = _make_minimal_config(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml as _yaml
    cfg_path.write_text(_yaml.dump(cfg))

    cat_dir = tmp_path / "isp" / "LSM_1"
    aug = cat_dir / "augmented_dataset"
    (aug / "baseline" / "train" / "good").mkdir(parents=True)
    (aug / "baseline" / "test" / "good").mkdir(parents=True)
    (aug / "aroma_full" / "train" / "good").mkdir(parents=True)
    (aug / "aroma_full" / "train" / "defect").mkdir(parents=True)
    # defect/ 존재하지만 비어있음

    results = run_benchmark(
        config_path=str(cfg_path),
        cat_dir=str(cat_dir),
        groups=["aroma_full"],
        models=["yolo11"],
        resume=False,
    )

    aroma_result = results["yolo11"]["aroma_full"]
    assert aroma_result["error"] == "NoDefectImages"


# ---------------------------------------------------------------------------
# Test 11: _train_yolo 항상 임시 YAML 생성 (test/ 존재 여부 무관)
# ---------------------------------------------------------------------------

def test_yolo_always_creates_yaml(tmp_path):
    """test/ 디렉터리가 존재하더라도 _train_yolo 는 val=train YAML 을 생성해야 한다."""
    from stage7_benchmark import _train_yolo

    # augmented_dataset/{group}/ 구조 구성
    data_dir = tmp_path / "augmented_dataset" / "aroma_full"
    train_dir = data_dir / "train"
    (train_dir / "good").mkdir(parents=True)
    (train_dir / "defect").mkdir(parents=True)
    # 이전 실행에서 남은 test/ 심볼릭 링크 시뮬레이션
    (data_dir / "test" / "good").mkdir(parents=True)

    mock_model = MagicMock()
    cfg = {"models": {"yolo11": {"epochs": 1, "lr": 0.01}},
           "dataset": {"image_size": 256, "train_batch_size": 8}}

    with patch.dict("sys.modules", {"torch": MagicMock()}):
        _train_yolo(mock_model, train_dir / "good",
                    group="aroma_full", config=cfg, seed=42)

    # model.train 호출 시 data 인자가 .yaml 파일 경로여야 함
    mock_model.train.assert_called_once()
    call_kwargs = mock_model.train.call_args
    data_arg = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
    assert data_arg is not None
    assert data_arg.endswith(".yaml"), (
        f"data 인자가 YAML 파일이어야 함 (test/ 존재 시에도), got: {data_arg}"
    )


# ---------------------------------------------------------------------------
# Test 12: build_report.json stage4_status="incomplete" + defect_count=0 경고
# ---------------------------------------------------------------------------

def test_build_report_stage4_incomplete_warning(tmp_path):
    """build_report.json 에 stage4_status='incomplete' 이고 defect_count=0 이면
    (1) stage4_status 관련 UserWarning 발생,
    (2) aroma_full 결과에 error='NoDefectImages' 반환."""
    import yaml as _yaml
    from stage7_benchmark import run_benchmark

    cfg = _make_minimal_config(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_yaml.dump(cfg))

    # 카테고리 디렉터리 구성
    cat_dir = tmp_path / "isp" / "LSM_1"
    aug = cat_dir / "augmented_dataset"

    # Stage 6 사전 검증을 통과하기 위한 최소 구조
    (aug / "baseline" / "test" / "good").mkdir(parents=True)

    # aroma_full: good 만 존재, defect 없음
    good_dir = aug / "aroma_full" / "train" / "good"
    good_dir.mkdir(parents=True)
    (good_dir / "img_001.png").write_bytes(b"\x89PNG dummy")
    (good_dir / "img_002.png").write_bytes(b"\x89PNG dummy")

    # build_report.json — stage4 미완료, defect_count=0
    build_report = {
        "stage4_status": "incomplete",
        "aroma_full": {"defect_count": 0},
    }
    (aug / "build_report.json").write_text(json.dumps(build_report))

    # 실행 + 경고 캡처
    with pytest.warns(UserWarning, match="stage4_status"):
        results = run_benchmark(
            config_path=str(cfg_path),
            cat_dir=str(cat_dir),
            groups=["aroma_full"],
            models=["yolo11"],
            resume=False,
        )

    # aroma_full 결과: defect 이미지 없으므로 NoDefectImages
    aroma_result = results["yolo11"]["aroma_full"]
    assert aroma_result is not None
    assert aroma_result["error"] == "NoDefectImages"


# ---------------------------------------------------------------------------
# Test 13: EfficientDet baseline good_loader — config eval_batch_size 사용
# ---------------------------------------------------------------------------

def test_effdet_baseline_good_loader_uses_config_batch_size():
    """_eval_effdet 의 baseline good_loader 가 하드코딩 32 대신
    config['dataset']['eval_batch_size'] 를 사용하는지 확인."""
    import ast
    from pathlib import Path

    source = Path("stage7_benchmark.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # _eval_effdet 함수 내의 DataLoader 호출을 모두 찾는다
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # DataLoader(...) 호출 확인
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name != "DataLoader":
                continue

            # batch_size 키워드 인자 확인
            for kw in node.keywords:
                if kw.arg == "batch_size":
                    # 하드코딩된 숫자가 아닌지 확인
                    assert not isinstance(kw.value, ast.Constant), (
                        f"stage7_benchmark.py:{node.lineno} — "
                        f"DataLoader batch_size 가 하드코딩됨 ({kw.value.value}). "
                        f"config['dataset']['eval_batch_size'] 를 사용해야 함."
                    )
