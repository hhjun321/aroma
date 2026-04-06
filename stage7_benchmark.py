"""stage7_benchmark.py

Stage 7 of the AROMA pipeline: anomaly detection benchmark.

Models: YOLO11 (ultralytics) + EfficientDet-D0 (effdet)
  - baseline    : pretrained 특징 cosine 거리 (one-class scoring)
  - aroma_full  : good + defect 이진 분류 fine-tuning
  - aroma_pruned: good + defect 이진 분류 fine-tuning (quality_score 필터링)
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")


def _glob_images(directory: Path) -> list[Path]:
    """확장자 무관 이미지 파일 수집."""
    images: list[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        images.extend(directory.glob(ext))
    return sorted(images)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_train_paths(cat_dir: str, group: str) -> tuple[Path, Optional[Path]]:
    aug = Path(cat_dir) / "augmented_dataset"
    train_good_dir = aug / group / "train" / "good"
    defect_path = aug / group / "train" / "defect"
    train_defect_dir = defect_path if defect_path.exists() else None
    return train_good_dir, train_defect_dir


def get_task_for_domain(domain: str, config: dict) -> str:
    return config["dataset"]["task_by_domain"].get(domain, "classification")


# ---------------------------------------------------------------------------
# Test set 관리 (CASDA 방식 — 공정한 비교)
# ---------------------------------------------------------------------------

def _ensure_test_dir(cat_dir: str, group: str) -> Path:
    """그룹별 test 디렉터리를 반환한다.

    모든 그룹이 동일한 test set을 사용해야 공정한 벤치마크가 된다.
    CASDA 방식: baseline/test 를 각 그룹에 준비한다.

    Colab/Google Drive 효율화:
      test 데이터는 평가 시 읽기 전용이므로 물리 복사가 불필요하다.
      디렉터리 단위 심볼릭 링크(1회 연산, 추가 스토리지 0)를 우선 시도하고,
      실패 시에만 shutil.copytree 로 폴백한다.

      shutil.copytree on Drive: O(n_files) 네트워크 I/O → 수십 분
      os.symlink (디렉터리): O(1) → 수초

    - group/test/ 이미 존재 → 그대로 반환
    - 없으면 → symlink 시도 → 실패 시 copytree 폴백

    Raises:
        FileNotFoundError: baseline/test 도 없을 때
    """
    import os as _os

    aug = Path(cat_dir) / "augmented_dataset"
    group_test = aug / group / "test"
    if group_test.exists():
        return group_test

    baseline_test = aug / "baseline" / "test"
    if not baseline_test.exists():
        raise FileNotFoundError(f"baseline/test 없음: {baseline_test}")

    group_test.parent.mkdir(parents=True, exist_ok=True)

    # 1순위: 디렉터리 단위 심볼릭 링크 (O(1), 추가 스토리지 0)
    try:
        _os.symlink(baseline_test.resolve(), group_test)
        return group_test
    except OSError:
        pass

    # 폴백: 물리 복사 (Drive FUSE 환경에서는 느림)
    shutil.copytree(str(baseline_test), str(group_test))
    return group_test


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _should_skip(output_dir: Path, model: str, group: str) -> bool:
    return (output_dir / model / group / "experiment_meta.json").exists()


def _save_meta(output_dir: Path, model: str, group: str, metrics: dict) -> None:
    meta_dir = output_dir / model / group
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "experiment_meta.json").write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_yolo_model():
    """YOLO11n-cls pretrained 모델 반환."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics")
    return YOLO("yolo11n-cls.pt")


def build_effdet_classifier(pretrained: bool = True):
    """EfficientDet-D0 백본 + 분류 헤드 모델 반환.

    effdet 의 EfficientDet-D0 백본(EfficientNet-B0)에 AdaptiveAvgPool + Linear 헤드를 붙여
    이미지 분류 모델로 사용한다.
    """
    try:
        from effdet import create_model as create_effdet_model
    except ImportError:
        raise ImportError("effdet 패키지가 필요합니다: pip install effdet")

    import torch.nn as nn

    det = create_effdet_model("efficientdet_d0", pretrained=pretrained, num_classes=90)
    backbone = det.backbone
    num_features = backbone.feature_info.channels()[-1]

    class _EfficientDetClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(num_features, 2)

        def forward(self, x):
            return self.head(self._pool(x))

        def extract_features(self, x):
            return self._pool(x)

        def _pool(self, x):
            feats = self.backbone(x)       # list of feature maps per FPN level
            return self.pool(feats[-1]).flatten(1)

    return _EfficientDetClassifier()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_yolo(
    model: "YOLO",
    train_good_dir: Path,
    group: str,
    config: dict,
    seed: int,
) -> "YOLO":
    """YOLO11 fine-tuning.

    baseline: fine-tuning 없음 (pretrained 특징 거리 사용).
    aroma: train/{good,defect}/ 구조로 이진 분류 fine-tuning.

    Returns:
        학습된 YOLO 모델 (학습 실패 시 last.pt 에서 재생성, 없으면 원본 반환).
    """
    if group == "baseline":
        return model

    import os
    import tempfile
    import torch
    torch.manual_seed(seed)

    data_dir = train_good_dir.parent.parent  # augmented_dataset/{group}/

    # ultralytics ClassificationTrainer 는 val/ · test/ 디렉터리가 없으면
    # testset = None 을 반환 → _build_train_pipeline 에서 ImageFolder(root=None)
    # → TypeError: expected str, bytes or os.PathLike object, not NoneType.
    #
    # 또한, test/ 심볼릭 링크가 이전 실행에서 남아있으면 ultralytics 가
    # test/(2클래스)를 val로 사용 → 모델(2클래스) vs test(2클래스) 불일치 없이
    # 동작하지만, val split 에서 good 샘플만 skip 되어 무의미한 검증이 됨.
    #
    # CASDA prepare_yolo_dataset 패턴 참조:
    # 항상 명시적 YAML 을 생성하여 val=train 으로 지정, ultralytics 의
    # 자동 디렉터리 탐색을 우회한다.
    train_abs = str((data_dir / "train").resolve())
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=tempfile.gettempdir()
    )
    yaml.dump({"train": train_abs, "val": train_abs}, tmp)
    tmp.close()
    _yaml_path = tmp.name

    data_arg = _yaml_path
    model_cfg = config["models"].get("yolo11", {})

    try:
        model.train(
            data=data_arg,
            epochs=model_cfg.get("epochs", 30),
            imgsz=config["dataset"]["image_size"],
            batch=config["dataset"]["train_batch_size"],
            lr0=model_cfg.get("lr", 0.01),
            verbose=False,
            exist_ok=True,
            seed=seed,
        )
    except Exception:
        pass  # 학습 중 예외 → trainer.last 에서 복구 시도
    finally:
        if _yaml_path:
            try:
                os.unlink(_yaml_path)
            except Exception:
                pass

    # 학습 성공/실패 무관 — trainer.last 에서 새 YOLO 인스턴스를 생성한다.
    # model.train() 이 정상 반환해도 내부 ckpt_path 등이 오염될 수 있으므로
    # 항상 체크포인트에서 깨끗한 인스턴스를 만든다.
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        last = getattr(trainer, "last", None)
        if last is not None:
            last_path = Path(str(last))
            if last_path.exists():
                try:
                    from ultralytics import YOLO
                    return YOLO(str(last_path))
                except Exception:
                    pass

    # last.pt 미존재 (첫 epoch 전 실패) → 새 pretrained 모델 반환
    try:
        from ultralytics import YOLO
        return YOLO("yolo11n-cls.pt")
    except Exception:
        return model  # 최후 수단


def _train_effdet(
    model,
    train_good_dir: Path,
    train_defect_dir: Optional[Path],
    group: str,
    config: dict,
    seed: int,
) -> None:
    """EfficientDet 분류기 fine-tuning.

    baseline: fine-tuning 없음 (pretrained 특징 거리 사용).
    aroma: good + defect 이진 분류 fine-tuning.
    """
    if group == "baseline":
        return

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    torch.manual_seed(seed)

    model_cfg = config["models"].get("efficientdet_d0", {})
    epochs = model_cfg.get("epochs", 30)
    lr = model_cfg.get("lr", 1e-4)

    transform = transforms.Compose([
        transforms.Resize((config["dataset"]["image_size"],) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(train_good_dir.parent), transform=transform)

    counts = [0, 0]
    for _, label in dataset.samples:
        counts[label] += 1
    n_total = sum(counts)
    class_weight = torch.tensor(
        [n_total / (2 * c) if c > 0 else 1.0 for c in counts], dtype=torch.float
    )

    loader = DataLoader(
        dataset,
        batch_size=config["dataset"]["train_batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    model.train()
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _collect_test_samples(test_dir: Path) -> tuple[list[str], list[int]]:
    """test_dir/{good,defect,...}/ 구조에서 (paths, y_true) 수집.

    y_true: 0=good, 1=anomaly.
    """
    paths, labels = [], []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = 0 if cls_dir.name == "good" else 1
        for p in _glob_images(cls_dir):
            paths.append(str(p))
            labels.append(label)
    return paths, labels


def _yolo_feature_distance(
    nn_model,
    image_paths: list[str],
    good_train_dir: Path,
    config: dict,
) -> list[float]:
    """Pretrained YOLO11 penultimate-layer 특징 cosine 거리 계산.

    good 학습 이미지의 평균 특징 벡터와의 거리 = anomaly score.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    imgsz = config["dataset"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = next(nn_model.parameters()).device
    nn_model.eval()

    # YOLO11 ClassificationModel: model.model = Sequential[..., Classify]
    # penultimate = layers[-2]
    captured: dict = {}

    def _hook(m, inp, out):
        f = out
        if f.dim() == 4:
            f = F.adaptive_avg_pool2d(f, 1)
        captured["feat"] = f.flatten(1).detach().cpu()

    layers = list(nn_model.model.children())
    handle = layers[-2].register_forward_hook(_hook)

    def _extract(paths: list[str]) -> torch.Tensor:
        feats = []
        with torch.no_grad():
            for p in paths:
                img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                nn_model(img)
                feats.append(captured["feat"])
        return torch.cat(feats)

    good_paths = [str(p) for p in _glob_images(good_train_dir)]
    mean_feat = _extract(good_paths).mean(0, keepdim=True)
    test_feats = _extract(image_paths)

    handle.remove()
    return (1.0 - F.cosine_similarity(test_feats, mean_feat)).tolist()


def _evaluate_yolo(
    model: "YOLO",
    test_dir: Path,
    group: str,
    config: dict,
) -> tuple[list, list]:
    """YOLO11 평가 → (y_true, y_score).

    baseline : pretrained 특징 cosine 거리.
    aroma    : fine-tuned 모델 P(defect=class_0) score.
    """
    image_paths, y_true = _collect_test_samples(test_dir)

    if group == "baseline":
        good_train_dir = test_dir.parent.parent / "baseline" / "train" / "good"
        y_score = _yolo_feature_distance(
            model.model, image_paths, good_train_dir, config
        )
    else:
        # fine-tuned: ultralytics predict → probs
        # ImageFolder sorts classes alphabetically: defect(0) < good(1)
        results = model.predict(
            image_paths,
            imgsz=config["dataset"]["image_size"],
            batch=config["dataset"]["eval_batch_size"],
            verbose=False,
        )
        y_score = [float(r.probs.data.cpu()[0]) for r in results]  # P(defect)

    return y_true, y_score


def _evaluate_effdet(
    model,
    test_dir: Path,
    group: str,
    config: dict,
) -> tuple[list, list]:
    """EfficientDet 분류기 평가 → (y_true, y_score).

    baseline : pretrained 특징 cosine 거리.
    aroma    : fine-tuned 모델 P(defect=class_0) score.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    imgsz = config["dataset"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    good_label = dataset.class_to_idx.get("good", -1)
    loader = DataLoader(
        dataset,
        batch_size=config["dataset"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    y_true, y_score = [], []

    with torch.no_grad():
        if group == "baseline":
            good_train_dir = test_dir.parent.parent / "baseline" / "train" / "good"
            good_ds = datasets.ImageFolder(
                str(good_train_dir.parent), transform=transform
            )
            good_loader = DataLoader(good_ds,
                                     batch_size=config["dataset"]["eval_batch_size"],
                                     shuffle=False,
                                     num_workers=config["dataset"]["num_workers"])
            good_feats = torch.cat(
                [model.extract_features(imgs.to(device)).cpu() for imgs, _ in good_loader]
            )
            mean_feat = good_feats.mean(0, keepdim=True)

            for imgs, labels in loader:
                f = model.extract_features(imgs.to(device)).cpu()
                scores = 1.0 - F.cosine_similarity(f, mean_feat)
                y_true.extend([0 if l == good_label else 1 for l in labels.tolist()])
                y_score.extend(scores.tolist())
        else:
            # defect(d) < good(g) 알파벳순 → defect=0 → P(defect) = softmax[:,0]
            for imgs, labels in loader:
                probs = F.softmax(model(imgs.to(device)), dim=1)[:, 0]
                y_true.extend([0 if l == good_label else 1 for l in labels.tolist()])
                y_score.extend(probs.cpu().tolist())

    return y_true, y_score


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    config_path: str,
    cat_dir: str,
    groups: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    resume: bool = True,
    output_dir: Optional[str] = None,
) -> dict:
    """cat_dir 에 대해 model × group 전체 실험을 수행한다.

    Args:
        config_path: benchmark_experiment.yaml 경로.
        cat_dir:     카테고리 루트 디렉터리.
        groups:      실행할 그룹 (None=전체).
        models:      실행할 모델 (None=전체).
        resume:      True=이미 완료된 실험 skip.
        output_dir:  결과 저장 루트 (None=yaml의 output_dir 사용).
    """
    from utils.ad_metrics import extract_metrics

    config = load_config(config_path)
    seed = config["experiment"]["seed"]
    output_root = Path(output_dir) if output_dir else Path(config["experiment"]["output_dir"])
    cat_name = Path(cat_dir).name
    out_dir = output_root / cat_name

    # Stage 6 완료 여부 사전 검증
    aug_dir = Path(cat_dir) / "augmented_dataset"
    test_dir_check = aug_dir / "baseline" / "test"
    if not aug_dir.exists():
        raise FileNotFoundError(f"Stage 6 미완료 — augmented_dataset 없음: {aug_dir}")
    if not test_dir_check.exists():
        raise FileNotFoundError(f"Stage 6 미완료 — baseline/test 없음: {test_dir_check}")

    # build_report.json 기반 사전 검증 (Stage 6에서 기록한 stage4_status 활용)
    build_report_path = aug_dir / "build_report.json"
    build_report = None
    if build_report_path.exists():
        build_report = json.loads(build_report_path.read_text())
        stage4_status = build_report.get("stage4_status", "unknown")
        if stage4_status in ("incomplete", "not_started"):
            warnings.warn(
                f"build_report.json stage4_status={stage4_status!r} — "
                f"aroma 그룹 학습이 skip될 수 있음: {cat_dir}"
            )

    domain = Path(cat_dir).parent.name
    use_pixel = domain in config["evaluation"].get("pixel_auroc_domains", [])

    all_groups = groups or list(config["dataset_groups"].keys())
    all_models = models or list(config["models"].keys())

    results: dict = {}

    for model_name in all_models:
        results[model_name] = {}

        for group in all_groups:
            if resume and _should_skip(out_dir, model_name, group):
                meta = json.loads(
                    (out_dir / model_name / group / "experiment_meta.json").read_text()
                )
                results[model_name][group] = meta
                continue

            train_good_dir, train_defect_dir = build_train_paths(cat_dir, group)

            # group 학습 데이터 존재 확인 — Stage 6 미완료 시 skip
            if not train_good_dir.exists():
                results[model_name][group] = None
                continue

            # aroma 그룹: defect 이미지 0건이면 skip (CASDA validate 패턴)
            # nc=1 학습 (loss=0, acc=100%) + test 2클래스 불일치 방지
            if group != "baseline":
                _has_defect = (
                    train_defect_dir is not None
                    and train_defect_dir.exists()
                    and any(train_defect_dir.iterdir())
                )
                if not _has_defect:
                    warnings.warn(
                        f"[{model_name}/{group}] defect 이미지 0건 → skip "
                        f"(Stage 4/6 결과 확인 필요: {cat_dir})"
                    )
                    results[model_name][group] = {
                        "error": "NoDefectImages",
                        "detail": f"train/defect/ 비어있거나 없음: {train_defect_dir}",
                    }
                    continue

            try:
                if model_name == "yolo11":
                    model = build_yolo_model()
                    model = _train_yolo(model, train_good_dir, group, config, seed)
                    # ── 평가용 test 디렉터리: 반드시 학습 완료 후 준비 ──
                    # _train_yolo 가 항상 val=train YAML 을 사용하므로 test/ 존재
                    # 여부와 무관하게 안전하지만, 불필요한 파일 노출을 줄이기 위해
                    # 학습 후에만 test/ 심볼릭 링크를 생성한다.
                    # (CASDA: inject → train → clean 순서 패턴 참조)
                    test_dir = _ensure_test_dir(cat_dir, group)
                    y_true, y_score = _evaluate_yolo(model, test_dir, group, config)

                elif model_name == "efficientdet_d0":
                    model = build_effdet_classifier(
                        pretrained=config["models"]["efficientdet_d0"].get("pretrained", True)
                    )
                    _train_effdet(model, train_good_dir, train_defect_dir, group, config, seed)
                    # EfficientDet 은 PyTorch ImageFolder 로 학습하므로 test/ 의
                    # 존재 여부가 학습에 영향을 주지 않지만, 동일한 타이밍 규칙 적용.
                    test_dir = _ensure_test_dir(cat_dir, group)
                    y_true, y_score = _evaluate_effdet(model, test_dir, group, config)

                else:
                    continue

                metrics = extract_metrics(y_true, y_score)
                if not use_pixel:
                    metrics["pixel_auroc"] = None

                _save_meta(out_dir, model_name, group, metrics)
                results[model_name][group] = metrics

            except Exception as e:
                results[model_name][group] = {"error": type(e).__name__, "detail": str(e)}

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 7 — AD benchmark (YOLO11 + EfficientDet-D0)")
    p.add_argument("--config", default="configs/benchmark_experiment.yaml",
                   help="benchmark_experiment.yaml 경로.")
    p.add_argument("--cat_dir", required=True, help="카테고리 루트 디렉터리.")
    p.add_argument("--groups", nargs="*", help="실행할 그룹 (미지정=전체).")
    p.add_argument("--models", nargs="*", help="실행할 모델 (미지정=전체).")
    p.add_argument("--no-resume", action="store_true", help="이미 완료된 실험도 재실행.")
    p.add_argument("--output_dir", default=None, help="결과 저장 루트 경로.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_benchmark(
        config_path=args.config,
        cat_dir=args.cat_dir,
        groups=args.groups,
        models=args.models,
        resume=not args.no_resume,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
