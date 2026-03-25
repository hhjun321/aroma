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

    # 재현성을 위한 seed 고정
    torch.manual_seed(seed)

    model_cfg = config["models"]
    epochs, lr = 30, 1e-4
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

    train_dir = train_good_dir.parent
    dataset = datasets.ImageFolder(str(train_dir), transform=transform)

    # 클래스 불균형 보정 — defect가 good보다 수십 배 많을 수 있음.
    # 각 클래스 가중치 = 전체 샘플 수 / (클래스 수 × 클래스 샘플 수)
    counts = [0, 0]
    for _, label in dataset.samples:
        counts[label] += 1
    n_total = sum(counts)
    class_weight = torch.tensor(
        [n_total / (2 * c) if c > 0 else 1.0 for c in counts],
        dtype=torch.float,
    )

    loader = DataLoader(dataset,
                        batch_size=config["dataset"]["train_batch_size"],
                        shuffle=True,
                        num_workers=config["dataset"]["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

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
    """timm 분류 모델 평가 → (y_true, y_score) 반환.

    y_true 규약: 0=정상(good), 1=이상(anomaly).
    y_score: 높을수록 이상 가능성 높음.

    ImageFolder 는 폴더명 알파벳순으로 class index 를 부여하므로
    "good" 폴더가 항상 label=0 이 아닐 수 있다 (대소문자·이름에 따라 다름).
    → dataset.class_to_idx["good"] 로 정상 클래스 index 를 확인 후 y_true 를 구성한다.

    aroma 그룹 학습 폴더는 defect(d) / good(g) 두 클래스이며 알파벳순 defect=0, good=1.
    → 이상 점수 = softmax[:, 0] (P(defect)).
    """
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
    # test 폴더에서 "good" 클래스의 index 확인 (이름·대소문자 무관)
    good_label = dataset.class_to_idx.get("good", -1)
    loader = DataLoader(dataset,
                        batch_size=config["dataset"]["eval_batch_size"],
                        shuffle=False,
                        num_workers=config["dataset"]["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    y_true, y_score = [], []

    with torch.no_grad():
        if group == "baseline":
            good_train_dir = test_dir.parent.parent / "baseline" / "train"
            good_dataset = datasets.ImageFolder(str(good_train_dir), transform=transform)
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
                # 1 - cosine_sim: 정상 평균과 멀수록 높은 이상 점수
                scores = 1.0 - F.cosine_similarity(f, mean_feat)
                # y_true: good→0, anomaly→1
                y_true.extend([0 if l == good_label else 1
                                for l in labels.tolist()])
                y_score.extend(scores.cpu().tolist())
        else:
            # aroma 학습 폴더: defect(d<g)=0, good=1 → P(defect) = softmax[:, 0]
            for imgs, labels in loader:
                out = model(imgs.to(device))
                probs = F.softmax(out, dim=1)[:, 0]   # P(defect) = 이상 점수
                y_true.extend([0 if l == good_label else 1
                                for l in labels.tolist()])
                y_score.extend(probs.cpu().tolist())

    return y_true, y_score


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
    output_dir: Optional[str] = None,
) -> dict:
    """cat_dir 에 대해 모든 model × group 실험을 수행한다.

    Args:
        output_dir: 결과 저장 루트 경로. 지정하면 yaml의 output_dir 를 덮어씀.
            Colab 환경에서 cat_dir 를 로컬 경로로, 결과는 Drive 에 저장할 때 사용.
    """
    config = load_config(config_path)
    seed = config["experiment"]["seed"]
    output_root = Path(output_dir) if output_dir else Path(config["experiment"]["output_dir"])
    cat_name = Path(cat_dir).name
    output_dir = output_root / cat_name

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
