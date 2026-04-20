"""stage6_dataset_builder.py

Stage 6 of the AROMA pipeline: augmented dataset construction.

Builds baseline / aroma_full / aroma_pruned dataset groups from
Stage 4 synthesis outputs and Stage 5 quality scores.
"""
from __future__ import annotations

import argparse
import yaml

from utils.dataset_builder import build_dataset_groups
from utils.io import validate_dir


def run_dataset_builder(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_ratio: float | None = 0.5,
    augmentation_ratio_full: float | None = None,
    augmentation_ratio_pruned: float | None = None,
    augmentation_ratio_by_domain: dict | None = None,
    split_ratio: float | None = None,
    split_seed: int = 42,
    workers: int = 0,
    balance_defect_types: bool = False,
    groups: list[str] | None = None,
    preselected_defect_pairs_full: list[tuple[str, str]] | None = None,
    preselected_defect_pairs_pruned: list[tuple[str, str]] | None = None,
) -> dict:
    """build_dataset_groups() 의 직접 래퍼.

    CLI 진입점과 노트북 셀 양쪽이 동일한 함수를 호출하도록 단일화.
    """
    # ── 전제조건 검증 ──────────────────────────────────────────────────────
    validate_dir(cat_dir, name="Category root directory")
    validate_dir(image_dir, name="Train/good image directory")

    return build_dataset_groups(
        cat_dir=cat_dir,
        image_dir=image_dir,
        seed_dirs=seed_dirs,
        pruning_ratio=pruning_ratio,
        augmentation_ratio_full=augmentation_ratio_full,
        augmentation_ratio_pruned=augmentation_ratio_pruned,
        augmentation_ratio_by_domain=augmentation_ratio_by_domain,
        split_ratio=split_ratio,
        split_seed=split_seed,
        workers=workers,
        balance_defect_types=balance_defect_types,
        groups=groups,
        preselected_defect_pairs_full=preselected_defect_pairs_full,
        preselected_defect_pairs_pruned=preselected_defect_pairs_pruned,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 6 — augmented dataset construction"
    )
    p.add_argument("--cat_dir", required=True,
                   help="카테고리 루트 디렉터리.")
    p.add_argument("--image_dir", required=True,
                   help="원본 train/good 이미지 디렉터리.")
    p.add_argument("--seed_dirs", nargs="+", required=True,
                   help="dataset_config.json 의 seed_dir 경로 목록 (공백 구분).")
    p.add_argument("--pruning_ratio", type=float, default=0.5,
                   help="aroma_pruned quality_score 상위 비율 (기본 0.5 = 상위 50%%).")
    p.add_argument("--augmentation_ratio_full", type=float, default=None,
                   help="aroma_full 원본 대비 합성 defect 비율 (None=모두 사용).")
    p.add_argument("--augmentation_ratio_pruned", type=float, default=None,
                   help="aroma_pruned 원본 대비 합성 defect 비율 (None=모두 사용).")
    p.add_argument("--split_ratio", type=float, default=None,
                   help="good 이미지 train/test 분할 비율 (0~1). "
                        "None=원본 데이터셋 분할 사용. "
                        "예: 0.8 → 전체 good 80%% train, 20%% test.")
    p.add_argument("--split_seed", type=int, default=42,
                   help="split_ratio 사용 시 결정적 셔플 시드 (기본 42).")
    p.add_argument("--config", type=str, default=None,
                   help="benchmark_experiment.yaml 경로 (비율·분할 설정 로드용).")
    p.add_argument("--workers", type=int, default=0,
                   help="병렬 워커 수 (0=순차, -1=자동, N>=2=N 프로세스).")
    p.add_argument("--groups", nargs="+", default=None,
                   choices=["baseline", "aroma_full", "aroma_pruned"],
                   help="빌드할 그룹 지정 (기본=전체). 예: --groups aroma_pruned")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── Config 파일에서 dataset 설정 로드 ─────────────────────────────────
    augmentation_ratio_by_domain = None
    pruning_ratio = args.pruning_ratio
    split_ratio = args.split_ratio
    split_seed = args.split_seed
    balance_defect_types = False
    if args.config:
        from pathlib import Path
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            ds = config.get("dataset", {})
            augmentation_ratio_by_domain = ds.get("augmentation_ratio_by_domain")
            # CLI 인자가 없을 때만 config 값 사용 (CLI 우선)
            if pruning_ratio is None:
                pruning_ratio = ds.get("pruning_ratio", 0.5)
            if split_ratio is None:
                split_ratio = ds.get("split_ratio")
            if split_seed == 42:
                split_seed = ds.get("split_seed", 42)
            balance_defect_types = bool(ds.get("defect_type_balance", False))

    result = run_dataset_builder(
        cat_dir=args.cat_dir,
        image_dir=args.image_dir,
        seed_dirs=args.seed_dirs,
        pruning_ratio=pruning_ratio,
        augmentation_ratio_full=args.augmentation_ratio_full,
        augmentation_ratio_pruned=args.augmentation_ratio_pruned,
        augmentation_ratio_by_domain=augmentation_ratio_by_domain,
        split_ratio=split_ratio,
        split_seed=split_seed,
        workers=args.workers,
        balance_defect_types=balance_defect_types,
        groups=args.groups,
    )
    print(f"Domain: {result.get('domain', 'unknown')}")
    print(f"Applied ratio_full: {result.get('augmentation_ratio_full')}")
    print(f"Applied ratio_pruned: {result.get('augmentation_ratio_pruned')}")
    print(f"pruning_ratio: {result.get('pruning_ratio')}")
    print(f"split_ratio: {result.get('split_ratio')}  split_seed: {result.get('split_seed')}")
    print(f"baseline   good={result['baseline']['good_count']}")
    print(f"aroma_full defect={result['aroma_full']['defect_count']}")
    print(f"aroma_pruned defect={result['aroma_pruned']['defect_count']}")


if __name__ == "__main__":
    main()
