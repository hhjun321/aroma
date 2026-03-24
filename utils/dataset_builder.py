"""utils/dataset_builder.py

Stage 6 증강 데이터셋 구성 라이브러리.

baseline / aroma_full / aroma_pruned 3개 group 을 MVTec-style 폴더 구조로 생성.
파일 복사 방식 (symlink 미사용) — Colab/Google Drive 환경 호환.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _copy_worker(args_tuple: tuple) -> str | None:
    """ProcessPoolExecutor pickle-safe 복사 워커.

    args_tuple: (src_path_str, dst_path_str)
    dst 이미 존재하면 스킵 (파일 단위 resume).
    tasks 구성 예: tasks = [(str(src), str(dst)) for src, dst in copy_pairs]
    """
    src_path_str, dst_path_str = args_tuple
    dst = Path(dst_path_str)
    if dst.exists():
        return dst_path_str
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path_str, dst_path_str)
    return dst_path_str


def _collect_defect_paths(
    cat_dir: str,
    pruning_threshold: float | None = None,
) -> list[tuple[str, str]]:
    """stage4_output/{seed_id}/defect/*.png 수집 (단일 깊이 *).

    Returns:
        List of (src_path_str, dst_filename) where
        dst_filename = "{seed_id}_{image_id}.png" (충돌 방지).
    """
    stage4_dir = Path(cat_dir) / "stage4_output"
    if not stage4_dir.exists():
        return []

    result = []
    for seed_dir in sorted(stage4_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        defect_dir = seed_dir / "defect"
        if not defect_dir.exists():
            continue

        img_paths = sorted(defect_dir.glob("*.png"))
        if not img_paths:
            continue

        if pruning_threshold is not None:
            quality_file = seed_dir / "quality_scores.json"
            if not quality_file.exists():
                warnings.warn(
                    f"quality_scores.json 없음, seed 스킵: {seed_dir.name}",
                    stacklevel=3,
                )
                continue
            quality_data = json.loads(quality_file.read_text())
            quality_map = {
                s["image_id"]: s["quality_score"]
                for s in quality_data.get("scores", [])
            }
            for p in img_paths:
                if quality_map.get(p.stem, 0.0) >= pruning_threshold:
                    result.append((str(p), f"{seed_dir.name}_{p.name}"))
        else:
            for p in img_paths:
                result.append((str(p), f"{seed_dir.name}_{p.name}"))

    return result


def _copy_images(src_dir: Path, dst_dir: Path, num_workers: int,
                 desc: str = "") -> int:
    """src_dir 아래 PNG/JPG 전체를 dst_dir 로 복사. 복사 파일 수 반환."""
    from utils.parallel import run_parallel
    imgs = (
        sorted(src_dir.glob("*.png"))
        + sorted(src_dir.glob("*.jpg"))
        + sorted(src_dir.glob("*.jpeg"))
    )
    if not imgs:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(s), str(dst_dir / s.name)) for s in imgs]
    run_parallel(_copy_worker, tasks, num_workers, desc=desc)
    return len(imgs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_groups(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_threshold: float = 0.6,
    workers: int = 0,
) -> dict:
    """3개 dataset group 을 구성하고 build_report.json 을 저장한다.

    Args:
        cat_dir: 카테고리 루트 디렉터리 (e.g. `.../mvtec/bottle`).
        image_dir: 원본 train/good 이미지 디렉터리.
        seed_dirs: dataset_config.json 에서 추출한 seed_dir 경로 목록.
            baseline/test/{defect_type}/ 구성에 사용.
        pruning_threshold: aroma_pruned 의 quality_score 최솟값.
        workers: 병렬 워커 수 (0=순차).

    Returns:
        build_report dict. build_report.json 으로도 저장.

    Skip:
        build_report.json 이 존재하고 저장된 pruning_threshold 와 일치하면
        재생성 없이 로드하여 반환. 불일치 시 전체 재생성.
    """
    cat_path = Path(cat_dir)
    aug_dir = cat_path / "augmented_dataset"
    report_path = aug_dir / "build_report.json"

    # Skip 조건 확인
    if report_path.exists():
        cached = json.loads(report_path.read_text())
        if abs(cached.get("pruning_threshold", -1) - pruning_threshold) < 1e-6:
            return cached

    from utils.parallel import resolve_workers, run_parallel
    num_workers = resolve_workers(workers)

    # ── baseline/train/good/ ───────────────────────────────────────────────
    baseline_good = aug_dir / "baseline" / "train" / "good"
    good_count = _copy_images(Path(image_dir), baseline_good, num_workers,
                               desc="baseline/train/good")

    # ── baseline/test/ ────────────────────────────────────────────────────
    # test/good/ (image_dir 기준: {cat}/train/good → {cat}/test/good)
    test_good_src = Path(image_dir).parents[1] / "test" / "good"
    if test_good_src.exists():
        _copy_images(test_good_src,
                     aug_dir / "baseline" / "test" / "good",
                     num_workers, desc="baseline/test/good")

    # test/{defect_type}/ (seed_dirs 에서 추출)
    for sd in seed_dirs:
        defect_type = Path(sd).name
        src = Path(sd)
        if src.exists():
            _copy_images(src,
                         aug_dir / "baseline" / "test" / defect_type,
                         num_workers, desc=f"baseline/test/{defect_type}")

    # ── aroma_full/train/ ─────────────────────────────────────────────────
    _copy_images(Path(image_dir),
                 aug_dir / "aroma_full" / "train" / "good",
                 num_workers, desc="aroma_full/train/good")

    full_defect_pairs = _collect_defect_paths(cat_dir, pruning_threshold=None)
    if full_defect_pairs:
        full_defect_dst = aug_dir / "aroma_full" / "train" / "defect"
        full_defect_dst.mkdir(parents=True, exist_ok=True)
        tasks = [(src, str(full_defect_dst / dst_name))
                 for src, dst_name in full_defect_pairs]
        run_parallel(_copy_worker, tasks, num_workers, desc="aroma_full/train/defect")

    # ── aroma_pruned/train/ ───────────────────────────────────────────────
    _copy_images(Path(image_dir),
                 aug_dir / "aroma_pruned" / "train" / "good",
                 num_workers, desc="aroma_pruned/train/good")

    pruned_defect_pairs = _collect_defect_paths(cat_dir,
                                                 pruning_threshold=pruning_threshold)
    if pruned_defect_pairs:
        pruned_defect_dst = aug_dir / "aroma_pruned" / "train" / "defect"
        pruned_defect_dst.mkdir(parents=True, exist_ok=True)
        tasks = [(src, str(pruned_defect_dst / dst_name))
                 for src, dst_name in pruned_defect_pairs]
        run_parallel(_copy_worker, tasks, num_workers, desc="aroma_pruned/train/defect")

    report = {
        "pruning_threshold": pruning_threshold,
        "baseline":     {"good_count": good_count, "defect_count": 0},
        "aroma_full":   {"good_count": good_count,
                         "defect_count": len(full_defect_pairs)},
        "aroma_pruned": {"good_count": good_count,
                         "defect_count": len(pruned_defect_pairs)},
    }
    aug_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report
