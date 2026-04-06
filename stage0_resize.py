"""stage0_resize.py

Stage 0 of the AROMA pipeline: in-place image resize.

Resizes all images in dataset directories to a uniform target size
(default 512x512) before the pipeline begins.  This ensures consistent
resolution across all downstream stages and reduces Google Drive storage.

Usage:
    python stage0_resize.py --config dataset_config.json --size 512
    python stage0_resize.py --config dataset_config.json --clean --dry-run
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import load_json, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

_STAGE_OUTPUT_DIRS = [
    "stage1_output",
    "stage1b_output",
    "stage2_output",
    "stage3_output",
    "stage4_output",        # Stage 5 output (quality_scores.json) lives here too
    "augmented_dataset",    # Stage 6 output
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def resize_directory(
    dir_path: Path | str,
    target_size: int = 512,
    dry_run: bool = False,
) -> dict[str, int]:
    """Resize all images in *dir_path* to *target_size* x *target_size* in-place.

    Returns dict with keys: resized, skipped, errors.
    """
    dir_path = Path(dir_path)
    stats = {"resized": 0, "skipped": 0, "errors": 0}

    if not dir_path.is_dir():
        logger.warning("Directory not found, skipping: %s", dir_path)
        return stats

    image_files: list[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        image_files.extend(dir_path.glob(ext))
    image_files.sort()

    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error("Failed to read image: %s", img_path)
                stats["errors"] += 1
                continue

            h, w = img.shape[:2]
            if h == target_size and w == target_size:
                stats["skipped"] += 1
                continue

            if dry_run:
                logger.info("[DRY-RUN] Would resize %s (%dx%d -> %dx%d)",
                            img_path.name, w, h, target_size, target_size)
                stats["resized"] += 1
                continue

            # Choose interpolation: INTER_AREA for downscale, INTER_LINEAR for upscale
            needs_downscale = (h > target_size) or (w > target_size)
            interp = cv2.INTER_AREA if needs_downscale else cv2.INTER_LINEAR

            resized = cv2.resize(img, (target_size, target_size), interpolation=interp)
            cv2.imwrite(str(img_path), resized)
            stats["resized"] += 1

        except Exception:
            logger.exception("Error processing %s", img_path)
            stats["errors"] += 1

    return stats


def resize_category(
    entry: dict[str, Any],
    target_size: int = 512,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resize all images for a single dataset category.

    Processes three directories: image_dir, seed_dir, cat_dir/test/good/.
    Creates a sentinel file on success to enable resume.

    Returns dict with: category, resized, skipped, errors, skipped_category.
    """
    seed_dir = Path(entry["seed_dir"])
    cat_dir = seed_dir.parents[1]
    image_dir = Path(entry["image_dir"])
    test_good_dir = cat_dir / "test" / "good"

    sentinel = cat_dir / f".stage0_resize_{target_size}_done"

    result: dict[str, Any] = {
        "category": cat_dir.name,
        "resized": 0,
        "skipped": 0,
        "errors": 0,
        "skipped_category": False,
    }

    # ── Resume: sentinel이 있으면 skip ────────────────────────────
    if sentinel.exists():
        logger.info("Sentinel exists, skipping: %s", cat_dir.name)
        result["skipped_category"] = True
        return result

    # ── 3개 디렉토리 리사이즈 ─────────────────────────────────────
    for label, dir_path in [
        ("image_dir", image_dir),
        ("seed_dir", seed_dir),
        ("test/good", test_good_dir),
    ]:
        logger.info("Resizing %s: %s", label, dir_path)
        stats = resize_directory(dir_path, target_size=target_size, dry_run=dry_run)
        result["resized"] += stats["resized"]
        result["skipped"] += stats["skipped"]
        result["errors"] += stats["errors"]

    # ── Sentinel 생성 (dry_run이 아니고 에러 0건일 때) ────────────
    if not dry_run and result["errors"] == 0:
        save_json(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_size": target_size,
                "resized": result["resized"],
                "skipped": result["skipped"],
            },
            sentinel,
        )

    return result


def clean_category(
    entry: dict[str, Any],
    dry_run: bool = False,
) -> list[str]:
    """Delete stage 1-6 outputs and sentinel files for a category.

    Returns list of deleted (or would-be-deleted in dry_run) paths.
    """
    seed_dir = Path(entry["seed_dir"])
    cat_dir = seed_dir.parents[1]

    deleted: list[str] = []

    # ── Stage 출력 디렉토리 삭제 ──────────────────────────────────
    for dirname in _STAGE_OUTPUT_DIRS:
        target = cat_dir / dirname
        if target.exists():
            if dry_run:
                logger.info("[DRY-RUN] Would delete: %s", target)
            else:
                shutil.rmtree(target)
                logger.info("Deleted: %s", target)
            deleted.append(str(target))

    # ── Sentinel 파일 삭제 ────────────────────────────────────────
    for sentinel in cat_dir.glob(".stage0_resize_*_done"):
        if dry_run:
            logger.info("[DRY-RUN] Would delete: %s", sentinel)
        else:
            sentinel.unlink()
            logger.info("Deleted: %s", sentinel)
        deleted.append(str(sentinel))

    return deleted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 0 — in-place image resize to uniform target size"
    )
    p.add_argument("--config", default="dataset_config.json",
                   help="Path to dataset_config.json (default: dataset_config.json)")
    p.add_argument("--size", type=int, default=512,
                   help="Target image size (default: 512)")
    p.add_argument("--domain-filter", default=None,
                   choices=["isp", "mvtec", "visa"],
                   help="Process only the specified domain")
    p.add_argument("--dry-run", action="store_true",
                   help="Report only, do not modify files")
    p.add_argument("--clean", action="store_true",
                   help="Delete stage 1-6 outputs before resize")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_json(args.config)

    # Deduplicate by cat_dir (multiple entries may share the same category)
    seen: set[str] = set()
    entries: list[tuple[str, dict]] = []
    for key, entry in config.items():
        if key.startswith("_"):
            continue
        if args.domain_filter and entry.get("domain") != args.domain_filter:
            continue
        cat_dir = str(Path(entry["seed_dir"]).parents[1])
        if cat_dir in seen:
            continue
        seen.add(cat_dir)
        entries.append((key, entry))

    if not entries:
        print("No categories to process.")
        return

    # ── Clean (if requested) ──────────────────────────────────────
    if args.clean:
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}Cleaning {len(entries)} categories...")
        for key, entry in entries:
            deleted = clean_category(entry, dry_run=args.dry_run)
            if deleted:
                print(f"  {key}: {len(deleted)} items {'would be ' if args.dry_run else ''}deleted")

    # ── Resize ────────────────────────────────────────────────────
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Resizing {len(entries)} categories to {args.size}x{args.size}...")
    total_resized, total_skipped, total_errors = 0, 0, 0

    for key, entry in entries:
        result = resize_category(entry, target_size=args.size, dry_run=args.dry_run)
        if result["skipped_category"]:
            print(f"  {key}: sentinel exists — skipped")
        else:
            print(f"  {key}: resized={result['resized']} skipped={result['skipped']} errors={result['errors']}")
        total_resized += result["resized"]
        total_skipped += result["skipped"]
        total_errors += result["errors"]

    print(f"\nTotal: resized={total_resized} skipped={total_skipped} errors={total_errors}")


if __name__ == "__main__":
    main()
