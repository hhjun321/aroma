"""stage5_quality_scoring.py

Stage 5 of the AROMA pipeline: synthesis quality scoring.

Computes artifact_score + blur_score for all defect images in a
Stage 4 seed output directory. Saves quality_scores.json alongside
the defect images.
"""
from __future__ import annotations

import argparse

from utils.quality_scoring import score_defect_images


def run_quality_scoring(
    stage4_seed_dir: str,
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
) -> dict:
    """score_defect_images()의 직접 래퍼.

    CLI 진입점과 작업일지 셀 양쪽이 동일한 함수를 호출하도록 단일화.
    """
    return score_defect_images(
        stage4_seed_dir=stage4_seed_dir,
        w_artifact=w_artifact,
        w_blur=w_blur,
        workers=workers,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 5 — synthesis quality scoring (artifact + blur)"
    )
    p.add_argument("--stage4_seed_dir", required=True,
                   help="Stage 4 seed output directory (contains defect/*.png).")
    p.add_argument("--w_artifact", type=float, default=0.5,
                   help="Weight for artifact_score (default 0.5).")
    p.add_argument("--w_blur", type=float, default=0.5,
                   help="Weight for blur_score (default 0.5).")
    p.add_argument("--workers", type=int, default=0,
                   help="병렬 워커 수 (0=순차, -1=자동, N>=2=N개 프로세스).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = run_quality_scoring(
        stage4_seed_dir=args.stage4_seed_dir,
        w_artifact=args.w_artifact,
        w_blur=args.w_blur,
        workers=args.workers,
    )
    stats = result.get("stats", {})
    count = stats.get("count", 0)
    print(f"Scored {count} images → quality_scores.json")
    if count:
        print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}"
              f"  p50={stats['p50']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}")


if __name__ == "__main__":
    main()
