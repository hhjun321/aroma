"""stage5_quality_scoring.py

Stage 5 of the AROMA pipeline: synthesis quality scoring.

Computes artifact_score + blur_score for all defect images in a
Stage 4 seed output directory. Saves quality_scores.json alongside
the defect images.

NOTE: Stage 5 output (quality_scores.json) is intentionally stored
inside stage4_output/{seed_id}/ rather than a separate stage5_output/.
This colocation simplifies Stage 6's defect collection and pruning —
each seed directory is self-contained with both images and quality data.
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

from utils.io import validate_dir
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
    # ── 전제조건 검증 ──────────────────────────────────────────────────────
    validate_dir(stage4_seed_dir, name="Stage 4 seed output directory")

    return score_defect_images(
        stage4_seed_dir=stage4_seed_dir,
        w_artifact=w_artifact,
        w_blur=w_blur,
        workers=workers,
    )


def _quality_scoring_worker(args_tuple: tuple) -> dict | None:
    """Pickle-safe worker for parallel batch processing.
    
    Args:
        args_tuple: (stage4_seed_dir, w_artifact, w_blur, workers)
    
    Returns:
        {"seed_id": str, "count": int, "mean": float} | None
    """
    from pathlib import Path
    stage4_seed_dir, w_artifact, w_blur, workers = args_tuple
    
    # Skip if already processed
    cache_path = Path(stage4_seed_dir) / "quality_scores.json"
    if cache_path.exists():
        return None
    
    result = score_defect_images(
        stage4_seed_dir=stage4_seed_dir,
        w_artifact=w_artifact,
        w_blur=w_blur,
        workers=workers,
    )
    
    stats = result.get("stats", {})
    seed_id = Path(stage4_seed_dir).name
    return {
        "seed_id": seed_id,
        "count": stats.get("count", 0),
        "mean": stats.get("mean", 0.0),
    }


def run_quality_scoring_batch(
    stage4_seed_dirs: List[str],
    w_artifact: float = 0.5,
    w_blur: float = 0.5,
    workers: int = 0,
    parallel_seeds: int = 0,
) -> List[dict]:
    """Run quality scoring on multiple seed directories in parallel.
    
    Args:
        stage4_seed_dirs: List of Stage 4 seed output directory paths.
        w_artifact: Weight for artifact_score (default 0.5).
        w_blur: Weight for blur_score (default 0.5).
        workers: Image-level parallel workers within each seed (0=sequential, -1=auto).
        parallel_seeds: Seed-level parallelism (0=sequential, -1=auto, N=N processes).
                       This controls how many seeds are processed in parallel.
    
    Returns:
        List of result dicts: [{"seed_id": str, "count": int, "mean": float}, ...]
        None entries (already processed seeds) are filtered out.
    
    Note:
        - parallel_seeds: controls seed-level parallelism (across directories)
        - workers: controls image-level parallelism (within each seed's defect images)
        - For best performance on large datasets, use parallel_seeds=-1 and workers=0 or 1
          to avoid nested parallelism overhead.
    """
    if not stage4_seed_dirs:
        return []
    
    from utils.parallel import resolve_workers, run_parallel
    
    num_parallel_seeds = resolve_workers(parallel_seeds)
    tasks = [
        (seed_dir, w_artifact, w_blur, workers)
        for seed_dir in stage4_seed_dirs
    ]
    
    results = run_parallel(
        _quality_scoring_worker,
        tasks,
        num_parallel_seeds,
        desc="Stage5 batch quality scoring",
    )
    
    # Filter out None (already processed)
    return [r for r in results if r is not None]


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
