"""임시 데이터셋 필터링 스크립트

현재 생성된 augmented_dataset의 aroma_full을 도메인별 목표 비율에 맞게 필터링합니다.
Quality score 기반으로 상위 이미지를 선택하여 과도한 증강 데이터를 제거합니다.

Usage:
    python scripts/filter_augmented_dataset.py --cat_dir /path/to/category --domain mvtec
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


# 도메인별 목표 증강 비율
AUGMENTATION_RATIO_BY_DOMAIN = {
    "isp": {
        "full": 1.0,
        "pruned": 0.5,
    },
    "mvtec": {
        "full": 2.0,
        "pruned": 1.5,
    },
    "visa": {
        "full": 2.0,
        "pruned": 1.5,
    },
}


def load_quality_scores(stage4_output: Path) -> Dict[str, float]:
    """Load quality scores from all seed directories.
    
    Returns:
        Dict mapping image_id to final_score (higher is better)
    """
    scores = {}
    
    # Stage 4 output structure: {seed_id}/quality_scores.json
    for seed_dir in stage4_output.iterdir():
        if not seed_dir.is_dir():
            continue
        
        score_file = seed_dir / "quality_scores.json"
        if not score_file.exists():
            continue
        
        with open(score_file) as f:
            seed_scores = json.load(f)
        
        # Merge scores (image_id.png -> final_score)
        for img_name, score_data in seed_scores.items():
            if isinstance(score_data, dict) and "final_score" in score_data:
                # Key: "{seed_id}/{image_id}.png" for uniqueness
                key = f"{seed_dir.name}/{img_name}"
                scores[key] = score_data["final_score"]
    
    return scores


def filter_dataset_group(
    group_dir: Path,
    target_ratio: float,
    quality_scores: Dict[str, float],
    dry_run: bool = False,
) -> Dict[str, int]:
    """Filter defect images in a dataset group to match target ratio.
    
    Args:
        group_dir: Path to aroma_full or aroma_pruned directory
        target_ratio: Target defect/good ratio
        quality_scores: Dict mapping seed_id/image_id to quality score
        dry_run: If True, only print actions without deleting
    
    Returns:
        Dict with statistics (kept, removed counts)
    """
    good_dir = group_dir / "good"
    if not good_dir.exists():
        print(f"  ⚠ Good directory not found: {good_dir}")
        return {"kept": 0, "removed": 0, "good_count": 0}
    
    # Count good images
    good_images = list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))
    good_count = len(good_images)
    
    if good_count == 0:
        print(f"  ⚠ No good images found in {good_dir}")
        return {"kept": 0, "removed": 0, "good_count": 0}
    
    # Target defect count
    target_defect_count = int(good_count * target_ratio)
    
    print(f"  Good images: {good_count}")
    print(f"  Target ratio: {target_ratio} → {target_defect_count} defects")
    
    # Find all defect directories (exclude 'good')
    defect_dirs = [d for d in group_dir.iterdir() if d.is_dir() and d.name != "good"]
    
    if not defect_dirs:
        print(f"  ⚠ No defect directories found in {group_dir}")
        return {"kept": 0, "removed": 0, "good_count": good_count}
    
    total_kept = 0
    total_removed = 0
    
    for defect_dir in defect_dirs:
        defect_images = list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))
        current_count = len(defect_images)
        
        print(f"\n  📁 {defect_dir.name}: {current_count} defects")
        
        if current_count <= target_defect_count:
            print(f"    ✓ Already within target ({current_count} ≤ {target_defect_count})")
            total_kept += current_count
            continue
        
        # Sort by quality score (descending)
        scored_images = []
        for img_path in defect_images:
            # Try multiple key formats to match quality_scores
            img_name = img_path.name
            possible_keys = [
                f"{img_path.parent.parent.name}/{img_name}",  # seed_id/image.png
                img_name,  # image.png
                img_path.stem,  # image (without extension)
            ]
            
            score = None
            for key in possible_keys:
                if key in quality_scores:
                    score = quality_scores[key]
                    break
            
            # Fallback: use random score if not found
            if score is None:
                score = 0.5  # neutral score
            
            scored_images.append((img_path, score))
        
        # Sort by score (high to low) and keep top N
        scored_images.sort(key=lambda x: x[1], reverse=True)
        keep_images = scored_images[:target_defect_count]
        remove_images = scored_images[target_defect_count:]
        
        print(f"    Keep: {len(keep_images)} (quality: {keep_images[0][1]:.3f} ~ {keep_images[-1][1]:.3f})")
        print(f"    Remove: {len(remove_images)} (quality: {remove_images[0][1]:.3f} ~ {remove_images[-1][1]:.3f})")
        
        # Remove low-quality images
        if not dry_run:
            for img_path, score in remove_images:
                img_path.unlink()
        
        total_kept += len(keep_images)
        total_removed += len(remove_images)
    
    return {
        "kept": total_kept,
        "removed": total_removed,
        "good_count": good_count,
    }


def filter_augmented_dataset(
    cat_dir: Path,
    domain: str,
    dry_run: bool = False,
) -> None:
    """Filter augmented dataset for a category.
    
    Args:
        cat_dir: Category root directory (e.g., .../mvtec/bottle)
        domain: Domain name (isp, mvtec, visa)
        dry_run: If True, only print actions without modifying files
    """
    augmented_dir = cat_dir / "augmented_dataset"
    
    if not augmented_dir.exists():
        print(f"❌ Augmented dataset not found: {augmented_dir}")
        return
    
    # Load domain-specific ratios
    if domain not in AUGMENTATION_RATIO_BY_DOMAIN:
        print(f"⚠ Unknown domain '{domain}', using default ratio 1.0")
        ratio_full = 1.0
        ratio_pruned = 0.8
    else:
        ratio_full = AUGMENTATION_RATIO_BY_DOMAIN[domain]["full"]
        ratio_pruned = AUGMENTATION_RATIO_BY_DOMAIN[domain]["pruned"]
    
    print(f"\n{'='*60}")
    print(f"Category: {cat_dir.name}")
    print(f"Domain: {domain}")
    print(f"Target ratios: full={ratio_full}, pruned={ratio_pruned}")
    print(f"{'='*60}")
    
    # Load quality scores from stage4_output
    stage4_output = cat_dir / "stage4_output"
    quality_scores = {}
    
    if stage4_output.exists():
        print("\n📊 Loading quality scores...")
        quality_scores = load_quality_scores(stage4_output)
        print(f"  Loaded {len(quality_scores)} scores")
    else:
        print("\n⚠ No quality scores found, using random filtering")
    
    # Filter aroma_full
    aroma_full = augmented_dir / "aroma_full"
    if aroma_full.exists():
        print("\n🔧 Filtering aroma_full...")
        stats_full = filter_dataset_group(aroma_full, ratio_full, quality_scores, dry_run)
        print(f"\n✓ aroma_full: kept={stats_full['kept']}, removed={stats_full['removed']}")
    
    # Filter aroma_pruned
    aroma_pruned = augmented_dir / "aroma_pruned"
    if aroma_pruned.exists():
        print("\n🔧 Filtering aroma_pruned...")
        stats_pruned = filter_dataset_group(aroma_pruned, ratio_pruned, quality_scores, dry_run)
        print(f"\n✓ aroma_pruned: kept={stats_pruned['kept']}, removed={stats_pruned['removed']}")
    
    # Update build_report.json
    if not dry_run:
        report_path = augmented_dir / "build_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            
            # Update counts
            if "aroma_full" in report and aroma_full.exists():
                report["aroma_full"]["defect_count"] = stats_full["kept"]
                report["ratio_full"] = ratio_full
            
            if "aroma_pruned" in report and aroma_pruned.exists():
                report["aroma_pruned"]["defect_count"] = stats_pruned["kept"]
                report["aroma_pruned"]["defect_pruned"] = stats_pruned["removed"]
                report["ratio_pruned"] = ratio_pruned
            
            # Add filtering metadata
            report["_filtered"] = True
            report["_filter_method"] = "quality_score_based"
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"\n✓ Updated {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter augmented dataset to target ratios")
    parser.add_argument(
        "--cat_dir",
        type=str,
        required=True,
        help="Category directory (e.g., /path/to/mvtec/bottle)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["isp", "mvtec", "visa"],
        help="Domain name",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print actions without modifying files",
    )
    
    args = parser.parse_args()
    
    cat_dir = Path(args.cat_dir)
    if not cat_dir.exists():
        print(f"❌ Category directory not found: {cat_dir}")
        return
    
    filter_augmented_dataset(cat_dir, args.domain, args.dry_run)
    print("\n✓ Filtering complete!")


if __name__ == "__main__":
    main()
