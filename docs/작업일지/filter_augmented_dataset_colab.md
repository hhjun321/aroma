# 임시 데이터셋 필터링 스크립트 (Colab 셀)

**목적:** Stage 6 결과물을 도메인별 목표 비율에 맞게 필터링

**방법:** Quality score 기반 상위 이미지 선택

---

## 셀 1: 필터링 함수 정의

```python
import json
from pathlib import Path
from typing import Dict

# 도메인별 목표 증강 비율
AUGMENTATION_RATIO_BY_DOMAIN = {
    "isp": {"full": 1.0, "pruned": 0.5},
    "mvtec": {"full": 2.0, "pruned": 1.5},
    "visa": {"full": 2.0, "pruned": 1.5},
}

def load_quality_scores(stage4_output: Path) -> Dict[str, float]:
    """Load quality scores from all seed directories."""
    scores = {}
    
    for seed_dir in stage4_output.iterdir():
        if not seed_dir.is_dir():
            continue
        
        score_file = seed_dir / "quality_scores.json"
        if not score_file.exists():
            continue
        
        with open(score_file) as f:
            seed_scores = json.load(f)
        
        for img_name, score_data in seed_scores.items():
            if isinstance(score_data, dict) and "final_score" in score_data:
                key = f"{seed_dir.name}/{img_name}"
                scores[key] = score_data["final_score"]
    
    return scores


def filter_dataset_group(group_dir: Path, target_ratio: float, quality_scores: Dict[str, float]):
    """Filter defect images to match target ratio."""
    good_dir = group_dir / "good"
    if not good_dir.exists():
        return {"kept": 0, "removed": 0, "good_count": 0}
    
    # Count good images
    good_images = list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))
    good_count = len(good_images)
    target_defect_count = int(good_count * target_ratio)
    
    print(f"  Good: {good_count}, Target defects: {target_defect_count} (ratio={target_ratio})")
    
    # Find defect directories
    defect_dirs = [d for d in group_dir.iterdir() if d.is_dir() and d.name != "good"]
    
    total_kept = 0
    total_removed = 0
    
    for defect_dir in defect_dirs:
        defect_images = list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))
        current_count = len(defect_images)
        
        if current_count <= target_defect_count:
            print(f"    {defect_dir.name}: {current_count} → skip (already within target)")
            total_kept += current_count
            continue
        
        # Sort by quality score
        scored_images = []
        for img_path in defect_images:
            img_name = img_path.name
            possible_keys = [
                f"{img_path.parent.parent.name}/{img_name}",
                img_name,
                img_path.stem,
            ]
            
            score = 0.5  # default
            for key in possible_keys:
                if key in quality_scores:
                    score = quality_scores[key]
                    break
            
            scored_images.append((img_path, score))
        
        # Keep top N by quality
        scored_images.sort(key=lambda x: x[1], reverse=True)
        keep_images = scored_images[:target_defect_count]
        remove_images = scored_images[target_defect_count:]
        
        # Delete low-quality images
        for img_path, score in remove_images:
            img_path.unlink()
        
        print(f"    {defect_dir.name}: {current_count} → {len(keep_images)} (removed {len(remove_images)})")
        total_kept += len(keep_images)
        total_removed += len(remove_images)
    
    return {"kept": total_kept, "removed": total_removed, "good_count": good_count}


def filter_category(cat_dir: Path, domain: str):
    """Filter augmented dataset for one category."""
    augmented_dir = cat_dir / "augmented_dataset"
    
    if not augmented_dir.exists():
        print(f"❌ {cat_dir.name}: augmented_dataset not found")
        return
    
    ratio_full = AUGMENTATION_RATIO_BY_DOMAIN[domain]["full"]
    ratio_pruned = AUGMENTATION_RATIO_BY_DOMAIN[domain]["pruned"]
    
    print(f"\n{'='*50}")
    print(f"{cat_dir.name} ({domain})")
    print(f"Target: full={ratio_full}, pruned={ratio_pruned}")
    print(f"{'='*50}")
    
    # Load quality scores
    stage4_output = cat_dir / "stage4_output"
    quality_scores = {}
    if stage4_output.exists():
        quality_scores = load_quality_scores(stage4_output)
        print(f"Loaded {len(quality_scores)} quality scores")
    
    # Filter aroma_full
    aroma_full = augmented_dir / "aroma_full"
    if aroma_full.exists():
        print("\n🔧 Filtering aroma_full...")
        stats_full = filter_dataset_group(aroma_full, ratio_full, quality_scores)
    
    # Filter aroma_pruned
    aroma_pruned = augmented_dir / "aroma_pruned"
    if aroma_pruned.exists():
        print("\n🔧 Filtering aroma_pruned...")
        stats_pruned = filter_dataset_group(aroma_pruned, ratio_pruned, quality_scores)
    
    # Update build_report.json
    report_path = augmented_dir / "build_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        
        if "aroma_full" in report:
            report["aroma_full"]["defect_count"] = stats_full["kept"]
            report["ratio_full"] = ratio_full
        
        if "aroma_pruned" in report:
            report["aroma_pruned"]["defect_count"] = stats_pruned["kept"]
            report["aroma_pruned"]["defect_pruned"] = stats_pruned["removed"]
            report["ratio_pruned"] = ratio_pruned
        
        report["_filtered"] = True
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    
    print(f"\n✓ {cat_dir.name} complete!")
    print(f"  aroma_full: {stats_full['kept']} defects")
    print(f"  aroma_pruned: {stats_pruned['kept']} defects")

print("✓ Filter functions defined")
```

---

## 셀 2: 3개 카테고리 필터링 실행

```python
from pathlib import Path

# 카테고리 정보 (smoke_test_3cat.md 참조)
CATEGORIES = {
    "isp_ASM": {
        "domain": "isp",
        "cat_dir": Path("/content/drive/MyDrive/data/Aroma/isp/unsupervised/ASM"),
    },
    "mvtec_bottle": {
        "domain": "mvtec",
        "cat_dir": Path("/content/drive/MyDrive/data/Aroma/mvtec/bottle"),
    },
    "visa_candle": {
        "domain": "visa",
        "cat_dir": Path("/content/drive/MyDrive/data/Aroma/visa/candle"),
    },
}

print("🚀 Starting dataset filtering...\n")

for key, info in CATEGORIES.items():
    filter_category(info["cat_dir"], info["domain"])

print("\n" + "="*50)
print("✓ All categories filtered!")
print("="*50)
```

---

## 셀 3: 결과 검증

```python
import json
from pathlib import Path

print("📊 Filtered Dataset Statistics\n")
print("="*70)

for key, info in CATEGORIES.items():
    report_path = info["cat_dir"] / "augmented_dataset" / "build_report.json"
    
    if not report_path.exists():
        print(f"{key}: ❌ No report found")
        continue
    
    with open(report_path) as f:
        report = json.load(f)
    
    baseline_good = report["baseline"]["good_count"]
    full_defect = report["aroma_full"]["defect_count"]
    pruned_defect = report["aroma_pruned"]["defect_count"]
    
    ratio_full_actual = full_defect / baseline_good if baseline_good > 0 else 0
    ratio_pruned_actual = pruned_defect / baseline_good if baseline_good > 0 else 0
    
    ratio_full_target = report.get("ratio_full", "N/A")
    ratio_pruned_target = report.get("ratio_pruned", "N/A")
    
    print(f"\n{key}:")
    print(f"  Good: {baseline_good}")
    print(f"  aroma_full:   {full_defect:4d} (target={ratio_full_target}, actual={ratio_full_actual:.2f})")
    print(f"  aroma_pruned: {pruned_defect:4d} (target={ratio_pruned_target}, actual={ratio_pruned_actual:.2f})")
    print(f"  Filtered: {report.get('_filtered', False)}")

print("\n" + "="*70)
```

---

## 예상 결과

### Before (현재 상태)
```
isp_ASM:      500 good, 4356 full, 4306 pruned (8.7:1)
mvtec_bottle: 209 good, 4180 full, 4180 pruned (20:1)
visa_candle:  900 good, 9000 full, 1110 pruned (10:1)
```

### After (필터링 후)
```
isp_ASM:      500 good,  500 full,  250 pruned (1:1, 0.5:1) ✓
mvtec_bottle: 209 good,  418 full,  313 pruned (2:1, 1.5:1) ✓
visa_candle:  900 good, 1800 full, 1350 pruned (2:1, 1.5:1) ✓
```

---

## 사용 방법

1. **셀 1 실행**: 필터링 함수 정의
2. **셀 2 실행**: 3개 카테고리 필터링 (약 1-2분 소요)
3. **셀 3 실행**: 결과 검증

필터링 후 바로 Stage 7 벤치마크 실행 가능합니다!
