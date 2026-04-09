"""visa_candle Stage 6 결과 확인 스크립트

Stage 7에서 aroma_pruned defect 이미지가 0건으로 나타난 원인 조사.

사용법 (Colab):
    from pathlib import Path
    DATA = Path("/content/drive/MyDrive/data/Aroma")
    exec(open("/content/drive/MyDrive/project/aroma/docs/작업일지/check_visa_candle_stage6.py").read())
"""

import json
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Colab 경로 (필요 시 수정)
try:
    DATA  # 이미 정의되어 있으면 사용
except NameError:
    DATA = Path("/content/drive/MyDrive/data/Aroma")

cat_dir = DATA / "visa/candle"
aug_dir = cat_dir / "augmented_dataset"

print("=" * 70)
print("visa_candle Stage 6 결과 확인")
print("=" * 70)
print(f"카테고리 디렉토리: {cat_dir}")
print(f"증강 데이터셋: {aug_dir}")
print()

# ============================================================================
# 1. build_report.json 확인
# ============================================================================

print("=" * 70)
print("1. build_report.json 확인")
print("=" * 70)

report_path = aug_dir / "build_report.json"
if not report_path.exists():
    print("❌ build_report.json 없음!")
    print(f"   경로: {report_path}")
    exit(1)

with open(report_path) as f:
    report = json.load(f)

print("\n### 기본 정보 ###")
print(f"Domain: {report.get('domain', 'N/A')}")
print(f"Augmentation ratio full: {report.get('augmentation_ratio_full', 'N/A')}")
print(f"Augmentation ratio pruned: {report.get('augmentation_ratio_pruned', 'N/A')}")
print(f"Pruning threshold: {report.get('pruning_threshold', 'N/A')}")

print("\n### Baseline ###")
baseline = report.get('baseline', {})
print(f"Good count: {baseline.get('good_count', 'N/A')}")
print(f"Defect count: {baseline.get('defect_count', 'N/A')}")
print(f"Test defect types: {baseline.get('test_defect_types', 'N/A')}")

print("\n### Aroma Full ###")
aroma_full = report.get('aroma_full', {})
print(f"Good count: {aroma_full.get('good_count', 'N/A')}")
print(f"Defect count: {aroma_full.get('defect_count', 'N/A')}")
print(f"Target defect count: {aroma_full.get('target_defect_count', 'N/A')}")
print(f"Applied ratio: {aroma_full.get('applied_augmentation_ratio', 'N/A')}")

print("\n### Aroma Pruned ###")
aroma_pruned = report.get('aroma_pruned', {})
print(f"Good count: {aroma_pruned.get('good_count', 'N/A')}")
print(f"Defect count: {aroma_pruned.get('defect_count', 'N/A')}")  # ← 여기가 0인지 확인!
print(f"Target defect count: {aroma_pruned.get('target_defect_count', 'N/A')}")
print(f"Applied ratio: {aroma_pruned.get('applied_augmentation_ratio', 'N/A')}")

# ============================================================================
# 2. 실제 파일 개수 확인
# ============================================================================

print("\n" + "=" * 70)
print("2. 실제 디렉토리 파일 개수 확인")
print("=" * 70)

for group in ["baseline", "aroma_full", "aroma_pruned"]:
    group_dir = aug_dir / group / "train"
    
    print(f"\n### {group} ###")
    
    if not group_dir.exists():
        print(f"❌ {group}/train 없음!")
        continue
    
    # Good 이미지
    good_dir = group_dir / "good"
    if good_dir.exists():
        good_files = list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))
        print(f"Good: {len(good_files)} files")
    else:
        print(f"Good: ❌ 디렉토리 없음")
    
    # Defect 이미지
    defect_dir = group_dir / "defect"
    if defect_dir.exists():
        defect_files = list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))
        print(f"Defect: {len(defect_files)} files")
        
        if len(defect_files) == 0:
            print("  ⚠ Defect 디렉토리는 존재하지만 파일이 0개!")
    else:
        print(f"Defect: ❌ 디렉토리 없음")

# ============================================================================
# 3. Stage 4 quality_scores.json 확인
# ============================================================================

print("\n" + "=" * 70)
print("3. Stage 4 quality_scores.json 확인")
print("=" * 70)

stage4_dir = cat_dir / "stage4_output"
if not stage4_dir.exists():
    print("❌ stage4_output 디렉토리 없음!")
else:
    # 모든 seed_dir의 quality_scores.json 수집
    total_scores = 0
    high_quality_count = 0  # threshold >= 0.6
    
    for seed_dir in stage4_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        
        score_file = seed_dir / "quality_scores.json"
        if not score_file.exists():
            continue
        
        with open(score_file) as f:
            scores = json.load(f)
        
        print(f"\n### {seed_dir.name} ###")
        
        if isinstance(scores, list):
            # 새 형식: [{"filename": "...", "final_score": ...}, ...]
            total_scores += len(scores)
            for item in scores:
                if isinstance(item, dict) and "final_score" in item:
                    if item["final_score"] >= 0.6:
                        high_quality_count += 1
            
            print(f"Total scores: {len(scores)}")
            print(f"High quality (>=0.6): {sum(1 for s in scores if isinstance(s, dict) and s.get('final_score', 0) >= 0.6)}")
            
        elif isinstance(scores, dict):
            # 구 형식: {"img.png": {"final_score": ...}, ...}
            total_scores += len(scores)
            for score_data in scores.values():
                if isinstance(score_data, dict) and score_data.get("final_score", 0) >= 0.6:
                    high_quality_count += 1
            
            print(f"Total scores: {len(scores)}")
            print(f"High quality (>=0.6): {sum(1 for v in scores.values() if isinstance(v, dict) and v.get('final_score', 0) >= 0.6)}")
    
    print(f"\n### 전체 요약 ###")
    print(f"Total quality scores: {total_scores}")
    print(f"High quality (>=0.6): {high_quality_count}")
    print(f"Ratio: {high_quality_count / total_scores * 100:.1f}%" if total_scores > 0 else "N/A")

# ============================================================================
# 4. 진단 및 권장사항
# ============================================================================

print("\n" + "=" * 70)
print("4. 진단 및 권장사항")
print("=" * 70)

# 문제 진단
issues = []

# aroma_pruned defect count 확인
aroma_pruned_defect = aroma_pruned.get('defect_count', 0)
if aroma_pruned_defect == 0:
    issues.append("🔴 aroma_pruned defect count = 0")

# applied ratio 확인
if aroma_pruned.get('applied_augmentation_ratio') in (None, 'N/A'):
    issues.append("🔴 aroma_pruned applied ratio = N/A (비율 미적용)")

# 실제 파일과 report 불일치 확인
aroma_pruned_dir = aug_dir / "aroma_pruned" / "train" / "defect"
if aroma_pruned_dir.exists():
    actual_files = list(aroma_pruned_dir.glob("*.png")) + list(aroma_pruned_dir.glob("*.jpg"))
    if len(actual_files) != aroma_pruned_defect:
        issues.append(f"⚠ Report vs 실제 파일 불일치: {aroma_pruned_defect} vs {len(actual_files)}")

if issues:
    print("\n### 발견된 문제 ###")
    for issue in issues:
        print(f"  {issue}")
    
    print("\n### 권장사항 ###")
    print("  1. Stage 6 재실행 권장:")
    print("     - 명시적 비율 전달 확인 (ratio_full=2.0, ratio_pruned=1.5)")
    print("     - 로컬 캐시 → Drive 업로드 확인")
    print()
    print("  2. Stage 4 quality scores 확인:")
    print(f"     - High quality (>=0.6): {high_quality_count} 개")
    print(f"     - Pruned target: good_count * 1.5 = {baseline.get('good_count', 0) * 1.5:.0f}")
    print()
    print("  3. 디버깅:")
    print("     - utils/dataset_builder.py _collect_defect_paths() 함수 로그 확인")
    print("     - augmentation_ratio_pruned 파라미터 전달 여부 확인")
else:
    print("\n✅ 문제 없음 - 모든 데이터가 정상적으로 생성됨")

print("\n" + "=" * 70)
print("확인 완료")
print("=" * 70)
