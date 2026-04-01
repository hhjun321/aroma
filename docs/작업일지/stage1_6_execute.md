# AROMA Stage 1–6 Colab 실행 셀

## 규칙

- **`DOMAIN_FILTER`** 를 `"isp"` / `"mvtec"` / `"visa"` 로 바꿔 각 도메인 실행
- **skip 로직** 내장 — sentinel 파일이 있으면 자동으로 건너뜀 (resume 가능)
- **병렬 설정** 은 셀 상단 `# 병렬 설정` 블록에서 조절
- `failed` 리스트로 실패 항목 추적 — 전체 실행 후 한 번에 확인 가능

## 진행 상황 확인

```python
# Stage 1–6 전체 진행 상황 확인 (실행 전/후 언제든 사용)
import sys, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO   = Path("/content/drive/MyDrive/project/aroma")
sys.path.insert(0, str(REPO))

from scripts.check_progress import check_category, print_report

CONFIG  = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
tasks   = [(k, v, None) for k, v in CONFIG.items() if not k.startswith("_")]

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(lambda t: check_category(*t), tasks))

print_report(results)
```

---

## Stage 1: ROI 추출

**Sentinel:** `{cat_dir}/stage1_output/roi_metadata.json`
**병렬:** 카테고리 단위 `ThreadPoolExecutor` (내부 `workers=-1` 병렬 이미지 처리)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage1_roi_extraction import run_extraction

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
CAT_THREADS = 2   # 카테고리 동시 처리 수 (Drive 동시 쓰기 안정성 고려)
IMG_WORKERS = -1  # run_extraction 내부 이미지 단위 병렬 (cpu_count-1 자동)

# 처리 대상 수집 (중복 cat_dir 제거 + skip 확인)
cat_tasks, skip = [], 0
seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir   = Path(entry["seed_dir"]).parents[1]
    image_dir = entry["image_dir"]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    sentinel = cat_dir / "stage1_output" / "roi_metadata.json"
    if sentinel.exists():
        skip += 1
    else:
        cat_tasks.append((cat_dir, image_dir, entry["domain"]))

if not cat_tasks:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(cat_tasks)} categories 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage1(args):
        cat_dir, image_dir, domain = args
        run_extraction(
            image_dir  = image_dir,
            output_dir = str(cat_dir / "stage1_output"),
            domain     = domain,
            roi_levels = "both",
            workers    = IMG_WORKERS,
        )
        return cat_dir.name

    with tqdm(total=len(cat_tasks), desc=f"Stage1 {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=CAT_THREADS) as ex:
            futs = {ex.submit(_run_stage1, t): t for t in cat_tasks}
            for fut in as_completed(futs):
                cat_dir, *_ = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": Path(cat_dir).name,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 1b: Seed 특성 분석

**Sentinel:** `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json`
**병렬:** seed 단위 `ThreadPoolExecutor` (카테고리 내 seed들 동시 처리)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage1b_seed_characterization import run_seed_characterization

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
NUM_WORKERS = 4   # seed 단위 동시 처리 (CPU-bound, Colab 코어 수 고려)

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

# 처리 대상 수집
all_seeds, skip = [], 0
for key, entry in ENTRIES:
    seed_dir = Path(entry["seed_dir"])
    seeds    = sorted(seed_dir.glob("*.png")) if seed_dir.exists() else []
    cat_dir  = seed_dir.parents[1]
    for seed in seeds:
        out = cat_dir / "stage1b_output" / seed.stem
        if (out / "seed_profile.json").exists():
            skip += 1
        else:
            all_seeds.append((cat_dir, seed))

if not all_seeds:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage1b(args):
        cat_dir, seed = args
        out = cat_dir / "stage1b_output" / seed.stem
        run_seed_characterization(
            seed_defect = str(seed),
            output_dir  = str(out),
        )
        return cat_dir.name, seed.stem

    with tqdm(total=len(all_seeds), desc=f"Stage1b {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = {ex.submit(_run_stage1b, t): t for t in all_seeds}
            for fut in as_completed(futs):
                cat_dir, seed = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": cat_dir.name, "seed": seed.stem,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}/{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 2: Defect Seed 변형 생성

**Sentinel:** `{cat_dir}/stage2_output/{seed_id}/` 에 PNG가 `NUM_VARIANTS`개 이상
**병렬:** seed 단위 `ThreadPoolExecutor` (내부 `workers=-1` 변형 이미지 병렬 생성)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage2_defect_seed_generation import run_seed_generation

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
SEED_THREADS = 2    # seed 단위 동시 처리 (각 seed 내부에서 workers=-1 사용)
IMG_WORKERS  = -1   # run_seed_generation 내부 변형 이미지 병렬 수
NUM_VARIANTS = 50   # 카테고리당 생성할 변형 수 (실험 후 조정)

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
for key, entry in ENTRIES:
    cat_dir     = Path(entry["seed_dir"]).parents[1]
    stage1b_dir = cat_dir / "stage1b_output"
    if not stage1b_dir.exists():
        continue
    for profile_path in sorted(stage1b_dir.glob("*/seed_profile.json")):
        seed_id = profile_path.parent.name
        out_dir = cat_dir / "stage2_output" / seed_id
        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= NUM_VARIANTS:
            skip += 1
        else:
            all_seeds.append((cat_dir, profile_path))

if not all_seeds:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage2(args):
        cat_dir, profile_path = args
        seed_id = profile_path.parent.name
        info    = json.loads(profile_path.read_text())
        run_seed_generation(
            seed_defect  = info["seed_path"],
            num_variants = NUM_VARIANTS,
            output_dir   = str(cat_dir / "stage2_output" / seed_id),
            seed_profile = str(profile_path),
            workers      = IMG_WORKERS,
        )
        return cat_dir.name, seed_id

    with tqdm(total=len(all_seeds), desc=f"Stage2 {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=SEED_THREADS) as ex:
            futs = {ex.submit(_run_stage2, t): t for t in all_seeds}
            for fut in as_completed(futs):
                cat_dir, profile_path = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": Path(cat_dir).name,
                                   "seed": profile_path.parent.name,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}/{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 3: 레이아웃 로직 (배치 평가)

**Sentinel:** `{cat_dir}/stage3_output/{seed_id}/placement_map.json`
**병렬:** GPU 공유로 seed 간 병렬 불가 → 순차 처리 (GPU 내부 배치 연산으로 보상)

```python
import json, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from utils.suitability import GPUSuitabilityEvaluator

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

evaluator = GPUSuitabilityEvaluator()  # GPU 공유 인스턴스 (순차 처리)

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
for key, entry in ENTRIES:
    cat_dir    = Path(entry["seed_dir"]).parents[1]
    stage2_dir = cat_dir / "stage2_output"
    if not stage2_dir.exists():
        continue
    for d in sorted(stage2_dir.iterdir()):
        if not d.is_dir():
            continue
        if (cat_dir / "stage3_output" / d.name / "placement_map.json").exists():
            skip += 1
        else:
            all_seeds.append((cat_dir, d.name))

if not all_seeds:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    for cat_dir, seed_id in tqdm(all_seeds, desc=f"Stage3 {LABEL}"):
        try:
            roi_meta_path = cat_dir / "stage1_output" / "roi_metadata.json"
            seed_dir      = cat_dir / "stage2_output" / seed_id
            output_dir    = cat_dir / "stage3_output" / seed_id
            profile_path  = cat_dir / "stage1b_output" / seed_id / "seed_profile.json"

            if not roi_meta_path.exists():
                continue

            meta_list = json.loads(roi_meta_path.read_text())
            subtype   = json.loads(profile_path.read_text()).get("subtype", "general") \
                        if profile_path.exists() else "general"
            seed_paths = [str(p) for p in sorted(seed_dir.glob("*.png"))]

            placement_map = []
            for entry in meta_list:
                roi_boxes = entry.get("roi_boxes", [])
                if not roi_boxes:
                    placement_map.append({"image_id": entry["image_id"], "placements": []})
                    continue

                scores   = evaluator.compute_batch(subtype, roi_boxes)
                best_idx = int(np.argmax(scores))
                best_box = roi_boxes[best_idx]
                bg_type  = best_box.get("background_type", "smooth")
                x, y, w, h = best_box["box"]

                rotation = (float(best_box["dominant_angle"])
                            if bg_type == "directional"
                               and subtype in ("linear_scratch", "elongated")
                               and best_box.get("dominant_angle") is not None
                            else float(np.random.uniform(0, 360)))

                placements = [
                    {"defect_path":             s_path,
                     "x":                       x + w // 4,
                     "y":                       y + h // 4,
                     "scale":                   1.0,
                     "rotation":                rotation,
                     "suitability_score":       round(float(scores[best_idx]), 4),
                     "matched_background_type": bg_type}
                    for s_path in seed_paths
                ]
                placement_map.append({"image_id": entry["image_id"], "placements": placements})

            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "placement_map.json").write_text(json.dumps(placement_map, indent=2))

        except Exception as e:
            failed.append({"category": cat_dir.name, "seed": seed_id,
                           "error": str(e), "type": type(e).__name__})

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}/{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 4: MPB 합성

**Sentinel:** `{cat_dir}/stage4_output/{seed_id}/defect/*.png` 존재
**병렬:** 카테고리 단위 `run_synthesis_batch` (배경 이미지 1회 로드 → 전체 seed 적용, `IMG_THREADS` 이미지 병렬)

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage4_mpb_synthesis import run_synthesis_batch

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER  = "isp"   # "isp" / "mvtec" / "visa"
LABEL          = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
USE_FAST_BLEND = True    # False → seamlessClone (느리지만 품질 높음)
IMG_THREADS    = 4       # 이미지 단위 병렬 (ThreadPoolExecutor)

# 도메인별 I/O 최적화
# - isp/mvtec : 이미지 작음 → 기본값(3) 유지
# - visa      : 이미지 ~2MB(3032×2016) → png_compression=1 (~4× 쓰기 빠름)
# ※ max_background_dim 은 사용하지 말 것:
#   Stage 4 출력이 축소 해상도로 저장되어 Stage 6 최종 데이터셋에서
#   good(원본 3032px) vs defect(축소 1024px) 해상도 불일치 발생 → 학습 불가
PNG_COMPRESSION    = {"isp": 3, "mvtec": 3, "visa": 1}[DOMAIN_FILTER]
MAX_BACKGROUND_DIM = None   # 해상도 불일치 방지 — 변경 금지

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

# 카테고리 단위로 묶기
categories = {}
for key, entry in ENTRIES:
    cat_dir = Path(entry["seed_dir"]).parents[1]
    categories.setdefault(str(cat_dir), (cat_dir, entry["image_dir"]))

all_cats, skip_total = [], 0
for cat_dir_str, (cat_dir, image_dir) in categories.items():
    stage3_dir = cat_dir / "stage3_output"
    if not stage3_dir.exists():
        continue
    seed_pm_pairs, skip = [], 0
    for d in sorted(stage3_dir.iterdir()):
        if not d.is_dir():
            continue
        pm_path = d / "placement_map.json"
        if not pm_path.exists():
            continue
        out_defect = cat_dir / "stage4_output" / d.name / "defect"
        if out_defect.exists() and any(out_defect.glob("*.png")):
            skip += 1
        else:
            seed_pm_pairs.append((d.name, str(pm_path)))
    skip_total += skip
    if seed_pm_pairs:
        all_cats.append((cat_dir, image_dir, seed_pm_pairs))

if not all_cats:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    total_seeds = sum(len(p) for _, _, p in all_cats)
    print(f"{LABEL}: {len(all_cats)} 카테고리 / {total_seeds} seeds 처리 예정 (skip {skip_total}건)")
    failed = []

    for cat_dir, image_dir, seed_pm_pairs in tqdm(all_cats, desc=f"Stage4 {LABEL}"):
        try:
            run_synthesis_batch(
                image_dir           = image_dir,
                seed_placement_maps = seed_pm_pairs,
                output_root         = str(cat_dir / "stage4_output"),
                format              = "cls",
                use_fast_blend      = USE_FAST_BLEND,
                workers             = IMG_THREADS,
                png_compression     = PNG_COMPRESSION,
                max_background_dim  = MAX_BACKGROUND_DIM,
            )
        except Exception as e:
            failed.append({"category": cat_dir.name,
                           "error": str(e), "type": type(e).__name__})

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 5: 합성 품질 점수

**Sentinel:** `{cat_dir}/stage4_output/{seed_id}/quality_scores.json`
**병렬:** seed 단위 `ThreadPoolExecutor` (CPU-bound 품질 메트릭 계산)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage5_quality_scoring import run_quality_scoring

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
SEED_THREADS = 4    # seed 단위 동시 처리 (I/O + CPU 혼합)
IMG_WORKERS  = -1   # run_quality_scoring 내부 이미지 단위 병렬 수

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
for key, entry in ENTRIES:
    cat_dir    = Path(entry["seed_dir"]).parents[1]
    stage4_dir = cat_dir / "stage4_output"
    if not stage4_dir.exists():
        continue
    for d in sorted(stage4_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "quality_scores.json").exists():
            skip += 1
        else:
            all_seeds.append((cat_dir, d.name))

if not all_seeds:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage5(args):
        cat_dir, seed_id = args
        run_quality_scoring(
            stage4_seed_dir = str(cat_dir / "stage4_output" / seed_id),
            workers         = IMG_WORKERS,
        )
        return cat_dir.name, seed_id

    with tqdm(total=len(all_seeds), desc=f"Stage5 {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=SEED_THREADS) as ex:
            futs = {ex.submit(_run_stage5, t): t for t in all_seeds}
            for fut in as_completed(futs):
                cat_dir, seed_id = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": Path(cat_dir).name, "seed": seed_id,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}/{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 6: 증강 데이터셋 구성

**Sentinel:** `{cat_dir}/augmented_dataset/build_report.json` (threshold 일치 + defect 존재)
**병렬:** 카테고리 단위 `ThreadPoolExecutor` (I/O-bound Drive 파일 복사)

```python
import json, sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage6_dataset_builder import run_dataset_builder

REPO   = Path("/content/drive/MyDrive/project/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
NUM_IO_THREADS = 8  # 카테고리 내부 파일 복사 스레드 수 (I/O-bound → Thread 유리)
CAT_THREADS    = 2  # 카테고리 단위 동시 처리 수 (Drive 동시 쓰기 안정성 고려)
PRUNING_THRESHOLD = 0.6  # aroma_pruned quality 임계값

# cat_dir 단위로 묶기 (카테고리당 여러 seed_dir 가능)
cat_map: dict[str, str] = {}
cat_seed_dirs: dict[str, list] = defaultdict(list)
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir   = str(Path(entry["seed_dir"]).parents[1])
    image_dir = entry["image_dir"]
    if cat_dir in cat_map and cat_map[cat_dir] != image_dir:
        raise ValueError(f"image_dir 불일치: {cat_dir}")
    cat_map[cat_dir] = image_dir
    cat_seed_dirs[cat_dir].append(entry["seed_dir"])

all_cats, skip = [], 0
for cat_dir, image_dir in cat_map.items():
    if (Path(cat_dir) / "augmented_dataset" / "build_report.json").exists():
        skip += 1
    else:
        all_cats.append((cat_dir, image_dir, cat_seed_dirs[cat_dir]))

if not all_cats:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage6(args):
        cat_dir, image_dir, seed_dirs = args
        run_dataset_builder(
            cat_dir           = cat_dir,
            image_dir         = image_dir,
            seed_dirs         = seed_dirs,
            pruning_threshold = PRUNING_THRESHOLD,
            workers           = NUM_IO_THREADS,
        )
        return Path(cat_dir).name

    with tqdm(total=len(all_cats), desc=f"Stage6 {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=CAT_THREADS) as ex:
            futs = {ex.submit(_run_stage6, t): t for t in all_cats}
            for fut in as_completed(futs):
                cat_dir, *_ = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": Path(cat_dir).name,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 7: 환경 설치

```python
# Stage 7 최초 실행 전 1회만 실행
!pip install -q ultralytics effdet
```

---

## Stage 7: 벤치마크

**Sentinel:** `{output_dir}/{cat_name}/{model}/{group}/experiment_meta.json` (전체 model × group 완료)
**병렬:** GPU 점유 → 카테고리 순차 처리 (`resume=True` 로 중단 재개 가능)

> **모델 구성:** `draem` (700 epoch, GPU 집약) · `efficientnet_b4` · `resnet50` (각 30 epoch)
> **그룹 구성:** `baseline` · `aroma_full` · `aroma_pruned`
> **전제 조건:** Stage 6 완료 (`augmented_dataset/` 존재)

```python
import json, sys, yaml
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import run_benchmark

REPO        = Path("/content/drive/MyDrive/project/aroma")
CONFIG_PATH = str(REPO / "configs" / "benchmark_experiment.yaml")
CONFIG      = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG   = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
OUTPUT_DIR    = str(REPO / "outputs" / "benchmark_results")

EXCLUDE = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
MODELS  = list(BENCH_CFG["models"].keys())
GROUPS  = list(BENCH_CFG["dataset_groups"].keys())

# 처리 대상 수집 (중복 cat_dir 제거 + exclude 필터 + sentinel 확인)
seen = set()
all_cats, skip = [], 0
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    cat_dir = Path(entry["seed_dir"]).parents[1]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    if cat_dir.name in EXCLUDE:
        continue
    # Stage 6 sentinel 확인
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        print(f"  ⚠ {cat_dir.name}: Stage 6 미완료 → skip")
        continue
    out_root = Path(OUTPUT_DIR) / cat_dir.name
    if all((out_root / m / g / "experiment_meta.json").exists()
           for m in MODELS for g in GROUPS):
        skip += 1
    else:
        all_cats.append(cat_dir)

if not all_cats:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    print(f"  모델: {MODELS}")
    print(f"  그룹: {GROUPS}")
    failed = []

    for cat_dir in tqdm(all_cats, desc=f"Stage7 {LABEL}"):
        try:
            run_benchmark(
                config_path = CONFIG_PATH,
                cat_dir     = str(cat_dir),
                resume      = True,
                output_dir  = OUTPUT_DIR,
            )
        except Exception as e:
            failed.append({"category": cat_dir.name,
                           "error": str(e), "type": type(e).__name__})

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 7: 결과 요약

```python
import json
from pathlib import Path

REPO          = Path("/content/drive/MyDrive/project/aroma")
OUTPUT_DIR    = REPO / "outputs" / "benchmark_results"
DOMAIN_FILTER = "isp"   # 필터링할 도메인 (없으면 전체 출력)

CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG = __import__("yaml").safe_load(
    (REPO / "configs" / "benchmark_experiment.yaml").read_text())
MODELS = list(BENCH_CFG["models"].keys())
GROUPS = list(BENCH_CFG["dataset_groups"].keys())

# 도메인 필터: dataset_config 에서 해당 도메인 cat_name 목록 추출
if DOMAIN_FILTER:
    valid_cats = {Path(v["seed_dir"]).parents[1].name
                  for k, v in CONFIG.items()
                  if not k.startswith("_") and v["domain"] == DOMAIN_FILTER}
else:
    valid_cats = None

rows = []
for cat_dir in sorted(OUTPUT_DIR.iterdir()):
    if not cat_dir.is_dir():
        continue
    if valid_cats is not None and cat_dir.name not in valid_cats:
        continue
    row = {"category": cat_dir.name}
    for model in MODELS:
        for group in GROUPS:
            meta_path = cat_dir / model / group / "experiment_meta.json"
            if meta_path.exists():
                m = json.loads(meta_path.read_text())
                row[f"{model}/{group}"] = round(m.get("image_auroc", 0.0), 4)
            else:
                row[f"{model}/{group}"] = "-"
    rows.append(row)

if not rows:
    print("결과 없음")
else:
    cols = ["category"] + [f"{m}/{g}" for m in MODELS for g in GROUPS]
    header = "\t".join(cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("\t".join(str(row.get(c, "-")) for c in cols))
```

---

## 병렬 설정 가이드

| 스테이지 | 외부 병렬 | 내부 병렬 | 근거 |
|---------|---------|---------|------|
| Stage 1  | `CAT_THREADS=2` | `IMG_WORKERS=-1` | CPU-bound 이미지 분석, Drive 동시 쓰기 제한 |
| Stage 1b | `NUM_WORKERS=4` | 없음 | seed 단위 독립, CPU-bound |
| Stage 2  | `SEED_THREADS=2` | `IMG_WORKERS=-1` | 변형 생성 CPU 집약, 내부 병렬로 코어 포화 |
| Stage 3  | 순차 | GPU 배치 | GPU 인스턴스 공유 불가 |
| Stage 4  | 카테고리 단위 순차 | `IMG_THREADS=4` (Thread) | 배경 1회 로드 + Gaussian 합성, I/O 절감 |
| Stage 5  | `SEED_THREADS=4` | `IMG_WORKERS=-1` | I/O + CPU 혼합, seed 단위 독립 |
| Stage 6  | `CAT_THREADS=2` | `NUM_IO_THREADS=8` | I/O-bound Drive 복사, Thread 유리 |
| Stage 7  | 순차 (카테고리) | 없음 | GPU 단일 점유, `resume=True` 중단 재개 |
