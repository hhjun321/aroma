# AROMA Phase 1 — Stage 1–6 Colab 실행 셀

> **합성 방식:** MPB (Masked Poisson Blending) — Stage 4: `stage4_mpb_synthesis.py`
> Phase 2 (Diffusion 교체) 실행은 `phase2_execute.md` 참조.

## 규칙

- **`DOMAIN_FILTER`** 를 `"isp"` / `"mvtec"` / `"visa"` 로 바꿔 각 도메인 실행
- **skip 로직** 내장 — sentinel 파일이 있으면 자동으로 건너뜀 (resume 가능)
- **병렬 설정** 은 셀 상단 `# 병렬 설정` 블록에서 조절
- `failed` 리스트로 실패 항목 추적 — 전체 실행 후 한 번에 확인 가능
- **SSD 캐시** — Stage 4·6·7은 Drive FUSE I/O 병목 해소를 위해 로컬 SSD(`/content/tmp_*`) 경유

---

## 진행 상황 확인

```python
import sys, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO   = Path("/content/aroma")
sys.path.insert(0, str(REPO))

from scripts.check_progress import check_category, print_report

CONFIG  = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
tasks   = [(k, v, None) for k, v in CONFIG.items() if not k.startswith("_")]

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(lambda t: check_category(*t), tasks))

print_report(results)
```

---

## Stage 0: 이미지 리사이즈 (512×512)

**Sentinel:** `{cat_dir}/.stage0_resize_512_done`
**병렬:** 카테고리 단위 `ThreadPoolExecutor`

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage0_resize import resize_category, clean_category

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"   # "isp" / "mvtec" / "visa"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
TARGET_SIZE = 512
CLEAN_FIRST = True

# 병렬 설정
CAT_THREADS = 4   # I/O bound

cat_tasks, skip = [], 0
seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    sentinel = cat_dir / f".stage0_resize_{TARGET_SIZE}_done"
    if sentinel.exists():
        skip += 1
    else:
        cat_tasks.append((key, entry))

if not cat_tasks:
    print(f"✓ {LABEL} Stage 0 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(cat_tasks)} categories 처리 예정 (skip {skip}건)")
    failed = []

    if CLEAN_FIRST:
        for key, entry in cat_tasks:
            deleted = clean_category(entry)
            if deleted:
                print(f"  🗑 {key}: {len(deleted)} items deleted")

    def _do(args):
        key, entry = args
        return key, resize_category(entry, target_size=TARGET_SIZE, workers=-1)

    with ThreadPoolExecutor(max_workers=CAT_THREADS) as ex:
        futs = {ex.submit(_do, t): t[0] for t in cat_tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Stage 0"):
            key = futs[fut]
            try:
                _, result = fut.result()
                print(f"  ✓ {key}: resized={result['resized']} skip={result['skipped']}")
            except Exception as e:
                print(f"  ✗ {key}: {e}")
                failed.append(key)

    if failed:
        print(f"\n⚠ Failed: {failed}")
    else:
        print(f"\n✓ {LABEL} Stage 0 완료")
```

---

## Stage 1: ROI 추출

**Sentinel:** `{cat_dir}/stage1_output/roi_metadata.json`
**병렬:** 카테고리 단위 `ThreadPoolExecutor` (내부 `workers=-1` 이미지 병렬)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage1_roi_extraction import run_extraction

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
CAT_THREADS = 2
IMG_WORKERS = -1

cat_tasks, skip = [], 0
seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir   = Path(seed_dirs_list[0]).parents[1]
    image_dir = entry["image_dir"]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    if (cat_dir / "stage1_output" / "roi_metadata.json").exists():
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
**병렬:** `run_seed_characterization_batch()` — seed 단위 내부 병렬 (`workers=-1`)

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage1b_seed_characterization import run_seed_characterization_batch

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_tasks, skip = [], 0
seen_cats = set()
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    use_prefix     = len(seed_dirs_list) > 1
    cat_dir        = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen_cats:
        continue
    seen_cats.add(str(cat_dir))
    for seed_dir_str in seed_dirs_list:
        seed_dir = Path(seed_dir_str)
        for seed in sorted(seed_dir.glob("*.png")) if seed_dir.exists() else []:
            seed_id = f"{seed_dir.name}_{seed.stem}" if use_prefix else seed.stem
            out = cat_dir / "stage1b_output" / seed_id
            if (out / "seed_profile.json").exists():
                skip += 1
            else:
                all_tasks.append((str(seed), str(out)))

if not all_tasks:
    print(f"✓ {LABEL} 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(all_tasks)} seeds 처리 예정 (skip {skip}건)")
    results = run_seed_characterization_batch(all_tasks, workers=-1)
    print(f"✓ {LABEL} 완료 — processed={len(results)}")
```

---

## Stage 2: Defect Seed 변형 생성

**Sentinel:** `{cat_dir}/stage2_output/{seed_id}/` 에 PNG가 `NUM_VARIANTS`개 이상
**병렬:** seed 단위 `ThreadPoolExecutor` (내부 `workers=-1`)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage2_defect_seed_generation import run_seed_generation

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
SEED_THREADS = 2
IMG_WORKERS  = -1
NUM_VARIANTS = 50

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
seen_cats = set()
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir     = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen_cats:
        continue
    seen_cats.add(str(cat_dir))
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
**병렬:** GPU 공유로 seed 간 병렬 불가 → 순차 처리

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage3_layout_logic import run_layout_logic

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
seen_cats = set()
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir    = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen_cats:
        continue
    seen_cats.add(str(cat_dir))
    stage2_dir = cat_dir / "stage2_output"
    if not stage2_dir.exists():
        continue
    for d in sorted(stage2_dir.iterdir()):
        if not d.is_dir():
            continue
        if (cat_dir / "stage3_output" / d.name / "placement_map.json").exists():
            skip += 1
        else:
            all_seeds.append((cat_dir, d.name, entry))

if not all_seeds:
    print(f"✓ {LABEL} 모든 작업 완료")
else:
    print(f"{LABEL}: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    for cat_dir, seed_id, entry in tqdm(all_seeds, desc=f"Stage3 {LABEL}"):
        try:
            profile_path = cat_dir / "stage1b_output" / seed_id / "seed_profile.json"
            run_layout_logic(
                roi_metadata     = str(cat_dir / "stage1_output" / "roi_metadata.json"),
                defect_seeds_dir = str(cat_dir / "stage2_output" / seed_id),
                output_dir       = str(cat_dir / "stage3_output" / seed_id),
                seed_profile     = str(profile_path) if profile_path.exists() else None,
                domain           = entry.get("domain", "mvtec"),
                use_gpu          = True,
            )
        except Exception as e:
            failed.append({"category": cat_dir.name, "seed": seed_id,
                           "error": str(e), "type": type(e).__name__})

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}/{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 4: MPB 합성 (로컬 SSD 캐시)

**Sentinel:** `{cat_dir}/stage4_output/{seed_id}/defect/*.png` 존재
**병렬:** 카테고리 순차 처리 — 각 카테고리를 로컬 SSD에서 합성 후 Drive 업로드

> Drive FUSE I/O 병목 해소: 배경 이미지·stage2 variants를 로컬 SSD에 복사 후 합성,
> 결과만 Drive로 업로드. placement_map의 `defect_path`를 로컬 경로로 치환.

```python
import json, sys, shutil, time, yaml
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage4_mpb_synthesis import run_synthesis_batch

REPO      = Path("/content/aroma")
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER      = "isp"
LABEL              = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
USE_FAST_BLEND     = True
IMG_THREADS        = 4
PNG_COMPRESSION    = {"isp": 3, "mvtec": 3, "visa": 1}[DOMAIN_FILTER]
MAX_BACKGROUND_DIM = None   # 해상도 불일치 방지 — 변경 금지
MAX_IMAGES_PER_SEED = BENCH_CFG.get("synthesis", {}).get("max_images_per_seed", 50)
LOCAL_TMP          = Path("/content/tmp_stage4")

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

categories = {}
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
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
        t0 = time.time()
        local_cat    = LOCAL_TMP / cat_dir.name
        local_imgdir = local_cat / "images"
        local_stage2 = local_cat / "stage2_output"
        local_stage3 = local_cat / "stage3_output"
        local_stage4 = local_cat / "stage4_output"
        stage2_dir   = cat_dir / "stage2_output"
        try:
            # 1) 배경 이미지 → 로컬 SSD
            if local_imgdir.exists():
                shutil.rmtree(local_imgdir)
            shutil.copytree(image_dir, str(local_imgdir))

            # 2) 필요한 stage2 variants → 로컬 SSD (미완료 seed만)
            local_stage2.mkdir(parents=True, exist_ok=True)
            for seed_id, _ in seed_pm_pairs:
                src = stage2_dir / seed_id
                dst = local_stage2 / seed_id
                if src.exists() and not dst.exists():
                    shutil.copytree(str(src), str(dst))

            # 3) placement_map.json → 로컬 경로로 defect_path 치환
            local_pm_pairs = []
            for seed_id, pm_path in seed_pm_pairs:
                dst_dir = local_stage3 / seed_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                pm_data = json.loads(Path(pm_path).read_text())
                for item in pm_data:
                    for p in item.get("placements", []):
                        if "defect_path" in p:
                            p["defect_path"] = p["defect_path"].replace(
                                str(stage2_dir), str(local_stage2))
                local_pm = dst_dir / "placement_map.json"
                local_pm.write_text(json.dumps(pm_data))
                local_pm_pairs.append((seed_id, str(local_pm)))

            copy_sec = time.time() - t0

            # 4) 로컬 SSD에서 합성
            t1 = time.time()
            run_synthesis_batch(
                image_dir            = str(local_imgdir),
                seed_placement_maps  = local_pm_pairs,
                output_root          = str(local_stage4),
                format               = "cls",
                use_fast_blend       = USE_FAST_BLEND,
                workers              = IMG_THREADS,
                png_compression      = PNG_COMPRESSION,
                max_background_dim   = MAX_BACKGROUND_DIM,
                max_images_per_seed  = MAX_IMAGES_PER_SEED,
            )
            synth_sec = time.time() - t1

            # 5) 결과 → Drive 업로드
            t2 = time.time()
            drive_stage4 = cat_dir / "stage4_output"
            for seed_id, _ in local_pm_pairs:
                src = local_stage4 / seed_id
                dst = drive_stage4 / seed_id
                if src.exists():
                    if dst.exists():
                        shutil.rmtree(str(dst))
                    shutil.copytree(str(src), str(dst))
            upload_sec = time.time() - t2

            total_sec = time.time() - t0
            print(f"  ✓ {cat_dir.name}: 캐시 {copy_sec:.0f}s + 합성 {synth_sec:.0f}s + 업로드 {upload_sec:.0f}s = {total_sec:.0f}s")

        except Exception as e:
            failed.append({"category": cat_dir.name, "error": str(e), "type": type(e).__name__})
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 5: 합성 품질 점수

**Sentinel:** `{cat_dir}/stage4_output/{seed_id}/quality_scores.json`
**병렬:** `run_quality_scoring_batch()` — seed 간 `parallel_seeds` + 이미지 간 `workers` 2중 병렬

```python
import json, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
from stage5_quality_scoring import run_quality_scoring_batch

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
PARALLEL_SEEDS = 4   # seed 간 병렬 수 (CPU 코어 고려)
IMG_WORKERS    = -1  # seed 내 이미지 병렬 수 (auto)

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seed_dirs, skip = [], 0
seen_cats = set()
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir    = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen_cats:
        continue
    seen_cats.add(str(cat_dir))
    stage4_dir = cat_dir / "stage4_output"
    if not stage4_dir.exists():
        continue
    for d in sorted(stage4_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "quality_scores.json").exists():
            skip += 1
        else:
            all_seed_dirs.append(str(d))

if not all_seed_dirs:
    print(f"✓ {LABEL} 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(all_seed_dirs)} seeds 처리 예정 (skip {skip}건)")
    results = run_quality_scoring_batch(
        stage4_seed_dirs = all_seed_dirs,
        workers          = IMG_WORKERS,
        parallel_seeds   = PARALLEL_SEEDS,
    )
    completed = len([r for r in results if r])
    print(f"✓ {LABEL} 완료 — scored={completed}")
    if results:
        avg_q = sum(r["mean"] for r in results if r) / max(1, completed)
        print(f"  평균 quality_score: {avg_q:.3f}")
```

---

## Stage 6: 증강 데이터셋 구성 (로컬 SSD 캐시)

**Sentinel:** `{cat_dir}/augmented_dataset/build_report.json` (pruning_ratio 일치 + defect 존재)
**병렬:** 카테고리 단위 — 각 카테고리를 로컬 SSD에서 구성 후 Drive 업로드

> Drive FUSE I/O 병목 해소: stage4_output·image_dir를 로컬 SSD에 `parallel_copytree`로 복사 후
> 데이터셋을 구성하고 augmented_dataset 결과만 Drive에 업로드한다.
> 도메인 추출 로직이 경로 구조에 의존하므로 `LOCAL_TMP/{domain_path}/{cat_name}` 구조 유지.

```python
import json, sys, shutil, time, yaml
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage6_dataset_builder import run_dataset_builder

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "isp"
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
CAT_ONLY = set()     # 특정 카테고리만 처리 — 예: {"candle"}. 빈 set() 이면 전체

# 병렬 설정
CAT_THREADS    = 2   # 카테고리 동시 처리 (Drive 동시 읽기/쓰기 안정성 고려)
NUM_IO_THREADS = 8   # 카테고리 내부 파일 복사 스레드
PRUNING_RATIO        = 0.5
SPLIT_RATIO          = 0.8
SPLIT_SEED           = 42
BALANCE_DEFECT_TYPES = True
LOCAL_TMP = Path("/content/tmp_stage6")

# 도메인별 증강 비율 (benchmark_experiment.yaml 로드)
_bench_cfg = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
AUGMENTATION_RATIO_BY_DOMAIN = _bench_cfg.get("dataset", {}).get("augmentation_ratio_by_domain")

# 로컬 경로에서도 도메인 추출이 동작하도록 경로 구조를 보존
# ISP: LOCAL_TMP/isp/unsupervised/{cat}  MVTec: LOCAL_TMP/mvtec/{cat}  VisA: LOCAL_TMP/visa/{cat}
_LOCAL_DOMAIN_PATH = {
    "isp":   LOCAL_TMP / "isp" / "unsupervised",
    "mvtec": LOCAL_TMP / "mvtec",
    "visa":  LOCAL_TMP / "visa",
}[DOMAIN_FILTER]

def parallel_copytree(src_dst_pairs, max_workers=4):
    """여러 (src, dst) 디렉터리 쌍을 ThreadPoolExecutor로 병렬 복사."""
    def _copy_one(pair):
        src, dst = Path(pair[0]), Path(pair[1])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            return str(dst)
        return None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_copy_one, src_dst_pairs))
    return [r for r in results if r is not None]

# cat_dir 단위로 묶기
cat_map: dict[str, str] = {}
cat_seed_dirs: dict[str, list] = defaultdict(list)
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir_path = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir_path.name not in CAT_ONLY:
        continue
    image_dir = entry["image_dir"]
    cat_map.setdefault(str(cat_dir_path), image_dir)
    cat_seed_dirs[str(cat_dir_path)].extend(seed_dirs_list)

all_cats, skip = [], 0
for cat_dir, image_dir in cat_map.items():
    if (Path(cat_dir) / "augmented_dataset" / "build_report.json").exists():
        skip += 1
    else:
        all_cats.append((cat_dir, image_dir, cat_seed_dirs[cat_dir]))

if not all_cats:
    print(f"✓ {LABEL} 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage6(args):
        cat_dir_str, image_dir, seed_dirs = args
        cat_dir = Path(cat_dir_str)
        t0 = time.time()

        local_cat     = _LOCAL_DOMAIN_PATH / cat_dir.name
        local_imgdir  = local_cat / "train" / "good"
        local_stage4  = local_cat / "stage4_output"

        try:
            # 병렬 복사: stage4_output + image_dir + test 디렉터리
            copy_pairs = []
            if (cat_dir / "stage4_output").exists():
                copy_pairs.append((cat_dir / "stage4_output", local_stage4))
            copy_pairs.append((Path(image_dir), local_imgdir))
            test_good = cat_dir / "test" / "good"
            if test_good.exists():
                copy_pairs.append((test_good, local_cat / "test" / "good"))
            # test/{defect_type} 디렉터리
            cat_test = Path(image_dir).parents[1] / "test"
            if cat_test.exists():
                for d in cat_test.iterdir():
                    if d.is_dir() and d.name != "good":
                        copy_pairs.append((d, local_cat / "test" / d.name))

            parallel_copytree(copy_pairs, max_workers=4)
            copy_sec = time.time() - t0

            # 로컬 seed_dirs (test defect용)
            local_seed_dirs = []
            for sd in seed_dirs:
                local_sd = local_cat / "test" / Path(sd).name
                if local_sd.exists():
                    local_seed_dirs.append(str(local_sd))

            # 로컬에서 데이터셋 구성
            t1 = time.time()
            result = run_dataset_builder(
                cat_dir                     = str(local_cat),
                image_dir                   = str(local_imgdir),
                seed_dirs                   = local_seed_dirs or seed_dirs,
                pruning_ratio               = PRUNING_RATIO,
                augmentation_ratio_by_domain= AUGMENTATION_RATIO_BY_DOMAIN,
                split_ratio                 = SPLIT_RATIO,
                split_seed                  = SPLIT_SEED,
                workers                     = NUM_IO_THREADS,
                balance_defect_types        = BALANCE_DEFECT_TYPES,
            )
            build_sec = time.time() - t1

            # Drive 업로드
            t2 = time.time()
            drive_aug = cat_dir / "augmented_dataset"
            if drive_aug.exists():
                shutil.rmtree(str(drive_aug))
            shutil.copytree(str(local_cat / "augmented_dataset"), str(drive_aug))
            upload_sec = time.time() - t2

            total_sec = time.time() - t0
            print(f"  ✓ {cat_dir.name}: 캐시 {copy_sec:.0f}s + 구성 {build_sec:.0f}s + 업로드 {upload_sec:.0f}s = {total_sec:.0f}s"
                  f"  [full={result['aroma_full']['defect_count']} pruned={result['aroma_pruned']['defect_count']}]")
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

        return cat_dir.name

    with tqdm(total=len(all_cats), desc=f"Stage6 {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=CAT_THREADS) as ex:
            futs = {ex.submit(_run_stage6, t): t for t in all_cats}
            for fut in as_completed(futs):
                cat_dir_str, *_ = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": Path(cat_dir_str).name,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
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

## Stage 7: 데이터 구조 확인

```python
import json, yaml, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")

REPO          = Path("/content/aroma")
CONFIG        = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG     = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
DOMAIN_FILTER = "isp"
EXCLUDE       = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
GROUPS        = list(BENCH_CFG["dataset_groups"].keys())

seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))

    tag = "[EXCL]" if cat_dir.name in EXCLUDE else "     "
    aug = cat_dir / "augmented_dataset"
    if not aug.exists():
        print(f"{tag} {cat_dir.name}: augmented_dataset 없음")
        continue

    group_status = []
    for g in GROUPS:
        gdir = aug / g
        n_good   = len(list((gdir / "train" / "good").glob("*")))   if (gdir / "train" / "good").exists()   else 0
        n_defect = len(list((gdir / "train" / "defect").glob("*"))) if (gdir / "train" / "defect").exists() else 0
        has_test = (gdir / "test").exists()
        n_test   = sum(len(list(d.glob("*"))) for d in (gdir / "test").iterdir()
                       if d.is_dir()) if has_test else 0
        group_status.append(
            f"{g}: train(good={n_good}, defect={n_defect}) test={n_test if has_test else '(auto-symlink)'}"
        )
    print(f"{tag} {cat_dir.name}")
    for s in group_status:
        print(f"       {s}")
```

---

## Stage 7: test set 준비

```python
import json, sys, yaml
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import _ensure_test_dir

REPO          = Path("/content/aroma")
CONFIG        = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG     = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
DOMAIN_FILTER = "isp"
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
EXCLUDE       = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
NON_BASELINE  = [g for g in BENCH_CFG["dataset_groups"] if g != "baseline"]

seen, tasks, failed_prep = set(), [], []
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen or cat_dir.name in EXCLUDE:
        continue
    seen.add(str(cat_dir))
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        continue
    for g in NON_BASELINE:
        if not (cat_dir / "augmented_dataset" / g / "test").exists():
            tasks.append((cat_dir, g))

if not tasks:
    print(f"✓ {LABEL} test set 준비 완료")
else:
    print(f"{LABEL}: {len(tasks)}개 그룹 test set 준비")
    for cat_dir, group in tqdm(tasks, desc=f"test set {LABEL}"):
        try:
            result = _ensure_test_dir(str(cat_dir), group)
            method = "symlink" if result.is_symlink() else "copy"
            print(f"  ✓ {cat_dir.name}/{group}/test [{method}]")
        except Exception as e:
            failed_prep.append(f"{cat_dir.name}/{group}: {e}")
            print(f"  ✗ {cat_dir.name}/{group}: {e}")

    print(f"\n{'✓' if not failed_prep else '✗'} {LABEL} test set 준비 {'완료' if not failed_prep else f'{len(failed_prep)}건 실패'}")
```

---

## Stage 7: 벤치마크 (로컬 SSD 캐시)

**Sentinel:** `{output_dir}/{cat_name}/{model}/{group}/experiment_meta.json`
**병렬:** GPU 점유 → 카테고리 순차 처리 — augmented_dataset을 로컬 SSD로 복사 후 DataLoader 가속

> **모델:** `yolo11` · `efficientdet_d0` **그룹:** `baseline` · `aroma_full` · `aroma_pruned`

```python
import json, sys, yaml, shutil, time
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import run_benchmark, _ensure_test_dir

REPO        = Path("/content/aroma")
CONFIG_PATH = str(REPO / "configs" / "benchmark_experiment.yaml")
CONFIG      = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG   = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())

DOMAIN_FILTER = "isp"
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
OUTPUT_DIR    = str(REPO / "outputs" / "benchmark_results")
LOCAL_TMP     = Path("/content/tmp_stage7")

EXCLUDE = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
MODELS  = list(BENCH_CFG["models"].keys())
GROUPS  = list(BENCH_CFG["dataset_groups"].keys())
NON_BASELINE = [g for g in GROUPS if g != "baseline"]

seen = set()
all_cats, skip = [], 0
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    if cat_dir.name in EXCLUDE:
        continue
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
    print(f"✓ {LABEL} 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    print(f"  모델: {MODELS}  그룹: {GROUPS}")
    failed = []

    for cat_dir in tqdm(all_cats, desc=f"Stage7 {LABEL}"):
        t0 = time.time()
        local_cat = LOCAL_TMP / cat_dir.name
        try:
            # augmented_dataset → 로컬 SSD (DataLoader I/O 가속)
            local_aug = local_cat / "augmented_dataset"
            if local_cat.exists():
                shutil.rmtree(str(local_cat))
            shutil.copytree(str(cat_dir / "augmented_dataset"), str(local_aug))
            copy_sec = time.time() - t0

            # test set 준비 (로컬에서 symlink/copy)
            for g in NON_BASELINE:
                if not (local_aug / g / "test").exists():
                    _ensure_test_dir(str(local_cat), g)

            # 벤치마크 실행 (데이터=로컬 SSD, 결과=Drive)
            t1 = time.time()
            results = run_benchmark(
                config_path = CONFIG_PATH,
                cat_dir     = str(local_cat),
                resume      = True,
                output_dir  = OUTPUT_DIR,
            )
            bench_sec = time.time() - t1
            total_sec = time.time() - t0

            print(f"  ✓ {cat_dir.name}: 캐시 {copy_sec:.0f}s + 벤치마크 {bench_sec:.0f}s = {total_sec:.0f}s")
            for model_name, group_results in results.items():
                for group, val in group_results.items():
                    if isinstance(val, dict) and "error" in val:
                        failed.append({
                            "category": cat_dir.name,
                            "error": f"{model_name}/{group}: {val.get('detail', '')[:80]}",
                            "type": val["error"],
                        })
                    elif isinstance(val, dict):
                        auroc = val.get("image_auroc")
                        if auroc is not None:
                            print(f"    {model_name}/{group}: AUROC={auroc:.4f}")
        except Exception as e:
            failed.append({"category": cat_dir.name, "error": str(e), "type": type(e).__name__})
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

    shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 7: 결과 요약

```python
import json, yaml
from pathlib import Path

REPO          = Path("/content/aroma")
OUTPUT_DIR    = REPO / "outputs" / "benchmark_results"
DOMAIN_FILTER = "isp"

CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
MODELS    = list(BENCH_CFG["models"].keys())
GROUPS    = list(BENCH_CFG["dataset_groups"].keys())

if DOMAIN_FILTER:
    valid_cats = {
        Path((v.get("seed_dirs") or [v["seed_dir"]])[0]).parents[1].name
        for k, v in CONFIG.items()
        if not k.startswith("_") and v["domain"] == DOMAIN_FILTER
    }
else:
    valid_cats = None

cols = [f"{m}/{g}" for m in MODELS for g in GROUPS]
header = f"{'category':<20}" + "".join(f"{c:<30}" for c in cols)
print(header)
print("-" * len(header))

for cat_dir in sorted(OUTPUT_DIR.iterdir()):
    if not cat_dir.is_dir():
        continue
    if valid_cats is not None and cat_dir.name not in valid_cats:
        continue
    row = f"{cat_dir.name:<20}"
    for m in MODELS:
        for g in GROUPS:
            meta = cat_dir / m / g / "experiment_meta.json"
            if meta.exists():
                data = json.loads(meta.read_text())
                if "error" in data:
                    val = f"ERR:{data['error']}"
                else:
                    auroc = data.get("image_auroc", 0.0)
                    f1    = data.get("image_f1", 0.0)
                    val   = f"AUROC={auroc:.3f} F1={f1:.3f}"
            else:
                val = "—"
            row += f"{val:<30}"
    print(row)
```

---

## 병렬 설정 가이드

| 스테이지 | 외부 병렬 | 내부 병렬 | SSD 캐시 | 근거 |
|---------|---------|---------|---------|------|
| Stage 1  | `CAT_THREADS=2` | `IMG_WORKERS=-1` | — | CPU-bound 이미지 분석 |
| Stage 1b | `run_seed_characterization_batch` | `workers=-1` | — | 배치 함수로 단순화 |
| Stage 2  | `SEED_THREADS=2` | `IMG_WORKERS=-1` | — | 변형 생성 CPU 집약 |
| Stage 3  | 순차 | GPU 배치 | — | GPU 인스턴스 공유 불가 |
| Stage 4  | 카테고리 순차 | `IMG_THREADS=4` | **O** | 배경·variants Drive FUSE 병목 해소 |
| Stage 5  | `run_quality_scoring_batch` | `IMG_WORKERS=-1` | — | 배치 함수, `parallel_seeds=4` |
| Stage 6  | `CAT_THREADS=2` | `NUM_IO_THREADS=8` | **O** | `parallel_copytree` + 로컬 구성 후 업로드 |
| Stage 7  | 순차 (카테고리) | — | **O** | DataLoader I/O 가속, GPU 단일 점유 |
