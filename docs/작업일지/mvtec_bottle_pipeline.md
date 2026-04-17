# mvtec_bottle 단독 파이프라인 실행 셀

> **목적:** mvtec_bottle 카테고리만 Stage 0 → 7 까지 재실행한다.
> seed_id 체계가 `000` → `broken_large_000` 으로 변경되었으므로
> Stage 1b 이후 출력물을 먼저 삭제한 뒤 실행해야 한다.
>
> **대상 경로:** `/content/drive/MyDrive/data/Aroma/mvtec/bottle`

---

## 공통 설정

```python
import json, sys
from pathlib import Path

REPO       = Path("/content/aroma")
DATA_ROOT  = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR    = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"

sys.path.insert(0, str(REPO))
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY  = CONFIG[CATEGORY_KEY]

seed_dirs_list = ENTRY.get("seed_dirs") or [ENTRY["seed_dir"]]
image_dir      = ENTRY["image_dir"]

print(f"cat_dir   : {CAT_DIR}")
print(f"image_dir : {image_dir}")
print(f"seed_dirs : {seed_dirs_list}")
```

---

## 기존 출력 정리

> **주의:** seed_id 체계 변경(`000` → `broken_large_000`)으로 인해
> Stage 1b 이후 출력물을 삭제 후 재생성해야 한다.
> Stage 0, Stage 1 출력은 영향 없으므로 유지.

```python
import shutil

CLEAN_TARGETS = [
    CAT_DIR / "stage1b_output",
    CAT_DIR / "stage2_output",
    CAT_DIR / "stage3_output",
    CAT_DIR / "stage4_output",
    CAT_DIR / "augmented_dataset",
]

for target in CLEAN_TARGETS:
    if target.exists():
        shutil.rmtree(target)
        print(f"  삭제: {target}")
    else:
        print(f"  없음: {target}")
print("✓ 정리 완료")
```

---

## Stage 0: 이미지 리사이즈 (512×512)

**Sentinel:** `{cat_dir}/.stage0_resize_512_done`

```python
import json, sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage0_resize import resize_category, clean_category

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"
CONFIG       = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY        = CONFIG[CATEGORY_KEY]

TARGET_SIZE  = 512
CLEAN_FIRST  = False   # True: stage1~6 출력물 먼저 삭제

sentinel = CAT_DIR / f".stage0_resize_{TARGET_SIZE}_done"
if sentinel.exists():
    print("✓ Stage 0 완료 (skip)")
else:
    if CLEAN_FIRST:
        deleted = clean_category(ENTRY)
        if deleted:
            print(f"  🗑 {len(deleted)} items deleted")
    result = resize_category(ENTRY, target_size=TARGET_SIZE, workers=-1)
    print(f"✓ Stage 0 완료: resized={result['resized']} skip={result['skipped']}")
```

---

## Stage 1: ROI 추출

**Sentinel:** `{cat_dir}/stage1_output/roi_metadata.json`

```python
import json, sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage1_roi_extraction import run_extraction

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"
CONFIG       = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY        = CONFIG[CATEGORY_KEY]

IMG_WORKERS = -1

sentinel = CAT_DIR / "stage1_output" / "roi_metadata.json"
if sentinel.exists():
    print("✓ Stage 1 완료 (skip)")
else:
    run_extraction(
        image_dir  = ENTRY["image_dir"],
        output_dir = str(CAT_DIR / "stage1_output"),
        domain     = ENTRY["domain"],
        roi_levels = "both",
        workers    = IMG_WORKERS,
    )
    print("✓ Stage 1 완료")
```

---

## Stage 1b: Seed 특성 분석

**Sentinel:** `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json`
> broken_large / broken_small / contamination 각 20개 = 60 seeds
>
> `run_seed_characterization_batch(workers=-1)` 사용 — ProcessPool을 내부에서 단일 계층으로
> 관리하므로 Thread+ProcessPool 중첩 없이 안전하게 병렬 처리.

```python
import json, sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage1b_seed_characterization import run_seed_characterization_batch

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"
CONFIG       = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY        = CONFIG[CATEGORY_KEY]

seed_dirs_list = ENTRY.get("seed_dirs") or [ENTRY["seed_dir"]]
use_prefix     = len(seed_dirs_list) > 1   # True — 3개 유형

tasks, skip = [], 0
for seed_dir_str in seed_dirs_list:
    seed_dir = Path(seed_dir_str)
    seeds    = sorted(seed_dir.glob("*.png")) if seed_dir.exists() else []
    for seed in seeds:
        seed_id = f"{seed_dir.name}_{seed.stem}" if use_prefix else seed.stem
        out = CAT_DIR / "stage1b_output" / seed_id
        if (out / "seed_profile.json").exists():
            skip += 1
        else:
            tasks.append((str(seed), str(out)))

if not tasks:
    print(f"✓ Stage 1b 완료 (skip {skip}건)")
else:
    print(f"Stage 1b: {len(tasks)} seeds 처리 예정 (skip {skip}건)")
    results = run_seed_characterization_batch(tasks, workers=-1)
    done    = len([r for r in results if r])
    failed  = len(tasks) - done
    print(f"✓ Stage 1b 완료: processed={done}" + (f" / 실패={failed}" if failed else ""))
```

---

## Stage 2: Defect Seed 변형 생성

**Sentinel:** `{cat_dir}/stage2_output/{seed_id}/` 에 PNG가 `NUM_VARIANTS`개 이상
**순차 처리** — `run_seed_generation` 내부에서 `workers=-1` 로 모든 코어 활용.
외부 ThreadPoolExecutor + 내부 ProcessPool 중첩은 Colab에서 fork 데드락을 유발하므로 제거.

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage2_defect_seed_generation import run_seed_generation

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"

IMG_WORKERS  = -1   # 내부 변형 이미지 병렬 (cpu_count-1 자동)
NUM_VARIANTS = 50

stage1b_dir = CAT_DIR / "stage1b_output"
all_seeds, skip = [], 0
for profile_path in sorted(stage1b_dir.glob("*/seed_profile.json")):
    seed_id = profile_path.parent.name
    out_dir = CAT_DIR / "stage2_output" / seed_id
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= NUM_VARIANTS:
        skip += 1
    else:
        all_seeds.append(profile_path)

if not all_seeds:
    print(f"✓ Stage 2 완료 (skip {skip}건)")
else:
    print(f"Stage 2: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    for profile_path in tqdm(all_seeds, desc="Stage2 bottle"):
        seed_id = profile_path.parent.name
        try:
            info = json.loads(profile_path.read_text())
            run_seed_generation(
                seed_defect  = info["seed_path"],
                num_variants = NUM_VARIANTS,
                output_dir   = str(CAT_DIR / "stage2_output" / seed_id),
                seed_profile = str(profile_path),
                workers      = IMG_WORKERS,
            )
        except Exception as e:
            failed.append({"seed": seed_id, "error": str(e), "type": type(e).__name__})

    print("\n" + (f"✓ Stage 2 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 3: 레이아웃 로직 (배치 평가)

**Sentinel:** `{cat_dir}/stage3_output/{seed_id}/placement_map.json`
**순차 처리** — GPU 공유 불가

```python
import json, sys
from pathlib import Path
from tqdm.auto import tqdm

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage3_layout_logic import run_layout_logic

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"
CONFIG       = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY        = CONFIG[CATEGORY_KEY]

stage2_dir = CAT_DIR / "stage2_output"
all_seeds, skip = [], 0
for d in sorted(stage2_dir.iterdir()):
    if not d.is_dir():
        continue
    if (CAT_DIR / "stage3_output" / d.name / "placement_map.json").exists():
        skip += 1
    else:
        all_seeds.append(d.name)

if not all_seeds:
    print(f"✓ Stage 3 완료 (skip {skip}건)")
else:
    print(f"Stage 3: {len(all_seeds)} seeds 처리 예정 (skip {skip}건)")
    failed = []

    for seed_id in tqdm(all_seeds, desc="Stage3 bottle"):
        try:
            profile_path = CAT_DIR / "stage1b_output" / seed_id / "seed_profile.json"
            run_layout_logic(
                roi_metadata     = str(CAT_DIR / "stage1_output" / "roi_metadata.json"),
                defect_seeds_dir = str(CAT_DIR / "stage2_output" / seed_id),
                output_dir       = str(CAT_DIR / "stage3_output" / seed_id),
                seed_profile     = str(profile_path) if profile_path.exists() else None,
                domain           = ENTRY.get("domain", "mvtec"),
                use_gpu          = True,
            )
        except Exception as e:
            failed.append({"seed": seed_id, "error": str(e), "type": type(e).__name__})

    print("\n" + (f"✓ Stage 3 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['seed']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 4: MPB 합성 (로컬 SSD 캐시)

**Sentinel:** `{cat_dir}/stage4_output/{seed_id}/defect/*.png` 존재

> Drive FUSE I/O 병목 해소: 입력 데이터를 Colab 로컬 SSD(`/content/tmp_stage4/`)에
> 복사한 뒤 합성을 수행하고, 결과만 Drive로 업로드합니다.
>
> `workers=4` ProcessPool이 Drive 직접 접근 시 FUSE 레이턴시 × 동시 워커 수 →
> RAM 스파이크 → OOM → `BrokenProcessPool` 발생.
> 로컬 SSD 캐시 후 실행하면 워커가 빠르게 완료되어 OOM 없이 병렬 처리 가능.

```python
import json, sys, shutil, time
from pathlib import Path
from tqdm.auto import tqdm

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage4_mpb_synthesis import run_synthesis_batch

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"
CONFIG       = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY        = CONFIG[CATEGORY_KEY]

USE_FAST_BLEND     = True
WORKERS            = 4      # 로컬 SSD 캐시 후 ProcessPool 병렬 처리
PNG_COMPRESSION    = 3
MAX_BACKGROUND_DIM = None   # 변경 금지
LOCAL_TMP          = Path("/content/tmp_stage4")

# ── 미완료 seed 수집 ───────────────────────────────────────────
stage3_dir = CAT_DIR / "stage3_output"
stage2_dir = CAT_DIR / "stage2_output"
image_dir  = Path(ENTRY["image_dir"])

seed_pm_pairs, skip = [], 0
for d in sorted(stage3_dir.iterdir()):
    if not d.is_dir():
        continue
    pm_path    = d / "placement_map.json"
    if not pm_path.exists():
        continue
    out_defect = CAT_DIR / "stage4_output" / d.name / "defect"
    if out_defect.exists() and any(out_defect.glob("*.png")):
        skip += 1
    else:
        seed_pm_pairs.append((d.name, str(pm_path)))

if not seed_pm_pairs:
    print(f"✓ Stage 4 완료 (skip {skip}건)")
else:
    print(f"Stage 4: {len(seed_pm_pairs)} seeds 처리 예정 (skip {skip}건)")

    local_cat       = LOCAL_TMP / CAT_DIR.name
    local_image_dir = local_cat / "images"
    local_stage2    = local_cat / "stage2_output"
    local_stage3    = local_cat / "stage3_output"
    local_stage4    = local_cat / "stage4_output"

    try:
        # ── 1) 배경 이미지 복사 ────────────────────────────────
        t0 = time.time()
        if local_image_dir.exists():
            shutil.rmtree(local_image_dir)
        print("  배경 이미지 복사 중...", end=" ", flush=True)
        shutil.copytree(str(image_dir), str(local_image_dir))
        print(f"완료 ({time.time()-t0:.1f}s)")

        # ── 2) Stage 2 variants 복사 (미완료 seed만) ──────────
        local_stage2.mkdir(parents=True, exist_ok=True)
        print(f"  Stage 2 variants 복사 중... ({len(seed_pm_pairs)} seeds)")
        t1 = time.time()
        for seed_id, _ in tqdm(seed_pm_pairs, desc="    복사", leave=False):
            src = stage2_dir / seed_id
            dst = local_stage2 / seed_id
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))
        print(f"  완료 ({time.time()-t1:.1f}s)")

        # ── 3) placement_map 복사 + defect_path 로컬 경로 치환 ─
        local_pm_pairs = []
        drive_stage2_prefix = str(stage2_dir)
        local_stage2_prefix = str(local_stage2)
        for seed_id, pm_path in seed_pm_pairs:
            dst_dir = local_stage3 / seed_id
            dst_dir.mkdir(parents=True, exist_ok=True)

            pm_data = json.loads(Path(pm_path).read_text())
            for entry in pm_data:
                for p in entry.get("placements", []):
                    if "defect_path" in p:
                        p["defect_path"] = p["defect_path"].replace(
                            drive_stage2_prefix, local_stage2_prefix)

            local_pm = dst_dir / "placement_map.json"
            local_pm.write_text(json.dumps(pm_data))
            local_pm_pairs.append((seed_id, str(local_pm)))

        copy_sec = time.time() - t0
        print(f"  로컬 캐시 완료 ({copy_sec:.1f}s)")

        # ── 4) 합성 (로컬 SSD) ────────────────────────────────
        t2 = time.time()
        run_synthesis_batch(
            image_dir           = str(local_image_dir),
            seed_placement_maps = local_pm_pairs,
            output_root         = str(local_stage4),
            format              = "cls",
            use_fast_blend      = USE_FAST_BLEND,
            workers             = WORKERS,
            png_compression     = PNG_COMPRESSION,
            max_background_dim  = MAX_BACKGROUND_DIM,
        )
        synth_sec = time.time() - t2
        print(f"  합성 완료 ({synth_sec:.1f}s)")

        # ── 5) 결과를 Drive로 업로드 ──────────────────────────
        t3 = time.time()
        drive_stage4 = CAT_DIR / "stage4_output"
        print(f"  Drive 업로드 중... ({len(local_pm_pairs)} seeds)")
        for seed_id, _ in tqdm(local_pm_pairs, desc="    업로드", leave=False):
            src = local_stage4 / seed_id
            dst = drive_stage4 / seed_id
            if src.exists():
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.copytree(str(src), str(dst))
        upload_sec = time.time() - t3
        print(f"  Drive 업로드 완료 ({upload_sec:.1f}s)")

        total = time.time() - t0
        print(f"\n✓ Stage 4 완료 (총 {total:.1f}s = 캐시 {copy_sec:.1f}s + 합성 {synth_sec:.1f}s + 업로드 {upload_sec:.1f}s)")

    except Exception as e:
        print(f"✗ Stage 4 실패: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(str(local_cat), ignore_errors=True)
```

---

## Stage 5: 합성 품질 점수

**Sentinel:** `{cat_dir}/stage4_output/{seed_id}/quality_scores.json`
>
> `run_quality_scoring_batch(workers=0, parallel_seeds=-1)` 사용.
> `workers=0`: seed 내부 이미지 단위 ProcessPool 비활성화 (중첩 방지).
> `parallel_seeds=-1`: seed 단위 자동 병렬 (단일 계층 ProcessPool).

```python
import sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage5_quality_scoring import run_quality_scoring_batch

DATA_ROOT = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR   = DATA_ROOT / "mvtec" / "bottle"

stage4_dir = CAT_DIR / "stage4_output"
all_seed_dirs, skip = [], 0
for d in sorted(stage4_dir.iterdir()):
    if not d.is_dir():
        continue
    if (d / "quality_scores.json").exists():
        skip += 1
    else:
        all_seed_dirs.append(str(d))

if not all_seed_dirs:
    print(f"✓ Stage 5 완료 (skip {skip}건)")
else:
    print(f"Stage 5: {len(all_seed_dirs)} seeds 처리 예정 (skip {skip}건)")
    results = run_quality_scoring_batch(
        stage4_seed_dirs = all_seed_dirs,
        workers          = 0,    # 이미지 단위 순차 (ProcessPool 중첩 방지)
        parallel_seeds   = -1,   # seed 단위 자동 병렬
    )
    done = len([r for r in results if r])
    if done:
        avg_count   = sum(r["count"] for r in results if r) / done
        avg_quality = sum(r["mean"]  for r in results if r) / done
        print(f"✓ Stage 5 완료: {done} seeds (평균 {avg_count:.0f}장/seed, quality={avg_quality:.3f})")
    else:
        print("✗ Stage 5: 처리 결과 없음")
```

---

## Stage 6: 증강 데이터셋 구성 (로컬 SSD 캐시)

**Sentinel:** `{cat_dir}/augmented_dataset/build_report.json`

> 순수 파일 복사 Stage — Drive FUSE I/O가 극심한 병목.
> stage4_output + image_dir + seed_dirs를 로컬 SSD에 복사 후 구성,
> 완성된 augmented_dataset만 Drive로 업로드.

```python
import json, sys, shutil, time
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage6_dataset_builder import run_dataset_builder

DATA_ROOT    = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR      = DATA_ROOT / "mvtec" / "bottle"
CATEGORY_KEY = "mvtec_bottle"
CONFIG       = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
ENTRY        = CONFIG[CATEGORY_KEY]

PRUNING_THRESHOLD            = 0.6
PRUNING_THRESHOLD_BY_DOMAIN  = {"isp": 0.6, "mvtec": 0.6, "visa": 0.4}
SPLIT_RATIO                  = 0.8
SPLIT_SEED                   = 42
NUM_IO_THREADS               = 8
BALANCE_DEFECT_TYPES         = True   # 3 유형 균등 샘플링
LOCAL_TMP                    = Path("/content/tmp_stage6")

seed_dirs_list = ENTRY.get("seed_dirs") or [ENTRY["seed_dir"]]
image_dir      = Path(ENTRY["image_dir"])

sentinel = CAT_DIR / "augmented_dataset" / "build_report.json"
if sentinel.exists():
    print("✓ Stage 6 완료 (skip)")
else:
    # 로컬 경로: Drive 경로 구조 보존 (Aroma/mvtec/bottle)
    # → dataset_builder 내부 도메인 추출 로직이 경로 구조에 의존
    relative_path = CAT_DIR.relative_to(DATA_ROOT.parent)  # Aroma/mvtec/bottle
    local_cat     = LOCAL_TMP / relative_path

    if local_cat.exists():
        shutil.rmtree(str(local_cat))

    try:
        t0 = time.time()

        # ── 1) stage4_output 복사 (defect 이미지 + quality_scores.json) ──
        print("  stage4_output 복사 중...", end=" ", flush=True)
        shutil.copytree(str(CAT_DIR / "stage4_output"), str(local_cat / "stage4_output"))
        print(f"완료 ({time.time()-t0:.1f}s)")

        # ── 2) image_dir (train/good) 복사 ─────────────────────────────
        local_image_dir = local_cat / "train" / "good"
        print("  image_dir 복사 중...", end=" ", flush=True)
        t1 = time.time()
        shutil.copytree(str(image_dir), str(local_image_dir))
        print(f"완료 ({time.time()-t1:.1f}s)")

        # ── 3) seed_dirs (test/broken_large 등) 복사 ────────────────────
        local_seed_dirs = []
        for sd_str in seed_dirs_list:
            sd_path  = Path(sd_str)
            local_sd = local_cat / "test" / sd_path.name
            if sd_path.exists():
                shutil.copytree(str(sd_path), str(local_sd))
            local_seed_dirs.append(str(local_sd))

        copy_sec = time.time() - t0
        print(f"  로컬 캐시 완료 ({copy_sec:.1f}s)")

        # ── 4) 로컬에서 데이터셋 구성 ───────────────────────────────────
        t2 = time.time()
        run_dataset_builder(
            cat_dir                     = str(local_cat),
            image_dir                   = str(local_image_dir),
            seed_dirs                   = local_seed_dirs,
            pruning_threshold           = PRUNING_THRESHOLD,
            pruning_threshold_by_domain = PRUNING_THRESHOLD_BY_DOMAIN,
            split_ratio                 = SPLIT_RATIO,
            split_seed                  = SPLIT_SEED,
            workers                     = NUM_IO_THREADS,
            balance_defect_types        = BALANCE_DEFECT_TYPES,
        )
        build_sec = time.time() - t2
        print(f"  데이터셋 구성 완료 ({build_sec:.1f}s)")

        # ── 5) augmented_dataset을 Drive로 업로드 ───────────────────────
        t3 = time.time()
        drive_aug = CAT_DIR / "augmented_dataset"
        if drive_aug.exists():
            shutil.rmtree(str(drive_aug))
        shutil.copytree(str(local_cat / "augmented_dataset"), str(drive_aug))
        upload_sec = time.time() - t3

        total = time.time() - t0
        print(f"  Drive 업로드 완료 ({upload_sec:.1f}s)")
        print(f"\n✓ Stage 6 완료 (총 {total:.1f}s = 캐시 {copy_sec:.1f}s + 구성 {build_sec:.1f}s + 업로드 {upload_sec:.1f}s)")

    except Exception as e:
        print(f"✗ Stage 6 실패: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(str(local_cat), ignore_errors=True)
```

---

## Stage 7: 데이터 구조 확인

```python
import json, yaml, sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))

DATA_ROOT = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR   = DATA_ROOT / "mvtec" / "bottle"

BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
GROUPS    = list(BENCH_CFG["dataset_groups"].keys())

aug = CAT_DIR / "augmented_dataset"
if not aug.exists():
    print("augmented_dataset 없음 — Stage 6 미완료")
else:
    for g in GROUPS:
        gdir     = aug / g
        n_good   = len(list((gdir / "train" / "good").glob("*")))   if (gdir / "train" / "good").exists()   else 0
        n_defect = len(list((gdir / "train" / "defect").glob("*"))) if (gdir / "train" / "defect").exists() else 0
        has_test = (gdir / "test").exists()
        n_test   = sum(len(list(d.glob("*"))) for d in (gdir / "test").iterdir() if d.is_dir()) if has_test else 0
        print(f"  {g}: train(good={n_good}, defect={n_defect}) test={n_test if has_test else '(없음)'}")
```

---

## Stage 7: test set 준비

```python
import json, yaml, sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage7_benchmark import _ensure_test_dir

DATA_ROOT = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR   = DATA_ROOT / "mvtec" / "bottle"

BENCH_CFG          = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
NON_BASELINE_GROUPS = [g for g in BENCH_CFG["dataset_groups"] if g != "baseline"]

if not (CAT_DIR / "augmented_dataset" / "baseline" / "test").exists():
    print("✗ baseline/test 없음 — Stage 6 미완료")
else:
    for g in NON_BASELINE_GROUPS:
        group_test = CAT_DIR / "augmented_dataset" / g / "test"
        if group_test.exists():
            print(f"  skip: {g}/test 이미 존재")
        else:
            try:
                result = _ensure_test_dir(str(CAT_DIR), g)
                method = "symlink" if result.is_symlink() else "copy"
                print(f"  ✓ {g}/test [{method}]")
            except Exception as e:
                print(f"  ✗ {g}/test: {e}")
```

---

## Stage 7: 벤치마크 (로컬 SSD 캐시)

> `augmented_dataset`을 로컬 SSD로 복사하여 DataLoader I/O 가속.
> 결과 JSON은 소량이므로 Drive에 직접 저장.

```python
import json, yaml, sys, shutil, time
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))
from stage7_benchmark import run_benchmark, _ensure_test_dir

DATA_ROOT  = Path("/content/drive/MyDrive/data/Aroma")
CAT_DIR    = DATA_ROOT / "mvtec" / "bottle"
OUTPUT_DIR = str(REPO / "outputs" / "benchmark_results")
LOCAL_TMP  = Path("/content/tmp_stage7")

CONFIG_PATH = str(REPO / "configs" / "benchmark_experiment.yaml")
BENCH_CFG   = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
MODELS      = list(BENCH_CFG["models"].keys())
GROUPS      = list(BENCH_CFG["dataset_groups"].keys())
NON_BASELINE = [g for g in GROUPS if g != "baseline"]

if not (CAT_DIR / "augmented_dataset" / "baseline" / "test").exists():
    print("✗ Stage 6 미완료 — 벤치마크 불가")
else:
    out_root     = Path(OUTPUT_DIR) / CAT_DIR.name
    already_done = all(
        (out_root / m / g / "experiment_meta.json").exists()
        for m in MODELS for g in GROUPS
    )
    if already_done:
        print("✓ 벤치마크 완료 (skip)")
    else:
        local_cat = LOCAL_TMP / CAT_DIR.name
        if local_cat.exists():
            shutil.rmtree(str(local_cat))

        try:
            # ── 1) augmented_dataset 로컬 복사 ──────────────────────
            t0 = time.time()
            print("  augmented_dataset 복사 중...", end=" ", flush=True)
            shutil.copytree(
                str(CAT_DIR / "augmented_dataset"),
                str(local_cat / "augmented_dataset"),
            )
            copy_sec = time.time() - t0
            print(f"완료 ({copy_sec:.1f}s)")

            # ── 2) test set 준비 (로컬에서) ─────────────────────────
            for g in NON_BASELINE:
                group_test = local_cat / "augmented_dataset" / g / "test"
                if not group_test.exists():
                    result = _ensure_test_dir(str(local_cat), g)
                    method = "symlink" if result.is_symlink() else "copy"
                    print(f"  test set: {g}/test [{method}]")

            # ── 3) 벤치마크 실행 (로컬 데이터, 결과는 Drive) ────────
            print(f"  벤치마크 실행: 모델={MODELS}, 그룹={GROUPS}")
            t1 = time.time()
            results = run_benchmark(
                config_path = CONFIG_PATH,
                cat_dir     = str(local_cat),
                resume      = True,
                output_dir  = OUTPUT_DIR,   # Drive에 직접 저장
            )
            bench_sec = time.time() - t1

            failed = []
            for model_name, group_results in results.items():
                for group, val in group_results.items():
                    if isinstance(val, dict) and "error" in val:
                        failed.append(f"{model_name}/{group}: {val.get('detail','')[:80]}")
                    elif isinstance(val, dict):
                        auroc = val.get("image_auroc", "?")
                        print(f"  ✓ {model_name}/{group}: AUROC={auroc:.4f}" if isinstance(auroc, float) else f"  ✓ {model_name}/{group}")

            total = time.time() - t0
            if failed:
                print(f"✗ {len(failed)}건 실패")
                for f in failed:
                    print(f"  {f}")
            else:
                print(f"\n✓ 벤치마크 완료 (총 {total:.1f}s = 캐시 {copy_sec:.1f}s + 벤치마크 {bench_sec:.1f}s)")

        except Exception as e:
            print(f"✗ {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)
```

---

## Stage 7: 결과 요약

```python
import json, yaml
from pathlib import Path

REPO       = Path("/content/aroma")
CAT_DIR    = Path("/content/drive/MyDrive/data/Aroma") / "mvtec" / "bottle"
OUTPUT_DIR = REPO / "outputs" / "benchmark_results"

BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment.yaml").read_text())
MODELS    = list(BENCH_CFG["models"].keys())
GROUPS    = list(BENCH_CFG["dataset_groups"].keys())

out_root = OUTPUT_DIR / CAT_DIR.name
cols     = ["category"] + [f"{m}/{g}" for m in MODELS for g in GROUPS]
row      = {"category": CAT_DIR.name}

for model in MODELS:
    for group in GROUPS:
        meta_path = out_root / model / group / "experiment_meta.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            if "error" in m:
                row[f"{model}/{group}"] = f"ERR:{m['error']}"
            else:
                row[f"{model}/{group}"] = round(m.get("image_auroc", 0.0), 4)
        else:
            row[f"{model}/{group}"] = "-"

header = "\t".join(cols)
print(header)
print("-" * len(header))
print("\t".join(str(row.get(c, "-")) for c in cols))
```
