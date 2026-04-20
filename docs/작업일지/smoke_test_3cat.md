# AROMA Smoke Test — 2개 카테고리 전체 파이프라인 검증

> **목적:** 512 리사이즈 후 Stage 0~7 전체 파이프라인이 정상 동작하는지 빠르게 검증
> **대상:** ISP/ASM, MVTec/bottle (2개 카테고리)
> **예상 소요:** ~20-30분 (Colab T4 GPU 기준)
> 
> **제외:** VisA/candle - Stage 4 defect 생성 실패 (가용 defect 0개)

### 최적화 적용 현황

| Stage | workers | SSD 캐시 | 비고 |
|-------|---------|----------|------|
| 0 (resize) | 셀 내 직접 구현 | O | Colab Drive repo에 `workers` 파라미터 미지원 |
| 1 (ROI) | `-1` (auto) | — | CPU-bound, 캐시 불필요 |
| 1b (seed) | `-1` (auto, batch) | — | `run_seed_characterization_batch()` 사용, seed_dirs 배열 + prefix 명명 지원 |
| 2 (variants) | `-1` (auto) | — | CPU-bound |
| 3 (layout) | `-1` (auto) | — | GPU 모드(`use_gpu=True`) 우선 |
| 4 (synthesis) | `4` (fixed) | O | I/O-bound, ~5,260 ops |
| 5 (quality) | `-1` (auto) | — | CPU-bound |
| 6 (dataset) | `8` (fixed) | O | I/O-bound 극심, ~9,590 ops |
| 7 (benchmark) | — | O | GPU-bound, DataLoader 가속 |

---

## 셀 0: 환경 설정 (1회)

```python
# 0-1. Drive 마운트
from google.colab import drive
drive.mount("/content/drive")

# 0-2. Git repo 클론/업데이트 (선택사항 — Drive repo에서 직접 실행도 가능)
from pathlib import Path

REPO_LOCAL = Path("/content/aroma")
REPO_DRIVE = Path("/content/aroma")

# Drive repo가 있으면 로컬 클론 없이 직접 사용 가능
# 로컬 클론이 필요하면 아래 주석 해제:
# if REPO_LOCAL.exists():
#     !git -C {REPO_LOCAL} pull --ff-only
# else:
#     !git clone <YOUR_REPO_URL> {REPO_LOCAL}

# 0-3. 패키지 설치
!pip install -q ultralytics effdet timm
```

---

## 셀 1: Smoke Test 공통 설정

**이 셀을 먼저 실행해야 이후 모든 셀이 동작합니다.**

```python
import json, sys, yaml
from pathlib import Path

# ━━━ 코드 경로 자동 탐지 ━━━
# 우선순위: 로컬 클론 > Drive repo
REPO_LOCAL = Path("/content/aroma")
REPO_DRIVE = Path("/content/aroma")

if REPO_LOCAL.exists() and (REPO_LOCAL / "stage0_resize.py").exists():
    CODE_ROOT = str(REPO_LOCAL)
elif REPO_DRIVE.exists() and (REPO_DRIVE / "stage0_resize.py").exists():
    CODE_ROOT = str(REPO_DRIVE)
else:
    raise FileNotFoundError(
        "stage0_resize.py를 찾을 수 없습니다. "
        f"확인된 경로: {REPO_LOCAL}, {REPO_DRIVE}")

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
print(f"코드 경로: {CODE_ROOT}")

REPO   = REPO_DRIVE  # dataset_config, configs 등은 항상 Drive에 위치
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

# ━━━ Smoke Test 대상 카테고리 (2개) ━━━
SMOKE_KEYS = ["isp_ASM", "visa_candle"]  # visa_candle 제외 (Stage 4 defect=0)

SMOKE_ENTRIES = {k: CONFIG[k] for k in SMOKE_KEYS}

# 카테고리별 기본 경로 계산
SMOKE_CATS = {}
for key, entry in SMOKE_ENTRIES.items():
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir   = Path(seed_dirs_list[0]).parents[1]
    image_dir = entry["image_dir"]
    domain    = entry["domain"]
    SMOKE_CATS[key] = {
        "cat_dir": cat_dir,
        "image_dir": image_dir,
        "seed_dirs": seed_dirs_list,
        "domain": domain,
    }

print("=== Smoke Test 대상 ===")
for key, info in SMOKE_CATS.items():
    exists = info["cat_dir"].exists()
    print(f"  {key}: {info['cat_dir']} {'✓' if exists else '✗ NOT FOUND'}")
    print(f"    seed_dirs: {[Path(s).name for s in info['seed_dirs']]}")
```

---

## 셀 2: Stage 0 — 이미지 리사이즈 (512×512, 로컬 SSD 캐시)

> Drive FUSE I/O 병목 해소: 이미지를 로컬 SSD로 복사 → 리사이즈 → Drive에 덮어쓰기.
> in-place 리사이즈이므로 원본이 변경됩니다.

```python
import shutil, time, cv2
from stage0_resize import clean_category

TARGET_SIZE = 512
CLEAN_FIRST = True
LOCAL_TMP = Path("/content/tmp_stage0")
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

def resize_dir_local(src_dir, target_size):
    """src_dir의 이미지를 로컬 SSD에서 리사이즈 후 Drive에 덮어쓰기."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        return {"resized": 0, "skipped": 0, "errors": 0}

    local_dir = LOCAL_TMP / src_dir.name
    if local_dir.exists():
        shutil.rmtree(local_dir)
    shutil.copytree(str(src_dir), str(local_dir))

    stats = {"resized": 0, "skipped": 0, "errors": 0}
    for ext in IMAGE_EXTS:
        for img_path in sorted(local_dir.glob(ext)):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["errors"] += 1
                    continue
                h, w = img.shape[:2]
                if h == target_size and w == target_size:
                    stats["skipped"] += 1
                    continue
                interp = cv2.INTER_AREA if max(h, w) > target_size else cv2.INTER_LINEAR
                resized = cv2.resize(img, (target_size, target_size), interpolation=interp)
                cv2.imwrite(str(img_path), resized)
                stats["resized"] += 1
            except Exception as e:
                print(f"    ERR: {img_path.name}: {e}")
                stats["errors"] += 1

    # 리사이즈된 이미지를 Drive로 덮어쓰기
    if stats["resized"] > 0:
        for ext in IMAGE_EXTS:
            for img_path in local_dir.glob(ext):
                shutil.copy2(str(img_path), str(src_dir / img_path.name))

    shutil.rmtree(local_dir, ignore_errors=True)
    return stats

for key, info in SMOKE_CATS.items():
    entry = SMOKE_ENTRIES[key]
    cat_dir  = info["cat_dir"]
    sentinel = cat_dir / f".stage0_resize_{TARGET_SIZE}_done"

    if sentinel.exists():
        print(f"  ⏭ {key}: 이미 완료 (sentinel 존재)")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 0: {key}")
    t0 = time.time()

    # Clean
    if CLEAN_FIRST:
        # clean_category expects 'seed_dir' (single str); adapt for seed_dirs array
        clean_entry = dict(entry)
        if "seed_dir" not in clean_entry:
            clean_entry["seed_dir"] = info["seed_dirs"][0]
        deleted = clean_category(clean_entry)
        if deleted:
            print(f"  {len(deleted)} items deleted")

    # Resize (로컬 SSD 캐시)
    total = {"resized": 0, "skipped": 0, "errors": 0}
    resize_targets = [
        ("image_dir", Path(info["image_dir"])),
        ("test/good", cat_dir / "test" / "good"),
    ]
    for sd in info["seed_dirs"]:
        resize_targets.append((f"seed/{Path(sd).name}", Path(sd)))
    for label, dir_path in resize_targets:
        s = resize_dir_local(dir_path, TARGET_SIZE)
        for k in total:
            total[k] += s[k]
        print(f"    {label}: resized={s['resized']} skipped={s['skipped']}")

    # Sentinel 생성
    if total["errors"] == 0:
        from datetime import datetime, timezone
        sentinel.write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_size": TARGET_SIZE,
            "resized": total["resized"],
            "skipped": total["skipped"],
        }))

    elapsed = time.time() - t0
    print(f"  ✓ resized={total['resized']} skipped={total['skipped']} ({elapsed:.1f}s)")

# 임시 디렉토리 정리
shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
print("\n✓ Stage 0 완료")
```

---

## 셀 3: Stage 1 — ROI 추출

```python
from stage1_roi_extraction import run_extraction

for key, info in SMOKE_CATS.items():
    cat_dir = info["cat_dir"]
    sentinel = cat_dir / "stage1_output" / "roi_metadata.json"

    if sentinel.exists():
        print(f"  ⏭ {key}: 이미 완료")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 1: {key}")

    run_extraction(
        image_dir  = info["image_dir"],
        output_dir = str(cat_dir / "stage1_output"),
        domain     = info["domain"],
        roi_levels = "both",
        workers    = -1,
    )
    print(f"  ✓ {key} 완료")

print("\n✓ Stage 1 완료")
```

---

## 셀 4: Stage 1b — Seed 특성 분석

```python
from stage1b_seed_characterization import run_seed_characterization_batch

for key, info in SMOKE_CATS.items():
    seed_dirs_list = info["seed_dirs"]
    cat_dir        = info["cat_dir"]
    use_prefix     = len(seed_dirs_list) > 1

    print(f"\n{'='*50}")
    print(f"Stage 1b: {key} (seed_dirs={len(seed_dirs_list)}개, prefix={use_prefix})")

    # 미완료 seed만 필터링
    tasks = []
    skipped = 0
    for sd_str in seed_dirs_list:
        sd = Path(sd_str)
        seeds = sorted(sd.glob("*.png")) if sd.exists() else []
        for seed in seeds:
            seed_id = f"{sd.name}_{seed.stem}" if use_prefix else seed.stem
            out = cat_dir / "stage1b_output" / seed_id
            if (out / "seed_profile.json").exists():
                skipped += 1
            else:
                tasks.append((str(seed), str(out)))

    if not tasks:
        print(f"  ⏭ {key}: 모두 완료 (skip {skipped})")
        continue

    results = run_seed_characterization_batch(tasks, workers=-1)
    print(f"  ✓ {key}: processed={len(results)} skipped={skipped}")

print("\n✓ Stage 1b 완료")
```

---

## 셀 5: Stage 2 — Defect Seed 변형 생성

```python
from stage2_defect_seed_generation import run_seed_generation

NUM_VARIANTS = 50

for key, info in SMOKE_CATS.items():
    cat_dir     = info["cat_dir"]
    stage1b_dir = cat_dir / "stage1b_output"

    if not stage1b_dir.exists():
        print(f"  ⚠ {key}: stage1b_output 없음 → skip")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 2: {key}")

    for profile_path in sorted(stage1b_dir.glob("*/seed_profile.json")):
        seed_id = profile_path.parent.name
        out_dir = cat_dir / "stage2_output" / seed_id

        if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= NUM_VARIANTS:
            print(f"  ⏭ {seed_id}: 이미 완료")
            continue

        profile = json.loads(profile_path.read_text())
        run_seed_generation(
            seed_defect  = profile["seed_path"],
            num_variants = NUM_VARIANTS,
            output_dir   = str(out_dir),
            seed_profile = str(profile_path),
            workers      = -1,
        )
        print(f"  ✓ {seed_id}")

print("\n✓ Stage 2 완료")
```

---

## 셀 6: Stage 3 — 레이아웃 로직

```python
from stage3_layout_logic import run_layout_logic

for key, info in SMOKE_CATS.items():
    cat_dir    = info["cat_dir"]
    stage2_dir = cat_dir / "stage2_output"

    if not stage2_dir.exists():
        print(f"  ⚠ {key}: stage2_output 없음 → skip")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 3: {key}")

    for d in sorted(stage2_dir.iterdir()):
        if not d.is_dir():
            continue
        seed_id = d.name
        pm_path = cat_dir / "stage3_output" / seed_id / "placement_map.json"

        if pm_path.exists():
            print(f"  ⏭ {seed_id}: 이미 완료")
            continue

        profile_path = cat_dir / "stage1b_output" / seed_id / "seed_profile.json"
        run_layout_logic(
            roi_metadata     = str(cat_dir / "stage1_output" / "roi_metadata.json"),
            defect_seeds_dir = str(d),
            output_dir       = str(cat_dir / "stage3_output" / seed_id),
            seed_profile     = str(profile_path) if profile_path.exists() else None,
            domain           = info["domain"] if info["domain"] != "visa" else "mvtec",
            workers          = -1,
            use_gpu          = True,
        )
        print(f"  ✓ {seed_id}")

print("\n✓ Stage 3 완료")
```

---

## 셀 7: Stage 4 — MPB 합성 (로컬 SSD 캐시)

> Drive FUSE I/O 병목 해소: 입력 데이터를 Colab 로컬 SSD(`/content/tmp_stage4/`)에
> 복사한 뒤 합성을 수행하고, 결과만 Drive로 업로드합니다.

```python
import shutil, time
from stage4_mpb_synthesis import run_synthesis_batch
from tqdm import tqdm

USE_FAST_BLEND = True
LOCAL_TMP = Path("/content/tmp_stage4")  # Colab 로컬 SSD
MAX_CANDLE_SEEDS = 10  # candle smoke test: 10 seeds만 처리 (100개 전체는 ~90K 이미지)

for key, info in SMOKE_CATS.items():
    cat_dir    = info["cat_dir"]
    image_dir  = Path(info["image_dir"])
    stage3_dir = cat_dir / "stage3_output"
    stage2_dir = cat_dir / "stage2_output"

    if not stage3_dir.exists():
        print(f"  ⚠ {key}: stage3_output 없음 → skip")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 4: {key}")

    # 미완료 seed 수집
    seed_pm_pairs = []
    skip = 0
    for d in sorted(stage3_dir.iterdir()):
        if not d.is_dir():
            continue
        pm = d / "placement_map.json"
        if not pm.exists():
            continue
        out_defect = cat_dir / "stage4_output" / d.name / "defect"
        if out_defect.exists() and any(out_defect.glob("*.png")):
            skip += 1
        else:
            seed_pm_pairs.append((d.name, str(pm)))

    # candle smoke test: 첫 10개 seed만 처리
    if key == "visa_candle" and len(seed_pm_pairs) > MAX_CANDLE_SEEDS:
        print(f"  ℹ candle smoke test: {len(seed_pm_pairs)} seeds → {MAX_CANDLE_SEEDS}로 제한")
        seed_pm_pairs = seed_pm_pairs[:MAX_CANDLE_SEEDS]

    if not seed_pm_pairs:
        print(f"  ⏭ {key}: 모두 완료 (skip {skip})")
        continue

    print(f"  {len(seed_pm_pairs)} seeds 처리 예정 (skip {skip})")

    # ── 로컬 SSD 캐시 ─────────────────────────────────────────
    t0 = time.time()
    local_cat = LOCAL_TMP / cat_dir.name
    local_image_dir = local_cat / "images"
    local_stage2    = local_cat / "stage2_output"
    local_stage3    = local_cat / "stage3_output"
    local_stage4    = local_cat / "stage4_output"

    # 1) 배경 이미지 복사
    if local_image_dir.exists():
        shutil.rmtree(local_image_dir)
    print(f"  배경 이미지 복사 중...", end=" ", flush=True)
    shutil.copytree(str(image_dir), str(local_image_dir))
    print("완료")

    # 2) Stage 2 output (defect variants) — 필요한 seed만 복사
    local_stage2.mkdir(parents=True, exist_ok=True)
    print(f"  Stage 2 variants 복사 중... ({len(seed_pm_pairs)} seeds)")
    for seed_id, _ in tqdm(seed_pm_pairs, desc="    복사", leave=False):
        src = stage2_dir / seed_id
        dst = local_stage2 / seed_id
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))

    # 3) placement_map.json 복사 + defect_path 로컬 경로 치환
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

    # ── 합성 (로컬 SSD) ──────────────────────────────────────
    png_comp = 1 if info["domain"] == "visa" else 3
    t1 = time.time()

    run_synthesis_batch(
        image_dir           = str(local_image_dir),
        seed_placement_maps = local_pm_pairs,
        output_root         = str(local_stage4),
        format              = "cls",
        use_fast_blend      = USE_FAST_BLEND,
        workers             = 4,
        png_compression     = png_comp,
        max_background_dim  = None,
    )
    synth_sec = time.time() - t1
    print(f"  합성 완료 ({synth_sec:.1f}s)")

    # ── 결과를 Drive로 복사 ───────────────────────────────────
    t2 = time.time()
    drive_stage4 = cat_dir / "stage4_output"
    print(f"  Drive 업로드 중... ({len(local_pm_pairs)} seeds)")
    for seed_id, _ in tqdm(local_pm_pairs, desc="    업로드", leave=False):
        src = local_stage4 / seed_id
        dst = drive_stage4 / seed_id
        if src.exists():
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))
    upload_sec = time.time() - t2
    print(f"  Drive 업로드 완료 ({upload_sec:.1f}s)")

    # ── 임시 파일 정리 ────────────────────────────────────────
    shutil.rmtree(str(local_cat), ignore_errors=True)

    total = time.time() - t0
    print(f"  ✓ {key} 완료 (총 {total:.1f}s = 캐시 {copy_sec:.1f}s + 합성 {synth_sec:.1f}s + 업로드 {upload_sec:.1f}s)")

print("\n✓ Stage 4 완료")
```

---

## 셀 8: Stage 5 — 합성 품질 점수

> Batch 병렬 처리: parallel_seeds로 시드 간 병렬 처리, workers로 시드 내 이미지 병렬 처리.

```python
from stage5_quality_scoring import run_quality_scoring_batch

for key, info in SMOKE_CATS.items():
    cat_dir    = info["cat_dir"]
    stage4_dir = cat_dir / "stage4_output"

    if not stage4_dir.exists():
        print(f"  ⚠ {key}: stage4_output 없음 → skip")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 5: {key}")

    # Collect seed directories
    seed_dirs = [
        str(d) for d in sorted(stage4_dir.iterdir())
        if d.is_dir()
    ]

    if not seed_dirs:
        print(f"  ⚠ {key}: 시드 없음")
        continue

    # Batch processing with seed-level parallelism
    results = run_quality_scoring_batch(
        stage4_seed_dirs = seed_dirs,
        workers          = 2,   # Image-level parallelism within each seed
        parallel_seeds   = 2,   # Seed-level parallelism (Colab 2-CPU 기준)
    )

    # Report results
    completed = len([r for r in results if r])
    skipped = len(seed_dirs) - completed
    print(f"  ✓ {key}: {completed} seeds scored, {skipped} already completed")
    
    if results:
        avg_count = sum(r["count"] for r in results) / len(results)
        avg_quality = sum(r["mean"] for r in results) / len(results)
        print(f"    평균: {avg_count:.0f} images/seed, quality={avg_quality:.3f}")

print("\n✓ Stage 5 완료")
```

---

## 셀 9: Stage 6 — 증강 데이터셋 구성 (로컬 SSD 캐시)

> 순수 파일 복사 Stage (~9,590 I/O ops). 로컬 SSD에서 구성 후 Drive로 일괄 업로드.
> ThreadPoolExecutor 병렬 복사 + tqdm 진행률 표시.

```python
import shutil, time, yaml
from stage6_dataset_builder import run_dataset_builder
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRUNING_RATIO = 0.5   # quality_score 상위 50% rank 기반 선택 (domain-agnostic)
LOCAL_TMP = Path("/content/tmp_stage6")

# 도메인별 증강 비율 설정 로드
CONFIG_PATH = REPO / "configs" / "benchmark_experiment.yaml"
augmentation_ratio_by_domain = None
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        augmentation_ratio_by_domain = config.get("dataset", {}).get("augmentation_ratio_by_domain")
    if augmentation_ratio_by_domain:
        print("도메인별 증강 비율 설정:")
        for domain, ratios in augmentation_ratio_by_domain.items():
            print(f"  {domain}: full={ratios['full']}, pruned={ratios['pruned']}")
    else:
        print("⚠ Config에 augmentation_ratio_by_domain 없음")
else:
    print("⚠ Config 파일 없음, 기본 동작 사용")

# 도메인별 증강 비율 명시적 매핑 (로컬 경로 도메인 추출 실패 대비)
RATIO_MAP = {
    "isp": {"full": 1.0, "pruned": 0.5},
    "mvtec": {"full": 2.0, "pruned": 1.5},
    "visa": {"full": 2.0, "pruned": 1.5},
}

# 병렬 디렉토리 복사 헬퍼
def parallel_copytree(src_dst_pairs, max_workers=4):
    """병렬로 여러 디렉토리 복사"""
    def _copy_one(pair):
        src, dst = pair
        if Path(src).exists():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            return dst
        return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_copy_one, src_dst_pairs))
    return [r for r in results if r is not None]

# cat_dir별 seed_dirs 수집 (복수 유형 배열 지원)
cat_seed_dirs = defaultdict(list)
for key, info in SMOKE_CATS.items():
    cat_seed_dirs[str(info["cat_dir"])].extend(info["seed_dirs"])

for key, info in SMOKE_CATS.items():
    cat_dir   = info["cat_dir"]
    image_dir = Path(info["image_dir"])
    domain    = info["domain"]  # 도메인 정보 추출
    sentinel  = cat_dir / "augmented_dataset" / "build_report.json"

    if sentinel.exists():
        print(f"  ⏭ {key}: 이미 완료")
        continue

    print(f"\n{'='*50}")
    print(f"Stage 6: {key} (domain={domain})")
    print(f"  목표 비율: full={RATIO_MAP[domain]['full']}, pruned={RATIO_MAP[domain]['pruned']}")
    t0 = time.time()

    # ── 로컬 SSD에 cat_dir 구조 미러링 (경로 구조 보존) ──────────
    # Drive: /content/drive/.../Aroma/isp/unsupervised/ASM
    # Local: /content/tmp_stage6/isp/unsupervised/ASM (구조 동일)
    # 이유: utils/dataset_builder.py의 도메인 추출 로직이 경로 구조에 의존
    relative_path = cat_dir.relative_to(cat_dir.parents[3])  # "isp/unsupervised/ASM"
    local_cat = LOCAL_TMP / relative_path
    if local_cat.exists():
        shutil.rmtree(str(local_cat))

    print(f"  로컬 캐시 복사 중... (경로: {relative_path})")
    
    # 병렬로 모든 필요한 디렉토리 복사
    copy_tasks = []
    
    # 1) stage4_output 복사 (defect 이미지 + quality_scores.json)
    stage4_src = cat_dir / "stage4_output"
    if stage4_src.exists():
        copy_tasks.append((stage4_src, local_cat / "stage4_output"))

    # 2) image_dir (train/good) 복사
    local_image_dir = local_cat / "train" / "good"
    copy_tasks.append((image_dir, local_image_dir))

    # 3) test/good 복사
    test_good_src = cat_dir / "test" / "good"
    if test_good_src.exists():
        copy_tasks.append((test_good_src, local_cat / "test" / "good"))

    # 4) seed_dirs (test defect) 복사
    local_seed_dirs = []
    for sd in cat_seed_dirs[str(cat_dir)]:
        sd_path = Path(sd)
        local_sd = local_cat / "test" / sd_path.name
        if sd_path.exists():
            copy_tasks.append((sd_path, local_sd))
            local_seed_dirs.append(str(local_sd))
    
    # 병렬 복사 실행 (4개 스레드로 동시 복사)
    parallel_copytree(copy_tasks, max_workers=4)
    
    copy_sec = time.time() - t0
    print(f"  로컬 캐시 완료 ({copy_sec:.1f}s, {len(copy_tasks)}개 디렉토리)")

    # ── 로컬에서 데이터셋 구성 (병렬 복사 + 명시적 비율 전달) ──────
    t1 = time.time()
    
    # 도메인별 비율 명시적 전달 (로컬 경로에서 도메인 추출 실패 대비)
    ratio_full = RATIO_MAP[domain]["full"]
    ratio_pruned = RATIO_MAP[domain]["pruned"]
    
    result = run_dataset_builder(
        cat_dir                      = str(local_cat),
        image_dir                    = str(local_image_dir),
        seed_dirs                    = local_seed_dirs,
        pruning_ratio                = PRUNING_RATIO,
        augmentation_ratio_full      = ratio_full,      # 명시적 전달!
        augmentation_ratio_pruned    = ratio_pruned,    # 명시적 전달!
        augmentation_ratio_by_domain = None,            # 이미 적용했으므로 None
        workers                      = 8,               # I/O-bound 작업에 최적화
        balance_defect_types         = True,            # 복수 결함 유형 균등 샘플링
    )
    build_sec = time.time() - t1
    print(f"  데이터셋 구성 완료 ({build_sec:.1f}s)")
    print(f"    추출된 domain: {result.get('domain', 'unknown')}")
    print(f"    applied ratio_full: {result.get('augmentation_ratio_full', 'N/A')}")
    print(f"    applied ratio_pruned: {result.get('augmentation_ratio_pruned', 'N/A')}")
    print(f"    baseline good={result['baseline']['good_count']}")
    print(f"    aroma_full defect={result['aroma_full']['defect_count']}")
    print(f"    aroma_pruned defect={result['aroma_pruned']['defect_count']}")
    # 예상 결과:
    # - isp_ASM (500 good): full=500, pruned=250
    # - mvtec_bottle (209 good): full=418, pruned=313
    # - visa_candle (900 good): full=1800, pruned=1350

    # ── 결과를 Drive로 업로드 ─────────────────────────────────
    t2 = time.time()
    print(f"  Drive 업로드 중...")
    local_aug = local_cat / "augmented_dataset"
    drive_aug = cat_dir / "augmented_dataset"
    if drive_aug.exists():
        shutil.rmtree(str(drive_aug))
    shutil.copytree(str(local_aug), str(drive_aug))
    upload_sec = time.time() - t2
    print(f"  Drive 업로드 완료 ({upload_sec:.1f}s)")

    # ── 임시 파일 정리 ────────────────────────────────────────
    shutil.rmtree(str(local_cat), ignore_errors=True)

    total = time.time() - t0
    print(f"  ✓ {key} 완료 (총 {total:.1f}s = 캐시 {copy_sec:.1f}s + 구성 {build_sec:.1f}s + 업로드 {upload_sec:.1f}s)")

shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
print("\n✓ Stage 6 완료")
```

---

## 셀 10: Stage 6 검증 — 데이터 구조 확인

```python
BENCH_CFG = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment.yaml").read_text())
GROUPS = list(BENCH_CFG["dataset_groups"].keys())

for key, info in SMOKE_CATS.items():
    cat_dir = info["cat_dir"]
    aug = cat_dir / "augmented_dataset"

    print(f"\n{'='*50}")
    print(f"{key}: {cat_dir.name}")

    if not aug.exists():
        print("  ✗ augmented_dataset 없음")
        continue

    for g in GROUPS:
        gdir = aug / g
        n_good   = len(list((gdir / "train" / "good").glob("*")))   if (gdir / "train" / "good").exists()   else 0
        n_defect = len(list((gdir / "train" / "defect").glob("*"))) if (gdir / "train" / "defect").exists() else 0
        has_test = (gdir / "test").exists()
        if has_test:
            test_counts = {d.name: len(list(d.glob("*")))
                          for d in (gdir / "test").iterdir() if d.is_dir()}
        else:
            test_counts = "(auto-symlink at Stage 7)"
        print(f"  {g}: train(good={n_good}, defect={n_defect}) test={test_counts}")
```

---

## 셀 9.5: Stage 6 재실행 준비 (기존 augmented_dataset 삭제)

> **목적:** 잘못된 비율로 생성된 augmented_dataset을 삭제하고 Stage 6 재실행 준비
> **실행 시점:** Stage 6 결과가 도메인별 목표 비율과 맞지 않을 때만 실행

### 옵션 A: 전체 카테고리 삭제

```python
import shutil

print("기존 augmented_dataset 삭제 중...\n")

for key, info in SMOKE_CATS.items():
    aug_dir = info["cat_dir"] / "augmented_dataset"
    if aug_dir.exists():
        shutil.rmtree(str(aug_dir))
        print(f"✓ {key}: augmented_dataset 삭제 완료")
    else:
        print(f"  {key}: augmented_dataset 없음 (스킵)")

print("\n✓ 삭제 완료. 이제 셀 9를 다시 실행하세요.")
```

### 옵션 B: 특정 카테고리만 삭제 (visa_candle)

```python
import shutil

# 삭제할 카테고리 선택
TARGETS = ["visa_candle"]  # 필요에 따라 추가: ["visa_candle", "isp_ASM"]

print(f"선택된 카테고리만 augmented_dataset 삭제 중: {TARGETS}\n")

for key, info in SMOKE_CATS.items():
    if key not in TARGETS:
        print(f"  ⏭ {key}: 건너뜀")
        continue
    
    aug_dir = info["cat_dir"] / "augmented_dataset"
    if aug_dir.exists():
        shutil.rmtree(str(aug_dir))
        print(f"✓ {key}: augmented_dataset 삭제 완료")
    else:
        print(f"  {key}: augmented_dataset 없음 (이미 삭제됨)")

print("\n✓ 삭제 완료. 이제 셀 9를 다시 실행하세요.")
```

**다음 단계:**
1. 위 셀 실행으로 기존 augmented_dataset 삭제
2. **셀 9** 재실행 (Stage 6 - 도메인별 비율 적용)
   - 삭제한 카테고리만 재실행됨 (나머지는 스킵)
3. **셀 10** 실행 (검증)

---

## 셀 10.5: Stage 7 — 벤치마크 결과 초기화 (재실행 시)

> seed_dirs 변경으로 augmented_dataset이 재생성되었으므로 이전 벤치마크 결과(sentinel)를 삭제한다.
> 삭제하지 않으면 셀 11에서 "모든 모델/그룹 완료 (skip)" 처리됨.

```python
import shutil
from pathlib import Path

OUTPUT_DIR = REPO / "outputs" / "benchmark_results"

for key, info in SMOKE_CATS.items():
    cat_name = info["cat_dir"].name
    cat_out  = OUTPUT_DIR / cat_name
    if cat_out.exists():
        shutil.rmtree(str(cat_out))
        print(f"✓ 삭제: {cat_out}")
    else:
        print(f"  없음: {cat_out}")

print("\n✓ 완료 — 셀 11(벤치마크)을 실행하세요")
```

---

## 셀 11: Stage 7 — test set 준비 + 벤치마크 (로컬 SSD 캐시)

> 학습 데이터(augmented_dataset)를 로컬 SSD로 복사하여 DataLoader I/O 가속.
> 결과 JSON은 Drive에 직접 저장 (소량이므로 Drive 직접 쓰기 OK).

```python
import shutil, time, yaml
from stage7_benchmark import run_benchmark, _ensure_test_dir

LOCAL_TMP = Path("/content/tmp_stage7")

BENCH_CFG  = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment.yaml").read_text())
CONFIG_PATH = str(REPO / "configs" / "benchmark_experiment.yaml")
OUTPUT_DIR  = str(REPO / "outputs" / "benchmark_results")

MODELS = list(BENCH_CFG["models"].keys())
GROUPS = list(BENCH_CFG["dataset_groups"].keys())
NON_BASELINE = [g for g in GROUPS if g != "baseline"]

for key, info in SMOKE_CATS.items():
    cat_dir = info["cat_dir"]
    print(f"\n{'='*60}")
    print(f"Stage 7: {key} ({cat_dir.name})")

    # Stage 6 완료 확인
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        print(f"  ⚠ baseline/test 없음 → Stage 6 미완료, skip")
        continue

    # 완료 확인
    out_root = Path(OUTPUT_DIR) / cat_dir.name
    if all((out_root / m / g / "experiment_meta.json").exists()
           for m in MODELS for g in GROUPS):
        print(f"  ⏭ 모든 모델/그룹 완료")
        continue

    # ── augmented_dataset를 로컬 SSD로 복사 ───────────────────
    t0 = time.time()
    local_cat = LOCAL_TMP / cat_dir.name
    local_aug = local_cat / "augmented_dataset"
    if local_cat.exists():
        shutil.rmtree(str(local_cat))
    local_aug.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(cat_dir / "augmented_dataset"), str(local_aug))
    copy_sec = time.time() - t0
    print(f"  로컬 캐시 완료 ({copy_sec:.1f}s)")

    # test set 준비 (로컬에서)
    for g in NON_BASELINE:
        group_test = local_aug / g / "test"
        if not group_test.exists():
            result = _ensure_test_dir(str(local_cat), g)
            method = "symlink" if result.is_symlink() else "copy"
            print(f"  test set: {g}/test [{method}]")

    print(f"  모델: {MODELS}")
    print(f"  그룹: {GROUPS}")

    # ── 벤치마크 실행 (로컬 데이터, 결과는 Drive) ─────────────
    t1 = time.time()
    try:
        results = run_benchmark(
            config_path = CONFIG_PATH,
            cat_dir     = str(local_cat),
            resume      = True,
            output_dir  = OUTPUT_DIR,
        )
        for model_name, group_results in results.items():
            for group, val in group_results.items():
                if isinstance(val, dict) and "error" in val:
                    print(f"  ✗ {model_name}/{group}: {val['error']} — {val.get('detail', '')[:80]}")
                elif isinstance(val, dict):
                    auroc = val.get("image_auroc", "?")
                    print(f"  ✓ {model_name}/{group}: AUROC={auroc:.4f}" if isinstance(auroc, float) else f"  ✓ {model_name}/{group}: {auroc}")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")

    bench_sec = time.time() - t1
    total = time.time() - t0
    print(f"  ✓ {key} 완료 (총 {total:.1f}s = 캐시 {copy_sec:.1f}s + 벤치마크 {bench_sec:.1f}s)")

    # 임시 파일 정리
    shutil.rmtree(str(local_cat), ignore_errors=True)

shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
print(f"\n{'='*60}")
print("✓ Stage 7 완료")
```

---

## 셀 11.5: ASM YOLO OOM 수정 테스트 (Phase 3 적용 — 선택사항)

> **목적:** ASM YOLO CUDA OOM 수정(청크 분할 평가 + `PYTORCH_ALLOC_CONF` + `cache=False`) 단일 실험 검증  
> **대상:** isp_ASM + yolo11 + aroma_full (1개 실험만)  
> **소요 시간:** ~10분  
> **실행 조건:** 셀 11에서 ASM YOLO OOM 발생 시에만 실행

### 사용 시나리오

셀 11 실행 중 다음 에러 발생 시:
```
ERROR:stage7_benchmark:yolo11/aroma_full: Runtime error - CUDA out of memory.
Tried to allocate 5.71 GiB...
```

### Step 1: Git Pull + 런타임 재시작

```python
# Git pull로 최신 수정 적용
!cd /content/aroma && git pull
!cd /content/aroma && git log --oneline -3

# 예상 출력:
# (latest) fix: YOLO OOM - 청크 분할 평가 + PYTORCH_ALLOC_CONF + cache=False
# a026ae2 fix: YOLO OOM 추가 대응 - validation 비활성화, batch 크기 축소, 평가 전 메모리 정리
# e2296e4 docs: 셀 9.5 개선 - 선택적 카테고리 삭제 옵션 추가
```

**중요:** Git pull 후 반드시 **런타임 재시작** 필요!
- Colab 메뉴: `런타임` → `런타임 다시 시작`

### Step 2: 환경 재설정 (런타임 재시작 후)

```python
# 셀 0 재실행 (Drive 마운트 + 패키지 설치)
from google.colab import drive
drive.mount("/content/drive")
!pip install -q ultralytics effdet timm

# 셀 1 재실행 (경로 설정)
import sys
from pathlib import Path

REPO = Path("/content/aroma")
sys.path.insert(0, str(REPO))

DATA_BASE = Path("/content/drive/MyDrive/data/Aroma")
```

### Step 3: ASM YOLO aroma_full 단일 실험

```python
import yaml
from stage7_benchmark import run_benchmark

# Config 로드
config_path = REPO / "configs" / "benchmark_experiment.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# ASM 카테고리 경로
asm_cat = DATA_BASE / "isp" / "unsupervised" / "ASM"

print("="*60)
print("Phase 1: ASM YOLO aroma_full 단일 실험 테스트")
print("="*60)
print(f"카테고리: {asm_cat}")
print(f"모델: yolo11")
print(f"그룹: aroma_full")
print(f"예상 시간: ~10분\n")

# 단일 실험 실행
results = run_benchmark(
    config_path=str(config_path),
    cat_dir=str(asm_cat),
    models=["yolo11"],
    groups=["aroma_full"],
    output_dir=str(REPO / "outputs" / "benchmark_results"),
    resume=True,
)

# 결과 확인
print("\n" + "="*60)
print("테스트 결과:")
print("="*60)

for model, groups in results.items():
    for group, val in groups.items():
        if isinstance(val, dict) and "error" in val:
            print(f"❌ {model}/{group}")
            print(f"   에러: {val['error']}")
            print(f"   상세: {val.get('detail', 'N/A')[:200]}")
            print("\n⚠️ 수정 실패 - Phase 2 또는 Phase 3 진행 필요")
        elif isinstance(val, dict):
            auroc = val.get("image_auroc", "N/A")
            if auroc != "N/A":
                print(f"✅ {model}/{group}: AUROC={auroc:.4f}")
                print("\n✓ 수정 성공! 셀 11 재실행 가능")
            else:
                print(f"✅ {model}/{group}: {auroc}")
        else:
            print(f"? {model}/{group}: {val}")
```

### Step 4: 성공 시 전체 Smoke Test 재실행

Phase 1 테스트 성공 시:
1. **셀 11 재실행** (전체 Stage 7)
2. **셀 12 실행** (결과 요약)

### 실패 시 대안

Phase 3 (청크 분할) 포함 수정 후에도 OOM 시:
- train_batch_size를 16 → 4 로 줄이기 (`configs/benchmark_experiment.yaml`)
- eval_chunk_size를 64 → 32 로 줄이기
- image_size를 512 → 256 으로 줄이기 (성능 하락 감수)

---

## 셀 12: 결과 요약

```python
import json
from pathlib import Path

OUTPUT_DIR = REPO / "outputs" / "benchmark_results"

BENCH_CFG = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment.yaml").read_text())
MODELS = list(BENCH_CFG["models"].keys())
GROUPS = list(BENCH_CFG["dataset_groups"].keys())

print(f"\n{'='*80}")
print("SMOKE TEST 결과 요약")
print(f"{'='*80}")

cols = [f"{m}/{g}" for m in MODELS for g in GROUPS]
header = f"{'category':<20}" + "".join(f"{c:<25}" for c in cols)
print(header)
print("-" * len(header))

for key, info in SMOKE_CATS.items():
    cat_name = info["cat_dir"].name
    cat_out  = OUTPUT_DIR / cat_name
    row = f"{cat_name:<20}"

    for m in MODELS:
        for g in GROUPS:
            meta = cat_out / m / g / "experiment_meta.json"
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
            row += f"{val:<25}"

    print(row)

print(f"\n{'='*80}")
```

---

## 예상 실행 시간 (Colab T4 GPU)

| Stage | ASM | bottle | 합계 |
|-------|-----|--------|------|
| Stage 0 (resize) | ~10s | ~5s | ~15s |
| Stage 1 (ROI) | ~1분 | ~30s | ~1.5분 |
| Stage 1b (seed) | ~30s | ~10s | ~40s |
| Stage 2 (variants) | ~2분 | ~1분 | ~3분 |
| Stage 3 (layout) | ~1분 | ~30s | ~1.5분 |
| Stage 4 (synthesis) | ~3분 | ~2분 | ~5분 |
| Stage 5 (quality) | ~30s | ~20s | ~50s |
| Stage 6 (dataset) | ~1분 | ~30s | ~1.5분 |
| Stage 7 (benchmark) | ~5분 | ~3분 | ~8분 |
| **합계** | ~14분 | ~8.5분 | **~22-25분** |

---

## Smoke Test 통과 기준

1. **Stage 0**: 2개 카테고리 모두 sentinel 파일 생성 확인
2. **Stage 1~6**: 각 Stage 오류 없이 완료
3. **Stage 6 검증**: 각 카테고리 × 3개 그룹에 train/test 데이터 존재
4. **Stage 7**: 2개 카테고리 × 2개 모델 × 3개 그룹 = **12개 실험** 모두 완료

---

## 제외 카테고리: visa_candle

**제외 사유:** Stage 4 defect 생성 실패

**증상:**
```
⚠️ 증강 비율 1.5×900=1350개 요청, 가용 defect 0개만 사용 (카테고리: candle)
aroma_pruned defect=0
```

**원인:**
- Stage 4 (`stage4_synthesis.py`) 실행 시 defect 이미지가 생성되지 않음
- `stage4_output/` 디렉토리는 존재하나 defect 파일 0개
- 원인 미상 (Stage 1b-3 결과는 정상, Stage 4 로직 이슈 추정)

**해결 시도:**
1. Stage 6 재실행 (명시적 비율 전달) → 여전히 defect=0
2. 진단 스크립트 준비 → Stage 4 확인 필요
3. **결정:** 시간 관계상 visa_candle 제외하고 진행

**향후 조치:**
- Stage 4 로직 디버깅 필요
- visa 도메인 특화 이슈 가능성 조사

---

## 디버깅: visa_candle Stage 6 결과 확인 (참고용)

Stage 7 실행 중 `visa_candle`의 `aroma_pruned` 그룹에서 defect 이미지가 0건으로 나타난 경우,
아래 스크립트로 Stage 6 결과를 확인합니다.

### 셀 12: visa_candle Stage 6 진단 (참고용)

```python
# visa_candle Stage 6 결과 확인 스크립트 실행
exec(open(REPO / "docs/작업일지/check_visa_candle_stage6.py").read())
```

**확인 내용:**
1. `build_report.json`에서 증강 비율 적용 여부 확인
2. `aroma_pruned/train/defect/` 실제 파일 개수 확인
3. Stage 4 `quality_scores.json`에서 고품질 이미지 개수 확인
4. 문제 진단 및 권장사항 출력

**예상 출력:**
```
======================================================================
visa_candle Stage 6 결과 확인
======================================================================

=== build_report.json ===
Domain: visa
Augmentation ratio full: 2.0
Augmentation ratio pruned: 1.5

### Aroma Pruned ###
Good count: 900
Defect count: 0  ← 문제!
Target defect count: 1350
Applied ratio: N/A  ← 비율 미적용!

======================================================================
4. 진단 및 권장사항
======================================================================

### 발견된 문제 ###
  🔴 aroma_pruned defect count = 0
  🔴 aroma_pruned applied ratio = N/A (비율 미적용)

### 권장사항 ###
  1. Stage 6 재실행 권장:
     - 명시적 비율 전달 확인 (ratio_full=2.0, ratio_pruned=1.5)
     - 로컬 캐시 → Drive 업로드 확인
```
5. **결과**: AUROC ≥ 0.5 (random baseline 이상) — 합성 데이터가 학습에 기여하는지 확인
