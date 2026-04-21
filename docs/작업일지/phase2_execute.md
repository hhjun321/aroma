# AROMA Phase 2 — Diffusion 합성 Colab 실행 셀

> **합성 방식:** SD Inpainting + ControlNet — Stage 4: `stage4_diffusion_synthesis.py`
> Phase 1 (MPB) 실행 결과(`stage4_output/`)는 그대로 보존하고,
> Diffusion 합성 결과는 `stage4_diffusion_output/` 에 별도 저장한다.
>
> **실험 그룹:** `baseline` / `aroma_mpb` (Phase 1 결과 재사용) / `aroma_diffusion` (Phase 2 신규)

## 규칙

- **`DOMAIN_FILTER`** 를 `"isp"` / `"mvtec"` / `"visa"` 로 바꿔 각 도메인 실행
- **`CAT_ONLY`** 로 특정 카테고리만 실행 — `["bottle"]` = bottle만, `None` = 전체
- **skip 로직** 내장 — sentinel 파일이 있으면 자동으로 건너뜀 (resume 가능)
- **병렬 설정** 은 셀 상단 `# 병렬 설정` 블록에서 조절
- `failed` 리스트로 실패 항목 추적 — 전체 실행 후 한 번에 확인 가능
- Stage 0–3 은 Phase 1 결과를 그대로 재사용 (재실행 불필요)

## 진행 상황 확인

```python
# Stage 1–6 전체 진행 상황 확인 (실행 전/후 언제든 사용)
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

## Stage 0–3: Phase 1 결과 재사용

Stage 0 (리사이즈), Stage 1 (ROI 추출), Stage 1b (Seed 특성 분석), Stage 2 (변형 생성),
Stage 3 (레이아웃 로직) 출력은 Phase 1 과 동일하게 사용한다.
해당 셀은 `phase1_execute.md` 를 참조하여 실행한다.

### Stage 2 / Stage 3 재사용 범위 및 제한

| 출력물 | Phase 2 사용 방식 | 비고 |
|--------|-----------------|------|
| `stage2_output/{seed_id}/variant_*.png` | **ROI 위치·크기 계산에만 사용** | `placement_map.json`의 `defect_path` → `_make_roi_mask()` 에서 패치 크기(ph, pw) 참조 |
| `stage3_output/{seed_id}/placement_map.json` | **그대로 재사용** | x, y 좌표·scale은 원본 seed 기준으로도 동일하게 유효 |
| `stage1b_output/{seed_id}/seed_profile.json` | **ControlNet 입력 소스** | `seed_path` 필드 → 원본 결함 이미지 → Canny 엣지 추출 |

> **설계 원칙 (CASDA 방식 준수)**
> CASDA 검증 파라미터(`strength=0.7` 등)는 **실제 결함 이미지**를 ControlNet 입력으로
> 실험하여 얻은 값이다. Stage 2 변형 이미지(elastic warp)에서 추출한 Canny 엣지는
> 원본 결함 형태가 왜곡되어 ControlNet 구조적 가이던스 품질이 저하된다.
> Phase 2의 핵심 질문("합성 방식 MPB vs Diffusion")을 순수하게 검증하려면
> Diffusion 경로에 MPB 고유 요소(Stage 2 변형)가 개입되어서는 안 된다.

> **경로 주의**
> `placement_map.json`의 `defect_path` 는 Stage 2 파일의 절대 경로를 저장한다.
> Colab Drive 마운트 경로가 Phase 1 실행 시와 동일해야 하며,
> 다를 경우 `smoke_test_3cat.md` 셀 7 방식의 경로 치환이 필요하다.

---

## Stage 4: Diffusion 합성

**출력:** `{cat_dir}/stage4_diffusion_output/{seed_id}/defect/*.png`
**병렬:** GPU 단일 점유 → 카테고리 순차 처리 (MPB의 IMG_THREADS 없음)
**CASDA 검증 파라미터:** `strength=0.7` · `guidance_scale=7.5` · `num_inference_steps=30` · `conditioning_scale=0.7`

> **로컬 SSD 캐시:** `image_dir` (배경 이미지)를 `/content/tmp_stage4` 에 복사 후 추론,
> 결과는 로컬에 쓰고 Drive 에 업로드. Drive FUSE I/O 병목 해소.

```python
import json, sys, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage4_diffusion_synthesis import run_synthesis_batch

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD ("isp" / "mvtec" / "visa")
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 실행 (None → 전체 MVTec AD)
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# Diffusion 파라미터 (CASDA 검증값)
CONTROLNET_MODEL    = None    # None → pretrained lllyasviel/sd-controlnet-canny 사용
                              # 파인튜닝된 모델 경로: "/content/drive/.../controlnet_output"
DEVICE              = "cuda"
RESOLUTION          = 512
NUM_INFERENCE_STEPS = 30
STRENGTH            = 0.7
GUIDANCE_SCALE      = 7.5
CONDITIONING_SCALE  = 0.7    # 학습 1.0 → 추론 0.7 (아티팩트 방지)
SEED                = 42

LOCAL_TMP = Path("/content/tmp_stage4")
_LOCAL_DOMAIN_PATH = {
    "isp":   LOCAL_TMP / "isp" / "unsupervised",
    "mvtec": LOCAL_TMP / "mvtec",
    "visa":  LOCAL_TMP / "visa",
}[DOMAIN_FILTER]

def _parallel_copy(src_dst_pairs, max_workers=4):
    def _copy(pair):
        src, dst = Path(pair[0]), Path(pair[1])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_copy, src_dst_pairs))

# 카테고리 단위로 묶기
categories = {}
ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
        continue
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
        out_defect = cat_dir / "stage4_diffusion_output" / d.name / "defect"
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

    for cat_dir, image_dir, seed_pm_pairs in tqdm(all_cats, desc=f"Stage4-Diffusion {LABEL}"):
        local_cat    = _LOCAL_DOMAIN_PATH / cat_dir.name
        local_imgdir = local_cat / "train" / "good"
        local_output = local_cat / "stage4_diffusion_output"
        t0 = time.time()
        try:
            # 로컬 SSD 복사: image_dir만 (stage1b seed_path는 Drive 직접 읽기 — seed당 1회)
            _parallel_copy([(image_dir, local_imgdir)], max_workers=4)
            copy_sec = time.time() - t0

            t1 = time.time()
            run_synthesis_batch(
                image_dir           = str(local_imgdir),
                seed_placement_maps = seed_pm_pairs,
                output_root         = str(local_output),
                cat_dir             = str(cat_dir),   # stage1b_output 경로용 (Drive)
                format              = "cls",
                controlnet_model    = CONTROLNET_MODEL,
                device              = DEVICE,
                resolution          = RESOLUTION,
                num_inference_steps = NUM_INFERENCE_STEPS,
                strength            = STRENGTH,
                guidance_scale      = GUIDANCE_SCALE,
                conditioning_scale  = CONDITIONING_SCALE,
                seed                = SEED,
            )
            infer_sec = time.time() - t1

            # Drive 업로드
            t2 = time.time()
            drive_output = cat_dir / "stage4_diffusion_output"
            if drive_output.exists():
                shutil.rmtree(str(drive_output))
            shutil.copytree(str(local_output), str(drive_output))
            upload_sec = time.time() - t2

            print(f"  ✓ {cat_dir.name}: 캐시 {copy_sec:.0f}s + 추론 {infer_sec:.0f}s"
                  f" + 업로드 {upload_sec:.0f}s = {time.time()-t0:.0f}s")
        except Exception as e:
            failed.append({"category": cat_dir.name,
                           "error": str(e), "type": type(e).__name__})
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

    shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## Stage 5: 합성 품질 점수 (Diffusion 출력 기준)

**Sentinel:** `{cat_dir}/stage4_diffusion_output/{seed_id}/quality_scores.json`
**병렬:** seed 단위 `ThreadPoolExecutor` (CPU-bound 품질 메트릭 계산)

```python
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage5_quality_scoring import run_quality_scoring

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD ("isp" / "mvtec" / "visa")
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 실행 (None → 전체 MVTec AD)
LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

# 병렬 설정
SEED_THREADS = 4    # seed 단위 동시 처리 (I/O + CPU 혼합)
IMG_WORKERS  = -1   # run_quality_scoring 내부 이미지 단위 병렬 수

ENTRIES = [(k, v) for k, v in CONFIG.items()
           if not k.startswith("_") and v["domain"] == DOMAIN_FILTER]

all_seeds, skip = [], 0
seen_cats = set()
for key, entry in ENTRIES:
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir    = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen_cats:
        continue
    seen_cats.add(str(cat_dir))
    stage4_dir = cat_dir / "stage4_diffusion_output"   # Phase 2: diffusion 출력 디렉터리
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
            stage4_seed_dir = str(cat_dir / "stage4_diffusion_output" / seed_id),
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

## Stage 6: 증강 데이터셋 구성 (aroma_diffusion 그룹 — 로컬 SSD 캐시)

**Sentinel:** `{cat_dir}/augmented_dataset/aroma_diffusion/train/defect` 존재
**병렬:** 카테고리 단위 — 로컬 SSD에서 구성 후 Drive 에 `aroma_diffusion` 만 업로드

> `baseline` / `aroma_mpb` 그룹은 Phase 1 에서 생성됨 — 건드리지 않는다.
> Phase 2 에서는 `aroma_diffusion` 그룹만 추가 생성한다.
> Drive FUSE I/O 병목 해소: `stage4_diffusion_output` · `image_dir` 를 로컬 SSD 에 복사 후
> 데이터셋을 구성하고 `aroma_diffusion` 결과만 Drive 에 업로드한다.
> 도메인 추출 로직이 경로 구조에 의존하므로 `LOCAL_TMP/{domain}/{cat}` 구조 유지.

```python
import json, sys, shutil, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage6_dataset_builder import run_dataset_builder

REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD ("isp" / "mvtec" / "visa")
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 실행 (None → 전체 MVTec AD)

# 병렬 설정
CAT_THREADS    = 2   # 카테고리 동시 처리 (Drive 동시 읽기/쓰기 안정성 고려)
NUM_IO_THREADS = 8   # 카테고리 내부 파일 복사 스레드
PRUNING_RATIO        = 0.5
SPLIT_RATIO          = 0.8
SPLIT_SEED           = 42
BALANCE_DEFECT_TYPES = True
LOCAL_TMP = Path("/content/tmp_stage6")

# Phase 2: Diffusion 출력 디렉터리 지정
STAGE4_SUBDIR = "stage4_diffusion_output"   # stage4_output (MPB) 과 병존
DATASET_GROUP = "aroma_diffusion"           # augmented_dataset 하위 그룹명

# 도메인 경로 구조 보존 (dataset_builder 도메인 추출 로직 호환)
_LOCAL_DOMAIN_PATH = {
    "isp":   LOCAL_TMP / "isp" / "unsupervised",
    "mvtec": LOCAL_TMP / "mvtec",
    "visa":  LOCAL_TMP / "visa",
}[DOMAIN_FILTER]

def parallel_copytree(src_dst_pairs, max_workers=4):
    """여러 (src, dst) 디렉터리 쌍을 ThreadPoolExecutor 로 병렬 복사."""
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
    defect_dir = Path(cat_dir) / "augmented_dataset" / DATASET_GROUP / "train" / "defect"
    if defect_dir.exists() and any(defect_dir.iterdir()):
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

        local_cat    = _LOCAL_DOMAIN_PATH / cat_dir.name
        local_imgdir = local_cat / "train" / "good"
        local_stage4 = local_cat / STAGE4_SUBDIR

        try:
            # 로컬 SSD 복사: stage4_diffusion_output + image_dir
            copy_pairs = []
            if (cat_dir / STAGE4_SUBDIR).exists():
                copy_pairs.append((cat_dir / STAGE4_SUBDIR, local_stage4))
            copy_pairs.append((Path(image_dir), local_imgdir))
            parallel_copytree(copy_pairs, max_workers=4)
            copy_sec = time.time() - t0

            # 로컬에서 aroma_diffusion 그룹 구성
            t1 = time.time()
            result = run_dataset_builder(
                cat_dir              = str(local_cat),
                image_dir            = str(local_imgdir),
                seed_dirs            = seed_dirs,       # baseline 미생성 → Drive 경로 그대로
                pruning_ratio        = PRUNING_RATIO,
                split_ratio          = SPLIT_RATIO,
                split_seed           = SPLIT_SEED,
                workers              = NUM_IO_THREADS,
                balance_defect_types = BALANCE_DEFECT_TYPES,
                stage4_subdir        = STAGE4_SUBDIR,
                dataset_group        = DATASET_GROUP,
            )
            build_sec = time.time() - t1

            # Drive 업로드: aroma_diffusion 그룹만
            t2 = time.time()
            drive_group = cat_dir / "augmented_dataset" / DATASET_GROUP
            if drive_group.exists():
                shutil.rmtree(str(drive_group))
            shutil.copytree(
                str(local_cat / "augmented_dataset" / DATASET_GROUP),
                str(drive_group),
            )
            upload_sec = time.time() - t2

            total_sec = time.time() - t0
            defect_count = result.get(DATASET_GROUP, {}).get("defect_count", "?")
            print(
                f"  ✓ {cat_dir.name}: 캐시 {copy_sec:.0f}s + 구성 {build_sec:.0f}s"
                f" + 업로드 {upload_sec:.0f}s = {total_sec:.0f}s  [defect={defect_count}]"
            )
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

        return cat_dir.name

    with tqdm(total=len(all_cats), desc=f"Stage6-Diffusion {LABEL}") as bar:
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

벤치마크 실행 전 각 카테고리의 `augmented_dataset` 구조를 점검한다.
Phase 2 에서는 `baseline` / `aroma_mpb` / `aroma_diffusion` 세 그룹이 존재해야 한다.

```python
import json, yaml, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")

REPO          = Path("/content/aroma")
CONFIG        = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG     = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())
DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD ("isp" / "mvtec" / "visa")
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 실행 (None → 전체 MVTec AD)
EXCLUDE       = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
GROUPS        = list(BENCH_CFG["dataset_groups"].keys())

seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))

    tag = "[EXCL]" if cat_dir.name in EXCLUDE else "     "
    aug = cat_dir / "augmented_dataset"
    if not aug.exists():
        print(f"{tag} {cat_dir.name}: augmented_dataset 없음 (Stage 6 미완료)")
        continue

    group_status = []
    for g in GROUPS:
        gdir = aug / g
        has_train = (gdir / "train" / "good").exists()
        has_defect = (gdir / "train" / "defect").exists()
        has_test = (gdir / "test").exists()
        n_good   = len(list((gdir / "train" / "good").glob("*")))   if has_train  else 0
        n_defect = len(list((gdir / "train" / "defect").glob("*"))) if has_defect else 0
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

`baseline/test` → `aroma_mpb/test`, `aroma_diffusion/test` symlink (실패 시 copytree 폴백).

```python
import json, sys, yaml
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import _ensure_test_dir

REPO          = Path("/content/aroma")
CONFIG        = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG     = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())
DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD ("isp" / "mvtec" / "visa")
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 실행 (None → 전체 MVTec AD)
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]  # → "MVTec AD"
EXCLUDE       = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))

NON_BASELINE_GROUPS = [g for g in BENCH_CFG["dataset_groups"] if g != "baseline"]

seen, tasks, failed_prep = set(), [], []
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen or cat_dir.name in EXCLUDE:
        continue
    seen.add(str(cat_dir))
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        continue
    for g in NON_BASELINE_GROUPS:
        group_test = cat_dir / "augmented_dataset" / g / "test"
        if not group_test.exists():
            tasks.append((cat_dir, g))

if not tasks:
    print(f"✓ {LABEL} test set 준비 완료 (모든 그룹에 이미 존재)")
else:
    print(f"{LABEL}: {len(tasks)}개 그룹 test set 준비")
    for cat_dir, group in tqdm(tasks, desc=f"test set 준비 {LABEL}"):
        try:
            result = _ensure_test_dir(str(cat_dir), group)
            method = "symlink" if result.is_symlink() else "copy"
            print(f"  ✓ {cat_dir.name}/{group}/test [{method}]")
        except Exception as e:
            failed_prep.append(f"{cat_dir.name}/{group}: {e}")
            print(f"  ✗ {cat_dir.name}/{group}: {e}")

    if failed_prep:
        print(f"\n✗ {len(failed_prep)}건 실패")
    else:
        print(f"\n✓ {LABEL} test set 준비 완료")
```

---

## Stage 7: 벤치마크

**Sentinel:** `{output_dir}/{cat_name}/{model}/{group}/experiment_meta.json`
**병렬:** GPU 점유 → 카테고리 순차 처리 (`resume=True` 로 중단 재개 가능)

> **모델 구성:** `yolo11` (YOLO11n-cls, 30 epoch) · `efficientdet_d0` (EfficientDet-D0, 30 epoch)
> **그룹 구성:** `baseline` · `aroma_mpb` (Phase 1 MPB) · `aroma_diffusion` (Phase 2 Diffusion)
> **핵심 비교 지표:** EfficientDet `aroma_mpb` vs `aroma_diffusion` — 블러 해소 검증
>
> **전제 조건:** `configs/benchmark_experiment_phase2.yaml` 에 세 그룹 정의 필요

```python
import json, sys, yaml
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import run_benchmark

REPO        = Path("/content/aroma")
CONFIG_PATH = str(REPO / "configs" / "benchmark_experiment_phase2.yaml")
CONFIG      = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG   = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())

DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD ("isp" / "mvtec" / "visa")
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 실행 (None → 전체 MVTec AD)
LABEL         = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]  # → "MVTec AD"
OUTPUT_DIR    = str(REPO / "outputs" / "benchmark_results_phase2")

EXCLUDE = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
MODELS  = list(BENCH_CFG["models"].keys())
GROUPS  = list(BENCH_CFG["dataset_groups"].keys())

seen = set()
all_cats, skip = [], 0
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    if cat_dir.name in EXCLUDE:
        continue
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        print(f"  ⚠ {cat_dir.name}: Stage 6 미완료 (baseline/test 없음) → skip")
        continue
    if not (cat_dir / "augmented_dataset" / "aroma_diffusion" / "train" / "defect").exists():
        print(f"  ⚠ {cat_dir.name}: aroma_diffusion 미완료 (Stage 4/6 Phase2 필요) → skip")
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

    for cat_dir in tqdm(all_cats, desc=f"Stage7-Phase2 {LABEL}"):
        try:
            results = run_benchmark(
                config_path = CONFIG_PATH,
                cat_dir     = str(cat_dir),
                resume      = True,
                output_dir  = OUTPUT_DIR,
            )
            for model_name, group_results in results.items():
                for group, val in group_results.items():
                    if isinstance(val, dict) and "error" in val:
                        failed.append({
                            "category": cat_dir.name,
                            "error": f"{model_name}/{group}: {val['detail'][:80]}",
                            "type": val["error"],
                        })
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

REPO          = Path("/content/aroma")
OUTPUT_DIR    = REPO / "outputs" / "benchmark_results_phase2"
DOMAIN_FILTER = "mvtec"   # Phase 2 실험 대상: MVTec AD (없으면 전체 출력)
CAT_ONLY      = ["bottle"]  # 1차 테스트: bottle만 (None → 전체)

CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG = __import__("yaml").safe_load(
    (REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())
MODELS = list(BENCH_CFG["models"].keys())
GROUPS = list(BENCH_CFG["dataset_groups"].keys())

if DOMAIN_FILTER:
    valid_cats = {Path(v.get("seed_dirs", [v["seed_dir"]])[0]).parents[1].name
                  for k, v in CONFIG.items()
                  if not k.startswith("_") and v["domain"] == DOMAIN_FILTER}
else:
    valid_cats = None
if CAT_ONLY:
    valid_cats = (valid_cats or set()) & set(CAT_ONLY)

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
                if "error" in m:
                    row[f"{model}/{group}"] = f"ERR:{m['error']}"
                else:
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
| Stage 4  | 카테고리 단위 순차 | 없음 | GPU 단일 점유; image_dir 로컬 SSD 캐시 + 업로드 |
| Stage 5  | `SEED_THREADS=4` | `IMG_WORKERS=-1` | I/O + CPU 혼합, seed 단위 독립 |
| Stage 6  | `CAT_THREADS=2` | `NUM_IO_THREADS=8` | I/O-bound Drive 복사, Thread 유리 |
| Stage 7  | 순차 (카테고리) | 없음 | GPU 단일 점유, `resume=True` 중단 재개 |

## Phase 2 실행 전 체크리스트

- [ ] `configs/benchmark_experiment_phase2.yaml` 에 `dataset_groups: baseline / aroma_mpb / aroma_diffusion` 정의
- [ ] Phase 1 `stage4_output/` (MPB 결과) 보존 확인
- [ ] `stage4_diffusion_synthesis.py` 에서 `run_synthesis_batch` 인터페이스 확인
- [ ] ControlNet 파인튜닝 완료 시 `CONTROLNET_MODEL` 경로 지정 (없으면 pretrained canny 사용)
- [ ] `stage6_dataset_builder.py` 에 `stage4_subdir` / `dataset_group` 파라미터 지원 확인
