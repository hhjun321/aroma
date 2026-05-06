# AROMA Phase 2 — 원클릭 실행

> **합성 방식:** SD Inpainting + ControlNet — Stage 4: `stage4_diffusion_synthesis.py`
> Phase 1 (MPB) 결과(`stage4_output/`)는 그대로 보존하고,
> Diffusion 합성 결과는 `stage4_diffusion_output/` 에 별도 저장한다.

## 실행 방법

1. **셀 0**: 환경 설치 (최초 1회만)
2. **셀 1**: 공통 설정 수정 (`DOMAIN_FILTER`, `CAT_ONLY`) → **여기서만 수정**
3. **셀 3**: Stage 4 — Diffusion 합성 (GPU)
4. **셀 4**: Stage 5 — 품질 점수 + Drive 체크포인트 업데이트
5. **셀 5**: Stage 6 — 데이터셋 구성 + Drive 업로드
6. **셀 7**: Stage 7 — 벤치마크
7. **런타임 → 모두 실행** 또는 셀 순서대로 실행

**재실행이 필요한 경우:**
- Stage 4-5-6 전체 재실행: **셀 3-0** (합성 이미지 + 데이터셋 전체 초기화)
- Stage 5만 재실행 (quality_scores.json 버그 수정 등): **셀 4-0** (quality_scores.json만 삭제, 합성 이미지 보존)
- Stage 6만 재실행 (데이터셋 구성 재실행): **셀 5-0** (augmented_dataset 그룹 삭제)
- Stage 7 재실행: **셀 6-0** (`RESET_GROUPS` 지정 후 experiment_meta.json 삭제)

## 설계 원칙

| 기존 (별도 셀) | 원클릭 (통합) |
|---|---|
| Stage 4: local 추론 → **Drive 업로드** | Stage 4: local 추론 → Drive 체크포인트 |
| Stage 5: **Drive에서 직접** 읽기 | Stage 5: local stage4 직접 읽기 |
| Stage 6: **Drive → local 재다운로드** → 빌드 → Drive | Stage 6: local stage4 그대로 사용 → Drive 업로드 |

Drive 왕복 (`stage4_diffusion_output`: 업로드→재다운로드)을 제거.
Stage 4 체크포인트 업로드는 GPU 결과 보존을 위해 유지 (재실행 시 재추론 방지).

## Skip 로직

- **셀 3 (Stage 4):** Drive `stage4_diffusion_output/{seed}/defect/*.png` 존재 → seed skip (Drive 체크포인트 → 로컬 복사만)
- **셀 4 (Stage 5):** Drive `stage4_diffusion_output/{seed}/quality_scores.json` 존재 → 카테고리 전체 skip
- **셀 5 (Stage 6):** `augmented_dataset/aroma_diffusion/train/defect` 존재 → 카테고리 전체 skip
- **셀 7 (Stage 7):** `experiment_meta.json` 존재 → 모델/그룹 skip (`resume=True`)

---

## 셀 0: 환경 설치 (최초 1회)

```python
!pip install -q ultralytics effdet
```

---

## 셀 1: 공통 설정

```python
import json, sys, yaml
from pathlib import Path

sys.path.insert(0, "/content/aroma")
REPO      = Path("/content/aroma")
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())
CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

# ── 여기서만 수정 ────────────────────────────────────────────────────
DOMAIN_FILTER = "mvtec"     # "isp" / "mvtec" / "visa"
CAT_ONLY      = ["bottle"]  # 특정 카테고리만 실행, None → 전체

# Diffusion 파라미터 (CASDA 검증값)
CONTROLNET_MODEL    = None   # None → pretrained lllyasviel/sd-controlnet-canny
                             # 파인튜닝 모델 경로: "/content/drive/.../controlnet_output"
DEVICE              = "cuda"
RESOLUTION          = 512
NUM_INFERENCE_STEPS = 30
STRENGTH            = 0.7
GUIDANCE_SCALE      = 7.5
CONDITIONING_SCALE  = 0.7   # 학습 1.0 → 추론 0.7 (아티팩트 방지)
DIFFUSION_SEED      = 42

# Stage 5 병렬
SEED_THREADS = 4    # seed 단위 동시 처리 (ThreadPoolExecutor)
IMG_WORKERS  = 0    # 이미지 단위: 반드시 0(순차) — ThreadPoolExecutor 내부에서
                    # ProcessPoolExecutor를 추가 생성하면 Linux fork 시 deadlock 발생
                    # (다른 스레드가 보유한 C 레벨 락이 자식 프로세스에 잠긴 채 복사됨)

# Stage 6 설정
PRUNING_RATIO        = 0.5
SPLIT_RATIO          = 0.8
SPLIT_SEED           = 42
BALANCE_DEFECT_TYPES = True
NUM_IO_THREADS       = 8
STAGE4_SUBDIR        = "stage4_diffusion_output"
DATASET_GROUP        = "aroma_diffusion"

# 로컬 SSD 루트 (Stage 4-6 통합 캐시)
LOCAL_TMP = Path("/content/tmp_phase2")

# Stage 7 선택 실행 (None → 전체)
STAGE7_MODELS = None   # 예: ["yolo11"] / ["efficientdet_d0"] / None → 전체
STAGE7_GROUPS = None   # 예: ["baseline", "aroma_diffusion"] / ["aroma_ratio_50"] / None → 전체
# ─────────────────────────────────────────────────────────────────────

# YAML에서 자동 로드 (수정 불필요)
MAX_IMAGES_PER_SEED = BENCH_CFG.get("synthesis", {}).get("max_images_per_seed", 50)
SEED_RATIO          = BENCH_CFG.get("synthesis", {}).get("seed_ratio")  # None → 전체
_ar_cfg             = BENCH_CFG.get("aroma_ratio", {})
AROMA_RATIOS        = _ar_cfg.get("ratios", []) if _ar_cfg.get("enabled") else []
AROMA_RATIO_GROUPS  = [f"aroma_ratio_{int(r * 100)}" for r in AROMA_RATIOS]

LABEL = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]
_LOCAL_DOMAIN_PATH = {
    "isp":   LOCAL_TMP / "isp" / "unsupervised",
    "mvtec": LOCAL_TMP / "mvtec",
    "visa":  LOCAL_TMP / "visa",
}[DOMAIN_FILTER]

print(f"설정 완료: {LABEL} / {'전체' if not CAT_ONLY else CAT_ONLY}")
```

---

## 셀 2: 진행 상황 확인 (선택)

```python
import sys, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/content/aroma")
from scripts.check_progress import check_category, print_report

tasks = [(k, v, None) for k, v in CONFIG.items() if not k.startswith("_")]

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(lambda t: check_category(*t), tasks))

print_report(results)
```

---

## 셀 3-0: 재실행 전 데이터 초기화 (선택)

> 특정 카테고리의 Stage 4-5-6 결과를 삭제하고 처음부터 재실행할 때 사용.
> `stage4_diffusion_output/` (Drive 체크포인트) 와 `augmented_dataset/aroma_diffusion/` (Stage 6 출력) 을 제거.

```python
import shutil, json, yaml
from pathlib import Path

REPO      = Path("/content/aroma")
CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())

DOMAIN_FILTER = "mvtec"     # 셀 1과 동일하게 맞출 것
CAT_ONLY      = ["bottle"]  # 초기화할 카테고리

_ar = BENCH_CFG.get("aroma_ratio", {})
AROMA_RATIO_GROUPS_ = (
    [f"aroma_ratio_{int(r*100)}" for r in _ar.get("ratios", [])]
    if _ar.get("enabled") else []
)

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

    targets = [
        cat_dir / "stage4_diffusion_output",
        cat_dir / "augmented_dataset" / "aroma_diffusion",
    ] + [cat_dir / "augmented_dataset" / g for g in AROMA_RATIO_GROUPS_]
    for t in targets:
        if t.exists():
            shutil.rmtree(str(t))
            print(f"  삭제: {t}")
        else:
            print(f"  없음 (skip): {t}")

print("초기화 완료 → 셀 3 재실행 가능")
```

---

## 셀 3: Stage 4 — Diffusion 합성

> **로컬 SSD 경로 구조** (`dataset_builder` 도메인 추출 로직 호환)
> ```
> /content/tmp_phase2/{domain}/{cat}/
>   ├── train/good/               ← image_dir 복사
>   ├── stage1b_output/           ← Drive에서 복사 (추론 중 Drive 읽기 제거)
>   └── stage4_diffusion_output/  ← 합성 출력 → Drive 체크포인트
> ```
>
> **skip:** Drive `stage4_diffusion_output/{seed}/defect/*.png` 전체 존재 → 로컬 복사만

```python
import random as _random
import shutil, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from stage4_diffusion_synthesis import run_synthesis_batch


def _parallel_copytree(src_dst_pairs, max_workers=4):
    def _copy(pair):
        src, dst = Path(pair[0]), Path(pair[1])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_copy, src_dst_pairs))


# ── 카테고리 목록 구성 ─────────────────────────────────────────────────
cat_map: dict = {}
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir_path = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]  # "visa_pcb" → "pcb"
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    cat_map.setdefault(str(cat_dir_path), entry["image_dir"])

# skip: Drive 체크포인트 내 모든 seed 완료 여부
all_cats, skip = [], 0
for cat_dir_str, image_dir in cat_map.items():
    cat_dir      = Path(cat_dir_str)
    drive_stage4 = cat_dir / STAGE4_SUBDIR
    stage3_dir   = cat_dir / "stage3_output"
    if not stage3_dir.exists():
        continue
    _all_seeds = [d for d in sorted(stage3_dir.iterdir())
                  if d.is_dir() and (d / "placement_map.json").exists()]
    if SEED_RATIO is not None:
        _k = max(1, int(len(_all_seeds) * SEED_RATIO))
        _all_seeds = _random.Random(DIFFUSION_SEED).sample(_all_seeds, min(_k, len(_all_seeds)))
    seeds_todo = [d for d in _all_seeds
                  if not any((drive_stage4 / d.name / "defect").glob("*.png"))]
    if not seeds_todo:
        skip += 1
    else:
        all_cats.append((cat_dir, image_dir))

if not all_cats:
    print(f"✓ {LABEL} Stage 4 완료 (skip {skip}건)")
else:
    print(f"{LABEL} Stage 4: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []

    for cat_dir, image_dir in tqdm(all_cats, desc=f"Stage4 {LABEL}"):
        local_cat     = _LOCAL_DOMAIN_PATH / cat_dir.name
        local_imgdir  = local_cat / "train" / "good"
        local_stage1b = local_cat / "stage1b_output"
        local_stage4  = local_cat / STAGE4_SUBDIR
        drive_stage4  = cat_dir / STAGE4_SUBDIR

        t_total = time.time()
        try:
            # ── 로컬 SSD 복사: image_dir + stage1b_output + Drive 체크포인트 ──
            t = time.time()
            copy_pairs = [
                (image_dir,                  local_imgdir),
                (cat_dir / "stage1b_output", local_stage1b),
            ]
            if drive_stage4.exists():
                copy_pairs.append((drive_stage4, local_stage4))
            _parallel_copytree(copy_pairs, max_workers=4)
            copy_sec = time.time() - t

            # ── 미완료 seed 목록 (로컬 기준) ──────────────────────────────
            stage3_dir = cat_dir / "stage3_output"
            _all_seeds_local = [d for d in sorted(stage3_dir.iterdir())
                                 if d.is_dir() and (d / "placement_map.json").exists()]
            if SEED_RATIO is not None:
                _k = max(1, int(len(_all_seeds_local) * SEED_RATIO))
                _all_seeds_local = _random.Random(DIFFUSION_SEED).sample(
                    _all_seeds_local, min(_k, len(_all_seeds_local)))
            seed_pm_pairs = [
                (d.name, str(d / "placement_map.json"))
                for d in _all_seeds_local
                if not any((local_stage4 / d.name / "defect").glob("*.png"))
            ]

            if not seed_pm_pairs:
                print(f"  ✓ {cat_dir.name}: Drive캐시 사용 ({copy_sec:.0f}s)")
            else:
                t_infer = time.time()
                run_synthesis_batch(
                    image_dir            = str(local_imgdir),
                    seed_placement_maps  = seed_pm_pairs,
                    output_root          = str(local_stage4),
                    cat_dir              = str(local_cat),   # 로컬 stage1b_output 사용
                    format               = "cls",
                    controlnet_model     = CONTROLNET_MODEL,
                    device               = DEVICE,
                    resolution           = RESOLUTION,
                    num_inference_steps  = NUM_INFERENCE_STEPS,
                    strength             = STRENGTH,
                    guidance_scale       = GUIDANCE_SCALE,
                    conditioning_scale   = CONDITIONING_SCALE,
                    seed                 = DIFFUSION_SEED,
                    max_images_per_seed  = MAX_IMAGES_PER_SEED,
                )
                infer_sec = time.time() - t_infer

                # ── Drive 체크포인트: 합성 이미지 보존 ──────────────────────
                t_ckpt = time.time()
                shutil.copytree(str(local_stage4), str(drive_stage4), dirs_exist_ok=True)
                ckpt_sec = time.time() - t_ckpt

                print(
                    f"  ✓ {cat_dir.name} [{time.time()-t_total:.0f}s 합계]  "
                    f"복사:{copy_sec:.0f}s  추론:{infer_sec:.0f}s  체크포인트:{ckpt_sec:.0f}s"
                )

        except Exception as e:
            import traceback
            failed.append({"category": cat_dir.name, "error": str(e), "type": type(e).__name__})
            traceback.print_exc()
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

    print("\n" + (f"✓ {LABEL} Stage 4 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## 셀 4-0: Stage 5 초기화 (선택)

> `quality_scores.json`만 삭제. Stage 4 합성 이미지는 보존.
> `_score_sharpness` 버그 수정 후 재계산 시 사용.

```python
import json, yaml
from pathlib import Path

REPO      = Path("/content/aroma")
CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

# ── 초기화 범위 설정 ──────────────────────────────────────────────────
DOMAIN_FILTER = "mvtec"     # 셀 1과 동일하게 맞출 것
CAT_ONLY      = ["bottle"]  # None → 전체
STAGE4_SUBDIR = "stage4_diffusion_output"
# ─────────────────────────────────────────────────────────────────────

deleted, missing = 0, 0
seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))

    stage4_root = cat_dir / STAGE4_SUBDIR
    if not stage4_root.exists():
        print(f"  없음 (skip): {stage4_root}")
        continue
    for seed_dir in sorted(stage4_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        qs = seed_dir / "quality_scores.json"
        if qs.exists():
            qs.unlink()
            print(f"  삭제: {cat_dir.name}/{STAGE4_SUBDIR}/{seed_dir.name}/quality_scores.json")
            deleted += 1
        else:
            missing += 1

print(f"\n초기화 완료: 삭제 {deleted}개, 없음(skip) {missing}개 → 셀 4 재실행 가능")
```

---

## 셀 4: Stage 5 — 품질 점수

> **skip:** Drive `stage4_diffusion_output/{seed}/quality_scores.json` 전체 존재 → 카테고리 skip
>
> **Drive 직접 처리:** 이미지 읽기·`quality_scores.json` 쓰기 모두 Drive 경로에서 직접 수행.
> 이미지를 로컬로 복사하지 않으며, JSON 파일만 Drive에 생성됨.

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from stage5_quality_scoring import run_quality_scoring


# ── 카테고리 목록 구성 ─────────────────────────────────────────────────
cat_map: dict = {}
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir_path = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]  # "visa_pcb" → "pcb"
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    cat_map.setdefault(str(cat_dir_path), entry["image_dir"])

# skip: Drive의 모든 seed에 quality_scores.json 존재
all_cats, skip = [], 0
for cat_dir_str in cat_map:
    cat_dir      = Path(cat_dir_str)
    drive_stage4 = cat_dir / STAGE4_SUBDIR
    if not drive_stage4.exists():
        continue
    seed_dirs_in_drive = [d for d in drive_stage4.iterdir() if d.is_dir()]
    if seed_dirs_in_drive and all(
        (d / "quality_scores.json").exists() for d in seed_dirs_in_drive
    ):
        skip += 1
    else:
        all_cats.append(cat_dir)

if not all_cats:
    print(f"✓ {LABEL} Stage 5 완료 (skip {skip}건)")
else:
    print(f"{LABEL} Stage 5: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []

    for cat_dir in tqdm(all_cats, desc=f"Stage5 {LABEL}"):
        drive_stage4 = cat_dir / STAGE4_SUBDIR

        t_total = time.time()
        try:
            # ── quality_scores.json 없는 seed만 Drive에서 직접 처리 ────────
            # 이미지 읽기·JSON 쓰기 모두 Drive 경로 직접 사용 (로컬 복사 없음)
            seeds_s5 = [
                d.name for d in sorted(drive_stage4.iterdir())
                if d.is_dir() and not (d / "quality_scores.json").exists()
            ]

            # IMG_WORKERS=0 필수 — ThreadPoolExecutor 스레드 내에서 ProcessPoolExecutor를
            # 추가 생성하면 Linux fork 시 deadlock 발생 (셀 1 주석 참고)
            def _run_s5(seed_id, _stage4=drive_stage4):
                run_quality_scoring(
                    stage4_seed_dir = str(_stage4 / seed_id),
                    workers         = IMG_WORKERS,  # 반드시 0
                )
                return seed_id

            t_score = time.time()
            if seeds_s5:
                with ThreadPoolExecutor(max_workers=SEED_THREADS) as ex:
                    futs = {ex.submit(_run_s5, s): s for s in seeds_s5}
                    for fut in as_completed(futs):
                        fut.result()
            score_sec = time.time() - t_score

            print(f"  ✓ {cat_dir.name} [{time.time()-t_total:.0f}s]  점수:{score_sec:.0f}s  ({len(seeds_s5)}seed)")

        except Exception as e:
            import traceback
            failed.append({"category": cat_dir.name, "error": str(e), "type": type(e).__name__})
            traceback.print_exc()

    print("\n" + (f"✓ {LABEL} Stage 5 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## 셀 5-0: Stage 6 초기화 (선택)

> `augmented_dataset/{group}` 삭제. Stage 4-5 결과는 보존.
> 기본값은 `aroma_diffusion` + `aroma_ratio_*` 삭제 (baseline 제외).
> `RESET_BASELINE=True` 로 test set 포함 baseline도 삭제 가능.

```python
import shutil, json, yaml
from pathlib import Path

REPO      = Path("/content/aroma")
CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())

# ── 초기화 범위 설정 ──────────────────────────────────────────────────
DOMAIN_FILTER  = "mvtec"     # 셀 1과 동일하게 맞출 것
CAT_ONLY       = ["bottle"]  # None → 전체
RESET_GROUPS   = None        # None → aroma_diffusion + aroma_ratio_* 전체 (baseline 제외)
RESET_BASELINE = False       # True → baseline도 삭제 (test set 포함)
# ─────────────────────────────────────────────────────────────────────

_ar = BENCH_CFG.get("aroma_ratio", {})
_ratio_groups = (
    [f"aroma_ratio_{int(r*100)}" for r in _ar.get("ratios", [])]
    if _ar.get("enabled") else []
)
groups_to_reset = RESET_GROUPS or (["aroma_diffusion"] + _ratio_groups)
if RESET_BASELINE:
    groups_to_reset = ["baseline"] + groups_to_reset

seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))

    for g in groups_to_reset:
        target = cat_dir / "augmented_dataset" / g
        if target.exists():
            shutil.rmtree(str(target))
            print(f"  삭제: {cat_dir.name}/augmented_dataset/{g}")
        else:
            print(f"  없음 (skip): {cat_dir.name}/augmented_dataset/{g}")

print("\n초기화 완료 → 셀 5 재실행 가능")
```

---

## 셀 5: Stage 6 — 데이터셋 구성

> **skip:** `augmented_dataset/aroma_diffusion/train/defect` 존재 → 카테고리 skip
>
> **병렬:** `CAT_THREADS=2` 카테고리 동시 처리 (Stage 4 종료 후 순수 I/O-bound)
>
> **선택적 복사:** `_collect_defect_paths`로 Drive의 `quality_scores.json`만 읽어 선택 대상 결정 →
> 선택된 PNG만 로컬 SSD로 복사. 전체 `stage4_diffusion_output` 복사 대비 파일 수 대폭 절감.
> `preselected_defect_pairs_full` 전달로 내부 재선택 없이 직접 빌드.

```python
import shutil, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from stage6_dataset_builder import run_dataset_builder
from utils.dataset_builder import _collect_defect_paths


CAT_THREADS = 2   # 카테고리 동시 처리 (Drive 동시 쓰기 안정성 고려)

# augmentation_ratio_by_domain: benchmark_experiment_phase2.yaml 에서 로드
AUGMENTATION_RATIO_BY_DOMAIN = BENCH_CFG.get("dataset", {}).get("augmentation_ratio_by_domain")
_ratio_full = (AUGMENTATION_RATIO_BY_DOMAIN or {}).get(DOMAIN_FILTER, {}).get("full")


# ── 카테고리 목록 구성 ─────────────────────────────────────────────────
cat_map: dict = {}
cat_seed_dirs: dict = defaultdict(list)
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir_path = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]  # "visa_pcb" → "pcb"
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    cat_map.setdefault(str(cat_dir_path), entry["image_dir"])
    cat_seed_dirs[str(cat_dir_path)].extend(seed_dirs_list)

all_cats, skip = [], 0
for cat_dir_str, image_dir in cat_map.items():
    cat_dir    = Path(cat_dir_str)
    groups_all = [DATASET_GROUP] + AROMA_RATIO_GROUPS
    baseline_test = cat_dir / "augmented_dataset" / "baseline" / "test"
    all_done   = (
        baseline_test.exists() and any(baseline_test.iterdir())
        and all(
            (cat_dir / "augmented_dataset" / g / "train" / "defect").exists()
            and any((cat_dir / "augmented_dataset" / g / "train" / "defect").iterdir())
            for g in groups_all
        )
    )
    if all_done:
        skip += 1
    else:
        all_cats.append((cat_dir, image_dir, cat_seed_dirs[cat_dir_str]))

if not all_cats:
    print(f"✓ {LABEL} Stage 6 완료 (skip {skip}건)")
else:
    print(f"{LABEL} Stage 6: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []

    def _run_stage6(args):
        cat_dir, image_dir, seed_dirs = args
        cat_name     = cat_dir.name
        local_cat    = _LOCAL_DOMAIN_PATH / cat_name
        local_imgdir = local_cat / "train" / "good"

        t_total = time.time()
        try:
            # ── 1단계: good 이미지 수 확인 ───────────────────────────────
            print(f"  [{cat_name}] 1/6 good 이미지 수 확인...")
            _exts = ("*.png", "*.jpg", "*.jpeg")
            good_count_drive = sum(len(list(Path(image_dir).glob(e))) for e in _exts)

            # ── 2단계: 각 그룹의 선택 대상 결정 (Drive quality_scores 기반) ─
            n_groups = 1 + len(AROMA_RATIOS)
            print(f"  [{cat_name}] 2/6 defect 경로 수집 ({n_groups}개 그룹, good={good_count_drive})...")
            pairs_diffusion = _collect_defect_paths(
                str(cat_dir),
                pruning_ratio        = PRUNING_RATIO,
                augmentation_ratio   = _ratio_full,
                good_count           = good_count_drive,
                balance_defect_types = BALANCE_DEFECT_TYPES,
                stage4_subdir        = STAGE4_SUBDIR,
            )
            aroma_ratio_pairs: dict = {}
            for ratio in AROMA_RATIOS:
                group_key = f"aroma_ratio_{int(ratio * 100)}"
                aug_ratio = ratio / (1 - ratio)
                aroma_ratio_pairs[group_key] = _collect_defect_paths(
                    str(cat_dir),
                    pruning_ratio        = PRUNING_RATIO,
                    augmentation_ratio   = aug_ratio,
                    good_count           = good_count_drive,
                    balance_defect_types = True,
                    stage4_subdir        = STAGE4_SUBDIR,
                )

            # ── 3단계: 필요한 PNG 합집합 + good 이미지 → 로컬 SSD 병렬 복사 ─
            t = time.time()
            all_src = {src for src, _ in pairs_diffusion}
            for pairs in aroma_ratio_pairs.values():
                all_src.update(src for src, _ in pairs)
            file_tasks = [
                (src, str(local_cat / Path(src).relative_to(cat_dir)))
                for src in sorted(all_src)
            ]
            print(f"  [{cat_name}] 3/6 defect {len(file_tasks)}개 + good {good_count_drive}장 병렬 복사 중...")

            def _copy_file(src_dst):
                src, dst = Path(src_dst[0]), Path(src_dst[1])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

            def _copy_good():
                shutil.copytree(str(image_dir), str(local_imgdir), dirs_exist_ok=True)

            # good 복사와 defect PNG 복사를 동시에 실행
            with ThreadPoolExecutor(max_workers=NUM_IO_THREADS + 1) as ex:
                good_fut = ex.submit(_copy_good)
                list(ex.map(_copy_file, file_tasks))
                good_fut.result()
            copy_sec = time.time() - t

            # ── 4단계: 경로 리매핑 (Drive → 로컬 SSD) ────────────────────
            print(f"  [{cat_name}] 4/6 경로 리매핑...")
            def _remap(pairs):
                return [
                    (str(local_cat / Path(src).relative_to(cat_dir)), dst_name)
                    for src, dst_name in pairs
                ]
            pairs_diffusion_local   = _remap(pairs_diffusion)
            aroma_ratio_pairs_local = {k: _remap(v) for k, v in aroma_ratio_pairs.items()}

            # ── 5단계: 데이터셋 구성 ─────────────────────────────────────
            t = time.time()
            # baseline: dataset_group 지정 시 자동 생성되지 않으므로 명시적 빌드
            baseline_local = local_cat / "augmented_dataset" / "baseline"
            baseline_rebuilt = not (baseline_local / "test").exists()
            if baseline_rebuilt:
                n_build = 2 + len(aroma_ratio_pairs_local)  # baseline 포함
                print(f"  [{cat_name}] 5/6 데이터셋 구성 중 ({n_build}개 그룹, baseline 포함)...")
                run_dataset_builder(
                    cat_dir       = str(local_cat),
                    image_dir     = str(local_imgdir),
                    seed_dirs     = seed_dirs,
                    split_ratio   = SPLIT_RATIO,
                    split_seed    = SPLIT_SEED,
                    workers       = NUM_IO_THREADS,
                    stage4_subdir = STAGE4_SUBDIR,
                    groups        = ["baseline"],
                )
            else:
                n_build = 1 + len(aroma_ratio_pairs_local)
                print(f"  [{cat_name}] 5/6 데이터셋 구성 중 ({n_build}개 그룹)...")
            result = run_dataset_builder(
                cat_dir                       = str(local_cat),
                image_dir                     = str(local_imgdir),
                seed_dirs                     = seed_dirs,
                pruning_ratio                 = PRUNING_RATIO,
                augmentation_ratio_by_domain  = AUGMENTATION_RATIO_BY_DOMAIN,
                split_ratio                   = SPLIT_RATIO,
                split_seed                    = SPLIT_SEED,
                workers                       = NUM_IO_THREADS,
                balance_defect_types          = BALANCE_DEFECT_TYPES,
                stage4_subdir                 = STAGE4_SUBDIR,
                dataset_group                 = DATASET_GROUP,
                preselected_defect_pairs_full = pairs_diffusion_local,
            )
            for group_key, pairs_local in aroma_ratio_pairs_local.items():
                ratio     = int(group_key.split("_")[-1]) / 100
                aug_ratio = ratio / (1 - ratio)
                run_dataset_builder(
                    cat_dir                       = str(local_cat),
                    image_dir                     = str(local_imgdir),
                    seed_dirs                     = seed_dirs,
                    pruning_ratio                 = PRUNING_RATIO,
                    augmentation_ratio_full       = aug_ratio,
                    split_ratio                   = SPLIT_RATIO,
                    split_seed                    = SPLIT_SEED,
                    workers                       = NUM_IO_THREADS,
                    balance_defect_types          = True,
                    stage4_subdir                 = STAGE4_SUBDIR,
                    dataset_group                 = group_key,
                    preselected_defect_pairs_full = pairs_local,
                )
            build_sec = time.time() - t

            # ── 6단계: Drive 업로드 (baseline + aroma_diffusion + aroma_ratio_* 그룹, 병렬) ─
            t = time.time()
            grp_list = ([("baseline")] if baseline_rebuilt else []) + \
                       [DATASET_GROUP] + list(aroma_ratio_pairs_local.keys())
            print(f"  [{cat_name}] 6/6 Drive 업로드 중 ({len(grp_list)}개 그룹 병렬)...")

            def _upload_grp(grp):
                drive_grp = cat_dir / "augmented_dataset" / grp
                if drive_grp.exists():
                    shutil.rmtree(str(drive_grp))
                shutil.copytree(str(local_cat / "augmented_dataset" / grp), str(drive_grp))

            with ThreadPoolExecutor(max_workers=len(grp_list)) as ex:
                list(ex.map(_upload_grp, grp_list))
            up_sec = time.time() - t

            defect_count = result.get(DATASET_GROUP, {}).get("defect_count", "?")
            ratio_counts = {k: len(v) for k, v in aroma_ratio_pairs_local.items()}
            print(
                f"  ✓ [{cat_name}] 완료 [{time.time()-t_total:.0f}s]  "
                f"복사:{copy_sec:.0f}s  구성:{build_sec:.0f}s  업로드:{up_sec:.0f}s  "
                f"diffusion={defect_count}  ratio={ratio_counts}"
            )
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

        return cat_name

    with tqdm(total=len(all_cats), desc=f"Stage6 {LABEL}") as bar:
        with ThreadPoolExecutor(max_workers=CAT_THREADS) as ex:
            futs = {ex.submit(_run_stage6, t): t for t in all_cats}
            for fut in as_completed(futs):
                cat_dir, *_ = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append({"category": cat_dir.name,
                                   "error": str(e), "type": type(e).__name__})
                bar.update(1)

    print("\n" + (f"✓ {LABEL} Stage 6 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## 셀 6: Stage 7 데이터 구조 확인 (선택)

> 벤치마크 실행 전 `baseline` / `aroma_mpb` / `aroma_diffusion` 세 그룹 존재 여부 점검.

```python
import json, yaml, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
BENCH_CFG = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text()
)
EXCLUDE = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
_ar = BENCH_CFG.get("aroma_ratio", {})
GROUPS  = list(BENCH_CFG["dataset_groups"].keys()) + (
    [f"aroma_ratio_{int(r*100)}" for r in _ar.get("ratios", [])] if _ar.get("enabled") else []
)

seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]  # "visa_pcb" → "pcb"
    if CAT_ONLY and cat_name not in CAT_ONLY:
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
        n_good   = len(list((gdir / "train" / "good").glob("*")))   if (gdir / "train" / "good").exists()   else 0
        n_defect = len(list((gdir / "train" / "defect").glob("*"))) if (gdir / "train" / "defect").exists() else 0
        has_test = (gdir / "test").exists()
        n_test   = sum(len(list(d.glob("*"))) for d in (gdir / "test").iterdir() if d.is_dir()) if has_test else 0
        group_status.append(f"{g}: good={n_good} defect={n_defect} test={n_test if has_test else '(auto)'}")
    print(f"{tag} {cat_dir.name}")
    for s in group_status:
        print(f"       {s}")
```

---

## 셀 6-1: Stage 4 합성 이미지 수 확인 (선택)

> `aroma_ratio_*` 그룹이 동일한 defect 수를 보이면, 합성 풀이 요청량보다 적어 상한에 걸리거나
> pruning_ratio 적용 후 모두 같은 수가 선택된 것.
> 실제 학습 good 수(baseline/train/good) 기준으로 ratio별 필요량과 공급량을 비교한다.

```python
import yaml, json
from pathlib import Path

REPO      = Path("/content/aroma")
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())
CONFIG    = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))

# ── 설정 ─────────────────────────────────────────────────────────────
DOMAIN_FILTER   = "visa"        # 확인할 도메인
CAT_ONLY        = ["pcb4"]      # None → 전체
STAGE4_SUBDIR   = "stage4_diffusion_output"
# ─────────────────────────────────────────────────────────────────────

_ar  = BENCH_CFG.get("aroma_ratio", {})
RATIOS        = _ar.get("ratios", []) if _ar.get("enabled") else []
PRUNING_RATIO = BENCH_CFG["dataset"].get("pruning_ratio", 1.0)

seen = set()
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))

    # good 수: Stage 6의 good_count_drive와 동일한 image_dir (split 전 원본)
    image_dir = Path(entry["image_dir"])
    _exts = ("*.png", "*.jpg", "*.jpeg")
    n_good = sum(len(list(image_dir.glob(e))) for e in _exts) if image_dir.exists() else 0

    # stage4 합성 이미지 수 (seed별 집계)
    stage4_root = cat_dir / STAGE4_SUBDIR
    seed_counts = {}
    if stage4_root.exists():
        for seed_dir in sorted(stage4_root.iterdir()):
            defect_dir = seed_dir / "defect"
            n = len(list(defect_dir.glob("*.png"))) if defect_dir.exists() else 0
            if n > 0:
                seed_counts[seed_dir.name] = n
    total_synth = sum(seed_counts.values())
    # pruning_ratio 적용 후 실제 선택 가능한 수
    after_pruning = int(total_synth * PRUNING_RATIO)

    print(f"\n{'='*60}")
    print(f"{cat_dir.name}  |  good={n_good}  |  합성 총계={total_synth}  |  pruning 후={after_pruning}  (pruning_ratio={PRUNING_RATIO})")
    print(f"  seed별: {seed_counts}")

    # ratio별 필요량 vs pruning 후 공급량
    if RATIOS and n_good > 0:
        print(f"\n  {'ratio':>6}  {'필요량(aug×good)':>18}  {'pruning 후 공급':>16}  {'상한 걸림':>10}")
        for r in RATIOS:
            aug_ratio = r / (1 - r)
            needed    = int(aug_ratio * n_good)
            capped    = "⚠ YES" if after_pruning < needed else "  no"
            print(f"  {r:>6.0%}  {needed:>18}  {after_pruning:>16}  {capped:>10}")
```

---

## 셀 6-0: Stage 7 초기화 (선택)

> 특정 카테고리·그룹의 벤치마크를 처음부터 재실행할 때 사용.
> Drive `DRIVE_OUTPUT_DIR/{cat}/{model}/{group}/experiment_meta.json` 을 삭제.
> `CAT_ONLY_RESET` / `RESET_GROUPS` 로 범위를 한정할 것.

```python
import yaml
from pathlib import Path

REPO      = Path("/content/aroma")
BENCH_CFG = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())

# ── 초기화 범위 설정 ──────────────────────────────────────────────────
CAT_ONLY_RESET = ["bottle"]          # None → 전체 카테고리
RESET_GROUPS   = ["aroma_diffusion"] # None → 전체 그룹 (baseline 포함 주의)
RESET_MODELS   = None                # None → 전체 모델
# ─────────────────────────────────────────────────────────────────────

DRIVE_OUTPUT_DIR_ = (
    BENCH_CFG["experiment"].get("results_dir")
    or "/content/drive/MyDrive/data/Aroma/benchmark_results_phase2"
)
_ar = BENCH_CFG.get("aroma_ratio", {})
_all_groups = list(BENCH_CFG["dataset_groups"].keys()) + (
    [f"aroma_ratio_{int(r*100)}" for r in _ar.get("ratios", [])] if _ar.get("enabled") else []
)
_models = RESET_MODELS or list(BENCH_CFG["models"].keys())
_groups = RESET_GROUPS if RESET_GROUPS is not None else _all_groups

deleted = 0
drive_root = Path(DRIVE_OUTPUT_DIR_)
for cat_dir in sorted(drive_root.iterdir()):
    if not cat_dir.is_dir():
        continue
    if CAT_ONLY_RESET and cat_dir.name not in CAT_ONLY_RESET:
        continue
    for model in _models:
        for group in _groups:
            meta = cat_dir / model / group / "experiment_meta.json"
            if meta.exists():
                meta.unlink()
                print(f"  삭제: {cat_dir.name}/{model}/{group}/experiment_meta.json")
                deleted += 1

print(f"\n초기화 완료 ({deleted}개 삭제) → 셀 7 재실행 가능")
```

---

## 셀 7: Stage 7 test set 준비 + 벤치마크 (로컬 SSD 캐시)

> **로컬 SSD 최적화:** `augmented_dataset/{group}/train/` + `baseline/test/` 를
> `/content/tmp_stage7` 에 복사 후 학습·평가. 30 epoch × 전체 이미지 반복 읽기가
> Drive FUSE가 아닌 로컬 NVMe에서 수행되어 학습 시간 단축.
>
> **resume 지원:** Drive의 기존 `experiment_meta.json` 을 local 에 먼저 복사하여
> `resume=True` 가 정상 동작.
>
> **Drive 쓰기 최소화:** 학습 중 쓰는 파일은 local 에만. 완료 후 meta JSON 만 업로드.
> `benchmark_results.json` / `benchmark_comparison.csv` 는 실험 1건 완료마다 Drive 에 직접 기록
> (`results_dir` 파라미터, yaml 설정값 사용).

```python
import json, shutil, sys, time, yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

sys.path.insert(0, "/content/aroma")
from stage7_benchmark import _ensure_test_dir, run_benchmark

CONFIG_PATH = str(REPO / "configs" / "benchmark_experiment_phase2.yaml")
BENCH_CFG   = yaml.safe_load((REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text())

EXCLUDE             = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
_ar = BENCH_CFG.get("aroma_ratio", {})
_all_models = list(BENCH_CFG["models"].keys())
_all_groups = list(BENCH_CFG["dataset_groups"].keys()) + (
    [f"aroma_ratio_{int(r*100)}" for r in _ar.get("ratios", [])] if _ar.get("enabled") else []
)
MODELS              = [m for m in _all_models if STAGE7_MODELS is None or m in STAGE7_MODELS]
GROUPS              = [g for g in _all_groups if STAGE7_GROUPS is None or g in STAGE7_GROUPS]
NON_BASELINE_GROUPS = [g for g in GROUPS if g != "baseline"]
print(f"모델: {MODELS}  /  그룹: {GROUPS}")

# Drive 저장 경로: experiment_meta.json + benchmark_results.json/csv 모두 여기에 저장
# yaml의 results_dir 값을 사용 (benchmark_experiment_phase2.yaml → experiment.results_dir)
DRIVE_OUTPUT_DIR = (
    BENCH_CFG["experiment"].get("results_dir")
    or "/content/drive/MyDrive/data/Aroma/benchmark_results_phase2"
)
print(f"Drive 저장 경로: {DRIVE_OUTPUT_DIR}")

LOCAL_TMP_S7 = Path("/content/tmp_stage7")
# {domain}/{cat} 구조: run_benchmark 내부 domain 감지 (pixel_auroc 여부) 에 사용
_LOCAL_S7_BASE = LOCAL_TMP_S7 / DOMAIN_FILTER


def _parallel_copytree_s7(src_dst_pairs, max_workers=4):
    def _copy(pair):
        src, dst = Path(pair[0]), Path(pair[1])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_copy, src_dst_pairs))


# ── Drive: test set symlink 준비 (O(1), Drive 영속 보존) ────────────────
seen, test_tasks, test_failed = set(), [], []
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]  # "visa_pcb" → "pcb"
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen or cat_dir.name in EXCLUDE:
        continue
    seen.add(str(cat_dir))
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        continue
    for g in NON_BASELINE_GROUPS:
        if not (cat_dir / "augmented_dataset" / g / "test").exists():
            test_tasks.append((cat_dir, g))

if test_tasks:
    print(f"Drive test set 준비: {len(test_tasks)}개")
    for cat_dir, group in test_tasks:
        try:
            result = _ensure_test_dir(str(cat_dir), group)
            method = "symlink" if result.is_symlink() else "copy"
            print(f"  ✓ {cat_dir.name}/{group} [{method}]")
        except Exception as e:
            test_failed.append(f"{cat_dir.name}/{group}: {e}")
            print(f"  ✗ {cat_dir.name}/{group}: {e}")
else:
    print("✓ Drive test set 준비 완료")

# ── 벤치마크 대상 카테고리 목록 ─────────────────────────────────────────
seen = set()
all_bench, skip_bench = [], 0
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir = Path(seed_dirs_list[0]).parents[1]
    cat_name = key[len(entry["domain"]) + 1:]  # "visa_pcb" → "pcb"
    if CAT_ONLY and cat_name not in CAT_ONLY:
        continue
    if str(cat_dir) in seen:
        continue
    seen.add(str(cat_dir))
    if cat_dir.name in EXCLUDE:
        continue
    if not (cat_dir / "augmented_dataset" / "baseline" / "test").exists():
        print(f"  ⚠ {cat_dir.name}: baseline/test 없음 → skip")
        continue
    if not (cat_dir / "augmented_dataset" / "aroma_diffusion" / "train" / "defect").exists():
        print(f"  ⚠ {cat_dir.name}: aroma_diffusion 미완료 → skip")
        continue
    out_root = Path(DRIVE_OUTPUT_DIR) / cat_dir.name
    # 데이터가 실제로 존재하는 그룹만 skip 판정 대상으로 한정
    # (aroma_mpb 등 미완료 그룹의 오래된 experiment_meta.json이 skip을 오유발하는 것을 방지)
    present_groups = [
        g for g in GROUPS
        if (cat_dir / "augmented_dataset" / g / "train" / "defect").exists()
    ]
    if present_groups and all(
        (out_root / m / g / "experiment_meta.json").exists()
        for m in MODELS for g in present_groups
    ):
        skip_bench += 1
    else:
        all_bench.append(cat_dir)

if not all_bench:
    print(f"✓ {LABEL} 벤치마크 완료 (skip {skip_bench}건)")
else:
    print(f"\n{LABEL} 벤치마크: {len(all_bench)} categories (skip {skip_bench}건)")
    print(f"  모델: {MODELS}  /  그룹: {GROUPS}")
    bench_failed = []

    for cat_dir in tqdm(all_bench, desc=f"Stage7 {LABEL}"):
        local_cat  = _LOCAL_S7_BASE / cat_dir.name              # 학습 데이터 캐시
        local_out  = LOCAL_TMP_S7 / "results" / cat_dir.name  # 실험 메타 로컬 임시 저장
        drive_out  = Path(DRIVE_OUTPUT_DIR) / cat_dir.name    # Drive 최종 저장

        t_total = time.time()
        try:
            # ── 학습 데이터 로컬 복사 ────────────────────────────────
            # {group}/train/ (학습 이미지) + baseline/test/ (평가 이미지)
            # test/{group}/ symlink 는 run_benchmark 내부 _ensure_test_dir 이 자동 생성
            t = time.time()
            copy_pairs = [
                (cat_dir / "augmented_dataset" / "baseline" / "test",
                 local_cat / "augmented_dataset" / "baseline" / "test"),
            ]
            for g in GROUPS:
                src_train = cat_dir / "augmented_dataset" / g / "train"
                if src_train.exists():
                    copy_pairs.append((src_train, local_cat / "augmented_dataset" / g / "train"))
            _parallel_copytree_s7(copy_pairs, max_workers=4)
            copy_sec = time.time() - t

            # ── Drive 기존 실험 meta 복사 (resume=True 지원) ──────────
            if drive_out.exists():
                shutil.copytree(str(drive_out), str(local_out), dirs_exist_ok=True)

            # ── 벤치마크 실행 (로컬 경로) ────────────────────────────
            # cat_dir=local_cat → 학습·평가 이미지를 로컬 NVMe에서 읽음
            # output_dir=local_out → experiment_meta.json 등 로컬 임시 기록
            # results_dir=DRIVE_OUTPUT_DIR → benchmark_results.json/csv 실험 1건마다 Drive 직접 기록
            t = time.time()
            results = run_benchmark(
                config_path = CONFIG_PATH,
                cat_dir     = str(local_cat),
                groups      = GROUPS,
                models      = MODELS,
                resume      = True,
                output_dir  = str(local_out),
                results_dir = DRIVE_OUTPUT_DIR,
            )
            bench_sec = time.time() - t

            # ── experiment_meta.json Drive 업로드 ────────────────────
            t = time.time()
            if local_out.exists():
                drive_out.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(local_out), str(drive_out), dirs_exist_ok=True)
            up_sec = time.time() - t

            print(
                f"  ✓ {cat_dir.name} [{time.time()-t_total:.0f}s 합계]  "
                f"복사:{copy_sec:.0f}s  학습+평가:{bench_sec:.0f}s  업로드:{up_sec:.0f}s"
            )

            for model_name, group_results in results.items():
                for group, val in group_results.items():
                    if isinstance(val, dict) and "error" in val:
                        bench_failed.append({
                            "category": cat_dir.name,
                            "error": f"{model_name}/{group}: {val.get('detail','')[:80]}",
                            "type": val["error"],
                        })

        except Exception as e:
            import traceback
            bench_failed.append({"category": cat_dir.name,
                                  "error": str(e), "type": type(e).__name__})
            traceback.print_exc()
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)
            shutil.rmtree(str(local_out), ignore_errors=True)

    shutil.rmtree(str(LOCAL_TMP_S7), ignore_errors=True)
    print("\n" + (f"✓ {LABEL} 벤치마크 완료" if not bench_failed else f"✗ {len(bench_failed)}건 실패"))
    for f in bench_failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## 셀 8: 결과 요약

```python
import json, yaml
from pathlib import Path

BENCH_CFG  = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text()
)
MODELS = list(BENCH_CFG["models"].keys())
_ar = BENCH_CFG.get("aroma_ratio", {})
GROUPS = list(BENCH_CFG["dataset_groups"].keys()) + (
    [f"aroma_ratio_{int(r*100)}" for r in _ar.get("ratios", [])] if _ar.get("enabled") else []
)

# 셀 7과 동일한 Drive 경로에서 결과 읽기
DRIVE_OUTPUT_DIR = Path(
    BENCH_CFG["experiment"].get("results_dir")
    or "/content/drive/MyDrive/data/Aroma/benchmark_results_phase2"
)

if CAT_ONLY:
    valid_cats = set(CAT_ONLY)
elif DOMAIN_FILTER:
    valid_cats = {
        Path(v.get("seed_dirs", [v["seed_dir"]])[0]).parents[1].name
        for k, v in CONFIG.items()
        if not k.startswith("_") and v["domain"] == DOMAIN_FILTER
    }
else:
    valid_cats = None

rows = []
for cat_dir in sorted(DRIVE_OUTPUT_DIR.iterdir()):
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
                row[f"{model}/{group}"] = (
                    f"ERR:{m['error']}" if "error" in m
                    else round(m.get("image_auroc", 0.0), 4)
                )
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

## 참고: 병렬 설정 가이드

| 스테이지 | 외부 병렬 | 내부 병렬 | 근거 |
|---------|---------|---------|------|
| Stage 4 | 카테고리 순차 | 없음 | GPU 단일 점유 Diffusion 추론 |
| Stage 5 | `SEED_THREADS=4` | `IMG_WORKERS=-1` | I/O + CPU 혼합, seed 단위 독립 |
| Stage 6 | 카테고리 순차 (셀 5) | `NUM_IO_THREADS=8` | I/O-bound 파일 복사 |
| Stage 7 | 카테고리 순차 | 없음 | GPU 단일 점유; augmented_dataset 로컬 SSD 캐시, meta만 Drive 업로드 |

## Phase 2 실행 전 체크리스트

- [ ] `configs/benchmark_experiment_phase2.yaml` 에 `dataset_groups: baseline / aroma_mpb / aroma_diffusion` 정의
- [ ] Phase 1 `stage4_output/` (MPB 결과) 보존 확인
- [ ] Stage 0-3 완료 확인 (`stage3_output/` 존재 여부)
- [ ] ControlNet 파인튜닝 완료 시 `CONTROLNET_MODEL` 경로 지정 (없으면 pretrained canny 사용)
