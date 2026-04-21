# AROMA Phase 2 — 원클릭 실행

> **합성 방식:** SD Inpainting + ControlNet — Stage 4: `stage4_diffusion_synthesis.py`
> Phase 1 (MPB) 결과(`stage4_output/`)는 그대로 보존하고,
> Diffusion 합성 결과는 `stage4_diffusion_output/` 에 별도 저장한다.

## 실행 방법

1. **셀 0**: 환경 설치 (최초 1회만)
2. **셀 1**: 공통 설정 수정 (`DOMAIN_FILTER`, `CAT_ONLY`) → **여기서만 수정**
3. **런타임 → 모두 실행** 또는 셀 순서대로 실행

## 설계 원칙

| 기존 (별도 셀) | 원클릭 (통합) |
|---|---|
| Stage 4: local 추론 → **Drive 업로드** | Stage 4: local 추론 → Drive 체크포인트 |
| Stage 5: **Drive에서 직접** 읽기 | Stage 5: local stage4 직접 읽기 |
| Stage 6: **Drive → local 재다운로드** → 빌드 → Drive | Stage 6: local stage4 그대로 사용 → Drive 업로드 |

Drive 왕복 (`stage4_diffusion_output`: 업로드→재다운로드)을 제거.
Stage 4 체크포인트 업로드는 GPU 결과 보존을 위해 유지 (재실행 시 재추론 방지).

## Skip 로직

- **셀 3 (4-5-6 통합):** `augmented_dataset/aroma_diffusion/train/defect` 존재 → 카테고리 전체 skip
- **Stage 4 내부:** Drive에 `stage4_diffusion_output` 존재 → local 복사만 (재추론 없음)
- **Stage 5 내부:** `quality_scores.json` 존재 → seed skip
- **Stage 7:** `experiment_meta.json` 존재 → 모델/그룹 skip (`resume=True`)

---

## 셀 0: 환경 설치 (최초 1회)

```python
!pip install -q ultralytics effdet
```

---

## 셀 1: 공통 설정

```python
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
SEED_THREADS = 4    # seed 단위 동시 처리
IMG_WORKERS  = -1   # run_quality_scoring 내부 이미지 단위 병렬 수

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
# ─────────────────────────────────────────────────────────────────────

import json, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
REPO   = Path("/content/aroma")
CONFIG = json.loads((REPO / "dataset_config.json").read_text(encoding="utf-8"))
LABEL  = {"isp": "ISP-AD", "mvtec": "MVTec AD", "visa": "VisA"}[DOMAIN_FILTER]

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

## 셀 3: Stage 4-5-6 통합 파이프라인

> **로컬 SSD 경로 구조** (`dataset_builder` 도메인 추출 로직 호환)
> ```
> /content/tmp_phase2/{domain}/{cat}/
>   ├── train/good/               ← image_dir 복사
>   ├── stage4_diffusion_output/  ← Stage 4 출력 (Stage 5, 6 공유)
>   └── augmented_dataset/
>       └── aroma_diffusion/      ← Stage 6 출력 → Drive 업로드
> ```

```python
import shutil, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from stage4_diffusion_synthesis import run_synthesis_batch
from stage5_quality_scoring import run_quality_scoring
from stage6_dataset_builder import run_dataset_builder


def _parallel_copytree(src_dst_pairs, max_workers=4):
    """여러 (src, dst) 쌍을 병렬 복사. src 없으면 건너뜀."""
    def _copy(pair):
        src, dst = Path(pair[0]), Path(pair[1])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_copy, src_dst_pairs))


# ── 카테고리 목록 구성 ─────────────────────────────────────────────────
cat_map: dict = {}
cat_seed_dirs: dict = defaultdict(list)
for key, entry in CONFIG.items():
    if key.startswith("_") or entry["domain"] != DOMAIN_FILTER:
        continue
    seed_dirs_list = entry.get("seed_dirs") or [entry["seed_dir"]]
    cat_dir_path = Path(seed_dirs_list[0]).parents[1]
    if CAT_ONLY and cat_dir_path.name not in CAT_ONLY:
        continue
    cat_map.setdefault(str(cat_dir_path), entry["image_dir"])
    cat_seed_dirs[str(cat_dir_path)].extend(seed_dirs_list)

# aroma_diffusion 완료 여부로 skip 판단
all_cats, skip = [], 0
for cat_dir_str, image_dir in cat_map.items():
    cat_dir = Path(cat_dir_str)
    defect_dir = cat_dir / "augmented_dataset" / DATASET_GROUP / "train" / "defect"
    if defect_dir.exists() and any(defect_dir.iterdir()):
        skip += 1
    else:
        all_cats.append((cat_dir, image_dir, cat_seed_dirs[cat_dir_str]))

if not all_cats:
    print(f"✓ {LABEL} 모든 작업 완료 (skip {skip}건)")
else:
    print(f"{LABEL}: {len(all_cats)} categories 처리 예정 (skip {skip}건)")
    failed = []

    for cat_dir, image_dir, seed_dirs in tqdm(all_cats, desc=f"Phase2 {LABEL}"):
        local_cat    = _LOCAL_DOMAIN_PATH / cat_dir.name
        local_imgdir = local_cat / "train" / "good"
        local_stage4 = local_cat / STAGE4_SUBDIR
        drive_stage4 = cat_dir / STAGE4_SUBDIR

        t_total = time.time()
        timing = {}

        try:
            # ── Stage 4: Diffusion 합성 ──────────────────────────────
            # Drive에 결과가 부분/완료 존재하면 일단 로컬로 복사.
            # 완료 여부와 무관하게 로컬 기준으로 미완료 seed만 추론 → 중단 재개 지원.
            t = time.time()
            copy_pairs = [(image_dir, local_imgdir)]
            if drive_stage4.exists():
                copy_pairs.append((drive_stage4, local_stage4))
            _parallel_copytree(copy_pairs, max_workers=4)
            copy_sec = time.time() - t

            # 로컬 기준 미완료 seed 목록 (sentinel: {seed}/defect/*.png)
            stage3_dir = cat_dir / "stage3_output"
            seed_pm_pairs = [
                (d.name, str(d / "placement_map.json"))
                for d in sorted(stage3_dir.iterdir())
                if d.is_dir()
                and (d / "placement_map.json").exists()
                and not any((local_stage4 / d.name / "defect").glob("*.png"))
            ]

            if not seed_pm_pairs:
                timing["s4"] = f"Drive캐시({copy_sec:.0f}s)"
            else:
                t_infer = time.time()
                run_synthesis_batch(
                    image_dir           = str(local_imgdir),
                    seed_placement_maps = seed_pm_pairs,
                    output_root         = str(local_stage4),
                    cat_dir             = str(cat_dir),  # stage1b_output 경로용 (Drive)
                    format              = "cls",
                    controlnet_model    = CONTROLNET_MODEL,
                    device              = DEVICE,
                    resolution          = RESOLUTION,
                    num_inference_steps = NUM_INFERENCE_STEPS,
                    strength            = STRENGTH,
                    guidance_scale      = GUIDANCE_SCALE,
                    conditioning_scale  = CONDITIONING_SCALE,
                    seed                = DIFFUSION_SEED,
                )
                infer_sec = time.time() - t_infer

                # Drive 체크포인트: GPU 결과 보존 (재실행 시 재추론 방지)
                t_ckpt = time.time()
                shutil.copytree(str(local_stage4), str(drive_stage4), dirs_exist_ok=True)
                timing["s4"] = (
                    f"복사{copy_sec:.0f}s+추론{infer_sec:.0f}s"
                    f"+체크포인트{time.time()-t_ckpt:.0f}s"
                )

            # ── Stage 5: 품질 점수 ───────────────────────────────────
            # local_stage4 기준으로 처리 (Drive 재다운로드 없음)
            t = time.time()
            seeds_s5 = [
                d.name for d in sorted(local_stage4.iterdir())
                if d.is_dir() and not (d / "quality_scores.json").exists()
            ]

            def _run_s5(seed_id):
                run_quality_scoring(
                    stage4_seed_dir = str(local_stage4 / seed_id),
                    workers         = IMG_WORKERS,
                )
                return seed_id

            if seeds_s5:
                with ThreadPoolExecutor(max_workers=SEED_THREADS) as ex:
                    futs = {ex.submit(_run_s5, s): s for s in seeds_s5}
                    for fut in as_completed(futs):
                        fut.result()
            timing["s5"] = f"{time.time()-t:.0f}s"

            # ── Stage 6: 데이터셋 구성 ──────────────────────────────
            # local_stage4 + local_imgdir 그대로 사용 (Drive 재다운로드 없음)
            t = time.time()
            result = run_dataset_builder(
                cat_dir              = str(local_cat),
                image_dir            = str(local_imgdir),
                seed_dirs            = seed_dirs,
                pruning_ratio        = PRUNING_RATIO,
                split_ratio          = SPLIT_RATIO,
                split_seed           = SPLIT_SEED,
                workers              = NUM_IO_THREADS,
                balance_defect_types = BALANCE_DEFECT_TYPES,
                stage4_subdir        = STAGE4_SUBDIR,
                dataset_group        = DATASET_GROUP,
            )
            timing["s6"] = f"{time.time()-t:.0f}s"

            # ── Drive 업로드: aroma_diffusion 그룹만 ────────────────
            t = time.time()
            drive_aroma = cat_dir / "augmented_dataset" / DATASET_GROUP
            if drive_aroma.exists():
                shutil.rmtree(str(drive_aroma))
            shutil.copytree(
                str(local_cat / "augmented_dataset" / DATASET_GROUP),
                str(drive_aroma),
            )
            timing["up"] = f"{time.time()-t:.0f}s"

            defect_count = result.get(DATASET_GROUP, {}).get("defect_count", "?")
            print(
                f"  ✓ {cat_dir.name} [{time.time()-t_total:.0f}s 합계]  "
                f"S4:{timing['s4']}  S5:{timing['s5']}  S6:{timing['s6']}  업로드:{timing['up']}  "
                f"[defect={defect_count}]"
            )

        except Exception as e:
            import traceback
            failed.append({"category": cat_dir.name, "error": str(e), "type": type(e).__name__})
            traceback.print_exc()
        finally:
            shutil.rmtree(str(local_cat), ignore_errors=True)

    shutil.rmtree(str(LOCAL_TMP), ignore_errors=True)
    print("\n" + (f"✓ {LABEL} 완료" if not failed else f"✗ {len(failed)}건 실패"))
    for f in failed:
        print(f"  [{f['category']}] {f['type']}: {f['error'][:120]}")
```

---

## 셀 4: Stage 7 데이터 구조 확인 (선택)

> 벤치마크 실행 전 `baseline` / `aroma_mpb` / `aroma_diffusion` 세 그룹 존재 여부 점검.

```python
import json, yaml, sys
from pathlib import Path

sys.path.insert(0, "/content/aroma")
BENCH_CFG = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text()
)
EXCLUDE = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
GROUPS  = list(BENCH_CFG["dataset_groups"].keys())

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

## 셀 5: Stage 7 test set 준비 + 벤치마크 (로컬 SSD 캐시)

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
OUTPUT_DIR  = str(REPO / "outputs" / "benchmark_results_phase2")

EXCLUDE             = set(BENCH_CFG.get("category_filter", {}).get("exclude", {}).get(DOMAIN_FILTER, []))
MODELS              = list(BENCH_CFG["models"].keys())
GROUPS              = list(BENCH_CFG["dataset_groups"].keys())
NON_BASELINE_GROUPS = [g for g in GROUPS if g != "baseline"]

# results_dir: benchmark_results.json / benchmark_comparison.csv 저장 (yaml 설정값 우선)
DRIVE_RESULTS_DIR = BENCH_CFG["experiment"].get("results_dir") or OUTPUT_DIR

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
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
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
    if CAT_ONLY and cat_dir.name not in CAT_ONLY:
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
    out_root = Path(OUTPUT_DIR) / cat_dir.name
    if all((out_root / m / g / "experiment_meta.json").exists()
           for m in MODELS for g in GROUPS):
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
        local_cat  = _LOCAL_S7_BASE / cat_dir.name          # 학습 데이터 캐시
        local_out  = LOCAL_TMP_S7 / "results" / cat_dir.name  # 실험 메타 로컬 저장
        drive_out  = Path(OUTPUT_DIR) / cat_dir.name          # Drive 업로드 대상

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
            # output_dir=local_out → experiment_meta.json 등 로컬에만 기록
            # results_dir=DRIVE_RESULTS_DIR → benchmark_results.json/csv 는 Drive 직접 기록
            t = time.time()
            results = run_benchmark(
                config_path = CONFIG_PATH,
                cat_dir     = str(local_cat),
                resume      = True,
                output_dir  = str(local_out),
                results_dir = DRIVE_RESULTS_DIR,
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

## 셀 6: 결과 요약

```python
import json, yaml
from pathlib import Path

OUTPUT_DIR = REPO / "outputs" / "benchmark_results_phase2"
BENCH_CFG  = yaml.safe_load(
    (REPO / "configs" / "benchmark_experiment_phase2.yaml").read_text()
)
MODELS = list(BENCH_CFG["models"].keys())
GROUPS = list(BENCH_CFG["dataset_groups"].keys())

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
| Stage 6 | 없음 (Stage 4-5-6 통합 루프) | `NUM_IO_THREADS=8` | I/O-bound 파일 복사 |
| Stage 7 | 카테고리 순차 | 없음 | GPU 단일 점유; augmented_dataset 로컬 SSD 캐시, meta만 Drive 업로드 |

## Phase 2 실행 전 체크리스트

- [ ] `configs/benchmark_experiment_phase2.yaml` 에 `dataset_groups: baseline / aroma_mpb / aroma_diffusion` 정의
- [ ] Phase 1 `stage4_output/` (MPB 결과) 보존 확인
- [ ] Stage 0-3 완료 확인 (`stage3_output/` 존재 여부)
- [ ] ControlNet 파인튜닝 완료 시 `CONTROLNET_MODEL` 경로 지정 (없으면 pretrained canny 사용)
