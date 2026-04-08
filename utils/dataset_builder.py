"""utils/dataset_builder.py

Stage 6 증강 데이터셋 구성 라이브러리.

baseline / aroma_full / aroma_pruned 3개 group 을 MVTec-style 폴더 구조로 생성.
파일 복사 방식 (symlink 미사용) — Colab/Google Drive 환경 호환.
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _copy_worker(args_tuple: tuple) -> str | None:
    """ProcessPoolExecutor pickle-safe 복사 워커.

    args_tuple: (src_path_str, dst_path_str)
    dst 이미 존재하면 스킵 (파일 단위 resume).
    tasks 구성 예: tasks = [(str(src), str(dst)) for src, dst in copy_pairs]
    """
    src_path_str, dst_path_str = args_tuple
    dst = Path(dst_path_str)
    if dst.exists():
        return dst_path_str
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path_str, dst_path_str)
    return dst_path_str


def _collect_defect_paths(
    cat_dir: str,
    pruning_threshold: float | None = None,
    augmentation_ratio: float | None = None,
    good_count: int = 0,
) -> list[tuple[str, str]]:
    """stage4_output/{seed_id}/defect/*.png 수집 (단일 깊이 *).

    Args:
        cat_dir: 카테고리 루트 디렉터리.
        pruning_threshold: quality_score 최솟값 (None이면 품질 필터링 없음).
        augmentation_ratio: 원본 대비 합성 defect 비율 (None이면 모든 이미지 사용).
            예: 2.0 → 원본 good 209개면 합성 defect 418개 선택.
        good_count: 원본 good 이미지 개수 (augmentation_ratio 계산에 사용).

    Returns:
        List of (src_path_str, dst_filename) where
        dst_filename = "{seed_id}_{image_id}.png" (충돌 방지).
    """
    stage4_dir = Path(cat_dir) / "stage4_output"
    if not stage4_dir.exists():
        return []

    # ── 1단계: 모든 defect 이미지 수집 (quality_score 포함) ──────────────
    candidates = []  # List of (src_path_str, dst_filename, quality_score)
    for seed_dir in sorted(stage4_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        defect_dir = seed_dir / "defect"
        if not defect_dir.exists():
            continue

        img_paths = sorted(defect_dir.glob("*.png"))
        if not img_paths:
            continue

        # quality_scores.json 로드 (pruning_threshold 또는 augmentation_ratio 사용 시 필요)
        quality_map = {}
        if pruning_threshold is not None or augmentation_ratio is not None:
            quality_file = seed_dir / "quality_scores.json"
            if not quality_file.exists():
                warnings.warn(
                    f"quality_scores.json 없음, seed 스킵: {seed_dir.name}",
                    stacklevel=3,
                )
                continue
            quality_data = json.loads(quality_file.read_text())
            quality_map = {
                s["image_id"]: s["quality_score"]
                for s in quality_data.get("scores", [])
            }

        for p in img_paths:
            quality = quality_map.get(p.stem, 0.0) if quality_map else 1.0
            candidates.append((str(p), f"{seed_dir.name}_{p.name}", quality))

    if not candidates:
        return []

    # ── 2단계: pruning_threshold 필터링 ──────────────────────────────────
    if pruning_threshold is not None:
        candidates = [
            (src, dst, q) for src, dst, q in candidates
            if q >= pruning_threshold
        ]

    # ── 3단계: augmentation_ratio 기반 샘플링 ────────────────────────────
    if augmentation_ratio is not None and good_count > 0:
        target_count = int(good_count * augmentation_ratio)
        
        # quality_score 기준 내림차순 정렬 후 상위 K개 선택
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        
        if len(candidates) > target_count:
            candidates = candidates[:target_count]
        elif len(candidates) < target_count:
            warnings.warn(
                f"증강 비율 {augmentation_ratio}×{good_count}={target_count}개 요청, "
                f"가용 defect {len(candidates)}개만 사용 (카테고리: {Path(cat_dir).name})",
                stacklevel=3,
            )

    # ── 4단계: 최종 결과 반환 ───────────────────────────────────────────
    return [(src, dst) for src, dst, _ in candidates]


def _check_stage4_status(cat_dir: str) -> tuple[str, int, int]:
    """stage4_output 의 완료 상태를 점검한다.

    Returns:
        (status, n_total_seeds, n_seeds_with_defects)
        status: "not_started" | "incomplete" | "partial" | "complete"
    """
    stage4_dir = Path(cat_dir) / "stage4_output"
    if not stage4_dir.exists():
        return ("not_started", 0, 0)

    seed_dirs = [d for d in sorted(stage4_dir.iterdir()) if d.is_dir()]
    n_total = len(seed_dirs)
    if n_total == 0:
        return ("not_started", 0, 0)

    n_with_defects = sum(
        1 for d in seed_dirs
        if (d / "defect").is_dir() and any((d / "defect").glob("*.png"))
    )

    if n_with_defects == 0:
        return ("incomplete", n_total, 0)
    if n_with_defects < n_total:
        return ("partial", n_total, n_with_defects)
    return ("complete", n_total, n_with_defects)


def _copy_images(src_dir: Path, dst_dir: Path, num_workers: int,
                 desc: str = "") -> int:
    """src_dir 아래 PNG/JPG 전체를 dst_dir 로 복사. 복사 파일 수 반환.

    Google Drive 같은 I/O-bound 환경에서는 ThreadPoolExecutor 가 효율적.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    imgs = (
        sorted(src_dir.glob("*.png"))
        + sorted(src_dir.glob("*.jpg"))
        + sorted(src_dir.glob("*.jpeg"))
    )
    if not imgs:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(s), str(dst_dir / s.name)) for s in imgs]
    
    # Progress tracking
    progress_desc = f"  {desc}" if desc else "  Copying images"
    
    if num_workers <= 1:
        for t in tqdm(tasks, desc=progress_desc, leave=False):
            _copy_worker(t)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(tqdm(ex.map(_copy_worker, tasks), total=len(tasks), 
                     desc=progress_desc, leave=False))
    return len(imgs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_groups(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_threshold: float = 0.6,
    augmentation_ratio_full: float | None = None,
    augmentation_ratio_pruned: float | None = None,
    workers: int = 0,
) -> dict:
    """3개 dataset group 을 구성하고 build_report.json 을 저장한다.

    Args:
        cat_dir: 카테고리 루트 디렉터리 (e.g. `.../mvtec/bottle`).
        image_dir: 원본 train/good 이미지 디렉터리.
        seed_dirs: dataset_config.json 에서 추출한 seed_dir 경로 목록.
            baseline/test/{defect_type}/ 구성에 사용.
        pruning_threshold: aroma_pruned 의 quality_score 최솟값.
        augmentation_ratio_full: aroma_full의 원본 대비 합성 defect 비율
            (None이면 모든 defect 사용). 예: 2.0 → good 209개면 defect 418개.
        augmentation_ratio_pruned: aroma_pruned의 원본 대비 합성 defect 비율.
        workers: 병렬 워커 수 (0=순차).

    Returns:
        build_report dict. build_report.json 으로도 저장.

    Skip:
        build_report.json 이 존재하고 저장된 파라미터들이 일치하면
        재생성 없이 로드하여 반환. 불일치 시 전체 재생성.
    """
    cat_path = Path(cat_dir)
    aug_dir = cat_path / "augmented_dataset"
    report_path = aug_dir / "build_report.json"

    # Skip 조건 확인
    # stage4_output 이 존재하고 이전 실행에서 실제 defect 가 수집됐을 때만 skip.
    # stage4 가 미실행인 채로 캐시된 경우(defect_count=0) stage4 완료 후 재실행 가능하도록 skip 하지 않음.
    stage4_dir = cat_path / "stage4_output"
    if report_path.exists():
        cached = json.loads(report_path.read_text())
        threshold_match = abs(cached.get("pruning_threshold", -1) - pruning_threshold) < 1e-6
        ratio_full_match = (
            cached.get("augmentation_ratio_full") == augmentation_ratio_full
        )
        ratio_pruned_match = (
            cached.get("augmentation_ratio_pruned") == augmentation_ratio_pruned
        )
        cached_has_defects = cached.get("aroma_full", {}).get("defect_count", 0) > 0
        stage4_now_exists = stage4_dir.exists() and any(stage4_dir.iterdir())
        
        if (threshold_match and ratio_full_match and ratio_pruned_match and
            (cached_has_defects or not stage4_now_exists)):
            return cached

    from concurrent.futures import ThreadPoolExecutor
    from utils.parallel import resolve_workers
    num_workers = resolve_workers(workers)

    # ── Stage 4 완료 전제 조건 검증 ────────────────────────────────────────
    stage4_status, stage4_seeds_total, stage4_seeds_with_defects = (
        _check_stage4_status(cat_dir)
    )
    if stage4_status == "incomplete":
        warnings.warn(
            f"Stage 4 미완료 — stage4_output/{stage4_seeds_total}개 seed 중 "
            f"defect 이미지 보유 0건: {cat_dir}",
            stacklevel=2,
        )

    # ── baseline/train/good/ ───────────────────────────────────────────────
    baseline_good = aug_dir / "baseline" / "train" / "good"
    good_count = _copy_images(Path(image_dir), baseline_good, num_workers,
                               desc="baseline/train/good")

    # ── baseline/test/ ────────────────────────────────────────────────────
    # test/good/ (image_dir 기준: {cat}/train/good → {cat}/test/good)
    test_good_src = Path(image_dir).parents[1] / "test" / "good"
    if test_good_src.exists():
        _copy_images(test_good_src,
                     aug_dir / "baseline" / "test" / "good",
                     num_workers, desc="baseline/test/good")

    # test/{defect_type}/ (seed_dirs 에서 추출)
    for sd in seed_dirs:
        defect_type = Path(sd).name
        src = Path(sd)
        if src.exists():
            _copy_images(src,
                         aug_dir / "baseline" / "test" / defect_type,
                         num_workers, desc=f"baseline/test/{defect_type}")

    # ── aroma_full/train/ ─────────────────────────────────────────────────
    _copy_images(Path(image_dir),
                 aug_dir / "aroma_full" / "train" / "good",
                 num_workers, desc="aroma_full/train/good")

    full_defect_pairs = _collect_defect_paths(
        cat_dir,
        pruning_threshold=None,
        augmentation_ratio=augmentation_ratio_full,
        good_count=good_count,
    )
    if not full_defect_pairs:
        # CASDA validate_yolo_dataset 패턴 — 학습 전 데이터 무결성 사전 검증
        detail = (
            "stage4_output 미존재"
            if not stage4_dir.exists()
            else "stage4_output 존재하나 defect 이미지 0건 — "
                 "quality_scores.json 누락 또는 Stage 4 미완료 가능성"
        )
        warnings.warn(
            f"aroma_* defect 이미지 0건 ({detail}): {cat_dir}",
            stacklevel=2,
        )
    if full_defect_pairs:
        from tqdm import tqdm
        full_defect_dst = aug_dir / "aroma_full" / "train" / "defect"
        full_defect_dst.mkdir(parents=True, exist_ok=True)
        tasks = [(src, str(full_defect_dst / dst_name))
                 for src, dst_name in full_defect_pairs]
        if num_workers <= 1:
            for t in tqdm(tasks, desc="  aroma_full/train/defect", leave=False):
                _copy_worker(t)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                list(tqdm(ex.map(_copy_worker, tasks), total=len(tasks),
                         desc="  aroma_full/train/defect", leave=False))

    # ── aroma_pruned/train/ ───────────────────────────────────────────────
    _copy_images(Path(image_dir),
                 aug_dir / "aroma_pruned" / "train" / "good",
                 num_workers, desc="aroma_pruned/train/good")

    pruned_defect_pairs = _collect_defect_paths(
        cat_dir,
        pruning_threshold=pruning_threshold,
        augmentation_ratio=augmentation_ratio_pruned,
        good_count=good_count,
    )
    if pruned_defect_pairs:
        from tqdm import tqdm
        pruned_defect_dst = aug_dir / "aroma_pruned" / "train" / "defect"
        pruned_defect_dst.mkdir(parents=True, exist_ok=True)
        tasks = [(src, str(pruned_defect_dst / dst_name))
                 for src, dst_name in pruned_defect_pairs]
        if num_workers <= 1:
            for t in tqdm(tasks, desc="  aroma_pruned/train/defect", leave=False):
                _copy_worker(t)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                list(tqdm(ex.map(_copy_worker, tasks), total=len(tasks),
                         desc="  aroma_pruned/train/defect", leave=False))

    report = {
        "pruning_threshold": pruning_threshold,
        "augmentation_ratio_full": augmentation_ratio_full,
        "augmentation_ratio_pruned": augmentation_ratio_pruned,
        "stage4_status": stage4_status,
        "stage4_seeds_total": stage4_seeds_total,
        "stage4_seeds_with_defects": stage4_seeds_with_defects,
        "baseline":     {"good_count": good_count, "defect_count": 0},
        "aroma_full":   {"good_count": good_count,
                         "defect_count": len(full_defect_pairs)},
        "aroma_pruned": {"good_count": good_count,
                         "defect_count": len(pruned_defect_pairs)},
    }
    aug_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report
