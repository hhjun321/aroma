"""utils/dataset_builder.py

Stage 6 증강 데이터셋 구성 라이브러리.

baseline / aroma_full / aroma_pruned 3개 group 을 MVTec-style 폴더 구조로 생성.
파일 복사 방식 (symlink 미사용) — Colab/Google Drive 환경 호환.
"""
from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_defect_type(seed_id: str) -> str:
    """seed_id 에서 결함 유형 접두사를 추출한다.

    Examples:
        'broken_large_000' → 'broken_large'
        '000'              → ''   (단일 유형 진입점, 접두사 없음)
    """
    m = re.match(r'^(.+)_\d+$', seed_id)
    return m.group(1) if m else ''


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
    pruning_ratio: float | None = None,
    augmentation_ratio: float | None = None,
    good_count: int = 0,
    balance_defect_types: bool = False,
) -> list[tuple[str, str]]:
    """stage4_output/{seed_id}/defect/*.png 수집 (단일 깊이 *).

    Args:
        cat_dir: 카테고리 루트 디렉터리.
        pruning_ratio: quality_score 기준 상위 X% 선택 비율 (None이면 필터링 없음).
            예: 0.5 → 전체 후보 중 상위 50% 선택.
            도메인·카테고리에 무관하게 동일하게 적용 (domain-agnostic rank 기반).
            augmentation_ratio 적용 전 품질 pre-filter로 동작.
        augmentation_ratio: 원본 대비 합성 defect 비율 (None이면 모든 이미지 사용).
            예: 2.0 → 원본 good 209개면 합성 defect 418개 선택.
        good_count: 원본 good 이미지 개수 (augmentation_ratio 계산에 사용).
        balance_defect_types: True 이면 결함 유형별 균등 샘플링을 적용한다.
            seed_id 의 접두사(예: broken_large, broken_small)를 유형으로 식별하고
            augmentation_ratio 로 계산된 할당량을 유형 수로 균등 분배한다.
            단일 유형(접두사 없음) 카테고리에는 영향 없음.

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

        # quality_scores.json 로드 (pruning_ratio 또는 augmentation_ratio 사용 시 필요)
        quality_map = {}
        if pruning_ratio is not None or augmentation_ratio is not None:
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

    # ── 2단계: pruning_ratio rank 기반 필터링 ────────────────────────────
    if pruning_ratio is not None and candidates:
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        keep = max(1, int(len(candidates) * pruning_ratio))
        candidates = candidates[:keep]

    # ── 3단계: augmentation_ratio 기반 샘플링 ────────────────────────────
    if augmentation_ratio is not None and good_count > 0:
        target_count = int(good_count * augmentation_ratio)

        if balance_defect_types:
            # 결함 유형별 균등 분배: seed_id 접두사로 유형 식별
            from collections import defaultdict
            type_groups: dict[str, list] = defaultdict(list)
            for item in candidates:
                src, dst, q = item
                seed_id = Path(src).parents[1].name  # e.g. 'broken_large_000'
                dtype = _get_defect_type(seed_id)    # e.g. 'broken_large'
                type_groups[dtype].append(item)

            n_types = max(1, len(type_groups))
            per_type = max(1, target_count // n_types)
            balanced: list = []
            for dtype in sorted(type_groups):
                items_sorted = sorted(type_groups[dtype], key=lambda x: x[2],
                                      reverse=True)
                if len(items_sorted) < per_type:
                    warnings.warn(
                        f"결함 유형 '{dtype}' 가용 {len(items_sorted)}개 < "
                        f"균형 할당 {per_type}개 "
                        f"(카테고리: {Path(cat_dir).name})",
                        stacklevel=3,
                    )
                balanced.extend(items_sorted[:per_type])
            candidates = balanced
        else:
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


def _copy_file_list(paths: list[Path], dst_dir: Path, num_workers: int,
                    desc: str = "") -> int:
    """특정 파일 경로 목록을 dst_dir 로 복사. 복사 파일 수 반환."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    if not paths:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(dst_dir / p.name)) for p in paths]

    progress_desc = f"  {desc}" if desc else "  Copying files"
    if num_workers <= 1:
        for t in tqdm(tasks, desc=progress_desc, leave=False):
            _copy_worker(t)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(tqdm(ex.map(_copy_worker, tasks), total=len(tasks),
                     desc=progress_desc, leave=False))
    return len(paths)


def _split_good_images(
    image_dir: Path,
    cat_test_good_dir: Path,
    split_ratio: float,
    split_seed: int,
) -> tuple[list[Path], list[Path]]:
    """train/good + test/good 전체를 pooling 후 결정적 분할.

    train/good 이미지와 test/good 이미지를 합산해 동일 비율로 분할한다.
    모든 카테고리·도메인에 걸쳐 동일한 split_ratio 를 보장하기 위해 사용한다.

    Args:
        image_dir:        원본 train/good 디렉터리.
        cat_test_good_dir: 원본 test/good 디렉터리 (없으면 train/good 만 사용).
        split_ratio:      train 비율 (0~1). 예: 0.8 → 80% train, 20% test.
        split_seed:       결정적 셔플 시드.

    Returns:
        (train_paths, test_paths) — 두 리스트는 상호 배타적 (no leakage).
    """
    import random

    _exts = ("*.png", "*.jpg", "*.jpeg")
    all_good: list[Path] = []
    for src in [image_dir, cat_test_good_dir]:
        if src.exists():
            for ext in _exts:
                all_good.extend(src.glob(ext))

    # 파일명 기준 정렬 → 시드 기반 셔플 (결정적)
    all_good = sorted(set(all_good), key=lambda p: p.name)
    rng = random.Random(split_seed)
    rng.shuffle(all_good)

    n_train = max(1, int(len(all_good) * split_ratio))
    return all_good[:n_train], all_good[n_train:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset_groups(
    cat_dir: str,
    image_dir: str,
    seed_dirs: list[str],
    pruning_ratio: float | None = 0.5,
    augmentation_ratio_full: float | None = None,
    augmentation_ratio_pruned: float | None = None,
    augmentation_ratio_by_domain: dict | None = None,
    split_ratio: float | None = None,
    split_seed: int = 42,
    workers: int = 0,
    balance_defect_types: bool = False,
) -> dict:
    """3개 dataset group 을 구성하고 build_report.json 을 저장한다.

    Args:
        cat_dir: 카테고리 루트 디렉터리 (e.g. `.../mvtec/bottle`).
        image_dir: 원본 train/good 이미지 디렉터리.
        seed_dirs: dataset_config.json 에서 추출한 seed_dir 경로 목록.
            baseline/test/{defect_type}/ 구성에 사용.
        pruning_ratio: aroma_pruned 의 quality_score 상위 비율 (0~1, None이면 비활성).
            예: 0.5 → 전체 후보 중 quality_score 기준 상위 50% 선택.
            도메인·카테고리 무관하게 동일하게 적용 (domain-agnostic rank 기반).
            augmentation_ratio 적용 전 품질 pre-filter로 동작.
        augmentation_ratio_full: aroma_full의 원본 대비 합성 defect 비율
            (None이면 모든 defect 사용). 예: 2.0 → good 209개면 defect 418개.
        augmentation_ratio_pruned: aroma_pruned의 원본 대비 합성 defect 비율.
        augmentation_ratio_by_domain: 도메인별 비율 설정 (dict).
            예: {"isp": {"full": 1.0, "pruned": 0.5}, "mvtec": {...}}
            이 값이 설정되면 도메인 추출 후 우선 적용됨.
        split_ratio: train/test good 이미지 분할 비율 (0~1).
            None=원본 데이터셋 분할 사용 (기본 동작).
            설정 시 train/good + test/good 전체를 pooling 후
            결정적으로 분할 → 모든 카테고리·도메인에 동일 비율 보장.
            예: 0.8 → 전체 good 의 80% 를 train, 20% 를 test 에 사용.
        split_seed: split_ratio 사용 시 결정적 셔플 시드 (기본 42).
        workers: 병렬 워커 수 (0=순차).
        balance_defect_types: True 이면 결함 유형별 균등 샘플링 적용.
            seed_id 접두사(예: broken_large, broken_small)로 유형을 식별하고
            augmentation_ratio 할당량을 유형 수로 균등 분배한다.
            단일 유형 카테고리(기존 ISP·VisA·MVTec 단일 시드 진입점)에는 영향 없음.

    Returns:
        build_report dict. build_report.json 으로도 저장.

    Skip:
        build_report.json 이 존재하고 저장된 파라미터들이 일치하면
        재생성 없이 로드하여 반환. 불일치 시 전체 재생성.
    """
    cat_path = Path(cat_dir)
    aug_dir = cat_path / "augmented_dataset"
    report_path = aug_dir / "build_report.json"

    # ── 도메인별 비율 우선 적용 ──────────────────────────────────────────
    # cat_dir 경로에서 도메인 추출
    # 경로 구조:
    #   MVTec/VisA: .../domain/category/ → parent.name = domain
    #   ISP: .../isp/unsupervised/category/ → parent.parent.name = domain
    domain = cat_path.parent.name
    if domain == "unsupervised":  # ISP special case
        domain = cat_path.parent.parent.name
    
    if augmentation_ratio_by_domain and domain in augmentation_ratio_by_domain:
        domain_ratios = augmentation_ratio_by_domain[domain]
        ratio_full = domain_ratios.get("full", augmentation_ratio_full)
        ratio_pruned = domain_ratios.get("pruned", augmentation_ratio_pruned)
    else:
        ratio_full = augmentation_ratio_full
        ratio_pruned = augmentation_ratio_pruned

    # Skip 조건 확인
    # stage4_output 이 존재하고 이전 실행에서 실제 defect 가 수집됐을 때만 skip.
    # stage4 가 미실행인 채로 캐시된 경우(defect_count=0) stage4 완료 후 재실행 가능하도록 skip 하지 않음.
    stage4_dir = cat_path / "stage4_output"
    if report_path.exists():
        cached = json.loads(report_path.read_text())
        pruning_ratio_match = cached.get("pruning_ratio") == pruning_ratio
        ratio_full_match = (
            cached.get("augmentation_ratio_full") == ratio_full
        )
        ratio_pruned_match = (
            cached.get("augmentation_ratio_pruned") == ratio_pruned
        )
        split_ratio_match = cached.get("split_ratio") == split_ratio
        split_seed_match = (
            split_ratio is None  # split_ratio=None 이면 seed 무관
            or cached.get("split_seed", 42) == split_seed
        )
        cached_has_defects = cached.get("aroma_full", {}).get("defect_count", 0) > 0
        stage4_now_exists = stage4_dir.exists() and any(stage4_dir.iterdir())

        balance_match = (
            cached.get("balance_defect_types", False) == balance_defect_types
        )
        if (pruning_ratio_match and ratio_full_match and ratio_pruned_match and
                split_ratio_match and split_seed_match and balance_match and
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

    # ── good 이미지 train/test 분할 ───────────────────────────────────────
    # split_ratio 가 설정된 경우: train/good + test/good 를 pooling 후 결정적 분할
    #   → 모든 카테고리·도메인에 동일한 비율 보장, test 가 train 에 포함되지 않음
    # split_ratio=None: 원본 데이터셋 분할 그대로 사용 (기존 동작)
    test_good_src = Path(image_dir).parents[1] / "test" / "good"

    if split_ratio is not None:
        train_good_paths, test_good_paths = _split_good_images(
            image_dir=Path(image_dir),
            cat_test_good_dir=test_good_src,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        good_count = _copy_file_list(
            train_good_paths, aug_dir / "baseline" / "train" / "good",
            num_workers, desc="baseline/train/good (split)"
        )
        _copy_file_list(
            test_good_paths, aug_dir / "baseline" / "test" / "good",
            num_workers, desc="baseline/test/good (split)"
        )
    else:
        # 원본 분할 사용
        good_count = _copy_images(
            Path(image_dir), aug_dir / "baseline" / "train" / "good",
            num_workers, desc="baseline/train/good"
        )
        if test_good_src.exists():
            _copy_images(test_good_src,
                         aug_dir / "baseline" / "test" / "good",
                         num_workers, desc="baseline/test/good")

    # test/{defect_type}/ — 원본 test 디렉터리 전체 스캔 (모든 defect 유형 포함)
    # 공정한 벤치마크를 위해 seed로 사용된 유형만이 아닌 전체 defect 유형을 포함한다.
    # seed_dirs는 원본 test 구조가 없을 때(이전 버전 호환) 폴백으로 사용.
    cat_test_dir = Path(image_dir).parents[1] / "test"
    if cat_test_dir.exists():
        for defect_dir in sorted(cat_test_dir.iterdir()):
            if defect_dir.is_dir() and defect_dir.name != "good":
                _copy_images(defect_dir,
                             aug_dir / "baseline" / "test" / defect_dir.name,
                             num_workers, desc=f"baseline/test/{defect_dir.name}")
    else:
        # 폴백: 원본 데이터셋 구조가 없을 때 seed_dirs 기반으로 구성
        for sd in seed_dirs:
            defect_type = Path(sd).name
            src = Path(sd)
            if src.exists():
                _copy_images(src,
                             aug_dir / "baseline" / "test" / defect_type,
                             num_workers, desc=f"baseline/test/{defect_type}")

    # ── aroma_full/train/ ─────────────────────────────────────────────────
    # good train 이미지는 baseline 과 동일한 분할을 사용 → 공정한 비교 보장
    if split_ratio is not None:
        _copy_file_list(
            train_good_paths, aug_dir / "aroma_full" / "train" / "good",
            num_workers, desc="aroma_full/train/good (split)"
        )
    else:
        _copy_images(Path(image_dir),
                     aug_dir / "aroma_full" / "train" / "good",
                     num_workers, desc="aroma_full/train/good")

    full_defect_pairs = _collect_defect_paths(
        cat_dir,
        pruning_ratio=None,
        augmentation_ratio=ratio_full,
        good_count=good_count,
        balance_defect_types=balance_defect_types,
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
    if split_ratio is not None:
        _copy_file_list(
            train_good_paths, aug_dir / "aroma_pruned" / "train" / "good",
            num_workers, desc="aroma_pruned/train/good (split)"
        )
    else:
        _copy_images(Path(image_dir),
                     aug_dir / "aroma_pruned" / "train" / "good",
                     num_workers, desc="aroma_pruned/train/good")

    pruned_defect_pairs = _collect_defect_paths(
        cat_dir,
        pruning_ratio=pruning_ratio,
        augmentation_ratio=ratio_pruned,
        good_count=good_count,
        balance_defect_types=balance_defect_types,
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
        "pruning_ratio": pruning_ratio,
        "augmentation_ratio_full": ratio_full,
        "augmentation_ratio_pruned": ratio_pruned,
        "split_ratio": split_ratio,
        "split_seed": split_seed if split_ratio is not None else None,
        "balance_defect_types": balance_defect_types,
        "domain": domain,
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
