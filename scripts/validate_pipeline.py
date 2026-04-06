"""scripts/validate_pipeline.py

AROMA 파이프라인 dry-run 검증 스크립트.

각 Stage의 입력/출력 전제조건을 실제 처리 없이 점검하여,
실행 전에 파이프라인이 성공할 수 있는지 미리 확인한다.

사용:
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --base /mnt/drive/data/Aroma
    python scripts/validate_pipeline.py --category mvtec_bottle --stage 4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# dataset_config.json 에 기록된 원본 base 경로
_CONFIG_BASE = "/content/drive/MyDrive/data/Aroma"


def _remap(path_str: str, new_base: str | None) -> Path:
    if new_base is None:
        return Path(path_str)
    rel = path_str[len(_CONFIG_BASE):]
    return Path(new_base) / rel.lstrip("/\\")


def _cat_dir(seed_dir: Path) -> Path:
    return seed_dir.parents[1]


# ---------------------------------------------------------------------------
# Per-stage validators
# ---------------------------------------------------------------------------

def _validate_stage1(cat: Path) -> list[str]:
    """Stage 1 출력: roi_metadata.json 존재 및 기본 구조."""
    errors = []
    meta_path = cat / "stage1_output" / "roi_metadata.json"
    if not meta_path.exists():
        errors.append(f"Stage 1 미완료: {meta_path} 없음")
    else:
        try:
            data = json.loads(meta_path.read_text())
            if not isinstance(data, list) or not data:
                errors.append(f"Stage 1 roi_metadata.json 비어있음: {meta_path}")
            elif "image_id" not in data[0]:
                errors.append(f"Stage 1 roi_metadata.json 스키마 오류 (image_id 누락): {meta_path}")
        except (json.JSONDecodeError, KeyError) as e:
            errors.append(f"Stage 1 roi_metadata.json 파싱 실패: {e}")
    return errors


def _validate_stage1b(cat: Path, seed_names: list[str]) -> list[str]:
    """Stage 1b 출력: 각 seed별 seed_profile.json 존재."""
    errors = []
    stage1b_dir = cat / "stage1b_output"
    if not stage1b_dir.exists():
        if seed_names:
            errors.append(f"Stage 1b 미완료: {stage1b_dir} 없음 ({len(seed_names)} seeds 예상)")
        return errors
    for s in seed_names:
        profile = stage1b_dir / s / "seed_profile.json"
        if not profile.exists():
            errors.append(f"Stage 1b 누락: {profile}")
    return errors


def _validate_stage2(cat: Path) -> list[str]:
    """Stage 2 출력: 각 seed별 variant PNG 존재."""
    errors = []
    stage1b_dir = cat / "stage1b_output"
    stage2_dir = cat / "stage2_output"
    if not stage1b_dir.exists():
        return errors  # Stage 1b 미완료 시 Stage 2 검증 불가
    seeds = sorted(d.name for d in stage1b_dir.iterdir() if d.is_dir())
    if not seeds:
        return errors
    if not stage2_dir.exists():
        errors.append(f"Stage 2 미완료: {stage2_dir} 없음 ({len(seeds)} seeds 예상)")
        return errors
    for s in seeds:
        seed_out = stage2_dir / s
        if not seed_out.exists() or not any(seed_out.glob("*.png")):
            errors.append(f"Stage 2 누락: {seed_out} (variant PNG 없음)")
    return errors


def _validate_stage3(cat: Path) -> list[str]:
    """Stage 3 출력: 각 seed별 placement_map.json 존재."""
    errors = []
    stage2_dir = cat / "stage2_output"
    stage3_dir = cat / "stage3_output"
    if not stage2_dir.exists():
        return errors
    seeds = sorted(d.name for d in stage2_dir.iterdir() if d.is_dir())
    if not seeds:
        return errors
    if not stage3_dir.exists():
        errors.append(f"Stage 3 미완료: {stage3_dir} 없음 ({len(seeds)} seeds 예상)")
        return errors
    for s in seeds:
        pm = stage3_dir / s / "placement_map.json"
        if not pm.exists():
            errors.append(f"Stage 3 누락: {pm}")
    return errors


def _validate_stage4(cat: Path) -> list[str]:
    """Stage 4 출력: 각 seed별 defect/*.png 존재."""
    errors = []
    stage3_dir = cat / "stage3_output"
    stage4_dir = cat / "stage4_output"
    if not stage3_dir.exists():
        return errors
    seeds = sorted(d.name for d in stage3_dir.iterdir() if d.is_dir())
    if not seeds:
        return errors
    if not stage4_dir.exists():
        errors.append(f"Stage 4 미완료: {stage4_dir} 없음 ({len(seeds)} seeds 예상)")
        return errors
    n_missing = 0
    for s in seeds:
        defect_dir = stage4_dir / s / "defect"
        if not defect_dir.exists() or not any(defect_dir.glob("*.png")):
            n_missing += 1
    if n_missing:
        errors.append(
            f"Stage 4: {n_missing}/{len(seeds)} seeds에 defect 이미지 없음 "
            f"(stage4_output)"
        )
    return errors


def _validate_stage5(cat: Path) -> list[str]:
    """Stage 5 출력: 각 seed별 quality_scores.json 존재 (stage4_output 내)."""
    errors = []
    stage4_dir = cat / "stage4_output"
    if not stage4_dir.exists():
        return errors
    seeds = [d for d in sorted(stage4_dir.iterdir()) if d.is_dir()]
    n_missing = 0
    for d in seeds:
        if not (d / "quality_scores.json").exists():
            n_missing += 1
    if n_missing:
        errors.append(
            f"Stage 5: {n_missing}/{len(seeds)} seeds에 quality_scores.json 없음"
        )
    return errors


def _validate_stage6(cat: Path) -> list[str]:
    """Stage 6 출력: build_report.json 존재 + defect_count > 0."""
    errors = []
    report_path = cat / "augmented_dataset" / "build_report.json"
    if not report_path.exists():
        errors.append(f"Stage 6 미완료: {report_path} 없음")
        return errors
    try:
        report = json.loads(report_path.read_text())
        defect_count = report.get("aroma_full", {}).get("defect_count", 0)
        stage4_status = report.get("stage4_status", "unknown")
        if defect_count == 0:
            errors.append(
                f"Stage 6: defect_count=0 (stage4_status={stage4_status}) — "
                f"Stage 7 실행 시 실패할 수 있음"
            )
    except (json.JSONDecodeError, KeyError) as e:
        errors.append(f"Stage 6 build_report.json 파싱 실패: {e}")
    return errors


_STAGE_VALIDATORS = {
    "1": _validate_stage1,
    "1b": _validate_stage1b,
    "2": _validate_stage2,
    "3": _validate_stage3,
    "4": _validate_stage4,
    "5": _validate_stage5,
    "6": _validate_stage6,
}

ALL_STAGES = ["1", "1b", "2", "3", "4", "5", "6"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_category(
    cat_key: str,
    entry: dict,
    base: str | None,
    stages: list[str] | None = None,
) -> dict:
    """카테고리 단위 dry-run 검증."""
    sd = _remap(entry["seed_dir"], base)
    cat = _cat_dir(sd)

    target_stages = stages or ALL_STAGES
    all_errors: list[str] = []

    for stage in target_stages:
        validator = _STAGE_VALIDATORS.get(stage)
        if validator is None:
            continue
        if stage == "1b":
            seed_names = sorted(p.stem for p in sd.glob("*.png")) if sd.exists() else []
            errs = validator(cat, seed_names)
        else:
            errs = validator(cat)
        all_errors.extend(errs)

    return {
        "cat_key": cat_key,
        "domain": entry["domain"],
        "cat_dir": str(cat),
        "errors": all_errors,
    }


def print_validation_report(results: list[dict]) -> None:
    by_domain: dict[str, list] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    total_ok = 0
    total_errors = 0

    for domain in sorted(by_domain):
        cats = by_domain[domain]
        print(f"\n{'='*62}")
        print(f"  {domain.upper()}  ({len(cats)} categories)")
        print(f"{'='*62}")

        for r in sorted(cats, key=lambda x: x["cat_key"]):
            if not r["errors"]:
                total_ok += 1
                print(f"  OK  {r['cat_key']}")
            else:
                total_errors += len(r["errors"])
                print(f"  !!  {r['cat_key']}  ({len(r['errors'])} issues)")
                for err in r["errors"]:
                    print(f"        {err}")

    print(f"\n{'='*62}")
    print(f"  결과: {total_ok}/{len(results)} categories 정상")
    if total_errors:
        print(f"  총 {total_errors}건 이슈 발견")
    else:
        print("  모든 카테고리 파이프라인 준비 완료")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AROMA 파이프라인 dry-run 검증 (처리 없이 입출력 점검)"
    )
    p.add_argument("--config", default="dataset_config.json",
                   help="dataset_config.json 경로 (기본: ./dataset_config.json)")
    p.add_argument("--base", default=None,
                   help=f"Drive 경로({_CONFIG_BASE!r})를 대체할 로컬 base 경로")
    p.add_argument("--domain", default=None,
                   help="특정 도메인만 확인 (isp / mvtec / visa)")
    p.add_argument("--category", default=None,
                   help="특정 카테고리만 확인 (e.g. mvtec_bottle)")
    p.add_argument("--stage", default=None,
                   help="특정 Stage만 확인 (e.g. 4 또는 1,2,3)")
    return p.parse_args()


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    target_stages = args.stage.split(",") if args.stage else None

    results = []
    for key, entry in config.items():
        if key.startswith("_"):
            continue
        if args.domain and entry.get("domain") != args.domain:
            continue
        if args.category and key != args.category:
            continue
        results.append(validate_category(key, entry, args.base, target_stages))

    print_validation_report(results)


if __name__ == "__main__":
    main()
