"""scripts/check_progress.py

AROMA 파이프라인 Stage 1–6 진행 상황 확인 스크립트.
도메인/카테고리/시드 단위로 완료 여부를 점검하고 미완료 목록을 출력한다.

사용:
    python scripts/check_progress.py
    python scripts/check_progress.py --domain isp
    python scripts/check_progress.py --base /mnt/drive/data/Aroma -v

--base: dataset_config.json 에 하드코딩된 Drive 경로
        (/content/drive/MyDrive/data/Aroma) 를 로컬 경로로 재매핑한다.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# dataset_config.json 에 기록된 원본 base 경로
_CONFIG_BASE = "/content/drive/MyDrive/data/Aroma"

STAGES = ["1", "1b", "2", "3", "4", "5", "6"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _remap(path_str: str, new_base: str | None) -> Path:
    """config 절대 경로를 new_base 로 재매핑한다."""
    if new_base is None:
        return Path(path_str)
    rel = path_str[len(_CONFIG_BASE):]          # e.g. "/isp/unsupervised/LSM_1/train/good"
    return Path(new_base) / rel.lstrip("/\\")


def _cat_dir(seed_dir: Path) -> Path:
    """seed_dir (e.g. .../LSM_1/test/area) → category root (.../LSM_1)."""
    return seed_dir.parents[1]


# ---------------------------------------------------------------------------
# Per-category check
# ---------------------------------------------------------------------------

def check_category(cat_key: str, entry: dict, base: str | None) -> dict:
    """카테고리 단위 진행 상황 dict 반환."""
    sd = _remap(entry["seed_dir"], base)
    cat = _cat_dir(sd)

    stages: dict[str, dict] = {}

    # ── Stage 1: roi_metadata.json ─────────────────────────────────────────
    stages["1"] = {"ok": (cat / "stage1_output" / "roi_metadata.json").exists()}

    # ── Stage 1b: seed_profile.json per seed ──────────────────────────────
    expected_seeds = sorted(p.stem for p in sd.glob("*.png")) if sd.exists() else []
    stage1b_dir = cat / "stage1b_output"
    missing_1b = [s for s in expected_seeds
                  if not (stage1b_dir / s / "seed_profile.json").exists()]
    stages["1b"] = {
        "ok": bool(expected_seeds) and not missing_1b,
        "total": len(expected_seeds),
        "missing": missing_1b,
    }

    # ── Stage 2: variant PNGs per seed ────────────────────────────────────
    seeds_1b = sorted(d.name for d in stage1b_dir.iterdir()
                      if d.is_dir()) if stage1b_dir.exists() else []
    stage2_dir = cat / "stage2_output"
    missing_2 = [s for s in seeds_1b
                 if not (stage2_dir / s).exists()
                 or not any((stage2_dir / s).glob("*.png"))]
    stages["2"] = {
        "ok": bool(seeds_1b) and not missing_2,
        "total": len(seeds_1b),
        "missing": missing_2,
    }

    # ── Stage 3: placement_map.json per seed ──────────────────────────────
    seeds_2 = sorted(d.name for d in stage2_dir.iterdir()
                     if d.is_dir()) if stage2_dir.exists() else []
    stage3_dir = cat / "stage3_output"
    missing_3 = [s for s in seeds_2
                 if not (stage3_dir / s / "placement_map.json").exists()]
    stages["3"] = {
        "ok": bool(seeds_2) and not missing_3,
        "total": len(seeds_2),
        "missing": missing_3,
    }

    # ── Stage 4: defect/*.png per seed ────────────────────────────────────
    seeds_3 = sorted(d.name for d in stage3_dir.iterdir()
                     if d.is_dir()) if stage3_dir.exists() else []
    stage4_dir = cat / "stage4_output"
    missing_4 = [s for s in seeds_3
                 if not (stage4_dir / s / "defect").exists()
                 or not any((stage4_dir / s / "defect").glob("*.png"))]
    stages["4"] = {
        "ok": bool(seeds_3) and not missing_4,
        "total": len(seeds_3),
        "missing": missing_4,
    }

    # ── Stage 5: quality_scores.json per seed ─────────────────────────────
    seeds_4 = sorted(d.name for d in stage4_dir.iterdir()
                     if d.is_dir()) if stage4_dir.exists() else []
    missing_5 = [s for s in seeds_4
                 if not (stage4_dir / s / "quality_scores.json").exists()]
    stages["5"] = {
        "ok": bool(seeds_4) and not missing_5,
        "total": len(seeds_4),
        "missing": missing_5,
    }

    # ── Stage 6: build_report.json ────────────────────────────────────────
    stages["6"] = {"ok": (cat / "augmented_dataset" / "build_report.json").exists()}

    return {
        "domain": entry["domain"],
        "cat_key": cat_key,
        "cat_dir": str(cat),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_cell(info: dict, stage: str) -> str:
    if info["ok"]:
        return "  OK"
    if stage in ("1", "6"):
        return "  --"
    total = info.get("total", 0)
    done = total - len(info.get("missing", []))
    return f"{done}/{total}"


def print_report(results: list[dict], verbose: bool = False) -> None:
    by_domain: dict[str, list] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    total_ok = 0

    for domain in sorted(by_domain):
        cats = by_domain[domain]
        print(f"\n{'='*62}")
        print(f"  {domain.upper()}  ({len(cats)} categories)")
        print(f"{'='*62}")
        hdr = f"  {'Category':<26}" + "".join(f"  {s:>4}" for s in STAGES)
        print(hdr)
        print(f"  {'-'*26}" + "  ----" * len(STAGES))

        for r in sorted(cats, key=lambda x: x["cat_key"]):
            all_ok = all(r["stages"][s]["ok"] for s in STAGES)
            marker = " " if all_ok else "!"
            cat_name = r["cat_key"].replace(f"{r['domain']}_", "", 1)
            cells = "".join(f"  {_fmt_cell(r['stages'][s], s):>4}" for s in STAGES)
            print(f"  {marker}{cat_name:<25}{cells}")
            if all_ok:
                total_ok += 1

            if verbose and not all_ok:
                for stg in STAGES:
                    info = r["stages"][stg]
                    if not info["ok"] and info.get("missing"):
                        for m in info["missing"][:5]:   # 최대 5개만 표시
                            print(f"      [S{stg}] {m}")
                        excess = len(info["missing"]) - 5
                        if excess > 0:
                            print(f"      [S{stg}] ... 외 {excess}건")

    print(f"\n{'='*62}")
    print(f"  완료: {total_ok}/{len(results)} categories")

    # 미완료 카테고리 목록 (재실행용)
    incomplete = [r for r in results
                  if not all(r["stages"][s]["ok"] for s in STAGES)]
    if not incomplete:
        return

    print(f"\n{'='*62}")
    print("  미완료 카테고리 (재실행 대상):")
    by_first: dict[str, list] = defaultdict(list)
    for r in incomplete:
        first = next(s for s in STAGES if not r["stages"][s]["ok"])
        by_first[first].append(r)

    for first_stage in STAGES:
        if first_stage not in by_first:
            continue
        print(f"\n  [Stage {first_stage}부터 재실행]")
        for r in by_first[first_stage]:
            print(f"    {r['domain']:6}  {r['cat_key']}")
            print(f"           cat_dir: {r['cat_dir']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AROMA Stage 1-6 진행 상황 확인")
    p.add_argument("--config", default="dataset_config.json",
                   help="dataset_config.json 경로 (기본: ./dataset_config.json)")
    p.add_argument("--base", default=None,
                   help=f"Drive 경로({_CONFIG_BASE!r})를 대체할 로컬 base 경로")
    p.add_argument("--domain", default=None,
                   help="특정 도메인만 확인 (isp / mvtec / visa)")
    p.add_argument("--workers", type=int, default=8,
                   help="병렬 워커 수 (기본: 8, Drive I/O 병렬화)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="누락 seed ID 상세 출력")
    return p.parse_args()


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    tasks = [
        (key, entry, args.base)
        for key, entry in config.items()
        if not key.startswith("_")
        and (args.domain is None or entry.get("domain") == args.domain)
    ]

    if args.workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda t: check_category(*t), tasks))
    else:
        results = [check_category(*t) for t in tasks]

    print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
