"""scripts/analyze_results.py

Stage 7 벤치마크 결과 집계 + 비교표 생성.

outputs/benchmark_results/ 를 순회하여:
  - benchmark_summary.json
  - comparison_table.md
  - comparison_table.csv
를 생성하고, baseline 대비 aroma_full / aroma_pruned 개선율을 출력한다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


GROUPS  = ["baseline", "aroma_full", "aroma_pruned"]
MODELS  = ["efficientnet_b4", "resnet50", "draem"]
METRICS = ["image_auroc", "image_f1", "pixel_auroc"]


def collect_results(output_root: Path) -> dict:
    """outputs/benchmark_results/{cat}/{model}/{group}/experiment_meta.json 수집."""
    summary: dict = {}
    for cat_dir in sorted(output_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        summary[cat] = {}
        for model in MODELS:
            summary[cat][model] = {}
            for group in GROUPS:
                meta = cat_dir / model / group / "experiment_meta.json"
                if meta.exists():
                    summary[cat][model][group] = json.loads(meta.read_text())
    return summary


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def build_comparison_table(summary: dict) -> list[dict]:
    """비교표 행 리스트 반환."""
    rows = []
    for cat, cat_data in sorted(summary.items()):
        for model, model_data in cat_data.items():
            row = {"category": cat, "model": model}
            for group in GROUPS:
                g_data = model_data.get(group, {})
                row[f"{group}_auroc"]    = g_data.get("image_auroc")
                row[f"{group}_f1"]       = g_data.get("image_f1")
                row[f"{group}_px_auroc"] = g_data.get("pixel_auroc")
            rows.append(row)
    return rows


def save_summary_json(summary: dict, output_root: Path) -> None:
    out = output_root / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out}")


def save_comparison_markdown(rows: list[dict], output_root: Path) -> None:
    lines = [
        "| Category | Model | baseline AUROC | aroma_full AUROC | aroma_pruned AUROC |",
        "|----------|-------|---------------|------------------|--------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['model']} "
            f"| {_fmt(r['baseline_auroc'])} "
            f"| {_fmt(r['aroma_full_auroc'])} "
            f"| {_fmt(r['aroma_pruned_auroc'])} |"
        )
    out = output_root / "comparison_table.md"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")


def save_comparison_csv(rows: list[dict], output_root: Path) -> None:
    fieldnames = ["category", "model"] + [
        f"{g}_{m}"
        for g in GROUPS
        for m in ["auroc", "f1", "px_auroc"]
    ]
    out = output_root / "comparison_table.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")


def print_improvement(summary: dict) -> None:
    """baseline 대비 aroma_full / aroma_pruned 평균 개선율 출력."""
    improvements: dict[str, list[float]] = {
        "aroma_full": [], "aroma_pruned": []
    }
    for cat_data in summary.values():
        for model_data in cat_data.values():
            base = model_data.get("baseline", {}).get("image_auroc")
            if base is None:
                continue
            for aug in ("aroma_full", "aroma_pruned"):
                aug_val = model_data.get(aug, {}).get("image_auroc")
                if aug_val is not None:
                    improvements[aug].append((aug_val - base) / (base + 1e-9) * 100)

    print("\n── 평균 Image-AUROC 개선율 (baseline 대비) ──")
    for aug, deltas in improvements.items():
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"  {aug:15s}: {avg:+.2f}%  (n={len(deltas)})")
        else:
            print(f"  {aug:15s}: 데이터 없음")


def main(output_root: str = "outputs/benchmark_results") -> None:
    root = Path(output_root)
    if not root.exists():
        print(f"결과 디렉터리 없음: {root}")
        return

    summary = collect_results(root)
    rows = build_comparison_table(summary)

    save_summary_json(summary, root)
    save_comparison_markdown(rows, root)
    save_comparison_csv(rows, root)
    print_improvement(summary)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="outputs/benchmark_results")
    args = p.parse_args()
    main(args.output_root)
