# -*- coding: utf-8 -*-
"""Shared helpers for AROMA paper figure scripts.

All experiment figures (Experiment 0/1/2) derive their numbers by parsing the
Markdown pipe-tables embedded in the manuscript (AROMA.txt), so figures stay in
sync with the single source of truth. The complexity figures parse the real
per-dataset complexity_report.json files.
"""
import os
import re
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = r"D:\project\aroma"
AROMA_TXT = os.path.join(PROJECT_ROOT, "AROMA연구분석", "Article", "AROMA.txt")
FIG_DIR = os.path.join(PROJECT_ROOT, "AROMA연구분석", "Article", "figure")
COMPLEXITY_DIR = os.path.join(PROJECT_ROOT, ".claude", ".etc", "complexity")

# ---------------------------------------------------------------------------
# Consistent color scheme
# ---------------------------------------------------------------------------
COLOR_BASELINE = "#7f7f7f"  # gray
COLOR_RANDOM = "#1f77b4"  # blue
COLOR_AROMA = "#d62728"  # red/orange
COLOR_AROMA_ALT = "#ff7f0e"  # orange (when 2-way)

DPI = 150


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
def setup_font():
    """Prefer Malgun Gothic (Windows Korean), fallback Arial, then DejaVu Sans."""
    preferred = ["Malgun Gothic", "Arial", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred if f in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ---------------------------------------------------------------------------
# Markdown table parsing
# ---------------------------------------------------------------------------
def _normalize_number(token):
    """Convert a table cell to float, handling unicode minus and +/- signs."""
    t = token.strip()
    t = t.replace("−", "-")  # unicode minus
    t = t.replace("−", "-")
    if t.startswith("+"):
        t = t[1:]
    try:
        return float(t)
    except ValueError:
        return t  # leave non-numeric (e.g., dataset/method names) as string


def parse_markdown_table(txt_path, caption_substring):
    """Find the first Markdown pipe-table that appears after a line containing
    `caption_substring`, returning (headers, rows) where rows is a list of dicts.

    Numeric cells are converted to float; the rest stay as strings.
    """
    with open(txt_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # locate caption line
    start = None
    for i, ln in enumerate(lines):
        if caption_substring in ln:
            start = i
            break
    if start is None:
        raise ValueError(f"Caption not found: {caption_substring!r}")

    # find the header row (first line beginning with '|' after the caption)
    hdr_idx = None
    for i in range(start, len(lines)):
        if lines[i].lstrip().startswith("|"):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError(f"No table after caption: {caption_substring!r}")

    def split_row(ln):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        return cells

    headers = split_row(lines[hdr_idx])
    # next line is the separator (|---|---|)
    rows = []
    for i in range(hdr_idx + 2, len(lines)):
        ln = lines[i]
        if not ln.lstrip().startswith("|"):
            break
        cells = split_row(ln)
        if len(cells) != len(headers):
            continue
        row = {h: _normalize_number(c) for h, c in zip(headers, cells)}
        rows.append(row)
    return headers, rows


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")
    return out
