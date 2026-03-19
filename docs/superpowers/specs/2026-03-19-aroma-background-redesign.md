# AROMA Design Revision — Background Type Redesign & Workers Parallelism
**Date:** 2026-03-19
**Supersedes:** Section 4 (Stage 1 background characterization) and Section 5 (data flow schema) of `2026-03-17-aroma-design.md`

---

## 1. Scope

This revision covers two changes to the existing AROMA design:

1. **Background type taxonomy redesign** — replace the 5 original types (`smooth | textured | vertical_stripe | horizontal_stripe | complex_pattern`) with a structure-based universal taxonomy that generalizes across ISP-AD, MVTec AD, and the newly added VisA dataset.
2. **Workers parallel processing** — add `--workers` CLI argument to all stages following the CASDA implementation pattern.

---

## 2. Dataset Scope Update

| Dataset | Domain | Status |
|---|---|---|
| ISP-AD | Screen Printing | Existing |
| MVTec AD | Object categories | Existing |
| **VisA** | Mixed industrial + food (12 categories) | **Added** |

DAGM 2007 and PKU-HRI PCB remain excluded from scope.

**Rationale for VisA over KolektorSDD:** VisA introduces background types (periodic grid patterns, organic textures) that are meaningfully distinct from ISP-AD and MVTec. KolektorSDD's single-domain metallic surface overlaps too heavily with ISP-AD. VisA creates a wider triangulation across domains, strengthening the domain generalization claim.

---

## 3. Background Type Taxonomy (Revised)

### 3.1 Type Definitions

| Type | Definition | ISP-AD Example | MVTec Example | VisA Example |
|---|---|---|---|---|
| `smooth` | Uniform solid area — no pattern | Solder surface | Bottle surface, capsule | Capsule, chewing gum |
| `directional` | Consistent directional pattern (lines, stripes, grain) | Circuit traces, print lines | Wood grain, fabric weave | Bolt thread |
| `periodic` | Repeating unit pattern (grid, dot array) | Solder pad array | Tile, grid texture | Macaroni grid, PCB via array |
| `organic` | Non-directional natural/random texture | — | Leather, hazelnut | Cashew, candle |
| `complex` | Two or more of the above coexisting | Mixed pad regions | Mixed object backgrounds | PCB + component overlay |

### 3.2 Detection Algorithm (Lazy Evaluation)

Replaces the original `variance → Sobel → FFT` pipeline:

```
1. Variance analysis      → low variance → smooth  (fast early exit)
2. Autocorrelation        → strong periodicity → periodic
3. Gradient direction entropy → low entropy (consistent direction) → directional
4. LBP texture entropy    → high entropy, non-directional → organic
5. Multiple signals active → complex
```

**Key changes from original:**
- `vertical_stripe` + `horizontal_stripe` → unified into `directional` (direction information preserved as `dominant_angle` metadata)
- `textured` → `organic` (semantic clarification)
- FFT → Autocorrelation (more direct for periodic pattern detection)

### 3.3 `dominant_angle` Field

Added to `roi_boxes` entries. Valid only when `background_type == "directional"`, otherwise `null`.

- Computed as the dominant gradient direction (0–180°) from Sobel analysis
- Used in Stage 3 to align `linear_scratch` and `elongated` defect rotations with background direction

---

## 4. Updated `roi_metadata.json` Schema

```json
[
  {
    "image_id": "img_001",
    "global_mask": "masks/global/img_001.png",
    "local_masks": ["masks/local/img_001_zone0.png"],
    "roi_boxes": [
      {
        "level": "local",
        "box": [120, 80, 64, 64],
        "zone_id": 0,
        "background_type": "directional",
        "dominant_angle": 90.0,
        "continuity_score": 0.82,
        "stability_score": 0.75
      },
      {
        "level": "local",
        "box": [300, 100, 64, 64],
        "zone_id": 1,
        "background_type": "smooth",
        "dominant_angle": null,
        "continuity_score": 0.91,
        "stability_score": 0.88
      }
    ]
  }
]
```

---

## 5. Unified Matching Rule Table

A single domain-agnostic table replaces the previous per-domain tables. The structural relationship between defect geometry and background structure is domain-invariant — a linear scratch fits a directional background regardless of whether that background is a circuit trace (ISP-AD), wood grain (MVTec), or bolt thread (VisA).

| Defect Subtype | smooth | directional | periodic | organic | complex |
|---|---|---|---|---|---|
| linear_scratch | 0.5 | **1.0** | 0.7 | 0.3 | 0.3 |
| elongated | 0.6 | **0.9** | 0.7 | 0.4 | 0.4 |
| compact_blob | **0.9** | 0.4 | 0.7 | 0.6 | 0.5 |
| irregular | 0.5 | 0.4 | 0.5 | **0.8** | **0.9** |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

**Paper contribution:** The unified rule table is the primary evidence for "training-free domain generalization" — no per-domain tuning is required. Ablation Study 2 (suitability-guided vs. Gram-only vs. random placement) uses identical experimental conditions across all three datasets.

### Stage 3 `dominant_angle` Integration

When `background_type == "directional"` and `defect_subtype in ("linear_scratch", "elongated")`:

```python
placement["rotation"] = dominant_angle  # align defect axis to background direction
```

Otherwise, existing random rotation logic applies.

### `--domain` Argument (Retained)

`--domain` is retained for Stage 1 (SAID domain hint for ROI extraction) and Stage 3 (dominant_angle rotation logic). It no longer affects matching rules.

---

## 6. Workers Parallel Processing

All stage CLIs add `--workers`. Implementation follows the CASDA pattern exactly.

### CLI Definition (All Stages)

```python
parser.add_argument(
    "--workers", type=int, default=0,
    help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리)"
)
```

### Worker Count Resolution

```python
num_workers = args.workers
if num_workers < 0:
    cpu_count = os.cpu_count() or 4
    num_workers = max(1, cpu_count - 1)
```

### Conditional Parallel Execution

```python
use_parallel = num_workers > 1 and len(items) > 10
```

Items ≤ 10: sequential (process creation overhead exceeds benefit).

### Per-Stage Parallelization Unit

| Stage | Unit | Worker Function |
|---|---|---|
| `stage1` | 1 image | `_process_single_image_worker` |
| `stage1b` | Single seed (N/A) | `--workers` accepted, ignored |
| `stage2` | 1 variant | `_generate_single_variant_worker` |
| `stage3` | 1 image × seed combination | `_compute_placement_worker` |
| `stage4` | 1 image | `_synthesize_single_image_worker` |

### Implementation Rules (CASDA Pattern)

1. **Module-level worker functions** — required for `ProcessPoolExecutor` pickle serialization; never nested inside classes
2. **Objects recreated inside worker** — `BackgroundAnalyzer`, `DefectCharacterizer`, blenders, etc. are instantiated per worker; no global state shared
3. **`as_completed()` + tqdm** — progress tracking consistent with CASDA
4. **JSON written by main process only** — workers return dicts; file I/O happens after all futures resolve, preventing race conditions
5. **Colab recommendation** — `--workers 2` (RAM constraint from model loading per worker)

### New Utility: `utils/parallel.py`

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any
from tqdm import tqdm
import os

def resolve_workers(workers: int) -> int:
    if workers < 0:
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 1)
    return workers

def run_parallel(fn: Callable, tasks: List[Any], num_workers: int,
                 desc: str = "") -> List[Any]:
    use_parallel = num_workers > 1 and len(tasks) > 10
    if not use_parallel:
        return [fn(t) for t in tasks]
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fn, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                results.append(result)
    return results
```

---

## 7. Impact on Existing Design

| Component | Change |
|---|---|
| `utils/background_characterization.py` | Full rewrite — new taxonomy + new detection algorithm |
| `utils/suitability.py` | Matching rule table replaced with unified single table; domain branching removed |
| `stage1_roi_extraction.py` | `dominant_angle` field added to output; `--workers` added |
| `stage2_defect_seed_generation.py` | `--workers` added |
| `stage3_layout_logic.py` | Dominant angle rotation logic added; `--workers` added |
| `stage4_mpb_synthesis.py` | `--workers` added |
| `utils/parallel.py` | **New file** |
| `roi_metadata.json` schema | `dominant_angle` field added per roi_box |
| Matching rules | Per-domain tables → single unified table |
| Ablation Study 2 | Unchanged in structure; now uses identical conditions across all 3 datasets |

---

## 8. Open Decisions

- **VisA matching rule values:** Initial values derived from structural reasoning; empirical calibration via ablation study recommended before final paper submission.
- **Autocorrelation window size:** Default not yet specified; recommend grid_size × 2 as starting point.
- **`dominant_angle` precision:** Binned (0°, 45°, 90°, 135°) vs. continuous float — to be decided during Stage 3 implementation.
- **LBP parameters:** `radius=1, n_points=8` as default (scikit-image standard); may require tuning for high-res MVTec images.
