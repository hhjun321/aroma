# AROMA Design Spec
**AROMA: Adaptive ROI-Aware Augmentation Framework**
Date: 2026-03-17
Last revised: 2026-03-17 (integrated CASDA background/defect analysis pipeline)

---

## 1. Overview

AROMA is a training-free industrial defect synthesis framework targeting **data scarcity** in visual inspection systems. It combines Modified Poisson Blending (MPB) and Training-Free Industrial Defect Generation (TF-IDG) within a 5-stage pipeline to produce physically realistic synthetic defect images.

**Core novelty claim:** Domain generalization across industrial inspection domains (ISP-AD → MVTec AD and vice versa) without retraining.

**Primary output:** SCI-level research paper + 5 standalone `.py` scripts runnable in Google Colab.

### Design revision rationale
The original 4-stage design used Gram matrix cosine similarity alone for ROI selection and placement. The revised design adds an explicit background characterization layer (grid-based texture classification) and a seed defect characterization layer (geometric indicator analysis) derived from the CASDA pipeline. This enables:
- **Interpretable ROI suitability scoring** (defect-background matching rules) complementing the Gram matrix
- **Subtype-aware variant generation** in Stage 2 (type-specific augmentation instead of uniform perturbation)
- **Additional ablation axis** (suitability-guided vs. Gram-only placement) for the paper

---

## 2. Constraints

- All training and experiments run in **Google Colab**
- Pipeline is split into **5 independent `.py` scripts**, one per stage
- All file paths are passed via **CLI arguments** (`argparse`)
- No shared global state or imports between scripts
- Stages communicate via **filesystem** only (JSON metadata + image files)
- Stage 1 and Stage 1b are **independent** and can run in parallel

---

## 3. Architecture

```
[Raw Images]                    [1 Defect Seed Sample]
      │                                  │
      ▼                                  ▼
stage1_roi_extraction.py        stage1b_seed_characterization.py
  → masks/, roi_metadata.json     → seed_profile.json
      │                                  │
      └──────────────┬───────────────────┘
                     ▼
         stage2_defect_seed_generation.py
           → defect_seeds/
                     │
                     ▼
         stage3_layout_logic.py
           → placement_map.json
                     │
                     ▼
         stage4_mpb_synthesis.py
           → augmented_dataset/
                     │
                     ▼
[Benchmarking: YOLO11, RT-DETRv2, ResNet-50, Swin-T]
```

---

## 4. Stage Definitions

### Stage 1 — `stage1_roi_extraction.py`
- **Purpose:** Unified two-level ROI extraction via **SAID (Segment All Industrial Defects, 2025)** + grid-based background characterization of each extracted ROI
- **CLI args:** `--image_dir`, `--output_dir`, `--domain [isp|mvtec]`, `--roi_levels [global|local|both]`, `--grid_size [default: 64]`
- **Backbone:** SAID — purpose-built for multi-domain industrial segmentation; no per-domain model switch required

**Level 1 — Global ROI (product boundary)**
SAID isolates the product boundary from background clutter. Applied identically across both domains.

**Level 2 — Local ROI (defect-prone sub-zones)**
SAID produces domain-aware sub-zone masks from the same forward pass:

| Domain | Local ROI Output |
|---|---|
| ISP-AD | Electrically significant zones — trace edges, solder pad centers, print line boundaries |
| MVTec AD | Semantic object sub-parts — screw heads, bottle caps, texture patch regions |

**Background characterization (post-SAID, per local ROI)**
After SAID extracts local ROIs, each ROI undergoes grid-based background analysis:

1. **Variance analysis** — distinguishes smooth vs. textured (cheap, runs first)
2. **Sobel edge direction** — identifies stripe patterns (vertical/horizontal)
3. **FFT frequency analysis** — distinguishes complex patterns (only when Sobel inconclusive)

Background types: `smooth` | `textured` | `vertical_stripe` | `horizontal_stripe` | `complex_pattern`

Each local ROI gets: `background_type`, `continuity_score` (uniformity of background within ROI), `stability_score` (average patch stability). These fields are written into `roi_metadata.json`.

- **Outputs:** `masks/global/<image_id>.png`, `masks/local/<image_id>_zone<n>.png`, `roi_metadata.json`

**Paper contribution:** Background characterization at extraction time allows suitability-guided ROI selection in Stage 3 without revisiting images — this is the key efficiency argument vs. pure Gram matrix approaches.

---

### Stage 1b — `stage1b_seed_characterization.py`
- **Purpose:** Characterize the single seed defect geometrically; produce `seed_profile.json` consumed by Stage 2 and Stage 3
- **CLI args:** `--seed_defect`, `--output_dir`, `--model_checkpoint [optional, for mask extraction]`
- **Runs independently of Stage 1** (can execute in parallel)

**Steps:**

1. **Mask extraction from seed image** — SAM (or Otsu thresholding fallback) generates a binary mask of the defect region in the seed image
2. **Geometric indicator computation** (4 indicators, ported from CASDA DefectCharacterizer):
   - `linearity` — eigenvalue ratio of covariance matrix; close to 1.0 = linear scratch
   - `solidity` — area / convex hull area; low = irregular boundary
   - `extent` — area / bounding box area; low = sparse defect
   - `aspect_ratio` — major axis / minor axis; high = elongated
3. **Subtype classification:**

| Subtype | Condition |
|---|---|
| `linear_scratch` | linearity > 0.85 AND aspect_ratio > 5.0 |
| `elongated` | aspect_ratio > 5.0 AND linearity > 0.6 |
| `compact_blob` | aspect_ratio < 2.0 AND solidity > 0.9 |
| `irregular` | solidity < 0.7 |
| `general` | otherwise |

- **Outputs:** `seed_profile.json`

---

### Stage 2 — `stage2_defect_seed_generation.py`
- **Purpose:** Generate N defect variants from a single seed sample using TF-IDG feature alignment, guided by the seed's geometric subtype
- **CLI args:** `--seed_defect`, `--seed_profile`, `--num_variants`, `--output_dir`
- **Logic:** Subtype-aware texture warping + geometric perturbation

**Subtype-specific augmentation strategy:**

| Subtype | Augmentation Strategy |
|---|---|
| `linear_scratch` | Direction-preserving warp (low transverse displacement, preserves dominant axis) |
| `elongated` | Axis-aligned elastic warp (stretch/compress along major axis) |
| `compact_blob` | Isotropic warp (equal radial displacement in all directions) |
| `irregular` | Heavy elastic warp (high-amplitude random displacement field) |
| `general` | Standard warp (original TF-IDG: random texture warp + rotation/flip) |

When `--seed_profile` is not provided, falls back to `general` strategy (backward-compatible).

- **Outputs:** `defect_seeds/<variant_id>.png`

---

### Stage 3 — `stage3_layout_logic.py`
- **Purpose:** Select the best ROI for each defect seed and compute placement coordinates using a hybrid suitability score (matching rules + continuity + Gram matrix)
- **CLI args:** `--roi_metadata`, `--defect_seeds_dir`, `--seed_profile`, `--output_dir`, `--image_dir [optional]`, `--domain [isp|mvtec]`

**Hybrid suitability scoring:**

```
suitability = 0.4 × matching_score
            + 0.3 × continuity_score
            + 0.2 × stability_score
            + 0.1 × gram_similarity
```

Where `matching_score` comes from domain-specific matching rule tables (see below), and `gram_similarity` is the Gram matrix cosine similarity from the original design (retained as a fine-grained signal).

**Domain-specific matching rules:**

ISP-AD:
| Defect Subtype | vertical_stripe | horizontal_stripe | smooth | textured | complex_pattern |
|---|---|---|---|---|---|
| linear_scratch | 1.0 | 1.0 | 0.5 | 0.4 | 0.3 |
| elongated | 0.9 | 0.9 | 0.6 | 0.5 | 0.4 |
| compact_blob | 0.5 | 0.5 | 1.0 | 0.6 | 0.4 |
| irregular | 0.4 | 0.4 | 0.5 | 0.8 | 1.0 |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

MVTec AD (domain-agnostic fallback — Gram matrix carries most of the signal):
| Defect Subtype | vertical_stripe | horizontal_stripe | smooth | textured | complex_pattern |
|---|---|---|---|---|---|
| linear_scratch | 0.8 | 0.8 | 0.6 | 0.5 | 0.3 |
| elongated | 0.7 | 0.7 | 0.6 | 0.6 | 0.4 |
| compact_blob | 0.4 | 0.4 | 0.9 | 0.7 | 0.5 |
| irregular | 0.4 | 0.4 | 0.5 | 0.8 | 0.9 |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

**ROI selection process** (replaces CASDA's position-search with a selection step):
For each defect seed, iterate over all candidate local ROIs from `roi_metadata.json`, compute suitability, select the highest-scoring ROI, then compute scale/rotation-matched placement coordinates within it.

- **Outputs:** `placement_map.json` (includes `suitability_score` per placement entry)

---

### Stage 4 — `stage4_mpb_synthesis.py`
- **Purpose:** Composite defect patches onto background images using MPB; remove boundary artifacts and match lighting
- **CLI args:** `--image_dir`, `--placement_map`, `--output_dir`, `--format [yolo|cls]`
- **Logic:** Solves Poisson equation with mixed boundary conditions: $\Delta f = \text{div}\,\mathbf{v}$ over $\Omega$, with $f|_{\partial\Omega} = f^*|_{\partial\Omega}$
- **Outputs:** `augmented_dataset/` in YOLO or classification folder format
- **No changes from original design**

---

## 5. Data Flow & Interface Schemas

| Stage | Key Inputs | Key Outputs |
|---|---|---|
| 1 | `--image_dir`, `--domain`, `--roi_levels` | `masks/global/*.png`, `masks/local/*.png`, `roi_metadata.json` |
| 1b | `--seed_defect` | `seed_profile.json` |
| 2 | `--seed_defect`, `--seed_profile` | `defect_seeds/*.png` |
| 3 | `--roi_metadata`, `--defect_seeds_dir`, `--seed_profile`, `--domain` | `placement_map.json` |
| 4 | `--image_dir`, `--placement_map` | `augmented_dataset/` |

**`roi_metadata.json` schema (updated):**
```json
[
  {
    "image_id": "img_001",
    "global_mask": "masks/global/img_001.png",
    "local_masks": ["masks/local/img_001_zone0.png", "masks/local/img_001_zone1.png"],
    "roi_boxes": [
      {
        "level": "local",
        "box": [120, 80, 64, 64],
        "zone_id": 0,
        "background_type": "horizontal_stripe",
        "continuity_score": 0.82,
        "stability_score": 0.75
      },
      {
        "level": "local",
        "box": [300, 100, 64, 64],
        "zone_id": 1,
        "background_type": "smooth",
        "continuity_score": 0.91,
        "stability_score": 0.88
      }
    ]
  }
]
```

**`seed_profile.json` schema (new):**
```json
{
  "seed_path": "seed_defect.png",
  "subtype": "linear_scratch",
  "linearity": 0.92,
  "solidity": 0.85,
  "extent": 0.31,
  "aspect_ratio": 7.4,
  "mask_path": "seed_mask.png"
}
```

**`placement_map.json` schema (updated):**
```json
[
  {
    "image_id": "img_001",
    "placements": [
      {
        "defect_path": "defect_seeds/v_001.png",
        "x": 120,
        "y": 85,
        "scale": 0.8,
        "rotation": 12,
        "suitability_score": 0.87,
        "matched_background_type": "horizontal_stripe"
      }
    ]
  }
]
```

---

## 6. Datasets

| Dataset | Domain | Role |
|---|---|---|
| ISP-AD | Screen Printing | Primary benchmark — domain-specific matching rules |
| MVTec AD | Object categories | Primary benchmark — domain-agnostic rules + Gram matrix |

DAGM 2007 excluded from scope.

---

## 7. Evaluation & Ablation Design

### Benchmarking Models

| Task | Model | Metric |
|---|---|---|
| Detection | YOLO11, RT-DETRv2 | mAP@0.5, F1-score |
| Classification | ResNet-50, Swin Transformer | F1-score |
| Generation quality | — | FID |

Detection and classification carry equal weight. Results reported as delta over baseline (Δ mAP, Δ F1, Δ FID).

### Ablation Study 1 — Synthesis Method

| Condition | Description |
|---|---|
| Baseline (no aug) | Train on real defects only |
| Copy-paste | Simple cut-and-paste, no blending |
| **AROMA (proposed)** | Full MPB synthesis |

### Ablation Study 2 — Placement Strategy

| Condition | Description |
|---|---|
| Random placement | Defects placed at random coordinates |
| Gram-only placement | ROI selection via Gram matrix cosine similarity only |
| **Suitability-guided placement (proposed)** | Hybrid: matching rules + continuity + Gram matrix |

### Ablation Study 3 — Seed Generation Strategy

| Condition | Description |
|---|---|
| Uniform augmentation | Same warp strategy for all defect subtypes (original design) |
| **Subtype-aware augmentation (proposed)** | Direction-preserving / isotropic / elastic warp per subtype |

### Domain Generalization Test (Core Novelty)
- Train augmented pipeline on ISP-AD → evaluate on MVTec AD
- Train augmented pipeline on MVTec AD → evaluate on ISP-AD
- Quantifies cross-domain robustness without retraining

---

## 8. SCI Paper Structure

| Section | Content |
|---|---|
| Abstract | AROMA overview, data scarcity motivation, domain generalization claim + key metric deltas |
| 1. Introduction | Industrial defect data scarcity, TF-IDG + MPB as training-free solution, domain generalization as headline |
| 2. Related Work | Standard Poisson blending limitations, existing defect augmentation, one-shot generation methods; unified industrial segmentation methods (Triad ICCV 2025, SAID 2025, UniCLIP-AD 2025) — positions AROMA as extending SAID's segmentation capability into synthesis |
| 3.1 ROI Extraction & Background Analysis | Stage 1 — unified two-level extraction via SAID; grid-based background characterization (variance → Sobel → FFT); `background_type`, `continuity_score` per ROI |
| 3.2 Seed Defect Characterization | Stage 1b — SAM-based mask extraction; 4 geometric indicators (linearity, solidity, extent, aspect_ratio); subtype classification table |
| 3.3 Subtype-Aware Seed Generation | Stage 2 — TF-IDG with subtype-specific warp strategies; comparison table of augmentation strategies per subtype |
| 3.4 Suitability-Guided Placement | Stage 3 — hybrid suitability score formulation; domain-specific matching rule tables; ROI selection algorithm |
| 3.5 MPB Synthesis | Stage 4 — full mathematical formulation of mixed-boundary Poisson equation |
| 4.1 Dataset Setup | ISP-AD, MVTec AD |
| 4.2 Ablation Study 1 | MPB vs. copy-paste vs. baseline |
| 4.3 Ablation Study 2 | Suitability-guided vs. Gram-only vs. random placement |
| 4.4 Ablation Study 3 | Subtype-aware vs. uniform seed generation |
| 4.5 Domain Generalization | Cross-domain results |
| 5. Conclusion | Training-free universality claim, limitations, future work |

---

## 9. Open Decisions

- **Publication venue:** Flexible for top-tier CV/ML or applied industrial vision journal
- **Matching rule calibration:** ISP-AD rules derived from domain knowledge; MVTec rules use conservative defaults pending empirical calibration — consider data-driven rule learning as a future extension
- **grid_size parameter:** Default 64 is calibrated for Severstal (1600×256); ISP-AD and MVTec resolutions may require per-domain tuning (recommend 32 for small images, 128 for high-res)
- **SAID availability:** Weights not yet publicly released as of 2026-03-17; SAM (ViT-B) used as backbone fallback via `--model_checkpoint`; swap to SAID checkpoint when available
- **Suitability weight tuning:** The hybrid weights (0.4 matching + 0.3 continuity + 0.2 stability + 0.1 gram) are initial estimates; ablation study will validate the weighting scheme
