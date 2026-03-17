# AROMA Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 5 standalone Colab-runnable `.py` scripts implementing the AROMA defect synthesis pipeline (ROI extraction + background analysis → seed characterization → subtype-aware seed generation → suitability-guided placement → MPB synthesis) targeting ISP-AD and MVTec AD datasets.

**Architecture:** Each stage is an independent script communicating via filesystem (JSON metadata + image files). SAID provides unified ROI extraction for both domains. Background characterization (grid-based texture analysis) enriches ROI metadata. Seed characterization (geometric indicator analysis) drives subtype-aware variant generation and suitability-guided placement. MPB composites patches using Poisson blending with mixed boundary conditions.

**Tech Stack:** Python 3.10+, PyTorch, OpenCV, SciPy (Poisson solver), scikit-image (regionprops), NumPy, Ultralytics (YOLO11), timm (ResNet-50/Swin-T), segment-anything (SAM fallback for SAID), cleanfid (FID), argparse, pytest

---

## File Structure

```
aroma/
├── stage1_roi_extraction.py          # SAID-based global+local ROI extraction + background analysis
├── stage1b_seed_characterization.py  # SAM/Otsu mask extraction + geometric indicator analysis
├── stage2_defect_seed_generation.py  # TF-IDG one-shot variant generation (subtype-aware)
├── stage3_layout_logic.py            # Hybrid suitability scoring + ROI selection + placement
├── stage4_mpb_synthesis.py           # MPB Poisson blending synthesis
├── ablation_copy_paste.py            # Ablation 1 baseline: copy-paste without blending
├── ablation_random_placement.py      # Ablation 2 baseline: random placement ignoring ROI
├── utils/
│   ├── __init__.py
│   ├── io.py                         # JSON read/write, path validation
│   ├── mask.py                       # Mask load/save helpers
│   ├── background_characterization.py  # Grid-based background texture analysis (ported from CASDA)
│   ├── defect_characterization.py      # Geometric indicator analysis (ported from CASDA)
│   └── suitability.py                  # Matching rules + suitability scoring (domain-aware)
├── tests/
│   ├── conftest.py                   # Shared fixtures (synthetic images, temp dirs)
│   ├── test_io.py
│   ├── test_mask.py
│   ├── test_background.py            # BackgroundAnalyzer unit tests
│   ├── test_defect_characterization.py  # DefectCharacterizer unit tests
│   ├── test_suitability.py           # Matching rules + suitability scoring tests
│   ├── test_stage1.py                # ROI extraction + background fields in output schema
│   ├── test_stage1b.py               # Seed characterization output schema tests
│   ├── test_stage2.py                # Seed generation output shape/count + subtype strategy tests
│   ├── test_stage3.py                # Placement map schema + suitability_score field tests
│   └── test_stage4.py                # MPB output image shape + boundary tests
├── requirements.txt
└── docs/
    └── superpowers/
        ├── specs/2026-03-17-aroma-design.md
        └── plans/2026-03-17-aroma-implementation.md
```

---

## Task 1: Project Scaffold

**Files:** `requirements.txt`, `utils/__init__.py`, `utils/io.py`, `utils/mask.py`, `tests/conftest.py`

- [ ] **Step 1: Write failing tests for `io.py`**

```python
# tests/test_io.py
import json, pytest
from pathlib import Path
from utils.io import save_json, load_json, validate_dir

def test_save_and_load_json(tmp_path):
    data = {"image_id": "img_001", "roi_boxes": []}
    path = tmp_path / "meta.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == data

def test_validate_dir_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dir(tmp_path / "nonexistent")

def test_validate_dir_passes_if_exists(tmp_path):
    validate_dir(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_io.py -v
```

- [ ] **Step 3: Implement `utils/io.py`**

```python
import json
from pathlib import Path

def save_json(data, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)

def validate_dir(path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Directory not found: {path}")
```

- [ ] **Step 4: Write failing tests for `mask.py`**

```python
# tests/test_mask.py
import numpy as np
from utils.mask import save_mask, load_mask

def test_save_and_load_mask(tmp_path):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    path = tmp_path / "mask.png"
    save_mask(mask, path)
    loaded = load_mask(path)
    assert loaded.shape == (64, 64)
    assert loaded[20, 20] == 255
    assert loaded[0, 0] == 0
```

- [ ] **Step 5: Run test to verify it fails**
```bash
pytest tests/test_mask.py -v
```

- [ ] **Step 6: Implement `utils/mask.py`**

```python
import cv2
import numpy as np
from pathlib import Path

def save_mask(mask: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)

def load_mask(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
```

- [ ] **Step 7: Create `utils/__init__.py`** (empty)

- [ ] **Step 8: Create `requirements.txt`**

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scipy>=1.11.0
scikit-image>=0.21.0
numpy>=1.24.0
ultralytics>=8.0.0
timm>=0.9.0
segment-anything>=1.0
cleanfid>=0.1.35
pytest>=7.4.0
```

- [ ] **Step 9: Create `tests/conftest.py`**

```python
import numpy as np
import pytest

@pytest.fixture
def synthetic_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:50, 10:50] = [120, 80, 60]
    return img

@pytest.fixture
def synthetic_defect():
    patch = np.ones((16, 16, 3), dtype=np.uint8) * 200
    patch[4:12, 4:12] = [30, 30, 30]
    return patch

@pytest.fixture
def temp_image_dir(tmp_path, synthetic_image):
    import cv2
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img_001.png"), synthetic_image)
    return img_dir
```

- [ ] **Step 10: Run all utils tests to verify they pass**
```bash
pytest tests/test_io.py tests/test_mask.py -v
```

- [ ] **Step 11: Commit**
```bash
git add utils/ tests/conftest.py tests/test_io.py tests/test_mask.py requirements.txt
git commit -m "feat: project scaffold — utils/io, utils/mask, fixtures, requirements"
```

---

## Task 2: Background Characterization Utility

**Files:** `utils/background_characterization.py`, `tests/test_background.py`

Ported from CASDA `src/analysis/background_characterization.py` with domain-agnostic adjustments (removed Steel-specific assumptions; `variance_threshold` and `grid_size` are constructor params, not hardcoded).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_background.py
import numpy as np
import pytest
from utils.background_characterization import BackgroundAnalyzer, BackgroundType

@pytest.fixture
def smooth_image():
    return np.full((128, 128, 3), 180, dtype=np.uint8)

@pytest.fixture
def stripe_image():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(0, 128, 8):
        img[:, i:i+4] = 200  # vertical stripes
    return img

def test_smooth_patch_classified_as_smooth(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    bg = result["background_map"][0, 0]
    assert bg == BackgroundType.SMOOTH.value

def test_stripe_patch_classified_as_vertical_stripe(stripe_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(stripe_image)
    bg = result["background_map"][0, 0]
    assert bg == BackgroundType.VERTICAL_STRIPE.value

def test_continuity_uniform_region_is_high():
    analyzer = BackgroundAnalyzer(grid_size=32)
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    result = analyzer.analyze_image(img)
    score = analyzer.check_continuity(result, (0, 0, 128, 128))
    assert score > 0.7

def test_background_at_location_returns_dict(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    assert info is not None
    assert "background_type" in info
    assert "stability_score" in info
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_background.py -v
```

- [ ] **Step 3: Implement `utils/background_characterization.py`**

Port CASDA's `BackgroundAnalyzer` class. Key adaptations:
- Remove Steel-specific comments; make it generic
- Keep lazy evaluation (variance → Sobel → FFT order)
- `variance_threshold` default: 100.0 (tunable per domain)
- `grid_size` default: 64 (recommend 32 for small images, 128 for high-res)

- [ ] **Step 4: Run tests**
```bash
pytest tests/test_background.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add utils/background_characterization.py tests/test_background.py
git commit -m "feat: background characterization utility — grid-based texture analysis"
```

---

## Task 3: Defect Characterization Utility

**Files:** `utils/defect_characterization.py`, `tests/test_defect_characterization.py`

Ported from CASDA `src/analysis/defect_characterization.py`. No domain-specific changes required — the 4 geometric indicators are fully domain-agnostic.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_defect_characterization.py
import numpy as np
import pytest
from utils.defect_characterization import DefectCharacterizer

@pytest.fixture
def linear_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:32, 5:60] = 1  # thin horizontal line
    return mask

@pytest.fixture
def blob_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # square blob
    return mask

def test_linear_mask_high_linearity(linear_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(linear_mask)
    assert metrics is not None
    assert metrics["linearity"] > 0.7

def test_blob_mask_low_aspect_ratio(blob_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(blob_mask)
    assert metrics is not None
    assert metrics["aspect_ratio"] < 3.0

def test_linear_mask_classified_as_linear_scratch(linear_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(linear_mask)
    subtype = dc.classify_defect_subtype(metrics)
    assert subtype == "linear_scratch"

def test_blob_mask_classified_as_compact_blob(blob_mask):
    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(blob_mask)
    subtype = dc.classify_defect_subtype(metrics)
    assert subtype == "compact_blob"

def test_empty_mask_returns_none():
    dc = DefectCharacterizer()
    mask = np.zeros((64, 64), dtype=np.uint8)
    result = dc.analyze_defect_region(mask)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_defect_characterization.py -v
```

- [ ] **Step 3: Implement `utils/defect_characterization.py`**

Port CASDA's `DefectCharacterizer` class. No changes required beyond removing Steel-specific comments.

- [ ] **Step 4: Run tests**
```bash
pytest tests/test_defect_characterization.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add utils/defect_characterization.py tests/test_defect_characterization.py
git commit -m "feat: defect characterization utility — geometric indicator analysis"
```

---

## Task 4: Suitability Scoring Utility

**Files:** `utils/suitability.py`, `tests/test_suitability.py`

New module (not in CASDA). Implements domain-specific matching rule tables and hybrid suitability score formula.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_suitability.py
import pytest
from utils.suitability import SuitabilityEvaluator

def test_isp_linear_scratch_on_stripe_is_high():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.matching_score("linear_scratch", "horizontal_stripe")
    assert score == 1.0

def test_isp_compact_blob_on_smooth_is_high():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.matching_score("compact_blob", "smooth")
    assert score == 1.0

def test_isp_linear_scratch_on_complex_is_low():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.matching_score("linear_scratch", "complex_pattern")
    assert score < 0.5

def test_mvtec_general_returns_neutral():
    ev = SuitabilityEvaluator(domain="mvtec")
    for bg in ["smooth", "textured", "vertical_stripe", "horizontal_stripe", "complex_pattern"]:
        score = ev.matching_score("general", bg)
        assert abs(score - 0.7) < 0.01

def test_compute_suitability_returns_float_in_range():
    ev = SuitabilityEvaluator(domain="isp")
    score = ev.compute_suitability(
        defect_subtype="linear_scratch",
        background_type="horizontal_stripe",
        continuity_score=0.8,
        stability_score=0.75,
        gram_similarity=0.6
    )
    assert 0.0 <= score <= 1.0

def test_unknown_domain_falls_back_to_general_rules():
    ev = SuitabilityEvaluator(domain="unknown_domain")
    score = ev.matching_score("linear_scratch", "smooth")
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_suitability.py -v
```

- [ ] **Step 3: Implement `utils/suitability.py`**

```python
from typing import Optional

MATCHING_RULES = {
    "isp": {
        "linear_scratch":  {"vertical_stripe": 1.0, "horizontal_stripe": 1.0, "smooth": 0.5, "textured": 0.4, "complex_pattern": 0.3},
        "elongated":       {"vertical_stripe": 0.9, "horizontal_stripe": 0.9, "smooth": 0.6, "textured": 0.5, "complex_pattern": 0.4},
        "compact_blob":    {"vertical_stripe": 0.5, "horizontal_stripe": 0.5, "smooth": 1.0, "textured": 0.6, "complex_pattern": 0.4},
        "irregular":       {"vertical_stripe": 0.4, "horizontal_stripe": 0.4, "smooth": 0.5, "textured": 0.8, "complex_pattern": 1.0},
        "general":         {"vertical_stripe": 0.7, "horizontal_stripe": 0.7, "smooth": 0.7, "textured": 0.7, "complex_pattern": 0.7},
    },
    "mvtec": {
        "linear_scratch":  {"vertical_stripe": 0.8, "horizontal_stripe": 0.8, "smooth": 0.6, "textured": 0.5, "complex_pattern": 0.3},
        "elongated":       {"vertical_stripe": 0.7, "horizontal_stripe": 0.7, "smooth": 0.6, "textured": 0.6, "complex_pattern": 0.4},
        "compact_blob":    {"vertical_stripe": 0.4, "horizontal_stripe": 0.4, "smooth": 0.9, "textured": 0.7, "complex_pattern": 0.5},
        "irregular":       {"vertical_stripe": 0.4, "horizontal_stripe": 0.4, "smooth": 0.5, "textured": 0.8, "complex_pattern": 0.9},
        "general":         {"vertical_stripe": 0.7, "horizontal_stripe": 0.7, "smooth": 0.7, "textured": 0.7, "complex_pattern": 0.7},
    },
}
# Fallback: use "general" row from "isp" for unknown domains
_FALLBACK_RULES = MATCHING_RULES["isp"]["general"]


class SuitabilityEvaluator:
    # Hybrid score weights
    W_MATCHING    = 0.4
    W_CONTINUITY  = 0.3
    W_STABILITY   = 0.2
    W_GRAM        = 0.1

    def __init__(self, domain: str = "isp"):
        self._rules = MATCHING_RULES.get(domain, MATCHING_RULES["isp"])

    def matching_score(self, defect_subtype: str, background_type: str) -> float:
        row = self._rules.get(defect_subtype, self._rules.get("general", _FALLBACK_RULES))
        return float(row.get(background_type, 0.5))

    def compute_suitability(
        self,
        defect_subtype: str,
        background_type: str,
        continuity_score: float,
        stability_score: float,
        gram_similarity: float = 0.0,
    ) -> float:
        m = self.matching_score(defect_subtype, background_type)
        score = (
            self.W_MATCHING   * m
            + self.W_CONTINUITY * continuity_score
            + self.W_STABILITY  * stability_score
            + self.W_GRAM       * gram_similarity
        )
        return float(min(max(score, 0.0), 1.0))
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/test_suitability.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add utils/suitability.py tests/test_suitability.py
git commit -m "feat: suitability scoring utility — domain-aware matching rules + hybrid score"
```

---

## Task 5: Stage 1 — ROI Extraction + Background Analysis (`stage1_roi_extraction.py`)

**Files:** `stage1_roi_extraction.py`, `tests/test_stage1.py`

Key additions vs. original: background analysis per local ROI; `background_type`, `continuity_score`, `stability_score` fields in each `roi_boxes` entry.

- [ ] **Step 1: Write failing tests for Stage 1 output schema**

```python
# tests/test_stage1.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

def test_roi_metadata_schema(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="both")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    assert isinstance(meta, list)
    entry = meta[0]
    for key in ("image_id", "global_mask", "local_masks", "roi_boxes"):
        assert key in entry

def test_roi_boxes_have_background_fields(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="both")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    for box in meta[0]["roi_boxes"]:
        assert "background_type" in box, "Missing background_type"
        assert "continuity_score" in box, "Missing continuity_score"
        assert "stability_score" in box, "Missing stability_score"
        assert 0.0 <= box["continuity_score"] <= 1.0
        assert 0.0 <= box["stability_score"] <= 1.0

def test_global_mask_is_binary(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="global")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    mask = cv2.imread(meta[0]["global_mask"], cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert set(np.unique(mask)).issubset({0, 255})

def test_roi_boxes_within_image_bounds(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="isp", roi_levels="local")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    for entry in meta:
        for box in entry["roi_boxes"]:
            x, y, w, h = box["box"]
            assert x >= 0 and y >= 0
            assert x + w <= 64 and y + h <= 64
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_stage1.py -v
```

- [ ] **Step 3: Implement `stage1_roi_extraction.py`**

Based on the original implementation. Key additions:
- Import `BackgroundAnalyzer` from `utils.background_characterization`
- After extracting each local ROI box, call `analyzer.analyze_image(roi_crop)` and `analyzer.get_background_at_location()` + `analyzer.check_continuity()`
- Write `background_type`, `continuity_score`, `stability_score` into each `roi_boxes` entry

- [ ] **Step 4: Run Stage 1 tests**
```bash
pytest tests/test_stage1.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add stage1_roi_extraction.py tests/test_stage1.py
git commit -m "feat: stage1 — ROI extraction + background analysis per local ROI"
```

---

## Task 6: Stage 1b — Seed Characterization (`stage1b_seed_characterization.py`)

**Files:** `stage1b_seed_characterization.py`, `tests/test_stage1b.py`

New stage. Extracts a binary mask from the seed defect image, computes 4 geometric indicators, classifies subtype, writes `seed_profile.json`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage1b.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

@pytest.fixture
def linear_seed_path(tmp_path):
    img = np.full((32, 64, 3), 180, dtype=np.uint8)
    img[14:18, 4:60] = [30, 30, 30]  # dark horizontal line = linear defect
    p = tmp_path / "seed_linear.png"
    cv2.imwrite(str(p), img)
    return p

def test_seed_profile_schema(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    for key in ("seed_path", "subtype", "linearity", "solidity", "extent", "aspect_ratio", "mask_path"):
        assert key in profile, f"Missing key: {key}"

def test_seed_subtype_is_valid_string(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    valid = {"linear_scratch", "elongated", "compact_blob", "irregular", "general"}
    assert profile["subtype"] in valid

def test_seed_mask_is_saved(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    assert Path(profile["mask_path"]).exists()

def test_linear_seed_classified_as_linear_or_elongated(tmp_path, linear_seed_path):
    from stage1b_seed_characterization import run_seed_characterization
    out = tmp_path / "out"
    run_seed_characterization(str(linear_seed_path), str(out))
    profile = json.loads((out / "seed_profile.json").read_text())
    assert profile["subtype"] in {"linear_scratch", "elongated", "general"}
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_stage1b.py -v
```

- [ ] **Step 3: Implement `stage1b_seed_characterization.py`**

```python
import argparse
import cv2
import numpy as np
from pathlib import Path
from utils.io import save_json
from utils.mask import save_mask
from utils.defect_characterization import DefectCharacterizer

def extract_seed_mask(image: np.ndarray, checkpoint: str = None) -> np.ndarray:
    """Extract binary defect mask from seed image via SAM or Otsu fallback."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        ckpt = checkpoint or "sam_vit_b_01ec64.pth"
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        generator = SamAutomaticMaskGenerator(sam)
        masks = generator.generate(image)
        if masks:
            # Select smallest non-background mask (likely the defect)
            sorted_masks = sorted(masks, key=lambda m: m["area"])
            return (sorted_masks[0]["segmentation"].astype(np.uint8)) * 255
    except Exception:
        pass
    # Otsu fallback
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def run_seed_characterization(seed_defect: str, output_dir: str,
                               model_checkpoint: str = None):
    image = cv2.imread(seed_defect)
    if image is None:
        raise FileNotFoundError(f"Seed image not found: {seed_defect}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mask_bin = extract_seed_mask(image, model_checkpoint)
    mask_path = out / "seed_mask.png"
    save_mask(mask_bin, mask_path)

    # Convert to 0/1 binary for regionprops
    mask_01 = (mask_bin > 0).astype(np.uint8)

    dc = DefectCharacterizer()
    metrics = dc.analyze_defect_region(mask_01)
    if metrics is None:
        # Fallback: whole image as single region
        metrics = {"linearity": 0.0, "solidity": 1.0, "extent": 1.0,
                   "aspect_ratio": 1.0, "region_id": 0, "area": mask_01.sum(),
                   "bbox": (0, 0, image.shape[1], image.shape[0]),
                   "centroid": (image.shape[1] / 2, image.shape[0] / 2)}

    subtype = dc.classify_defect_subtype(metrics)
    profile = {
        "seed_path": str(seed_defect),
        "subtype": subtype,
        "linearity": round(metrics["linearity"], 4),
        "solidity": round(metrics["solidity"], 4),
        "extent": round(metrics["extent"], 4),
        "aspect_ratio": round(metrics["aspect_ratio"], 4),
        "mask_path": str(mask_path),
    }
    save_json(profile, out / "seed_profile.json")


def main():
    parser = argparse.ArgumentParser(description="AROMA Stage 1b: Seed Characterization")
    parser.add_argument("--seed_defect", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_checkpoint", default=None)
    args = parser.parse_args()
    run_seed_characterization(args.seed_defect, args.output_dir, args.model_checkpoint)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run Stage 1b tests**
```bash
pytest tests/test_stage1b.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add stage1b_seed_characterization.py tests/test_stage1b.py
git commit -m "feat: stage1b — seed defect characterization, subtype classification, seed_profile.json"
```

---

## Task 7: Stage 2 — Subtype-Aware Seed Generation (`stage2_defect_seed_generation.py`)

**Files:** `stage2_defect_seed_generation.py`, `tests/test_stage2.py`

Adds `--seed_profile` argument. When provided, applies subtype-specific augmentation strategy. Falls back to `general` strategy when omitted (backward-compatible).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage2.py
import pytest, cv2, json
import numpy as np
from pathlib import Path

def test_output_count(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds"
    run_seed_generation(str(seed_path), num_variants=5, output_dir=str(out_dir))
    assert len(list(out_dir.glob("*.png"))) == 5

def test_output_shape_matches_seed(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds"
    run_seed_generation(str(seed_path), num_variants=3, output_dir=str(out_dir))
    for v_path in out_dir.glob("*.png"):
        assert cv2.imread(str(v_path)).shape == synthetic_defect.shape

def test_variants_differ_from_seed(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds"
    run_seed_generation(str(seed_path), num_variants=3, output_dir=str(out_dir))
    diffs = [np.mean(np.abs(cv2.imread(str(p)).astype(float) - synthetic_defect.astype(float)))
             for p in out_dir.glob("*.png")]
    assert any(d > 1.0 for d in diffs)

def test_seed_profile_subtype_used(tmp_path, synthetic_defect):
    """When seed_profile is provided, generation runs without error for each subtype."""
    from stage2_defect_seed_generation import run_seed_generation
    from utils.io import save_json
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    for subtype in ["linear_scratch", "compact_blob", "irregular", "elongated", "general"]:
        profile_path = tmp_path / f"profile_{subtype}.json"
        save_json({"subtype": subtype, "linearity": 0.9, "solidity": 0.8,
                   "extent": 0.5, "aspect_ratio": 6.0}, profile_path)
        out_dir = tmp_path / f"seeds_{subtype}"
        run_seed_generation(str(seed_path), num_variants=2,
                            output_dir=str(out_dir), seed_profile=str(profile_path))
        assert len(list(out_dir.glob("*.png"))) == 2
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_stage2.py -v
```

- [ ] **Step 3: Implement `stage2_defect_seed_generation.py`**

Extend the original implementation with:
- `--seed_profile` CLI arg
- `_load_subtype(seed_profile_path)` helper
- Subtype dispatch: `_direction_preserving_warp`, `_isotropic_warp`, `_heavy_elastic_warp`, `_axis_aligned_warp` alongside existing `_texture_warp` + `_geometric_perturb`
- `generate_variant(seed, rng_seed, subtype)` routes to the correct strategy

- [ ] **Step 4: Run Stage 2 tests**
```bash
pytest tests/test_stage2.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add stage2_defect_seed_generation.py tests/test_stage2.py
git commit -m "feat: stage2 — subtype-aware TF-IDG seed generation"
```

---

## Task 8: Stage 3 — Suitability-Guided Layout Logic (`stage3_layout_logic.py`)

**Files:** `stage3_layout_logic.py`, `tests/test_stage3.py`

Replaces pure Gram matrix selection with hybrid suitability scoring. Adds `--seed_profile` and `--domain` CLI args. Writes `suitability_score` and `matched_background_type` into each placement entry.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage3.py
import json, pytest, cv2
import numpy as np
from pathlib import Path

def _make_roi_metadata(tmp_path, synthetic_image):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    mask_path = tmp_path / "masks" / "local" / "img_001_zone0.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    meta = [{"image_id": "img_001",
             "global_mask": "",
             "local_masks": [str(mask_path)],
             "roi_boxes": [{"level": "local", "box": [10, 10, 40, 40], "zone_id": 0,
                            "background_type": "smooth",
                            "continuity_score": 0.8, "stability_score": 0.75}]}]
    meta_path = tmp_path / "roi_metadata.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path

def _make_seeds_dir(tmp_path, synthetic_defect):
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    cv2.imwrite(str(seeds_dir / "variant_0000.png"), synthetic_defect)
    return seeds_dir

def _make_seed_profile(tmp_path):
    from utils.io import save_json
    profile = {"subtype": "compact_blob", "linearity": 0.1, "solidity": 0.95,
               "extent": 0.8, "aspect_ratio": 1.2}
    p = tmp_path / "seed_profile.json"
    save_json(profile, p)
    return p

def test_placement_map_schema(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    profile = _make_seed_profile(tmp_path)
    out = tmp_path / "output"
    run_layout_logic(str(meta), str(seeds), str(out), seed_profile=str(profile), domain="isp")
    data = json.loads((out / "placement_map.json").read_text())
    assert isinstance(data, list)
    p = data[0]["placements"][0]
    for key in ("defect_path", "x", "y", "scale", "rotation", "suitability_score", "matched_background_type"):
        assert key in p, f"Missing key: {key}"

def test_suitability_score_in_range(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    profile = _make_seed_profile(tmp_path)
    out = tmp_path / "output"
    run_layout_logic(str(meta), str(seeds), str(out), seed_profile=str(profile), domain="isp")
    data = json.loads((out / "placement_map.json").read_text())
    for entry in data:
        for p in entry["placements"]:
            assert 0.0 <= p["suitability_score"] <= 1.0

def test_placement_within_roi_bounds(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    out = tmp_path / "output"
    run_layout_logic(str(meta), str(seeds), str(out))
    data = json.loads((out / "placement_map.json").read_text())
    for entry in data:
        for p in entry["placements"]:
            assert 0 <= p["x"] <= 64
            assert 0 <= p["y"] <= 64
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
pytest tests/test_stage3.py -v
```

- [ ] **Step 3: Implement `stage3_layout_logic.py`**

Key changes from original:
- Import `SuitabilityEvaluator` from `utils.suitability`
- Load `seed_profile.json` to get `defect_subtype`
- For each (seed, roi_box) pair: compute `gram_similarity` (original logic) + call `evaluator.compute_suitability()` with `background_type`, `continuity_score`, `stability_score`, `gram_similarity`
- Select the ROI box with the highest `suitability_score` (replaces argmax of Gram similarity alone)
- Write `suitability_score` and `matched_background_type` into each placement entry
- When `--seed_profile` is omitted, use `subtype="general"` (backward-compatible)

- [ ] **Step 4: Run Stage 3 tests**
```bash
pytest tests/test_stage3.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add stage3_layout_logic.py tests/test_stage3.py
git commit -m "feat: stage3 — hybrid suitability-guided ROI selection (matching rules + continuity + gram)"
```

---

## Task 9: Stage 4 — MPB Synthesis (`stage4_mpb_synthesis.py`)

No changes from original design. Implement as-is.

- [ ] **Step 1–5:** (same as original plan — omitted for brevity; see original Task 5)

- [ ] **Step 6: Commit**
```bash
git add stage4_mpb_synthesis.py tests/test_stage4.py
git commit -m "feat: stage4 — MPB Poisson synthesis with mixed boundary conditions"
```

---

## Task 10: Ablation Baselines

**Files:** `ablation_copy_paste.py`, `ablation_random_placement.py`

No changes from original design — these baselines remain as-is to isolate the contribution of MPB blending and suitability-guided placement respectively.

- [ ] **Step 1: Implement `ablation_copy_paste.py`** (Ablation Study 1 — synthesis method)
- [ ] **Step 2: Implement `ablation_random_placement.py`** (Ablation Study 2 — placement strategy)
- [ ] **Step 3: Commit**
```bash
git add ablation_copy_paste.py ablation_random_placement.py
git commit -m "feat: ablation baselines — copy-paste and random placement"
```

---

## Task 11: End-to-End Pipeline Verification

- [ ] **Step 1: Create test fixtures**
```bash
mkdir -p tests/fixtures/images
python -c "
import cv2, numpy as np
img = np.zeros((64,64,3), dtype=np.uint8); img[10:50,10:50]=[120,80,60]
cv2.imwrite('tests/fixtures/images/img_001.png', img)
defect = np.ones((16,16,3), dtype=np.uint8)*200; defect[4:12,4:12]=[30,30,30]
cv2.imwrite('tests/fixtures/seed_defect.png', defect)
"
```

- [ ] **Step 2: Run the full pipeline on synthetic test data**
```bash
# Stage 1 + Stage 1b (parallel)
python stage1_roi_extraction.py \
  --image_dir tests/fixtures/images \
  --output_dir /tmp/aroma_test/stage1 \
  --domain mvtec --roi_levels both

python stage1b_seed_characterization.py \
  --seed_defect tests/fixtures/seed_defect.png \
  --output_dir /tmp/aroma_test/stage1b

# Stage 2
python stage2_defect_seed_generation.py \
  --seed_defect tests/fixtures/seed_defect.png \
  --seed_profile /tmp/aroma_test/stage1b/seed_profile.json \
  --num_variants 5 \
  --output_dir /tmp/aroma_test/stage2

# Stage 3
python stage3_layout_logic.py \
  --roi_metadata /tmp/aroma_test/stage1/roi_metadata.json \
  --defect_seeds_dir /tmp/aroma_test/stage2 \
  --seed_profile /tmp/aroma_test/stage1b/seed_profile.json \
  --image_dir tests/fixtures/images \
  --domain mvtec \
  --output_dir /tmp/aroma_test/stage3

# Stage 4
python stage4_mpb_synthesis.py \
  --image_dir tests/fixtures/images \
  --placement_map /tmp/aroma_test/stage3/placement_map.json \
  --output_dir /tmp/aroma_test/stage4 \
  --format cls
```
Expected: `stage4/defect/*.png` created without errors.

- [ ] **Step 3: Run full test suite**
```bash
pytest tests/ -v --tb=short
```
Expected: All PASS

- [ ] **Step 4: Final commit**
```bash
git add tests/fixtures/
git commit -m "test: integration fixtures and full pipeline verification"
```

---

## Colab Usage Notes

Each stage maps to a Colab cell. Stage 1 and Stage 1b can run as concurrent cells.

```python
# Cell 1 — Mount Drive + install deps
from google.colab import drive
drive.mount('/content/drive')
!pip install -r requirements.txt

# Cell 2 — Stage 1 (background analysis included)
!python stage1_roi_extraction.py \
  --image_dir /content/drive/MyDrive/dataset/images \
  --output_dir /content/drive/MyDrive/aroma_out/stage1 \
  --domain isp --roi_levels both

# Cell 3 — Stage 1b (runs in parallel with Cell 2)
!python stage1b_seed_characterization.py \
  --seed_defect /content/drive/MyDrive/seed.png \
  --output_dir /content/drive/MyDrive/aroma_out/stage1b

# Cell 4 — Stage 2
!python stage2_defect_seed_generation.py \
  --seed_defect /content/drive/MyDrive/seed.png \
  --seed_profile /content/drive/MyDrive/aroma_out/stage1b/seed_profile.json \
  --num_variants 20 \
  --output_dir /content/drive/MyDrive/aroma_out/stage2

# Cell 5 — Stage 3
!python stage3_layout_logic.py \
  --roi_metadata /content/drive/MyDrive/aroma_out/stage1/roi_metadata.json \
  --defect_seeds_dir /content/drive/MyDrive/aroma_out/stage2 \
  --seed_profile /content/drive/MyDrive/aroma_out/stage1b/seed_profile.json \
  --image_dir /content/drive/MyDrive/dataset/images \
  --domain isp \
  --output_dir /content/drive/MyDrive/aroma_out/stage3

# Cell 6 — Stage 4
!python stage4_mpb_synthesis.py \
  --image_dir /content/drive/MyDrive/dataset/images \
  --placement_map /content/drive/MyDrive/aroma_out/stage3/placement_map.json \
  --output_dir /content/drive/MyDrive/aroma_out/stage4 \
  --format yolo
```
