# AROMA Background Redesign & Workers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite background characterization to a universal structure-based taxonomy (smooth | directional | periodic | organic | complex), replace per-domain suitability rules with a single unified table, add `dominant_angle` field to ROI metadata, and add `--workers` parallel processing to all stage CLIs following the CASDA pattern.

**Architecture:** `utils/background_characterization.py` is fully rewritten with a new 5-step lazy evaluation pipeline (variance → autocorrelation → gradient direction entropy → LBP → complex). `utils/suitability.py` drops domain branching in favor of one table. A new `utils/parallel.py` provides `resolve_workers()` and `run_parallel()` shared by all stages. All stage CLIs gain a `--workers` argument; worker functions are defined at module level for pickle compatibility.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, scikit-image (LBP, regionprops), SciPy, concurrent.futures (ProcessPoolExecutor), tqdm, pytest

**Execution context:** Work in `.worktrees/implement/` — all paths below are relative to that directory.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `utils/background_characterization.py` | **Rewrite** | New taxonomy + detection algorithm + `dominant_angle` |
| `utils/suitability.py` | **Rewrite** | Unified matching rule table, no domain branching |
| `utils/parallel.py` | **Create** | `resolve_workers()`, `run_parallel()` shared utility |
| `tests/test_background.py` | **Rewrite** | Tests for new types and `dominant_angle` |
| `tests/test_suitability.py` | **Rewrite** | Tests for unified table (no domain arg) |
| `tests/test_parallel.py` | **Create** | Tests for parallel utility |
| `stage1_roi_extraction.py` | **Modify** | Add `dominant_angle` to roi_boxes; add `--workers` |
| `stage1b_seed_characterization.py` | **Modify** | Add `--workers` (accept, ignored) |
| `stage2_defect_seed_generation.py` | **Modify** | Add `--workers`; parallelize variant generation |
| `stage3_layout_logic.py` | **Modify** | Add dominant_angle rotation logic; add `--workers` |
| `stage4_mpb_synthesis.py` | **Modify** | Add `--workers`; parallelize MPB synthesis |
| `tests/test_stage1.py` | **Modify** | Assert `dominant_angle` field in roi_boxes |
| `tests/test_stage3.py` | **Modify** | Assert rotation uses `dominant_angle` for directional+linear |

---

## Task 1: Rewrite `utils/background_characterization.py`

**Files:**
- Modify: `utils/background_characterization.py`
- Modify: `tests/test_background.py`

### Step 1: Rewrite `tests/test_background.py` with new taxonomy

- [ ] **Step 1: Write failing tests for new background types**

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
    """Vertical stripes — should classify as DIRECTIONAL."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(0, 128, 8):
        img[:, i:i + 4] = 200
    return img


@pytest.fixture
def periodic_image():
    """Grid of dots — should classify as PERIODIC."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            img[i:i + 8, j:j + 8] = 200
    return img


def test_smooth_classified_as_smooth(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    assert result["background_map"][0, 0] == BackgroundType.SMOOTH.value


def test_stripe_classified_as_directional(stripe_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(stripe_image)
    assert result["background_map"][0, 0] == BackgroundType.DIRECTIONAL.value


def test_periodic_classified_as_periodic(periodic_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(periodic_image)
    assert result["background_map"][0, 0] == BackgroundType.PERIODIC.value


def test_directional_has_dominant_angle(stripe_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(stripe_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    assert info["dominant_angle"] is not None
    assert 0.0 <= info["dominant_angle"] <= 180.0


def test_smooth_dominant_angle_is_none(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    assert info["dominant_angle"] is None


def test_continuity_uniform_region_is_high(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=32)
    result = analyzer.analyze_image(smooth_image)
    score = analyzer.check_continuity(result, (0, 0, 128, 128))
    assert score > 0.7


def test_get_background_at_location_has_required_fields(smooth_image):
    analyzer = BackgroundAnalyzer(grid_size=64)
    result = analyzer.analyze_image(smooth_image)
    info = analyzer.get_background_at_location(result, 32, 32)
    for key in ("background_type", "stability_score", "dominant_angle"):
        assert key in info


def test_old_type_names_not_present():
    """Ensure old taxonomy values are gone."""
    old_values = {"textured", "vertical_stripe", "horizontal_stripe", "complex_pattern"}
    new_values = {t.value for t in BackgroundType}
    assert old_values.isdisjoint(new_values), f"Old types still present: {old_values & new_values}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_background.py -v
```
Expected: Multiple FAILs — old enum values, missing `dominant_angle` in results.

- [ ] **Step 3: Rewrite `utils/background_characterization.py`**

```python
"""
Background Characterization Module

Universal structure-based background taxonomy (domain-agnostic):
  smooth | directional | periodic | organic | complex

Detection pipeline (lazy evaluation, cheapest first):
  1. Variance         → smooth
  2. Autocorrelation  → periodic
  3. Gradient entropy → directional  (also computes dominant_angle)
  4. LBP entropy      → organic
  5. Fallback         → complex
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from enum import Enum


class BackgroundType(Enum):
    SMOOTH      = "smooth"
    DIRECTIONAL = "directional"
    PERIODIC    = "periodic"
    ORGANIC     = "organic"
    COMPLEX     = "complex"


class BackgroundAnalyzer:
    """Grid-based background texture analyzer using universal structure-based taxonomy."""

    def __init__(
        self,
        grid_size: int = 64,
        variance_threshold: float = 100.0,
        periodic_threshold: float = 0.15,
        direction_entropy_threshold: float = 2.0,
        organic_entropy_threshold: float = 2.5,
    ):
        self.grid_size = grid_size
        self.variance_threshold = variance_threshold
        self.periodic_threshold = periodic_threshold
        self.direction_entropy_threshold = direction_entropy_threshold
        self.organic_entropy_threshold = organic_entropy_threshold

    def _compute_autocorrelation_peak(self, patch: np.ndarray) -> float:
        """Returns normalized off-origin autocorrelation peak (0-1). High = periodic."""
        f = np.fft.fft2(patch.astype(np.float32))
        ac = np.real(np.fft.ifft2(np.abs(f) ** 2))
        ac = ac / (ac[0, 0] + 1e-6)
        ac[0, 0] = 0.0
        return float(np.max(ac))

    def _compute_gradient_direction_entropy(
        self, patch: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """
        Returns (entropy_bits, dominant_angle_degrees).
        Low entropy → directional pattern. dominant_angle is None when entropy is high.
        Angles are in [0, 180) — orientation, not direction.
        """
        sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        angles = np.degrees(np.arctan2(sobel_y, sobel_x)) % 180.0  # orientation

        bins = np.linspace(0.0, 180.0, 9)  # 8 bins × 22.5°
        counts, _ = np.histogram(angles.flatten(), bins=bins,
                                 weights=magnitude.flatten())
        total = np.sum(counts) + 1e-6
        probs = counts / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        dominant_angle = None
        if entropy < self.direction_entropy_threshold:
            bin_idx = int(np.argmax(counts))
            dominant_angle = float(bins[bin_idx] + 11.25)  # bin centre

        return entropy, dominant_angle

    def _compute_lbp_entropy(self, patch: np.ndarray) -> float:
        """Returns LBP histogram entropy. High = organic/random texture."""
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(patch, P=8, R=1, method="uniform")
        counts, _ = np.histogram(lbp.ravel(), bins=10, density=True)
        counts = counts + 1e-12
        return float(-np.sum(counts * np.log2(counts)))

    def classify_patch(
        self, patch: np.ndarray
    ) -> Tuple[BackgroundType, float, Optional[float]]:
        """
        Classify a grayscale patch.

        Returns:
            (BackgroundType, stability_score 0-1, dominant_angle degrees or None)
            dominant_angle is only set when BackgroundType is DIRECTIONAL.
        """
        variance = float(np.var(patch))

        # Step 1: smooth
        if variance < self.variance_threshold:
            stability = float(np.clip(1.0 - variance / self.variance_threshold, 0.0, 1.0))
            return BackgroundType.SMOOTH, stability, None

        # Step 2: periodic
        ac_peak = self._compute_autocorrelation_peak(patch)
        if ac_peak > self.periodic_threshold:
            return BackgroundType.PERIODIC, float(np.clip(ac_peak, 0.0, 1.0)), None

        # Step 3: directional
        entropy, dominant_angle = self._compute_gradient_direction_entropy(patch)
        if entropy < self.direction_entropy_threshold:
            stability = float(np.clip(1.0 - entropy / self.direction_entropy_threshold, 0.0, 1.0))
            return BackgroundType.DIRECTIONAL, stability, dominant_angle

        # Step 4: organic
        lbp_entropy = self._compute_lbp_entropy(patch)
        if lbp_entropy > self.organic_entropy_threshold:
            return BackgroundType.ORGANIC, 0.5, None

        # Step 5: complex
        return BackgroundType.COMPLEX, 0.3, None

    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze full image using grid-based approach."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        gs = self.grid_size
        grid_h, grid_w = h // gs, w // gs

        background_map = np.empty((grid_h, grid_w), dtype=object)
        stability_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        angle_map = np.full((grid_h, grid_w), None, dtype=object)
        grid_info = []

        for i in range(grid_h):
            for j in range(grid_w):
                y1, x1 = i * gs, j * gs
                patch = gray[y1:y1 + gs, x1:x1 + gs]
                bg_type, stability, dominant_angle = self.classify_patch(patch)
                background_map[i, j] = bg_type.value
                stability_map[i, j] = stability
                angle_map[i, j] = dominant_angle
                grid_info.append({
                    "grid_id": (i, j),
                    "bbox": (x1, y1, x1 + gs, y1 + gs),
                    "background_type": bg_type.value,
                    "stability_score": float(stability),
                    "dominant_angle": dominant_angle,
                })

        return {
            "background_map": background_map,
            "stability_map": stability_map,
            "angle_map": angle_map,
            "grid_info": grid_info,
            "grid_size": gs,
            "grid_shape": (grid_h, grid_w),
        }

    def get_background_at_location(
        self, analysis_result: Dict, x: int, y: int
    ) -> Optional[Dict]:
        gs = analysis_result["grid_size"]
        grid_h, grid_w = analysis_result["grid_shape"]
        gi, gj = y // gs, x // gs
        if not (0 <= gi < grid_h and 0 <= gj < grid_w):
            return None
        return {
            "background_type": analysis_result["background_map"][gi, gj],
            "stability_score": float(analysis_result["stability_map"][gi, gj]),
            "dominant_angle": analysis_result["angle_map"][gi, gj],
            "grid_id": (gi, gj),
        }

    def check_continuity(
        self, analysis_result: Dict, bbox: Tuple[int, int, int, int]
    ) -> float:
        x1, y1, x2, y2 = bbox
        gs = analysis_result["grid_size"]
        grid_h, grid_w = analysis_result["grid_shape"]
        gi1, gj1 = y1 // gs, x1 // gs
        gi2 = min(y2 // gs, grid_h - 1)
        gj2 = min(x2 // gs, grid_w - 1)

        bg_map = analysis_result["background_map"]
        stab_map = analysis_result["stability_map"]
        region_bg = bg_map[gi1:gi2 + 1, gj1:gj2 + 1]
        region_stab = stab_map[gi1:gi2 + 1, gj1:gj2 + 1]

        if region_bg.size == 0:
            return 0.0

        _, counts = np.unique(region_bg, return_counts=True)
        uniformity = float(np.max(counts) / region_bg.size)
        avg_stability = float(np.mean(region_stab))
        return float(0.6 * uniformity + 0.4 * avg_stability)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_background.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/background_characterization.py tests/test_background.py
git commit -m "feat: rewrite background characterization — universal taxonomy (smooth|directional|periodic|organic|complex) + dominant_angle"
```

---

## Task 2: Rewrite `utils/suitability.py`

**Files:**
- Modify: `utils/suitability.py`
- Modify: `tests/test_suitability.py`

- [ ] **Step 1: Write failing tests for unified (domain-agnostic) table**

```python
# tests/test_suitability.py
import pytest
from utils.suitability import SuitabilityEvaluator


def test_linear_scratch_on_directional_is_max():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("linear_scratch", "directional")
    assert score == 1.0


def test_compact_blob_on_smooth_is_high():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("compact_blob", "smooth")
    assert score == 0.9


def test_linear_scratch_on_organic_is_low():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("linear_scratch", "organic")
    assert score < 0.5


def test_general_is_uniform_across_all_types():
    ev = SuitabilityEvaluator()
    bg_types = ["smooth", "directional", "periodic", "organic", "complex"]
    scores = [ev.matching_score("general", bg) for bg in bg_types]
    assert all(abs(s - 0.7) < 0.01 for s in scores)


def test_irregular_on_complex_is_highest_for_that_subtype():
    ev = SuitabilityEvaluator()
    score_complex = ev.matching_score("irregular", "complex")
    score_smooth = ev.matching_score("irregular", "smooth")
    assert score_complex > score_smooth


def test_compute_suitability_returns_float_in_range():
    ev = SuitabilityEvaluator()
    score = ev.compute_suitability(
        defect_subtype="linear_scratch",
        background_type="directional",
        continuity_score=0.8,
        stability_score=0.75,
        gram_similarity=0.6,
    )
    assert 0.0 <= score <= 1.0


def test_unknown_subtype_falls_back_to_general():
    ev = SuitabilityEvaluator()
    score = ev.matching_score("nonexistent_type", "smooth")
    assert abs(score - 0.7) < 0.01


def test_no_domain_argument_needed():
    """SuitabilityEvaluator takes no domain argument."""
    ev = SuitabilityEvaluator()
    assert ev is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_suitability.py -v
```
Expected: FAILs due to old domain-based API and old background type names.

- [ ] **Step 3: Rewrite `utils/suitability.py`**

```python
"""
Suitability Scoring — unified domain-agnostic matching rules.

A single table covers ISP-AD, MVTec AD, and VisA.
Structural relationships between defect geometry and background structure
are domain-invariant, so no per-domain branching is needed.
"""
from typing import Optional

# Unified matching rules — background types: smooth | directional | periodic | organic | complex
MATCHING_RULES = {
    "linear_scratch": {
        "smooth": 0.5, "directional": 1.0, "periodic": 0.7, "organic": 0.3, "complex": 0.3,
    },
    "elongated": {
        "smooth": 0.6, "directional": 0.9, "periodic": 0.7, "organic": 0.4, "complex": 0.4,
    },
    "compact_blob": {
        "smooth": 0.9, "directional": 0.4, "periodic": 0.7, "organic": 0.6, "complex": 0.5,
    },
    "irregular": {
        "smooth": 0.5, "directional": 0.4, "periodic": 0.5, "organic": 0.8, "complex": 0.9,
    },
    "general": {
        "smooth": 0.7, "directional": 0.7, "periodic": 0.7, "organic": 0.7, "complex": 0.7,
    },
}


class SuitabilityEvaluator:
    """Compute hybrid suitability score for defect-ROI placement."""

    W_MATCHING   = 0.4
    W_CONTINUITY = 0.3
    W_STABILITY  = 0.2
    W_GRAM       = 0.1

    def matching_score(self, defect_subtype: str, background_type: str) -> float:
        row = MATCHING_RULES.get(defect_subtype, MATCHING_RULES["general"])
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
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/suitability.py tests/test_suitability.py
git commit -m "feat: rewrite suitability — unified domain-agnostic matching rules (smooth|directional|periodic|organic|complex)"
```

---

## Task 3: Create `utils/parallel.py`

**Files:**
- Create: `utils/parallel.py`
- Create: `tests/test_parallel.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parallel.py
import os
import pytest
from utils.parallel import resolve_workers, run_parallel


def _double(x):
    return x * 2


def test_resolve_workers_zero_returns_zero():
    assert resolve_workers(0) == 0


def test_resolve_workers_negative_returns_positive():
    result = resolve_workers(-1)
    assert result >= 1


def test_resolve_workers_positive_unchanged():
    assert resolve_workers(4) == 4


def test_run_parallel_sequential_correctness():
    items = list(range(5))
    results = run_parallel(_double, items, num_workers=0)
    assert sorted(results) == [0, 2, 4, 6, 8]


def test_run_parallel_small_list_uses_sequential():
    """10 or fewer items always run sequentially regardless of workers."""
    items = list(range(5))
    results = run_parallel(_double, items, num_workers=4)
    assert sorted(results) == [0, 2, 4, 6, 8]


def test_run_parallel_large_list_parallel_correct():
    items = list(range(20))
    results = run_parallel(_double, items, num_workers=2)
    assert sorted(results) == sorted(x * 2 for x in items)


def test_run_parallel_filters_none():
    def maybe_none(x):
        return None if x % 2 == 0 else x
    items = list(range(10))
    results = run_parallel(maybe_none, items, num_workers=0)
    assert all(r is not None for r in results)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_parallel.py -v
```
Expected: ImportError — `utils.parallel` not found.

- [ ] **Step 3: Implement `utils/parallel.py`**

```python
"""
Parallel processing utility — CASDA pattern.

resolve_workers(workers):
  0  → sequential
  -1 → auto (cpu_count - 1)
  N  → N workers

run_parallel(fn, tasks, num_workers, desc):
  Sequential when num_workers <= 1 or len(tasks) <= 10.
  ProcessPoolExecutor otherwise.
  fn must be a module-level callable (pickle-safe).
  None results are filtered out.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any
from tqdm import tqdm
import os


def resolve_workers(workers: int) -> int:
    """Resolve --workers CLI value to actual worker count."""
    if workers < 0:
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 1)
    return workers


def run_parallel(
    fn: Callable,
    tasks: List[Any],
    num_workers: int,
    desc: str = "",
) -> List[Any]:
    """
    Run fn over tasks, sequentially or in parallel.

    Args:
        fn: Module-level callable. Receives one item from tasks.
        tasks: List of arguments, one per fn call.
        num_workers: Worker count from resolve_workers(). 0 = sequential.
        desc: tqdm progress bar label (parallel mode only).

    Returns:
        List of non-None results. Order is not guaranteed in parallel mode.
    """
    use_parallel = num_workers > 1 and len(tasks) > 10

    if not use_parallel:
        return [r for r in (fn(t) for t in tasks) if r is not None]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fn, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                results.append(result)
    return results
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_parallel.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/parallel.py tests/test_parallel.py
git commit -m "feat: add parallel utility — resolve_workers + run_parallel (CASDA pattern)"
```

---

## Task 4: Update `stage1_roi_extraction.py` — `dominant_angle` + `--workers`

**Files:**
- Modify: `stage1_roi_extraction.py`
- Modify: `tests/test_stage1.py`

- [ ] **Step 1: Add `dominant_angle` assertion to `tests/test_stage1.py`**

Add these two tests to the existing file:

```python
def test_roi_boxes_have_dominant_angle_field(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec", roi_levels="both")
    meta = json.loads((output_dir / "roi_metadata.json").read_text())
    for box in meta[0]["roi_boxes"]:
        assert "dominant_angle" in box, "Missing dominant_angle"
        # dominant_angle is float or None — both are valid
        assert box["dominant_angle"] is None or isinstance(box["dominant_angle"], float)


def test_workers_argument_accepted(tmp_path, temp_image_dir):
    from stage1_roi_extraction import run_extraction
    output_dir = tmp_path / "output"
    # workers=1 → sequential; should complete without error
    run_extraction(str(temp_image_dir), str(output_dir), domain="mvtec",
                   roi_levels="both", workers=1)
    assert (output_dir / "roi_metadata.json").exists()
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/test_stage1.py::test_roi_boxes_have_dominant_angle_field tests/test_stage1.py::test_workers_argument_accepted -v
```
Expected: FAIL.

- [ ] **Step 3: Update `stage1_roi_extraction.py`**

Key changes (apply to existing implementation):

1. Add `--workers` to argparse and `run_extraction` signature:
```python
parser.add_argument("--workers", type=int, default=0,
    help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리)")
```

2. Add module-level worker function. The existing `run_extraction` body processes each image sequentially — extract that per-image logic into this function:
```python
def _process_single_image_worker(args_tuple):
    """
    Module-level worker for ProcessPoolExecutor (pickle-safe).
    Processes one image: SAM/SAID ROI extraction + background analysis.
    All imports are inside the function so each subprocess initialises cleanly.
    """
    image_path_str, output_dir_str, domain, roi_levels, grid_size = args_tuple
    import cv2
    from pathlib import Path
    from utils.background_characterization import BackgroundAnalyzer
    from utils.mask import save_mask

    image_path = Path(image_path_str)
    output_dir = Path(output_dir_str)
    image_id = image_path.stem
    image = cv2.imread(image_path_str)
    if image is None:
        return None

    analyzer = BackgroundAnalyzer(grid_size=grid_size)

    # --- Global ROI (product boundary) via SAM/SAID fallback ---
    # Use the existing _extract_global_mask() helper (already in run_extraction).
    # Move that helper to module level so the worker can call it.
    global_mask = _extract_global_mask(image, domain)  # returns np.ndarray uint8
    global_mask_path = output_dir / "masks" / "global" / f"{image_id}.png"
    save_mask(global_mask, global_mask_path)

    # --- Local ROIs (defect-prone sub-zones) ---
    local_boxes = _extract_local_boxes(image, global_mask, domain, roi_levels)
    # _extract_local_boxes returns list of (x, y, w, h) tuples

    roi_boxes = []
    local_mask_paths = []
    for zone_id, (x, y, w, h) in enumerate(local_boxes):
        # Save local mask
        local_mask = _make_local_mask(image, x, y, w, h)
        local_mask_path = output_dir / "masks" / "local" / f"{image_id}_zone{zone_id}.png"
        save_mask(local_mask, local_mask_path)
        local_mask_paths.append(str(local_mask_path))

        # Background analysis on the cropped ROI region
        roi_crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY) \
                   if len(image.shape) == 3 else image[y:y + h, x:x + w]
        analysis = analyzer.analyze_image(image[y:y + h, x:x + w])
        cx, cy = w // 2, h // 2
        bg_info = analyzer.get_background_at_location(analysis, cx, cy)
        continuity = analyzer.check_continuity(analysis, (0, 0, w, h))

        roi_boxes.append({
            "level": "local",
            "box": [x, y, w, h],
            "zone_id": zone_id,
            "background_type": bg_info["background_type"],
            "dominant_angle": bg_info["dominant_angle"],   # float or None
            "continuity_score": round(continuity, 4),
            "stability_score": round(float(bg_info["stability_score"]), 4),
        })

    return {
        "image_id": image_id,
        "global_mask": str(global_mask_path),
        "local_masks": local_mask_paths,
        "roi_boxes": roi_boxes,
    }
```

> **Note for implementer:** `_extract_global_mask`, `_extract_local_boxes`, and `_make_local_mask` are helpers that must be defined at module level (not nested inside `run_extraction`). Move or create them there so the worker can import them. Each must accept only picklable arguments.

3. In `run_extraction`, use `run_parallel`:
```python
from utils.parallel import resolve_workers, run_parallel

num_workers = resolve_workers(workers)
tasks = [(str(p), output_dir_str, domain, roi_levels, grid_size) for p in image_paths]
results = run_parallel(_process_single_image_worker, tasks, num_workers,
                       desc=f"Stage1 ROI extraction (workers={num_workers})")
save_json([r for r in results if r], output_dir / "roi_metadata.json")
```

- [ ] **Step 4: Run all Stage 1 tests**

```bash
pytest tests/test_stage1.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add stage1_roi_extraction.py tests/test_stage1.py
git commit -m "feat: stage1 — add dominant_angle to roi_boxes + --workers parallel processing"
```

---

## Task 5: Update `stage1b_seed_characterization.py` — `--workers` (accepted, ignored)

**Files:**
- Modify: `stage1b_seed_characterization.py`

- [ ] **Step 1: Add `--workers` to argparse only**

```python
# In main():
parser.add_argument("--workers", type=int, default=0,
    help="병렬 워커 수 (0=순차 처리, -1=자동 감지, N>=2=N개 프로세스 병렬 처리)"
    " — Stage 1b processes a single seed; this argument is accepted for CLI"
    " consistency but has no effect.")
```

No other changes. `run_seed_characterization` signature unchanged.

- [ ] **Step 2: Verify existing Stage 1b tests still pass**

```bash
pytest tests/test_stage1b.py -v
```
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add stage1b_seed_characterization.py
git commit -m "feat: stage1b — add --workers argument (accepted, no effect; single-seed stage)"
```

---

## Task 6: Update `stage2_defect_seed_generation.py` — `--workers`

**Files:**
- Modify: `stage2_defect_seed_generation.py`
- Modify: `tests/test_stage2.py`

- [ ] **Step 1: Add workers test to `tests/test_stage2.py`**

```python
def test_workers_argument_parallel_produces_correct_count(tmp_path, synthetic_defect):
    from stage2_defect_seed_generation import run_seed_generation
    seed_path = tmp_path / "seed.png"
    cv2.imwrite(str(seed_path), synthetic_defect)
    out_dir = tmp_path / "seeds_parallel"
    # Use workers=1 (sequential) to keep test deterministic
    run_seed_generation(str(seed_path), num_variants=12, output_dir=str(out_dir), workers=1)
    assert len(list(out_dir.glob("*.png"))) == 12
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_stage2.py::test_workers_argument_parallel_produces_correct_count -v
```
Expected: FAIL — `workers` argument not accepted.

- [ ] **Step 3: Update `stage2_defect_seed_generation.py`**

1. Add `--workers` to argparse and `run_seed_generation(... workers=0)` signature.

2. Add module-level worker function:
```python
def _generate_single_variant_worker(args_tuple):
    """Module-level worker for variant generation (pickle-safe)."""
    seed_path_str, out_path_str, rng_seed, subtype = args_tuple
    import cv2
    import numpy as np
    seed = cv2.imread(seed_path_str)
    if seed is None:
        return None
    variant = generate_variant(seed, rng_seed=rng_seed, subtype=subtype)
    cv2.imwrite(out_path_str, variant)
    return out_path_str
```

3. In `run_seed_generation`, build task list and call `run_parallel`:
```python
from utils.parallel import resolve_workers, run_parallel

num_workers = resolve_workers(workers)
tasks = [
    (str(seed_defect), str(out_dir / f"variant_{i:04d}.png"), i, subtype)
    for i in range(num_variants)
]
run_parallel(_generate_single_variant_worker, tasks, num_workers,
             desc=f"Stage2 variant generation (workers={num_workers})")
```

- [ ] **Step 4: Run all Stage 2 tests**

```bash
pytest tests/test_stage2.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add stage2_defect_seed_generation.py tests/test_stage2.py
git commit -m "feat: stage2 — add --workers parallel variant generation"
```

---

## Task 7: Update `stage3_layout_logic.py` — `dominant_angle` rotation + `--workers`

**Files:**
- Modify: `stage3_layout_logic.py`
- Modify: `tests/test_stage3.py`

- [ ] **Step 1: Add tests to `tests/test_stage3.py`**

```python
def test_directional_linear_placement_uses_dominant_angle(tmp_path, synthetic_image, synthetic_defect):
    """When background is directional and defect is linear_scratch, rotation = dominant_angle."""
    from stage3_layout_logic import run_layout_logic
    from utils.io import save_json

    # Build roi_metadata with directional background + dominant_angle
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    mask_path = tmp_path / "masks" / "local" / "img_001_zone0.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)

    meta = [{"image_id": "img_001", "global_mask": "",
             "local_masks": [str(mask_path)],
             "roi_boxes": [{"level": "local", "box": [10, 10, 40, 40], "zone_id": 0,
                            "background_type": "directional", "dominant_angle": 45.0,
                            "continuity_score": 0.8, "stability_score": 0.75}]}]
    meta_path = tmp_path / "roi_metadata.json"
    meta_path.write_text(json.dumps(meta))

    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    cv2.imwrite(str(seeds_dir / "variant_0000.png"), synthetic_defect)

    profile_path = tmp_path / "seed_profile.json"
    save_json({"subtype": "linear_scratch", "linearity": 0.92, "solidity": 0.85,
               "extent": 0.31, "aspect_ratio": 7.4}, profile_path)

    out = tmp_path / "output"
    run_layout_logic(str(meta_path), str(seeds_dir), str(out),
                     seed_profile=str(profile_path), domain="isp")

    data = json.loads((out / "placement_map.json").read_text())
    placement = data[0]["placements"][0]
    assert placement["rotation"] == 45.0


def test_workers_argument_accepted(tmp_path, synthetic_image, synthetic_defect):
    from stage3_layout_logic import run_layout_logic
    # reuse _make_* helpers from existing tests
    meta = _make_roi_metadata(tmp_path, synthetic_image)
    seeds = _make_seeds_dir(tmp_path, synthetic_defect)
    out = tmp_path / "output_w"
    run_layout_logic(str(meta), str(seeds), str(out), workers=1)
    assert (out / "placement_map.json").exists()
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/test_stage3.py::test_directional_linear_placement_uses_dominant_angle tests/test_stage3.py::test_workers_argument_accepted -v
```
Expected: FAIL.

- [ ] **Step 3: Update `stage3_layout_logic.py`**

1. Add `--workers` to argparse and `run_layout_logic(... workers=0)` signature.

2. Add `dominant_angle` rotation logic in placement computation:
```python
# After selecting best ROI and computing placement:
rotation = compute_base_rotation(seed, roi_box)  # existing logic

# dominant_angle alignment for directional backgrounds
if (roi_box.get("background_type") == "directional"
        and defect_subtype in ("linear_scratch", "elongated")
        and roi_box.get("dominant_angle") is not None):
    rotation = float(roi_box["dominant_angle"])

placement["rotation"] = rotation
placement["matched_background_type"] = roi_box["background_type"]
```

3. Add module-level worker function and `run_parallel` call:
```python
def _compute_placement_worker(args_tuple):
    """Module-level worker for placement computation (pickle-safe)."""
    image_id, roi_boxes, seed_paths, defect_subtype, domain = args_tuple
    from utils.suitability import SuitabilityEvaluator
    evaluator = SuitabilityEvaluator()
    # ... suitability scoring + placement computation
    return {"image_id": image_id, "placements": [...]}
```

- [ ] **Step 4: Run all Stage 3 tests**

```bash
pytest tests/test_stage3.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add stage3_layout_logic.py tests/test_stage3.py
git commit -m "feat: stage3 — dominant_angle rotation alignment + --workers parallel placement"
```

---

## Task 8: Update `stage4_mpb_synthesis.py` — `--workers`

**Files:**
- Modify: `stage4_mpb_synthesis.py`
- Modify: `tests/test_stage4.py`

- [ ] **Step 1: Add workers test to `tests/test_stage4.py`**

```python
def test_workers_argument_accepted(tmp_path, temp_image_dir, synthetic_defect):
    """--workers=1 (sequential) must be accepted and produce output."""
    import json, cv2
    from stage4_mpb_synthesis import run_synthesis
    from utils.io import save_json

    # Minimal placement_map fixture
    defect_path = tmp_path / "defect_seeds" / "v_000.png"
    defect_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(defect_path), synthetic_defect)

    placement_map = [{"image_id": "img_001", "placements": [{
        "defect_path": str(defect_path),
        "x": 10, "y": 10, "scale": 1.0, "rotation": 0,
        "suitability_score": 0.8, "matched_background_type": "smooth",
    }]}]
    placement_path = tmp_path / "placement_map.json"
    save_json(placement_map, placement_path)

    out_dir = tmp_path / "output"
    run_synthesis(str(temp_image_dir), str(placement_path), str(out_dir),
                  format="cls", workers=1)
    assert len(list((out_dir / "defect").glob("*.png"))) > 0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_stage4.py::test_workers_argument_accepted -v
```
Expected: FAIL.

- [ ] **Step 3: Update `stage4_mpb_synthesis.py`**

1. Add `--workers` to argparse and `run_synthesis(... workers=0)` signature.

2. Add module-level worker function:
```python
def _synthesize_single_image_worker(args_tuple):
    """Module-level worker for MPB synthesis (pickle-safe)."""
    image_path_str, placements, output_dir_str, fmt = args_tuple
    import cv2
    from utils.mask import load_mask
    image = cv2.imread(image_path_str)
    if image is None:
        return None
    # ... existing MPB blending logic per placement
    return result_paths
```

3. In `run_synthesis`, call `run_parallel`:
```python
from utils.parallel import resolve_workers, run_parallel

num_workers = resolve_workers(workers)
tasks = [(str(img_path), placements, output_dir_str, fmt) for img_path, placements in ...]
run_parallel(_synthesize_single_image_worker, tasks, num_workers,
             desc=f"Stage4 MPB synthesis (workers={num_workers})")
```

- [ ] **Step 4: Run all Stage 4 tests**

```bash
pytest tests/test_stage4.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add stage4_mpb_synthesis.py tests/test_stage4.py
git commit -m "feat: stage4 — add --workers parallel MPB synthesis"
```

---

## Task 9: Full Test Suite Verification

- [ ] **Step 1: Run entire test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All PASS. No references to old background type names (`textured`, `vertical_stripe`, `horizontal_stripe`, `complex_pattern`) in any passing test.

- [ ] **Step 2: Verify old type names are gone from source**

```bash
grep -r "vertical_stripe\|horizontal_stripe\|TEXTURED\|VERTICAL_STRIPE\|HORIZONTAL_STRIPE" utils/ stage*.py
```
Expected: No output.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "test: full suite verification — background redesign + workers complete"
```
