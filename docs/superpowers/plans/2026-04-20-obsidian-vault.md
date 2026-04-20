# AROMA Obsidian Vault 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `D:\project\aroma\AROMA\` 에 13개 Obsidian 노트를 생성하여 AROMA 파이프라인 전체를 문서화한다.

**Architecture:** 통일 템플릿(목적 → 입출력/Sentinel → 스크립트 → 핵심 파라미터 → 계산로직) 적용. 코드 블록은 실행 코드 없이 설정값·계산식만 발췌. 영어 파일명, 한국어 내용, 숫자 접두사로 Obsidian 탐색기 순서 고정.

**Tech Stack:** Markdown, Obsidian wikilinks (`[[...]]`)

---

## Task 1: Vault 디렉터리 생성 + 00-INDEX.md

**Files:**
- Create: `D:\project\aroma\AROMA\00-INDEX.md`

- [ ] **Step 1: Vault 디렉터리 생성**

```bash
mkdir "D:\project\aroma\AROMA"
```

- [ ] **Step 2: 00-INDEX.md 작성**

`D:\project\aroma\AROMA\00-INDEX.md` 에 아래 내용 작성:

```markdown
# AROMA Pipeline — Index

**Colab 실행 참조:** `docs/작업일지/stage1_6_execute.md`
**설정 파일:** `dataset_config.json`, `configs/benchmark_experiment.yaml`

## 파이프라인 Stage 목록

| # | Stage | 스크립트 | Sentinel |
|---|-------|---------|---------|
| 0 | 이미지 리사이즈 | `stage0_resize.py` | `{cat_dir}/.stage0_resize_512_done` |
| 1 | ROI 추출 | `stage1_roi_extraction.py` | `{cat_dir}/stage1_output/roi_metadata.json` |
| 1b | Seed 특성 분석 | `stage1b_seed_characterization.py` | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 2 | 변형 생성 | `stage2_defect_seed_generation.py` | `{cat_dir}/stage2_output/{seed_id}/` PNG ≥ 50개 |
| 3 | 레이아웃 로직 | `stage3_layout_logic.py` | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |
| 4 | MPB 합성 | `stage4_mpb_synthesis.py` | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 5 | 품질 점수 | `stage5_quality_scoring.py` | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |
| 6 | 데이터셋 구성 | `stage6_dataset_builder.py` | `{cat_dir}/augmented_dataset/build_report.json` |
| 7a | 벤치마크 실행 | `stage7_benchmark.py` | `outputs/benchmark_results/{cat}/{model}/{group}/experiment_meta.json` |

## 도메인

| 도메인 | 태스크 | 카테고리 | 비고 |
|--------|--------|---------|------|
| isp | classification | ASM, LSM_1, LSM_2 | LSM_2 벤치마크 제외 |
| mvtec | segmentation | 15개 전 카테고리 | |
| visa | segmentation | candle, capsules, cashew, chewinggum, fryum, macaroni, pcb, pipe_fryum | |

## 핵심 경로 (cat_dir 기준)

```
{cat_dir}/stage1_output/roi_metadata.json
{cat_dir}/stage1b_output/{seed_id}/seed_profile.json
{cat_dir}/stage4_output/{seed_id}/quality_scores.json
{cat_dir}/augmented_dataset/{baseline,aroma_full,aroma_pruned}/
outputs/benchmark_results/{cat_name}/{model}/{group}/experiment_meta.json
```

## 노트 목록

- [[01-Stage0-Resize]]
- [[02-Stage1-ROI]]
- [[03-Stage1b-Seed]]
- [[04-Stage2-Variants]]
- [[05-Stage3-Layout]]
- [[06-Stage4-MPB]]
- [[07-Stage5-Quality]]
- [[08-Stage6-Dataset]]
- [[09-Stage7a-Benchmark]]
- [[09-Stage7b-Results]]
- [[10-Dataset-Structure]]
- [[11-Parallel-Guide]]
```

- [ ] **Step 3: 커밋**

```bash
git add "AROMA/00-INDEX.md"
git commit -m "docs: add AROMA Obsidian vault index"
```

---

## Task 2: 01-Stage0-Resize.md + 02-Stage1-ROI.md

**Files:**
- Create: `D:\project\aroma\AROMA\01-Stage0-Resize.md`
- Create: `D:\project\aroma\AROMA\02-Stage1-ROI.md`

- [ ] **Step 1: 01-Stage0-Resize.md 작성**

```markdown
# Stage 0 — 이미지 리사이즈

## 목적

모든 카테고리 이미지를 512×512로 통일. Stage 1~6 전체 처리의 전제 조건.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]` (원본 이미지) |
| 출력 | `{cat_dir}/stage0_output/` |
| Sentinel | `{cat_dir}/.stage0_resize_512_done` |

## 스크립트

[[stage0_resize]] → `stage0_resize.py`
- `resize_category(entry, target_size, workers)` — 카테고리 단위 리사이즈
- `clean_category(entry)` — Stage 1~6 출력물 삭제

## 핵심 파라미터

```python
TARGET_SIZE = 512
CAT_THREADS = 4   # ThreadPoolExecutor, I/O bound
CLEAN_FIRST = True
```

## 계산 로직 / 임계값

- `CLEAN_FIRST=True`: `clean_category()` 로 이전 Stage 출력 삭제 후 리사이즈
- 카테고리 단위 `ThreadPoolExecutor(max_workers=4)` 병렬 처리
- 완료 시 sentinel 파일 생성 → 재실행 자동 skip
```

- [ ] **Step 2: 02-Stage1-ROI.md 작성**

```markdown
# Stage 1 — ROI 추출

## 목적

배경 이미지에서 ROI 추출 및 배경 텍스처 분류. Stage 3 배치 로직의 핵심 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]/*.png` |
| 출력 | `{cat_dir}/stage1_output/roi_metadata.json` |
| 출력 | `{cat_dir}/stage1_output/masks/global|local/` |
| Sentinel | `{cat_dir}/stage1_output/roi_metadata.json` |

## 스크립트

[[stage1_roi_extraction]] → `stage1_roi_extraction.py`
- `run_extraction(image_dir, output_dir, domain, roi_levels, grid_size, workers)`

## 핵심 파라미터

```python
roi_levels  = "both"   # global | local | both
grid_size   = 64       # 배경 분석 그리드 셀 크기 (px)
CAT_THREADS = 2        # Drive 동시 쓰기 안정성 고려
IMG_WORKERS = -1       # 이미지 단위 병렬 (cpu_count - 1)
```

## 계산 로직 / 임계값

**세그멘테이션:**
- Otsu: `cv2.THRESH_BINARY + cv2.THRESH_OTSU` (기본)
- SAM(vit_b): 체크포인트 있으면 우선, 가장 작은 마스크 선택

**ROI 추출:**
- `connectedComponentsWithStats`, `min_area = 100 px`
- 조건 미달 컴포넌트 없으면 전체 이미지를 단일 ROI로 폴백

**배경 분류 (lazy evaluation, 비용 순):**

| 단계 | 방법 | 임계값 | 결과 |
|------|------|--------|------|
| 1 | 픽셀 분산 | `variance < 100.0` | smooth |
| 2 | 그래디언트 방향 엔트로피 (8-bin, 22.5°/bin) | `entropy < 1.0` | directional |
| 3 | 자기상관 FFT 피크 (off-origin 정규화) | `peak > 0.15` | periodic |
| 4 | LBP 엔트로피 (P=8, R=1, uniform, 10-bin) | `entropy > 2.5` | organic |
| 5 | fallback | — | complex |

**roi_metadata.json 주요 필드:**
```json
{
  "image_id": "string",
  "roi_boxes": [{
    "level": "local",
    "box": [x, y, w, h],
    "background_type": "smooth|directional|periodic|organic|complex",
    "continuity_score": 0.0,
    "stability_score": 0.0,
    "dominant_angle": null
  }]
}
```
```

- [ ] **Step 3: 커밋**

```bash
git add "AROMA/01-Stage0-Resize.md" "AROMA/02-Stage1-ROI.md"
git commit -m "docs: add Stage 0 and Stage 1 Obsidian notes"
```

---

## Task 3: 03-Stage1b-Seed.md

**Files:**
- Create: `D:\project\aroma\AROMA\03-Stage1b-Seed.md`

- [ ] **Step 1: 03-Stage1b-Seed.md 작성**

```markdown
# Stage 1b — Seed 결함 특성 분석

## 목적

결함 seed 이미지의 기하 지표를 분석해 서브타입 분류. Stage 3 적합도 계산의 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["seed_dirs"][i]/*.png` |
| 출력 | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 출력 | `{cat_dir}/stage1b_output/{seed_id}/seed_mask.png` |
| Sentinel | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |

**seed_id 명명:**
- seed_dirs 1개: `{seed.stem}` (예: `crack_001`)
- seed_dirs 복수: `{seed_dir.name}_{seed.stem}` (예: `broken_large_crack_001`)

## 스크립트

[[stage1b_seed_characterization]] → `stage1b_seed_characterization.py`
- `run_seed_characterization(seed_defect, output_dir, model_checkpoint)`
- `run_seed_characterization_batch(tasks, workers, model_checkpoint)`

`utils/defect_characterization.py`
- `DefectCharacterizer.analyze_defect_region(mask)` → 4개 기하 지표
- `DefectCharacterizer.classify_defect_subtype(metrics)` → 서브타입 문자열

## 핵심 파라미터

```python
NUM_WORKERS = 4   # seed 단위 ThreadPoolExecutor
```

## 계산 로직 / 임계값

**마스크 추출:**
- SAM(vit_b) 우선, 실패 시 Otsu(`THRESH_BINARY_INV + THRESH_OTSU`) 폴백
- `segmentation_method` 필드에 `"sam"` 또는 `"otsu"` 기록

**4개 기하 지표:**

| 지표 | 계산 | 의미 |
|------|------|------|
| linearity | `1 - (λ_min / λ_max)` — 픽셀 좌표 공분산 행렬의 고유값 비 | 직선=1.0, 원=0.0 |
| solidity | `region_area / convex_hull_area` | 볼록 채움 정도 |
| extent | `region_area / bounding_box_area` | 바운딩박스 대비 채움 |
| aspect_ratio | `major_axis_length / minor_axis_length` | 장단축 비 |

**결함 서브타입 분류 (우선순위 순):**

| 우선순위 | 서브타입 | 조건 |
|---------|---------|------|
| 1 | `linear_scratch` | linearity > **0.85** AND aspect_ratio > **5.0** |
| 2 | `elongated` | aspect_ratio > **5.0** AND linearity > **0.6** |
| 3 | `compact_blob` | aspect_ratio < **2.0** AND solidity > **0.9** |
| 4 | `irregular` | solidity < **0.7** |
| 5 | `general` | otherwise |

**seed_profile.json 구조:**
```json
{
  "seed_path": "절대경로",
  "subtype": "linear_scratch|elongated|compact_blob|irregular|general",
  "linearity": 0.0,
  "solidity": 0.0,
  "extent": 0.0,
  "aspect_ratio": 0.0,
  "mask_path": "절대경로",
  "segmentation_method": "sam|otsu"
}
```
```

- [ ] **Step 2: 커밋**

```bash
git add "AROMA/03-Stage1b-Seed.md"
git commit -m "docs: add Stage 1b Obsidian note"
```

---

## Task 4: 04-Stage2-Variants.md + 05-Stage3-Layout.md

**Files:**
- Create: `D:\project\aroma\AROMA\04-Stage2-Variants.md`
- Create: `D:\project\aroma\AROMA\05-Stage3-Layout.md`

- [ ] **Step 1: 04-Stage2-Variants.md 작성**

```markdown
# Stage 2 — Defect Seed 변형 생성

## 목적

각 결함 seed에서 다양한 변형 이미지 생성. Stage 3 배치 평가의 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 입력 | `seed_profile["seed_path"]` (원본 seed 이미지) |
| 출력 | `{cat_dir}/stage2_output/{seed_id}/*.png` |
| Sentinel | `{cat_dir}/stage2_output/{seed_id}/` 에 PNG ≥ `NUM_VARIANTS`개 |

## 스크립트

[[stage2_defect_seed_generation]] → `stage2_defect_seed_generation.py`
- `run_seed_generation(seed_defect, num_variants, output_dir, seed_profile, workers)`

## 핵심 파라미터

```python
NUM_VARIANTS = 50   # seed당 생성할 변형 수
SEED_THREADS = 2    # seed 단위 외부 병렬
IMG_WORKERS  = -1   # 변형 이미지 단위 내부 병렬 (cpu_count - 1)
```

## 계산 로직 / 임계값

- `seed_profile.json`의 `subtype` 기반 변형 전략 적용
- 각 seed당 `NUM_VARIANTS`개 PNG 생성 → Stage 3에서 배치 평가
- skip 조건: `stage2_output/{seed_id}/` 에 PNG ≥ `NUM_VARIANTS`개
```

- [ ] **Step 2: 05-Stage3-Layout.md 작성**

```markdown
# Stage 3 — 레이아웃 로직 (적합도 기반 배치)

## 목적

각 배경 이미지의 ROI에 결함 seed를 배치할 최적 위치 결정. Stage 4 합성 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/stage1_output/roi_metadata.json` |
| 입력 | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 입력 | `{cat_dir}/stage2_output/{seed_id}/*.png` |
| 출력 | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |
| Sentinel | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |

## 스크립트

[[stage3_layout_logic]] → `stage3_layout_logic.py`
- `run_layout_logic(roi_metadata, defect_seeds_dir, output_dir, seed_profile, domain, use_gpu)`

`utils/suitability.py`
- `SuitabilityEvaluator.compute_suitability(defect_subtype, background_type, continuity_score, stability_score)`
- `GPUSuitabilityEvaluator.compute_batch(defect_subtype, roi_boxes)` — PyTorch 배치 연산

## 핵심 파라미터

```python
use_gpu = True    # GPUSuitabilityEvaluator 사용 (PyTorch 필요)
workers = -1      # CPU 모드 시 이미지 단위 병렬
```

## 계산 로직 / 임계값

**적합도 점수 (0.0~1.0):**
```
suitability = 0.4 × matching
            + 0.3 × continuity_score
            + 0.2 × stability_score
            + 0.1 × gram_similarity   (기본값 0.0)
```

**MATCHING_RULES (defect_subtype × background_type):**

| subtype | smooth | directional | periodic | organic | complex |
|---------|--------|-------------|---------|---------|---------|
| linear_scratch | 0.5 | **1.0** | 0.7 | 0.3 | 0.3 |
| elongated | 0.6 | **0.9** | 0.7 | 0.4 | 0.4 |
| compact_blob | **0.9** | 0.4 | 0.7 | 0.6 | 0.5 |
| irregular | 0.5 | 0.4 | 0.5 | **0.8** | **0.9** |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

**회전 결정:**
- `directional` 배경 + `linear_scratch` 또는 `elongated` → `rotation = dominant_angle` (결 정렬)
- 그 외 모든 경우 → `rotation = uniform(0, 360)`

**배치 좌표:** `x = roi_box.x + w//4`, `y = roi_box.y + h//4`

**GPU 모드:** 이미지별로 모든 ROI 스코어를 1회 계산 후 전체 seed에 재사용.

**placement_map.json 주요 필드:**
```json
[{
  "image_id": "string",
  "placements": [{
    "defect_path": "절대경로",
    "x": 0, "y": 0,
    "scale": 1.0,
    "rotation": 0.0,
    "suitability_score": 0.0,
    "matched_background_type": "smooth"
  }]
}]
```
```

- [ ] **Step 3: 커밋**

```bash
git add "AROMA/04-Stage2-Variants.md" "AROMA/05-Stage3-Layout.md"
git commit -m "docs: add Stage 2 and Stage 3 Obsidian notes"
```

---

## Task 5: 06-Stage4-MPB.md

**Files:**
- Create: `D:\project\aroma\AROMA\06-Stage4-MPB.md`

- [ ] **Step 1: 06-Stage4-MPB.md 작성**

```markdown
# Stage 4 — MPB 합성 (Modified Poisson Blending)

## 목적

결함 patch를 배경 이미지에 합성해 증강 defect 이미지 생성. Stage 5 품질 평가의 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]/*.png` (배경 이미지) |
| 입력 | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |
| 출력 | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 출력 | `{cat_dir}/stage4_output/{seed_id}/defect/{image_id}_mask.png` |
| Sentinel | `{cat_dir}/stage4_output/{seed_id}/defect/` 에 PNG 존재 |

## 스크립트

[[stage4_mpb_synthesis]] → `stage4_mpb_synthesis.py`
- `run_synthesis_batch(image_dir, seed_placement_maps, output_root, format, use_fast_blend, workers, png_compression, max_background_dim)`

## 핵심 파라미터

```python
USE_FAST_BLEND     = True   # False → seamlessClone (느리지만 품질 높음)
IMG_THREADS        = 4      # ThreadPoolExecutor, I/O+CPU 혼합
format             = "cls"  # Stage 6 연동 필수값 — "yolo" 사용 불가
MAX_BACKGROUND_DIM = None   # 절대 변경 금지
PNG_COMPRESSION    = {"isp": 3, "mvtec": 3, "visa": 1}
```

## 계산 로직 / 임계값

**합성 방식 비교:**

| 방식 | 속도 | 품질 | 사용 조건 |
|------|------|------|---------|
| seamlessClone (NORMAL_CLONE) | 느림 | 높음 | `use_fast_blend=False` |
| Gaussian soft-mask | ~10-30× 빠름 | 낮음 | `use_fast_blend=True` |

**Gaussian soft-mask 계산:**
```
ksize   = max(3, (min(patch_h, patch_w) // 6) | 1)   # 홀수 보정
alpha   = GaussianBlur(ones(rh, rw), ksize)
blended = patch × alpha + bg × (1 - alpha)
```

**ROI 마스크 저장:** `{image_id}_mask.png` — Stage 5 품질 평가에서 결함 영역 집중 분석용.

**배치 최적화:** 배경 이미지 1회 로드 후 전 seed 합성 (seed 수만큼 I/O 절감).

**주의사항:**
- `MAX_BACKGROUND_DIM=None` 유지 필수 — 변경 시 good(원본 해상도) vs defect(축소 해상도) 불일치 → 학습 불가
- `format="yolo"` 사용 시 Stage 6 연동 불가 — 반드시 `format="cls"`
- VisA `png_compression=1`: 원본 ~2MB(3032×2016) → 쓰기 속도 최적화
```

- [ ] **Step 2: 커밋**

```bash
git add "AROMA/06-Stage4-MPB.md"
git commit -m "docs: add Stage 4 Obsidian note"
```

---

## Task 6: 07-Stage5-Quality.md

**Files:**
- Create: `D:\project\aroma\AROMA\07-Stage5-Quality.md`

- [ ] **Step 1: 07-Stage5-Quality.md 작성**

```markdown
# Stage 5 — 합성 품질 점수

## 목적

합성 defect 이미지의 품질을 수치화. Stage 6 pruning의 기준.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 입력 | `{cat_dir}/stage4_output/{seed_id}/defect/{image_id}_mask.png` (선택) |
| 출력 | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |
| Sentinel | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |

## 스크립트

[[stage5_quality_scoring]] → `stage5_quality_scoring.py`
- `run_quality_scoring_batch(stage4_seed_dirs, workers, parallel_seeds)`

`utils/quality_scoring.py`
- `score_defect_images(stage4_seed_dir, w_artifact, w_blur, workers)`

## 핵심 파라미터

```python
w_artifact     = 0.5
w_blur         = 0.5
workers        = -1   # 이미지 단위 병렬 (seed 내부)
parallel_seeds = -1   # seed 단위 병렬 (카테고리 내부)
# L4 Colab (8 CPU) 권장: parallel_seeds=-1, workers=0
```

## 계산 로직 / 임계값

**ROI 마스크 전처리:**
- `{stem}_mask.png` 로드 → 31×31 타원 커널 dilate → 결함 인접 픽셀만으로 통계 산출

**artifact_score (높을수록 아티팩트 없음):**
```
Sobel gradient magnitude:
  outlier_ratio = mean(mag > mean_mag + 3σ)
  edge_score    = 1 - min(outlier_ratio × 10, 1)   ← 10% 이상이면 최악

Laplacian energy:
  hf_ratio  = lap_energy / (mean_mag + std_mag + 1e-6)
  hf_score  = 1 - min(hf_ratio / 5.0, 1)   ← ratio 5.0 이상이면 최악

artifact = 0.6 × edge_score + 0.4 × hf_score
```

**blur_score (높을수록 선명):**
```
해상도 보정:
  scale     = num_pixels / (256 × 256)   ← 기준 해상도 256×256
  lap_score = clip(lap_var / (1000.0 × scale), 0, 1)

Gradient contrast:
  edge_sharp = clip((P90/P50 - 1) / 9.0, 0, 1)   ← ratio 1~10 → 0~1

blur = 0.5 × lap_score + 0.5 × edge_sharp
```

**최종 품질 점수:**
```
quality_score = 0.5 × artifact + 0.5 × blur
```

**quality_scores.json 구조:**
```json
{
  "weights": {"artifact": 0.5, "blur": 0.5},
  "scores": [{"image_id": "...", "artifact_score": 0.0, "blur_score": 0.0, "quality_score": 0.0}],
  "stats": {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
}
```

**병렬화 조건:** `run_parallel` 은 `num_workers > 1 AND len(tasks) >= 2` 일 때만 ProcessPoolExecutor 사용.
```

- [ ] **Step 2: 커밋**

```bash
git add "AROMA/07-Stage5-Quality.md"
git commit -m "docs: add Stage 5 Obsidian note"
```

---

## Task 7: 08-Stage6-Dataset.md

**Files:**
- Create: `D:\project\aroma\AROMA\08-Stage6-Dataset.md`

- [ ] **Step 1: 08-Stage6-Dataset.md 작성**

```markdown
# Stage 6 — 증강 데이터셋 구성

## 목적

Stage 4 합성 이미지와 Stage 5 품질 점수를 사용해 3개 그룹 데이터셋 구성.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]` (원본 good 이미지) |
| 입력 | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 입력 | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |
| 출력 | `{cat_dir}/augmented_dataset/{baseline,aroma_full,aroma_pruned}/` |
| Sentinel | `{cat_dir}/augmented_dataset/build_report.json` |

## 스크립트

[[stage6_dataset_builder]] → `stage6_dataset_builder.py`
- `run_dataset_builder(cat_dir, image_dir, seed_dirs, pruning_threshold, ...)`

`utils/dataset_builder.py`
- `build_dataset_groups(...)` — 실제 구성 로직

## 핵심 파라미터

```python
PRUNING_THRESHOLD = 0.6
PRUNING_THRESHOLD_BY_DOMAIN = {"isp": 0.6, "mvtec": 0.6, "visa": 0.4}
SPLIT_RATIO          = 0.8    # good 80% train / 20% test
SPLIT_SEED           = 42     # 결정적 분할
BALANCE_DEFECT_TYPES = True   # seed_dirs 결함 유형별 균등 샘플링
NUM_IO_THREADS = 8
CAT_THREADS    = 2
```

## 계산 로직 / 임계값

**3개 데이터셋 그룹:**

| 그룹 | 구성 | 설명 |
|------|------|------|
| `baseline` | good 이미지만 | pretrained 특징 거리 기반 평가 |
| `aroma_full` | good + 전체 합성 | Stage 4 전체 사용 |
| `aroma_pruned` | good + 품질 필터링 합성 | `quality_score ≥ pruning_threshold` |

**도메인별 증강 비율 (good 대비 defect 비율):**

| 도메인 | aroma_full | aroma_pruned | 태스크 |
|--------|-----------|-------------|--------|
| isp | 1.0 (1:1) | 0.5 | classification |
| mvtec | 2.0 (2:1) | 1.5 | segmentation |
| visa | 2.0 (2:1) | 1.5 | segmentation |

**VisA pruning_threshold=0.4 이유:**
artifact_score의 `hf_ratio` 정규화 상수(5.0)가 candle 왁스 매끄러운 표면(mean_mag 작음)에서 hf_ratio 과대평가 → 낮은 threshold 적용.

**BALANCE_DEFECT_TYPES=True:**
`seed_id` 접두사(`broken_large_`, `broken_small_` 등)로 결함 유형 식별 후 할당량을 유형 수로 균등 분배 → 특정 유형 편향 방지.

**디렉터리 구조:**
```
augmented_dataset/
├── baseline/
│   ├── train/good/
│   └── test/{good,defect}/
├── aroma_full/
│   ├── train/{good,defect}/
│   └── test/ → symlink → baseline/test
└── aroma_pruned/
    ├── train/{good,defect}/
    └── test/ → symlink → baseline/test
```
```

- [ ] **Step 2: 커밋**

```bash
git add "AROMA/08-Stage6-Dataset.md"
git commit -m "docs: add Stage 6 Obsidian note"
```

---

## Task 8: 09-Stage7a-Benchmark.md + 09-Stage7b-Results.md

**Files:**
- Create: `D:\project\aroma\AROMA\09-Stage7a-Benchmark.md`
- Create: `D:\project\aroma\AROMA\09-Stage7b-Results.md`

- [ ] **Step 1: 09-Stage7a-Benchmark.md 작성**

```markdown
# Stage 7a — 벤치마크 실행

## 목적

3개 그룹 데이터셋에 대해 2개 모델을 학습·평가해 AROMA 합성 효과를 수치로 검증.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/augmented_dataset/{group}/` |
| 입력 | `configs/benchmark_experiment.yaml` |
| 출력 | `outputs/benchmark_results/{cat_name}/{model}/{group}/` |
| Sentinel | `outputs/benchmark_results/{cat_name}/{model}/{group}/experiment_meta.json` |

**전제 조건:** `{cat_dir}/augmented_dataset/baseline/test` 존재 (Stage 6 완료)

## 스크립트

[[stage7_benchmark]] → `stage7_benchmark.py`
- `run_benchmark(config_path, cat_dir, resume, output_dir)`

## 핵심 파라미터

```python
DOMAIN_FILTER = "isp"   # isp | mvtec | visa
resume        = True    # 중단 재개 가능
```

## 계산 로직 / 임계값

**모델 설정:**

| 모델 | 백본 | epochs | lr | optimizer | 특이사항 |
|------|-----|--------|-----|-----------|---------|
| `yolo11` | YOLO11n-cls.pt | 30 | 0.01 | SGD | val/ 없으면 임시 YAML 자동 생성 |
| `efficientdet_d0` | EfficientDet-D0 (EfficientNet-B0) | 30 | 0.0001 | Adam | |

**평가 메트릭:**

| 메트릭 | 적용 도메인 |
|--------|------------|
| `image_auroc` | isp, mvtec, visa |
| `image_f1` | isp, mvtec, visa |
| `pixel_auroc` | mvtec, visa만 |

**실험 규모:** 2 models × 3 groups × 24 categories = **144 runs**

**카테고리 제외:**
- ISP/`LSM_2`: LSM_1과 동일 센서·라인, 샘플만 다름 → benchmark 제외
- VisA: `macaroni2`, `pcb2-4` → dataset_config에서 완전 제거됨

**공통 설정:**
```yaml
image_size: 512
train_batch_size: 16
eval_batch_size: 32
eval_chunk_size: 64   # YOLO predict OOM 방지
num_workers: 4
seed: 42
```

**test set 준비:**
- `aroma_full/test`, `aroma_pruned/test` 없으면 `baseline/test` symlink 자동 생성
- symlink 실패 환경: `copytree` 폴백 (Drive FUSE O(n_files))
```

- [ ] **Step 2: 09-Stage7b-Results.md 작성**

```markdown
# Stage 7b — 결과 해석 가이드

## 메트릭 해석

**Image AUROC (주요 지표):**

| 범위 | 해석 |
|------|------|
| > 0.95 | 우수 — 합성 데이터가 실제 결함 패턴을 잘 반영 |
| 0.80 ~ 0.95 | 양호 |
| 0.60 ~ 0.80 | 미흡 — 합성 품질 또는 파이프라인 설정 재검토 |
| < 0.60 | 불량 |
| ≈ 1.0 (완벽) | 데이터 누수 의심 — test/train 분리 확인 필요 |

## 비교 분석 기준

**그룹 간 AUROC 향상 의미:**

| 비교 | 검증 내용 |
|------|---------|
| `baseline` → `aroma_full` 향상 | AROMA 합성의 전반적 유효성 |
| `aroma_full` → `aroma_pruned` 향상 | quality scoring pruning 효과 |
| `aroma_pruned` < `aroma_full` | pruning_threshold 너무 엄격 → threshold 낮추기 |

**도메인별 특성:**
- `isp` (classification): 낮은 증강비율(1:1), 결함 유형 다양성 낮음 → 높은 AUROC 기대
- `mvtec`/`visa` (segmentation): 높은 증강비율(2:1), 다양한 표면 텍스처 → 편차 큼

## 결과 파일 위치

```
outputs/benchmark_results/
└── {cat_name}/
    └── {yolo11|efficientdet_d0}/
        └── {baseline|aroma_full|aroma_pruned}/
            └── experiment_meta.json
               {"image_auroc": 0.0, "image_f1": 0.0, "pixel_auroc": 0.0}
```

## 재실행 절차

결과 초기화 후 재실행:
```python
import shutil
from pathlib import Path
for d in Path("outputs/benchmark_results").glob("*"):
    shutil.rmtree(d, ignore_errors=True)
```
```

- [ ] **Step 3: 커밋**

```bash
git add "AROMA/09-Stage7a-Benchmark.md" "AROMA/09-Stage7b-Results.md"
git commit -m "docs: add Stage 7 Obsidian notes"
```

---

## Task 9: 10-Dataset-Structure.md + 11-Parallel-Guide.md

**Files:**
- Create: `D:\project\aroma\AROMA\10-Dataset-Structure.md`
- Create: `D:\project\aroma\AROMA\11-Parallel-Guide.md`

- [ ] **Step 1: 10-Dataset-Structure.md 작성**

```markdown
# 데이터셋 디렉터리 구조

## cat_dir 기준 전체 트리

```
{cat_dir}/
├── .stage0_resize_512_done          ← Stage 0 sentinel
│
├── stage1_output/
│   ├── roi_metadata.json            ← Stage 1 sentinel + Stage 3 입력
│   └── masks/
│       ├── global/{image_id}.png
│       └── local/{image_id}_zone{n}.png
│
├── stage1b_output/
│   └── {seed_id}/
│       ├── seed_profile.json        ← Stage 1b sentinel + Stage 2/3 입력
│       └── seed_mask.png
│
├── stage2_output/
│   └── {seed_id}/
│       └── *.png                    ← NUM_VARIANTS=50개 이상 시 Stage 2 완료
│
├── stage3_output/
│   └── {seed_id}/
│       └── placement_map.json       ← Stage 3 sentinel + Stage 4 입력
│
├── stage4_output/
│   └── {seed_id}/
│       ├── defect/
│       │   ├── {image_id}.png       ← Stage 4 sentinel (PNG 존재 시)
│       │   └── {image_id}_mask.png  ← ROI 마스크 (Stage 5 입력)
│       └── quality_scores.json      ← Stage 5 sentinel + Stage 6 입력
│
└── augmented_dataset/               ← Stage 6 출력
    ├── build_report.json            ← Stage 6 sentinel
    ├── baseline/
    │   ├── train/good/
    │   └── test/{good,defect}/      ← 고정 test set (split_seed=42)
    ├── aroma_full/
    │   ├── train/{good,defect}/
    │   └── test/ → symlink → baseline/test
    └── aroma_pruned/
        ├── train/{good,defect}/
        └── test/ → symlink → baseline/test
```

## Benchmark 출력

```
outputs/benchmark_results/
└── {cat_name}/
    └── {yolo11|efficientdet_d0}/
        └── {baseline|aroma_full|aroma_pruned}/
            └── experiment_meta.json
```

## seed_id 명명 규칙

| seed_dirs 수 | seed_id 형식 | 예시 |
|-------------|-------------|------|
| 1개 | `{seed.stem}` | `crack_001` |
| 복수 | `{seed_dir.name}_{seed.stem}` | `broken_large_crack_001` |

## 설정 파일

| 파일 | 내용 |
|------|------|
| `dataset_config.json` | 카테고리별 `image_dir`, `seed_dirs[]`, `domain` |
| `configs/benchmark_experiment.yaml` | 모델, 증강비율, pruning 임계값, 평가 메트릭 |
```

- [ ] **Step 2: 11-Parallel-Guide.md 작성**

```markdown
# 병렬 설정 가이드

## Stage별 병렬 설정

| Stage | 외부 병렬 변수 | 내부 병렬 변수 | 병렬 종류 | 근거 |
|-------|-------------|-------------|---------|------|
| 0 | `CAT_THREADS=4` | `workers=-1` | Thread+Process | I/O bound 리사이즈 |
| 1 | `CAT_THREADS=2` | `IMG_WORKERS=-1` | Thread+Process | CPU-bound, Drive 동시쓰기 제한 |
| 1b | `NUM_WORKERS=4` | — | Thread | seed 단위 독립, CPU-bound |
| 2 | `SEED_THREADS=2` | `IMG_WORKERS=-1` | Thread+Process | 내부 병렬로 코어 포화 |
| 3 | 순차 | GPU 배치 | GPU | GPU 인스턴스 공유 불가 |
| 4 | 카테고리 순차 | `IMG_THREADS=4` | Thread | 배경 1회 로드 I/O 절감 |
| 5 | `SEED_THREADS=4` | `IMG_WORKERS=-1` | Process | I/O+CPU 혼합, seed 독립 |
| 6 | `CAT_THREADS=2` | `NUM_IO_THREADS=8` | Thread | I/O-bound Drive 복사 |
| 7 | 순차 (카테고리) | — | — | GPU 단일 점유, resume=True |

## run_parallel 동작 조건

```python
# utils/parallel.py
use_parallel = num_workers > 1 and len(tasks) >= 2
# → False 이면 순차 실행 (ProcessPoolExecutor 미사용)
```

- `workers=0` → `num_workers=0` → 항상 순차
- `workers=-1` → `num_workers = max(1, cpu_count - 1)`
- `workers=N` → `num_workers=N`

## Colab GPU별 권장값

| GPU | CPU 수 | Stage 5 권장 | 비고 |
|-----|--------|------------|------|
| T4 | 2 | `parallel_seeds=2, workers=0` | CPU 적음 → 중첩 금지 |
| L4 | 8 | `parallel_seeds=-1, workers=0` | seed 수준만 자동 병렬 |
| A100 | 12 | `parallel_seeds=-1, workers=0` | seed 수준만 자동 병렬 |

**중첩 병렬화 주의:**
```
총 프로세스 = parallel_seeds × workers
L4 8-CPU: parallel_seeds=4, workers=2 → 8 프로세스 (적정)
L4 8-CPU: parallel_seeds=-1(7), workers=-1(7) → 49 프로세스 (과부하)
```

## 병렬 설정 디버그

```python
import os
print(f"CPU: {os.cpu_count()}")
# → resolve_workers(-1) = max(1, cpu_count - 1)
```
```

- [ ] **Step 3: 최종 커밋**

```bash
git add "AROMA/10-Dataset-Structure.md" "AROMA/11-Parallel-Guide.md"
git commit -m "docs: add dataset structure and parallel guide Obsidian notes"
```

---

## 자기 검토

**스펙 커버리지:**
- [x] 00-INDEX: pipeline stage 목록, 도메인, 핵심 경로
- [x] Stage 0~7: 모든 Stage 노트 완비
- [x] 계산 로직: artifact/blur/quality 공식, matching 표, subtype 임계값
- [x] 데이터셋 구조: 전체 트리, sentinel 경로
- [x] 병렬 가이드: Stage별 설정, GPU별 권장값

**Placeholder 점검:** 없음 — 모든 파라미터 값 명시, 수식 완전 표기.

**타입 일관성:** `seed_id`, `cat_dir`, `image_id` 표기 통일. `quality_score` 필드명 Stage 5·6 일치.
