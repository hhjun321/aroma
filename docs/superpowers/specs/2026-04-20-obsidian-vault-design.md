# AROMA Obsidian Vault 설계 문서

**날짜:** 2026-04-20  
**목적:** AROMA 프로젝트의 Obsidian vault 문서 구조 설계  
**참조:** `docs/작업일지/stage1_6_execute.md`

---

## 목표

1. **Claude 참조용** — 새 대화 시작 시 `00-INDEX.md` 하나로 파일경로·sentinel·파라미터 컨텍스트 파악
2. **사용자 위키** — 파이프라인 구조, 스크립트 파라미터, 임계값, 계산 로직 기록

---

## Vault 위치

```
D:\project\aroma\AROMA   ← Obsidian vault root
```

---

## 파일 목록

| 파일 | 내용 |
|------|------|
| `00-INDEX.md` | 파이프라인 전체 개요 (Claude 참조용, 기술 중심) |
| `01-Stage0-Resize.md` | 이미지 리사이즈 (512×512) |
| `02-Stage1-ROI.md` | ROI 추출 + 배경 분류 |
| `03-Stage1b-Seed.md` | Seed 결함 특성 분석 |
| `04-Stage2-Variants.md` | Defect seed 변형 생성 |
| `05-Stage3-Layout.md` | 적합도 기반 배치 로직 |
| `06-Stage4-MPB.md` | Modified Poisson Blending 합성 |
| `07-Stage5-Quality.md` | 합성 품질 점수 계산 |
| `08-Stage6-Dataset.md` | 증강 데이터셋 구성 |
| `09-Stage7a-Benchmark.md` | 벤치마크 실행 |
| `09-Stage7b-Results.md` | 결과 해석 가이드 |
| `10-Dataset-Structure.md` | 데이터셋 디렉터리 구조 |
| `11-Parallel-Guide.md` | 병렬 설정 가이드 |

---

## 네이밍 원칙

- 숫자 접두사(`00-`, `01-` ...) → Obsidian 파일 탐색기 순서 고정
- 영어 파일명 → Claude Glob/Read 접근 시 인코딩 문제 없음
- 노트 내부: 제목·설명은 한국어, 코드·파라미터·경로는 영어

---

## 각 Stage 노트 템플릿 (통일 A형)

모든 Stage 노트는 다음 섹션 순서를 따른다. 내용이 없는 섹션은 생략 가능.

```
## 목적
## 입력 / 출력 / Sentinel
## 스크립트
## 핵심 파라미터
## 계산 로직 / 임계값
```

코드 블록은 실행 코드가 아니라 **설정값만 발췌**한 파라미터 블록.

---

## 각 노트 콘텐츠 명세

### 00-INDEX.md
- Pipeline stage 목록: 번호, 스크립트명, sentinel 경로
- 도메인 목록: isp / mvtec / visa
- 핵심 출력 경로 테이블 (`cat_dir` 기준)
- 링크: 각 Stage 노트 wikilink

---

### 01-Stage0-Resize.md
**스크립트:** `stage0_resize.py`  
**파라미터:**
```python
TARGET_SIZE = 512
CAT_THREADS = 4
CLEAN_FIRST = True
```
**로직:**
- 카테고리 단위 ThreadPoolExecutor
- sentinel: `{cat_dir}/.stage0_resize_{TARGET_SIZE}_done`
- CLEAN_FIRST=True: Stage 1~6 출력물 먼저 삭제 후 리사이즈

---

### 02-Stage1-ROI.md
**스크립트:** `stage1_roi_extraction.py`  
**파라미터:**
```python
roi_levels  = "both"      # global | local | both
grid_size   = 64          # 배경 분석 그리드 셀 크기 (px)
CAT_THREADS = 2
IMG_WORKERS = -1
```
**계산 로직:**

*세그멘테이션:* Otsu (`THRESH_BINARY + THRESH_OTSU`). SAM(vit_b) 체크포인트 있으면 우선.  
*ROI 추출:* connectedComponentsWithStats, `min_area=100`. 조건 미달 시 전체 이미지 폴백.

*배경 분류 (lazy evaluation, 비용 순):*

| 단계 | 방법 | 판별 임계값 | 결과 |
|------|------|------------|------|
| 1 | 분산 | `variance < 100.0` | smooth |
| 2 | 그래디언트 방향 엔트로피 | `entropy < 1.0` | directional + dominant_angle |
| 3 | 자기상관 피크 | `peak > 0.15` | periodic |
| 4 | LBP 엔트로피 | `entropy > 2.5` | organic |
| 5 | fallback | — | complex |

*ROI 메타데이터 필드:*
```
continuity_score, stability_score, dominant_angle, background_type
```

---

### 03-Stage1b-Seed.md
**스크립트:** `stage1b_seed_characterization.py`  
**파라미터:**
```python
NUM_WORKERS = 4
```
**계산 로직:**

*마스크 추출:* SAM(vit_b) 또는 Otsu(`THRESH_BINARY_INV`) fallback.  
*4가지 기하 지표:*

| 지표 | 계산 | 의미 |
|------|------|------|
| linearity | `1 - (λ_min / λ_max)` | 선형성. 직선=1.0, 원=0.0 |
| solidity | region_area / convex_hull_area | 채움 정도 |
| extent | region_area / bbox_area | 바운딩박스 대비 채움 |
| aspect_ratio | major_axis / minor_axis | 장단축 비 |

*결함 서브타입 분류 (우선순위 순):*

| 우선순위 | 서브타입 | 조건 |
|---------|---------|------|
| 1 | linear_scratch | linearity > 0.85 AND aspect_ratio > 5.0 |
| 2 | elongated | aspect_ratio > 5.0 AND linearity > 0.6 |
| 3 | compact_blob | aspect_ratio < 2.0 AND solidity > 0.9 |
| 4 | irregular | solidity < 0.7 |
| 5 | general | otherwise |

*출력:* `seed_profile.json` (subtype, 4개 지표, mask_path, segmentation_method)

---

### 04-Stage2-Variants.md
**스크립트:** `stage2_defect_seed_generation.py`  
**파라미터:**
```python
NUM_VARIANTS = 50
SEED_THREADS = 2
IMG_WORKERS  = -1
```
**로직:**
- seed_profile.json 기반 변형 생성
- sentinel: `stage2_output/{seed_id}/` 에 PNG ≥ NUM_VARIANTS개

---

### 05-Stage3-Layout.md
**스크립트:** `stage3_layout_logic.py`  
**파라미터:**
```python
use_gpu = True
workers = -1
```
**계산 로직:**

*적합도 점수:*
```
suitability = 0.4×matching + 0.3×continuity + 0.2×stability + 0.1×gram_similarity
```

*MATCHING_RULES 표 (defect_subtype × background_type):*

| subtype | smooth | directional | periodic | organic | complex |
|---------|--------|-------------|---------|---------|---------|
| linear_scratch | 0.5 | **1.0** | 0.7 | 0.3 | 0.3 |
| elongated | 0.6 | **0.9** | 0.7 | 0.4 | 0.4 |
| compact_blob | **0.9** | 0.4 | 0.7 | 0.6 | 0.5 |
| irregular | 0.5 | 0.4 | 0.5 | **0.8** | **0.9** |
| general | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

*회전 결정:*
- `directional` 배경 + `linear_scratch`/`elongated` → `rotation = dominant_angle` (정렬)
- 그 외 → `rotation = uniform(0, 360)`

*GPU 모드:* PyTorch 텐서 배치 연산으로 이미지별 ROI 스코어를 1회 계산, 전체 seed 재사용.

---

### 06-Stage4-MPB.md
**스크립트:** `stage4_mpb_synthesis.py`  
**파라미터:**
```python
USE_FAST_BLEND  = True
IMG_THREADS     = 4
PNG_COMPRESSION = {"isp": 3, "mvtec": 3, "visa": 1}
MAX_BACKGROUND_DIM = None   # 변경 금지
```
**계산 로직:**

*seamlessClone (NORMAL_CLONE):* Poisson 방정식 반복 풀이, 경계 자연스러움. 느림.  
*Gaussian soft-mask (use_fast_blend=True):*
```
ksize = max(3, (min(rh, rw) // 6) | 1)
blended = patch × alpha + bg × (1 - alpha)   # alpha: Gaussian blur of ones
```
~10-30× 빠름. 합성 품질은 seamlessClone 대비 낮음.

*주의사항:*
- `MAX_BACKGROUND_DIM=None` 유지 — 변경 시 good(원본 해상도) vs defect(축소 해상도) 불일치 → 학습 불가
- `format='yolo'` 사용 시 Stage 6 연동 불가 → 반드시 `format='cls'`
- 배경 이미지 1회 로드 후 전체 seed 합성 (I/O 최적화)

---

### 07-Stage5-Quality.md
**스크립트:** `stage5_quality_scoring.py`, `utils/quality_scoring.py`  
**파라미터:**
```python
w_artifact      = 0.5
w_blur          = 0.5
workers         = -1    # 이미지 수준 병렬
parallel_seeds  = -1    # seed 수준 병렬
```
**계산 로직:**

*artifact_score (높을수록 아티팩트 없음):*
```
edge_score  = 1 - min(outlier_ratio × 10, 1)
             outlier_ratio = (gradient > mean + 3σ) 픽셀 비율
hf_ratio    = lap_energy / (mean_mag + std_mag + 1e-6)
hf_score    = 1 - min(hf_ratio / 5.0, 1)
artifact    = 0.6×edge_score + 0.4×hf_score
```

*blur_score (높을수록 선명):*
```
scale       = num_pixels / (256 × 256)   ← 기준 해상도 보정
lap_score   = clip(lap_var / (1000.0 × scale), 0, 1)
edge_sharp  = clip((P90/P50 - 1) / 9.0, 0, 1)
blur        = 0.5×lap_score + 0.5×edge_sharp
```

*최종:*
```
quality_score = 0.5×artifact + 0.5×blur
```

*ROI 마스크:* Stage 4가 저장한 `{stem}_mask.png` 로드 → 31×31 타원 dilate 후 결함 영역만 통계 산출.

---

### 08-Stage6-Dataset.md
**스크립트:** `stage6_dataset_builder.py`  
**파라미터:**
```python
PRUNING_THRESHOLD = 0.6
PRUNING_THRESHOLD_BY_DOMAIN = {"isp": 0.6, "mvtec": 0.6, "visa": 0.4}
SPLIT_RATIO          = 0.8   # good 80% train / 20% test
SPLIT_SEED           = 42
BALANCE_DEFECT_TYPES = True
NUM_IO_THREADS = 8
CAT_THREADS    = 2
```

*증강 비율 (augmentation_ratio):*

| 도메인 | aroma_full | aroma_pruned | 태스크 |
|--------|-----------|-------------|--------|
| isp | 1.0 (1:1) | 0.5 | classification |
| mvtec | 2.0 (2:1) | 1.5 | segmentation |
| visa | 2.0 (2:1) | 1.5 | segmentation |

*3개 그룹:*
- `baseline`: 원본 good만 (pretrained 특징 거리)
- `aroma_full`: 원본 + 전체 Stage 4 합성
- `aroma_pruned`: 원본 + `quality_score ≥ pruning_threshold` 필터링

*VisA pruning=0.4 이유:* artifact_score의 `hf_ratio` 상수(5.0)가 candle 왁스 매끄러운 표면에서 과대평가.

---

### 09-Stage7a-Benchmark.md
**스크립트:** `stage7_benchmark.py`  
**파라미터:**
```python
DOMAIN_FILTER = "isp"   # isp | mvtec | visa
resume        = True
```

*모델:*

| 모델 | 백본 | epochs | lr | optimizer |
|------|-----|--------|-----|-----------|
| yolo11 | YOLO11n-cls.pt | 30 | 0.01 | SGD |
| efficientdet_d0 | EfficientDet-D0 (EfficientNet-B0) | 30 | 0.0001 | Adam |

*실험 규모:* 2 models × 3 groups × 24 categories = **144 runs**  
*제외:* ISP/LSM_2 (LSM_1과 동일 센서·라인)

*메트릭:*
- `image_auroc`, `image_f1`: 전 도메인
- `pixel_auroc`: mvtec, visa만

*전제조건:* `augmented_dataset/baseline/test` 존재 (Stage 6 완료 sentinel)  
*sentinel:* `outputs/benchmark_results/{cat_name}/{model}/{group}/experiment_meta.json`

---

### 09-Stage7b-Results.md

**AUROC 해석:**

| 범위 | 의미 |
|------|------|
| > 0.95 | 우수 — 합성 데이터가 실제 결함 패턴을 잘 반영 |
| 0.80~0.95 | 양호 |
| < 0.80 | 부족 — 합성 품질 또는 파이프라인 설정 재검토 |
| ≈ 1.0 (완벽) | 데이터 누수 의심 — test/train 분리 확인 |

**비교 기준:**
- `baseline` → `aroma_full` AUROC 향상: AROMA 합성 유효성 확인
- `aroma_full` → `aroma_pruned` 향상: quality scoring pruning 효과 확인
- 도메인별 차이: classification(ISP) vs segmentation(MVTec/VisA) 과제 특성 반영

---

### 10-Dataset-Structure.md

*cat_dir 기준 디렉터리 트리:*
```
{cat_dir}/
├── stage0_output/           ← 리사이즈된 이미지 (512×512)
├── stage1_output/
│   ├── roi_metadata.json    ← sentinel
│   └── masks/global|local/
├── stage1b_output/
│   └── {seed_id}/
│       ├── seed_profile.json  ← sentinel
│       └── seed_mask.png
├── stage2_output/
│   └── {seed_id}/*.png      ← NUM_VARIANTS개
├── stage3_output/
│   └── {seed_id}/placement_map.json  ← sentinel
├── stage4_output/
│   └── {seed_id}/
│       ├── defect/*.png      ← sentinel (PNG 존재)
│       ├── defect/*_mask.png ← ROI 마스크 (Stage 5 사용)
│       └── quality_scores.json
└── augmented_dataset/
    ├── baseline/
    │   ├── train/good/
    │   └── test/{good,defect}/
    ├── aroma_full/
    │   ├── train/{good,defect}/
    │   └── test/ → symlink to baseline/test
    └── aroma_pruned/
        ├── train/{good,defect}/
        └── test/ → symlink to baseline/test
```

*benchmark 출력:*
```
outputs/benchmark_results/{cat_name}/{model}/{group}/experiment_meta.json
```

---

### 11-Parallel-Guide.md

*Stage별 최적 병렬 설정:*

| Stage | 외부 병렬 | 내부 병렬 | 근거 |
|-------|---------|---------|------|
| 0 | `CAT_THREADS=4` | `workers=-1` | I/O bound |
| 1 | `CAT_THREADS=2` | `IMG_WORKERS=-1` | CPU-bound + Drive 동시쓰기 제한 |
| 1b | `NUM_WORKERS=4` | — | seed 단위 독립 |
| 2 | `SEED_THREADS=2` | `IMG_WORKERS=-1` | 내부 병렬로 코어 포화 |
| 3 | 순차 | GPU 배치 | GPU 인스턴스 공유 불가 |
| 4 | 카테고리 순차 | `IMG_THREADS=4` (Thread) | 배경 1회 로드, I/O 절감 |
| 5 | `SEED_THREADS=4` | `IMG_WORKERS=-1` | I/O+CPU 혼합 |
| 6 | `CAT_THREADS=2` | `NUM_IO_THREADS=8` (Thread) | I/O-bound Drive 복사 |
| 7 | 순차 (카테고리) | — | GPU 단일 점유 |

*L4 GPU (8 CPU) 권장:*
- Stage 5: `parallel_seeds=-1, workers=0` (seed 수준만 병렬)
- `run_parallel` 병렬화 조건: `num_workers > 1 AND len(tasks) >= 2`

---

## 스펙 자기검토

- [ ] placeholder 없음
- [ ] 섹션 간 모순 없음
- [ ] 단일 구현 계획으로 충분한 범위
- [ ] 모호한 요구사항 없음
