# AROMA 파이프라인 연구 정리

작성일: 2026-04-17

---

## 1. 전체 파이프라인 개요 (Stage 0 ~ Stage 7)

| 단계 | 이름 | 핵심 행위 |
|------|------|-----------|
| Stage 0 | 이미지 리사이즈 | 파이프라인 시작 전 모든 이미지를 지정 크기(기본 512×512)로 일괄 변환 |
| Stage 1 | ROI 추출 | 정상(good) 이미지에서 Otsu로 ROI 후보 추출, 배경 특성 분석 → `roi_metadata.json` |
| Stage 1b | 시드 특성 분류 | 시드 결함 이미지 마스크를 분석해 5개 서브타입 중 하나로 분류 → `seed_profile.json` |
| Stage 2 | 결함 시드 변형 생성 | 서브타입에 맞는 elastic warping으로 시드 이미지를 N장 변형 (TF-IDG) |
| Stage 3 | 배치 위치 결정 | ROI 후보 × 시드 변형본 매칭, 적합도 점수로 최적 삽입 위치 결정 → `placement_map.json` |
| Stage 4 | 합성 이미지 생성 | Poisson blending으로 결함을 정상 이미지에 합성, ROI 마스크 함께 저장 |
| Stage 5 | 품질 점수 산정 | 합성 이미지의 artifact_score + blur_score 계산 → `quality_scores.json` |
| Stage 6 | 데이터셋 구성 | baseline / aroma_full / aroma_pruned 세 가지 데이터셋 구성 |
| Stage 7 | 벤치마크 | YOLO11 / EfficientDet-D0로 이상 탐지 성능(AUROC 등) 측정 및 비교 |

---

## 2. Stage 2 — 생성 모델 (TF-IDG)

### 핵심
딥러닝 생성 모델(GAN, Diffusion 등)을 사용하지 않는다.  
**TF-IDG(Training-Free Industrial Defect Generation)** — OpenCV 기반 elastic warping만으로 변형체를 생성한다.

### 변형 방식
1. 랜덤 displacement field 생성 (Gaussian smoothing 적용)
2. `cv2.remap()`으로 픽셀 재배치
3. 밝기/대비 jitter 추가 (× [0.85, 1.15])

### 서브타입별 전략

| 서브타입 | 변형 방식 | 의도 |
|----------|-----------|------|
| `linear_scratch` | Y 방향만 변위 | 선 방향 유지 |
| `elongated` | X 방향만 변위 | 축 방향 늘림/줄임 |
| `compact_blob` | X/Y 동일 변위 | 등방성 변형 |
| `irregular` | X/Y 각각 고진폭 변위 | 형태를 크게 뒤틀기 |
| `general` | 랜덤 변위 + 50% 좌우 반전 + ±15° 회전 | 범용 |

---

## 3. Stage 1b → Stage 2 — 서브타입 분류 흐름

### 분류 지표

| 지표 | 의미 |
|------|------|
| `linearity` | 결함이 선형에 가까운 정도 (픽셀 좌표 공분산 행렬의 고유값 비율: `1 - λ_min/λ_max`) |
| `aspect_ratio` | 장축 / 단축 길이 비율 |
| `solidity` | 결함 면적 / convex hull 면적 |

### 분류 규칙 (우선순위 순)

```
linearity > 0.85 AND aspect_ratio > 5.0  →  linear_scratch
aspect_ratio > 5.0 AND linearity > 0.6   →  elongated
aspect_ratio < 2.0 AND solidity > 0.9    →  compact_blob
solidity < 0.7                           →  irregular
그 외                                     →  general
```

분류 결과는 `seed_profile.json`의 `"subtype"` 필드에 저장된다.  
Stage 2는 이 파일을 읽어 해당 전략으로 변형을 수행한다.

---

## 4. Stage 1 — ROI 추출 상세

### 처리 흐름

```
good 이미지
   ↓
① Otsu 임계값 → global_mask (0/255 이진 마스크)
   ↓
② connected components → ROI bounding box 목록
   ↓
③ 배경 분석 (grid 기반, 64px 격자) → 각 ROI의 배경 특성
   ↓
roi_metadata.json 저장
```

### ① Otsu global_mask

`cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`  
픽셀 강도 히스토그램으로 전경/배경을 자동 분리.

### ② Bounding box 추출

`cv2.connectedComponentsWithStats(mask, connectivity=8)`  
연결된 픽셀 군집 → `(x, y, w, h)` 박스 추출.  
- `min_area=100` 미만은 노이즈로 제거
- 유효 box 없으면 전체 이미지 크기로 폴백

`--roi_levels` 옵션:

| 옵션 | 동작 |
|------|------|
| `local` | connected component 단위 box |
| `global` | 전체 마스크 외곽 box 하나 |
| `both` (기본값) | local box 사용 |

### ③ 배경 분석 (5단계 파이프라인)

이미지를 64px 격자로 분할, 각 셀을 비용 낮은 순으로 조기 종료:

| 단계 | 지표 | 판정 |
|------|------|------|
| 1 | 분산 < 100 | **smooth** |
| 2 | Gradient 방향 엔트로피 < 1.0 | **directional** (dominant_angle 기록) |
| 3 | FFT 자기상관 피크 > 0.15 | **periodic** |
| 4 | LBP 엔트로피 > 2.5 | **organic** |
| 5 | 해당 없음 | **complex** |

각 ROI box에 저장되는 점수:
- **`continuity_score`**: `0.6 × uniformity + 0.4 × stability`
- **`stability_score`**: 배경 유형의 전형성
- **`dominant_angle`**: directional 배경일 때 우세 방향 각도 (Stage 3에서 결함 회전 배치에 활용)

---

## 5. Stage 1b — SAM 사용 상세

### SAM의 역할

시드 결함 이미지에서 **결함 영역만 정밀하게 분리**하는 마스크를 추출한다.  
이 마스크로 `DefectCharacterizer`가 형태 지표를 계산한다.

### 동작 방식

```python
# SamAutomaticMaskGenerator: 프롬프트 없이 이미지 전체 자동 분할
masks = generator.generate(image)
# → 면적 오름차순 정렬 후 가장 작은 마스크 선택 = 결함
sorted_masks = sorted(masks, key=lambda m: m["area"])
return sorted_masks[0]["segmentation"]
```

**가장 작은 마스크를 결함으로 선택하는 근거:**  
결함은 배경·제품 외곽보다 훨씬 작은 영역을 차지한다는 가정.

### SAM 실패 시 Otsu 폴백

체크포인트(`sam_vit_b_01ec64.pth`) 없음 / import 오류 / 마스크 생성 실패 →  
`except Exception`으로 낙하 → Otsu(`THRESH_BINARY_INV + THRESH_OTSU`)로 대체.

| 구분 | SAM | Otsu |
|------|-----|------|
| 방식 | 딥러닝 자동 분할 | 픽셀 강도 임계값 |
| 정확도 | 복잡한 형태도 정밀 분리 | 밝기 대비 명확해야 유효 |
| 조건 | `sam_vit_b_01ec64.pth` 필요 | 추가 의존성 없음 |
| CLI | `--model_checkpoint` 경로 지정 | 지정 없으면 자동 사용 |

### SAM vs Otsu 마스크 품질 차이

| 특징 | SAM 마스크 | Otsu 마스크 |
|------|-----------|------------|
| 경계 | 결함 윤곽을 정밀하게 따라감 | 밝기 기반으로 뭉툭하게 잘림 |
| 배경 포함 | 거의 없음 | 배경 밝기 유사하면 함께 포함 |
| 형태 | 결함 모양에 충실 | 사각형·덩어리 형태로 나올 수 있음 |

### 출력 파일에서 사용 방법 확인 가능 여부

**불가능.** `seed_profile.json`에 `segmentation_method` 필드가 없다.  
간접 확인 방법: `seed_mask.png`를 육안으로 비교 (SAM은 정밀한 경계, Otsu는 뭉툭한 경계).

Colab에서 확인할 파일:
```
stage1b_output/{seed_name}/seed_profile.json   ← 형태 지표 + subtype
stage1b_output/{seed_name}/seed_mask.png        ← 마스크 품질 육안 확인
```

> **개선 완료:** `extract_seed_mask`가 `(mask, method)` 튜플을 반환하도록 수정.  
> `seed_profile.json`에 `"segmentation_method": "sam"` 또는 `"otsu"` 필드가 저장된다.

```json
{
  "seed_path": "...",
  "subtype": "general",
  "linearity": 0.1234,
  "solidity": 0.8765,
  "extent": 0.6543,
  "aspect_ratio": 2.1000,
  "mask_path": "...",
  "segmentation_method": "sam"
}
```

---

## 6. Stage 4 — 합성 수량 결정 방식

### 수량 결정 구조

Stage 4의 출력 수량은 **Stage 3의 `placement_map.json`이 결정**한다.

```
N_출력 = N_배경이미지 × N_seed_id
```

- **N_배경이미지**: Stage 1 `image_dir`의 good 이미지 수
- **N_seed_id**: Stage 2를 실행한 결함 시드 종류 수
- **N_variants** (Stage 2 `--num_variants`): 출력 이미지 수를 늘리지 않고, 한 출력 이미지에 합성되는 결함 패치 수 결정

### 모드별 차이

| 모드 | 출력 수량 | 출력 내용 |
|------|-----------|-----------|
| `run_synthesis` (단일) | 배경 이미지 수 | 모든 variants가 하나의 이미지에 누적 합성 |
| `run_synthesis_batch` (배치) | 배경 이미지 수 × seed_id 수 | seed_id별 별도 이미지 생성 |

합성 이미지 수를 늘리려면 **seed 종류를 늘려야** 하고,  
`num_variants`는 이미지당 결함 밀도(한 이미지에 몇 개의 결함이 합성되는가)에 영향을 준다.

---

## 7. Stage 5 — ROI 마스크 기반 품질 채점

### 왜 ROI 마스크가 필요한가

전체 이미지를 채점하면 **배경의 고주파 패턴이 점수를 오염**시킨다.  
예: 캔들(VisA)처럼 매끄러운 배경 → gradient가 낮아 `hf_ratio` 과대평가.

### 적용 방식

```
전체 이미지 → Sobel/Laplacian 계산 (공간 컨텍스트 유지)
        ↓
ROI 마스크 dilate (31×31 타원 커널)
        ↓
결함 영역 픽셀만 추출 → mean, std, variance 등 통계 계산
        ↓
artifact_score / blur_score 산출
```

**Gradient/Laplacian은 전체 이미지에서 계산** (공간 필터 연산이므로 경계 컨텍스트 필요).  
**통계 추출 시에만 ROI 픽셀로 제한** (경계 블렌딩 아티팩트 영역까지 포함하기 위해 31×31 dilate).

### 두 가지 개선 (Option A + B)

- **Option A**: `hf_ratio` 분모를 `mean_mag` → `mean_mag + std_mag`로 변경  
  매끄러운 배경에서 `mean_mag`가 작아 `hf_ratio`가 과대평가되던 문제 완화
- **Option B**: ROI 마스크로 결함 영역만 추출해 통계 산출

### 마스크 자동 로드

```python
mask_path = img_p.with_name(img_p.stem + "_mask.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
```

Stage 4가 저장한 `{image_id}_mask.png`를 자동 탐색. 없으면 전체 이미지 사용(폴백).

---

## 8. 추가 개선 사항 (논의된 내용)

- **Stage 1b `segmentation_method` 필드 추가**: `seed_profile.json`에 SAM/Otsu 사용 여부 기록 가능  
  → `extract_seed_mask`가 방법 정보를 함께 반환하도록 수정 필요
