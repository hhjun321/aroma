# 2차 연구 — AROMA SOTA 수준 고도화

## 연구 개요

**연구명:** AROMA Phase 2 — Multi-class 결함 분류 + MPB → Diffusion 교체 검증  
**실험 대상:** **MVTec AD** — 결함 유형 다양성 + 풍부한 선행 연구로 SOTA 비교 용이  
**1차 테스트 범위:** MVTec AD / bottle — 파이프라인 검증 후 전체 카테고리 확장  
**목표:** 1차 연구에서 식별된 MPB 블러 문제를 Diffusion 합성으로 해소하여 SOTA 수준 달성  
**핵심 논거:** 1차 연구에서 EfficientDet 성능 하락의 원인인 MPB 미세 블러를 Stage 4 교체(Diffusion)로 제거, 성능 회복 검증

→ [[12-Research-Phase1]] (선행 연구)

---

## 연구 축 1 — Multi-class 결함 분류 확장

### 목표

이진 분류(good / defect) → **결함 유형별 분류** (good / scratch / dent / contamination / ...)

### 타겟 데이터셋 선정 기준

| 기준 | MVTec AD | VisA |
|------|---------|------|
| 결함 유형 수 | 카테고리당 평균 5~7종 | 카테고리당 평균 3~5종 |
| 픽셀 수준 annotation | ✅ 있음 | ✅ 있음 |
| 카테고리 다양성 | 높음 (15개) | 중간 (8개) |
| 합성 난이도 | 중간 | 높음 (고해상도) |
| **추천** | **우선 타겟** | 검증용 확장 |

> 타겟: **MVTec AD** — 결함 유형 다양성 + 풍부한 선행 연구로 SOTA 비교 용이

### 평가 지표 변경

| 지표 | 1차 연구 | 2차 연구 |
|------|---------|---------|
| 분류 단위 | 이미지 (good/defect) | 이미지 (결함 유형별) |
| 주요 지표 | image_auroc | **macro-F1**, per-class accuracy |
| 보조 지표 | image_f1 | top-1 accuracy, confusion matrix |
| 픽셀 지표 | pixel_auroc | pixel_auroc (유지) |

### CASDA 참조 — 결함 특징 보존 생성

CASDA 연구에서 활용한 핵심 원리:
- 결함 서브타입(linear_scratch, compact_blob 등) 별 **형태 보존 변환**
- 생성 이미지가 원본 결함의 기하 지표(linearity, solidity, aspect_ratio)를 유지하도록 제약
- 결함 유형 간 혼동을 최소화하는 **class-discriminative synthesis**

**파이프라인 변경 포인트:**

| Stage | 1차 | 2차 |
|-------|-----|-----|
| Stage 1b | 서브타입 분류 (5종) | 서브타입 → **결함 유형 레이블** 매핑 |
| Stage 2 | 단순 변형 생성 | 유형별 특징 보존 변형 |
| Stage 6 | good/defect 이진 | **결함 유형별 클래스** 구성 |
| Stage 7 | AUROC | macro-F1 + per-class 분석 |

---

## 연구 축 2 — Stage 4 교체: MPB → Diffusion (핵심 논거)

### 1차 연구 한계 — MPB 블러와 EfficientDet 성능 하락

1차 연구에서 EfficientDet의 성능이 기대치 이하로 나타난 핵심 원인은 **MPB(Poisson Blending)가 결함 경계에 생성하는 미세 블러**였다.

| 현상 | 원인 | 결과 |
|------|------|------|
| EfficientDet 성능 하락 | MPB 블러 → 결함 경계 흐림 | 결함 특징이 분류기에 제대로 전달되지 않음 |
| YOLO11은 상대적 유지 | 경량 모델 특성상 블러 영향 덜 받음 | MPB 블러의 영향도 차이 확인 |

**1차 연구 Smoke Test 실측치 (bottle/ASM/candle, image_auroc):**

| 카테고리 | 모델 | baseline | aroma_full | aroma_pruned | Δ full | 비고 |
|---------|------|---------|-----------|-------------|--------|------|
| bottle | YOLO11 | 0.851 | 0.967 | 0.974 | **+0.116** | 합성 효과 명확 |
| bottle | EfficientDet | 0.994 | 0.947 | 0.925 | **−0.047** | ⚠ MPB 블러 영향 의심 |
| ASM | YOLO11 | 0.689 | 0.768 | 0.697 | +0.080 | |
| ASM | EfficientDet | 0.721 | 0.851 | 0.867 | +0.129 | |
| candle | YOLO11 | 0.625 | 0.890 | **0.877** | +0.265 | `pruning_ratio=0.5` 적용 |
| candle | EfficientDet | 0.764 | 0.750 | **0.672** | −0.014 | ⚠ EfficientDet 일관된 하락 |

> bottle EfficientDet은 baseline이 이미 0.994로 포화 상태에 가깝고, MPB 블러가 정밀 경계 분류기에 혼선을 줄 경우 실측치와 같이 하락 가능하다.

> MPB 위에 Diffusion을 적용하는 하이브리드 방식(Option B)은 마지막 단계에서 MPB를 사용하므로 **동일한 블러 문제에 직면한다**. 따라서 **Stage 4를 Diffusion으로 완전 교체(Option A)** 만이 근본 해결책이다.

---

### Latent Fusion 핵심 개념

Diffusion 기반 합성의 본질적 차별점은 **결함을 픽셀 공간이 아닌 Latent 공간에서 생성**한다는 것이다.

**전체 흐름:**

| 단계 | 설명 | AROMA 연결 |
|------|------|-----------|
| ① 정상 이미지 준비 | 결함이 없는 깨끗한 이미지 $I_{good}$ | Stage 0 리사이즈 완료 이미지 |
| ② ROI 지정 | 결함이 생길 법한 위치 $M_{roi}$ 선정 | Stage 1 roi_metadata.json |
| ③ 결함 설계 | 생성할 결함의 뼈대(Edge)와 특징(Metadata) 정의 | Stage 1b seed_profile.json (seed_path → Canny 엣지) + 기하 지표(linearity, solidity) |
| ④ Inpainting | $I_{good}$의 $M_{roi}$ 영역만 도려내어 AI에게 전달 | `mask_image_path` = `*_mask.png` |
| ⑤ Conditioning | "이 뼈대 모양대로, 이런 특징을 가진 결함을 그려라" | ControlNet(edge) + 텍스트 프롬프트(linearity, solidity) |
| ⑥ 결과 | 깨끗했던 이미지의 특정 부위에 실제와 동일한 결함이 심겨진 최종 이미지 | Stage 5 품질 필터링 대상 |

**MPB와의 근본 차이:**

```
MPB:       I_good ← 픽셀 복사 붙여넣기 → 경계 보정(Poisson) → 완성
Diffusion: I_good → Latent 인코딩 → M_roi 영역 Denoising → 픽셀 디코딩 → 완성
           (각 Denoising Step마다 마스크 기준으로 Latent 융합)
```

> MPB는 결함 픽셀을 "가져와서 붙이는" 방식이고,  
> Diffusion은 마스크 영역 안에서 결함을 "새로 그리는" 방식이다.  
> 결과적으로 배경 텍스처와의 경계 자연스러움이 근본적으로 다르다.

### 합성 방식 비교 구조

```
AROMA-MPB          (1차 연구 기준선 — 블러 문제 재현)
    ↓ 대조
AROMA-Diffusion    Stage 4 완전 교체 (Option A — 블러 문제 근본 해결)
```

### 합성 방식별 특성

| 항목 | MPB (Poisson Blending) | Diffusion (SD + ControlNet) |
|------|----------------------|----------------------------|
| 합성 속도 | 빠름 | 느림 |
| 학습 필요 | 없음 | ✅ 파인튜닝 필요 (또는 pretrained) |
| 경계 자연스러움 | 중간 | 매우 높음 |
| 결함 형태 보존 | 높음 | 높음 (ControlNet edge 조건) |
| 도메인 범용성 | 높음 | 중간 |
| VRAM | 불필요 | 높음 (≥ 8GB) |

### CASDA 참조 — Diffusion 합성 구현

CASDA 프로젝트 (`D:/project/CASDA`)는 철강 결함 이미지에 Diffusion 합성을 적용한 선행 파이프라인이다. AROMA Diffusion 합성은 CASDA의 세 핵심 구성요소를 차용한다.

#### ① 텍스트 프롬프트 생성 (`src/preprocessing/prompt_generator.py`)

CASDA의 `PromptGenerator.generate_technical_prompt()`는 결함 기하 지표를 자연어 조건으로 변환한다:

```python
# CASDA — generate_technical_prompt() 핵심 로직
linearity = defect_metrics.get('linearity', 0.0)
solidity  = defect_metrics.get('solidity', 0.0)
aspect_ratio = defect_metrics.get('aspect_ratio', 1.0)

if linearity > 0.8:   characteristics.append("highly linear")
elif linearity > 0.6: characteristics.append("moderately linear")
if solidity > 0.9:    characteristics.append("solid")
elif solidity < 0.7:  characteristics.append("irregular shape")
if aspect_ratio > 5:  characteristics.append("very elongated")

prompt = (
    f"Industrial steel defect: {char_str} defect "
    f"(class {class_id}) on {bg_info['surface']}, "
    f"background stability {stability_score:.2f}, "
    f"match quality {suitability_score:.2f}"
)
```

**AROMA 적용:** Stage 1b `seed_profile.json`의 `linearity`, `solidity` → 동일 로직으로 프롬프트 생성

```python
# next2 코드 (AROMA 적용판)
prompt = (
    f"A professional industrial photo of a {metadata['category']} defect, "
    f"with {metadata['linearity']:.2f} linearity and {metadata['solidity']:.2f} solidity, "
    f"highly detailed texture, sharp edges, 8k resolution, seamless integration"
)
negative_prompt = "blurry, low quality, color bleeding, shadow artifacts, unrealistic texture"
```

#### ② ControlNet 파인튜닝 (`scripts/train_controlnet.py`)

CASDA가 검증한 파라미터를 AROMA 도메인에 동일하게 적용:

| 파라미터 | CASDA 검증값 | 의미 |
|----------|------------|------|
| 기반 모델 | `runwayml/stable-diffusion-v1-5` | SD v1.5 UNet (동결) |
| ControlNet | `lllyasviel/sd-controlnet-canny` | Canny 힌트 기반 |
| `--learning_rate` | `1e-5` | ControlNet 가중치만 학습 |
| `--lr_scheduler` | `cosine` | 학습률 감쇠 |
| `--controlnet_conditioning_scale` | 학습 `1.0` / 추론 `0.7` | 추론 시 낮춰 아티팩트 방지 |
| `--snr_gamma` | `5.0` | Min-SNR-γ 손실 안정화 |
| `--mixed_precision` | `fp16` | VRAM 절약 (ControlNet은 fp32 유지) |

**멀티채널 힌트 구조 (CASDA Stage A 방식):**

| 채널 | 내용 | AROMA 매핑 |
|------|------|-----------|
| R | 결함 마스크 형태 | Stage 4 `*_mask.png` |
| G | Canny 엣지 | **Stage 1b 원본 seed 이미지** (`seed_profile["seed_path"]`) |
| B | Gabor 텍스처 | Stage 1 배경 텍스처 분석 |

> **Stage 2 변형 이미지를 사용하지 않는 이유:** CASDA는 실제 결함 이미지에서 Canny 엣지를 추출하여 검증하였다. Stage 2 elastic warp는 원본 형태를 왜곡하므로 ControlNet의 구조적 가이던스 품질이 저하된다. Stage 2 출력은 `placement_map.json`의 ROI 위치·크기 계산에만 활용된다.

#### ③ 합성 실행 (`scripts/test_controlnet.py` → `next2` 코드)

```python
# AROMA next2 구현 (StableDiffusionControlNetInpaintPipeline)
result_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,          # Stage 0 원본 good 이미지
    mask_image=mask_image,     # placement_map 좌표로 생성한 ROI 마스크
    control_image=control_image,  # Stage 1b seed_path → Canny 엣지 (CASDA 방식)
    num_inference_steps=30,    # CASDA와 동일
    strength=0.7,              # 경계 자연스러움 vs 형태 보존 균형
    guidance_scale=7.5,        # CASDA와 동일
).images[0]
```

> CASDA: 학습 `conditioning_scale=1.0` → 추론 `0.7` (네온 아티팩트 방지 검증 완료)  
> AROMA에서도 동일하게 `strength=0.7` 적용

### 대조 실험 설계

**실험 변수:**

```
독립변수: 합성 방식 (MPB / Diffusion)
종속변수: image_auroc, macro-F1, per-class accuracy
통제변수: 동일 ROI 배치 전략 (Stage 3), 동일 품질 필터링 (Stage 5)
```

**비교 그룹:**

| 그룹 | 합성 방식 | Stage 4 | 비고 |
|------|---------|---------|------|
| `baseline` | 없음 | — | |
| `aroma_mpb` | Poisson Blending | `seamlessClone` | 1차 연구 기준선 (블러 문제 재현) |
| `aroma_diffusion` | SD Inpainting + ControlNet | Diffusion (완전 교체) | 블러 문제 근본 해결 |

**`aroma_diffusion` 파이프라인 (Option A — Stage 4 교체):**

```
Stage 1b (seed_profile.json)
    ├─ seed_path    → 원본 결함 이미지 → Canny 엣지 → control_image  ← CASDA 방식
    └─ linearity, solidity, subtype → 텍스트 프롬프트 생성

Stage 3 (placement_map.json)
    ├─ x, y, scale → ROI 마스크 생성 (Stage 2 패치 크기 참조)
    └─ matched_background_type, suitability_score → 프롬프트 보강

Stage 4 (Diffusion 합성)
    - image         = Stage 0 원본 good 이미지
    - mask_image    = placement_map 좌표로 생성한 ROI 마스크
    - control_image = Stage 1b 원본 seed Canny 엣지  ← Stage 2 변형 이미지 미사용
    - prompt        = Stage 1b 기하 지표 → 자연어 변환
    - strength=0.7 / guidance_scale=7.5 / num_inference_steps=30
    ↓
aroma_diffusion 최종 이미지 (MPB 블러 없음, 실제 결함 형태 보존)
```

> **Stage 2 역할 한정:** Stage 2 변형 이미지(`variant_*.png`)는 `placement_map.json`의 `defect_path`를 통해 ROI 패치 크기(ph, pw) 계산에만 사용된다. ControlNet 입력(Canny 엣지)은 Stage 1b 원본 seed에서 추출한다.

---

## 파이프라인 변경 범위

| Stage | 변경 여부 | 내용 |
|-------|---------|------|
| Stage 0~3 | 유지 | ROI 추출·배치 전략 동일 |
| Stage 4 | **교체** | MPB → Diffusion 선택 가능하도록 모듈화 |
| Stage 5 | 유지 또는 확장 | quality_score에 perceptual loss 기반 지표 추가 검토 |
| Stage 6 | **확장** | 결함 유형별 클래스 디렉터리 구성 |
| Stage 7 | **확장** | macro-F1, confusion matrix, per-class AUROC 추가 |

---

## 결과 요약

> 수치는 실험 완료 후 기입

### Multi-class 분류 성능 (MVTec AD 타겟)

| 모델 | 합성 방식 | macro-F1 | top-1 accuracy | Δ vs baseline |
|------|---------|---------|---------------|--------------|
| YOLO11 | MPB | — | — | — |
| YOLO11 | Diffusion | — | — | — |
| EfficientDet | MPB | — | — | — |
| EfficientDet | Diffusion | — | — | — ↑ (블러 해소 기대) |

### 합성 방식 대조 (AUROC 기준)

**1차 연구 (MPB) 실측치 — 3카테고리 smoke test:**

| 모델 | 그룹 | bottle (MVTec) | candle (VisA) | ASM (ISP) |
|------|------|---------------|-------------|----------|
| YOLO11 | aroma_full | 0.967 (+0.116) | 0.890 (+0.265) | 0.768 (+0.080) |
| YOLO11 | aroma_pruned | 0.974 (+0.122) | 0.877 (+0.252) | 0.697 (+0.008) |
| EfficientDet | aroma_full | 0.947 (−0.047) | 0.750 (−0.014) | 0.851 (+0.129) |
| EfficientDet | aroma_pruned | 0.925 (−0.070) | 0.672 (−0.092) | 0.867 (+0.146) |

> Δ는 각 카테고리의 baseline 대비 변화량.
> EfficientDet의 candle 하락 패턴(full −0.014, pruned −0.092)은 bottle과 동일한 MPB 블러 영향으로 분석되며, Phase 2 Diffusion 교체의 핵심 검증 대상이다.

**2차 연구 (Diffusion) — 실험 완료 후 기입 (대상: MVTec AD):**

| 모델 | 그룹 | bottle | (다른 MVTec 카테고리...) | 비고 |
|------|------|--------|----------------------|------|
| YOLO11 | aroma_mpb (1차 기준선) | 0.967 | — | Phase 1 MPB 결과 재사용 |
| YOLO11 | aroma_diffusion | — | — | Phase 2 신규 |
| EfficientDet | aroma_mpb (1차 기준선) | 0.947 | — | ⚠ MPB 블러 재현 |
| EfficientDet | aroma_diffusion | — | — | 블러 해소 기대 ↑ |

---

## SOTA 비교 대상

| 논문/모델 | 방법론 | 참조 지표 |
|---------|--------|---------|
| DRAEM | Anomaly map + reconstruction | pixel_auroc |
| SimpleNet | 단순 특징 추출 기반 | image_auroc |
| PatchCore | Memory bank 기반 | image_auroc |
| CASDA | 컨텍스트 기반 적대적 합성 | image_auroc, macro-F1 |

---

## 결론 구조 (예상)

1. **Multi-class 유효성:** AROMA 합성 이미지로 결함 유형 분류 가능 → macro-F1 향상 확인
2. **블러 문제 해소:** Diffusion 기반 Stage 4에서 MPB 블러 제거 → EfficientDet 성능 회복 여부
3. **합성 방식 우열:** Diffusion > MPB (품질·경계 자연스러움 기준) / MPB > Diffusion (속도 기준)
4. **SOTA 달성 조건:** Diffusion 기반 AROMA가 PatchCore/DRAEM 대비 유의미한 성능 달성
5. **실용성:** CASDA 검증 파라미터(`strength=0.7`, `conditioning_scale=0.7`) 기준 추론 비용 vs 품질 트레이드오프 분석
