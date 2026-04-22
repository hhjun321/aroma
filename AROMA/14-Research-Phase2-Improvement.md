# Phase 2 개선 연구 — Diffusion 합성 성능 분석

## 배경

MVTec bottle 샘플 성능 평가에서 `stage4_diffusion_synthesis.py` 수행 시 baseline 대비 성능 하락 발생.
Phase 1 MPB 방식은 bottle에서 YOLO11 +0.116, EfficientDet −0.047이었으나,
Diffusion 방식에서도 유사하거나 더 큰 하락이 관측됨.

---

## 현재 파라미터 (CASDA 검증값)

```python
strength              = 0.7   # 마스크 영역 noise 강도
guidance_scale        = 7.5   # 텍스트 프롬프트 영향력 (CFG scale)
conditioning_scale    = 0.7   # ControlNet(Canny edge) 영향력
num_inference_steps   = 30
```

파이프라인: `StableDiffusionControlNetInpaintPipeline`
- Base model: `runwayml/stable-diffusion-inpainting`
- ControlNet: `lllyasviel/sd-controlnet-canny`
- ControlNet 입력: Stage 1b 원본 seed 이미지 → Canny 엣지

---

## 성능 하락 원인 분석

### 원인 1 — SD 모델의 도메인 미스매치

`runwayml/stable-diffusion-inpainting`은 자연 이미지로 학습된 범용 모델이다.
`guidance_scale=7.5`로 텍스트 프롬프트 의존도가 높을 때, 모델이 "산업용 유리병 결함"을
자연 이미지 기반으로 해석하여 bottle 표면 텍스처와 무관한 질감을 생성한다.

### 원인 2 — strength=0.7의 과도한 배경 훼손

```
strength=0.7 → 마스크 영역에 70% noise 주입 후 denoising
→ SD 모델이 해당 영역을 "창의적으로" 재생성
→ 유리병 고유의 투명/반사 텍스처가 훼손됨
```

### 원인 3 — conditioning_scale=0.7의 ControlNet 저활용

실제 결함 seed에서 추출한 Canny 엣지(신뢰할 수 있는 형태 정보)가
70%만 반영되어 결함 형태가 흐릿하게 재현됨.

### 원인 4 ★ — Grayscale 후처리로 인한 색상 정보 제거

```python
# stage4_diffusion_synthesis.py:191
result_pil = result_pil.convert("L").convert("RGB")
```

MVTec bottle은 **컬러(녹색 유리)** 이미지이나, 이 처리로 색상 정보가 완전히 소실된다.
합성 결과물이 흑백으로 변환되어 학습 데이터로서 현실감을 잃는다.
**이것이 성능 하락의 핵심 원인일 가능성이 높다.**

---

## 개선 방안

### A. Strength 튜닝 (Structure-Preserving 강도 조절)

`strength`를 낮추어 원본 배경 텍스처 보존을 강화한다.

| 값 | 효과 | 적합성 |
|----|------|--------|
| 0.7 (현재) | 마스크 영역 70% 재생성 | 유리 텍스처 훼손 위험 |
| 0.5 (권장) | 배경 보존↑, 자연스러운 융합 | bottle 유리 텍스처 유지 |
| 0.4 이하 | 변화 최소화 | 결함이 시각적으로 안 보일 위험 |

**권장 실험 구간: 0.45 ~ 0.55**

### B. ControlNet Guidance Scale 강화

`conditioning_scale`을 높이고 `guidance_scale`을 낮추어 텍스트 의존도를 줄이고
실제 결함 seed의 Canny 엣지를 픽셀 단위로 강제한다.

```
conditioning_scale ↑ → 실제 결함 엣지 형태를 강하게 투영 (신뢰할 수 있는 정보)
guidance_scale     ↓ → SD의 부정확한 텍스트 해석 영향 최소화
```

| 파라미터 | 현재 | 권장 구간 | 근거 |
|---------|------|----------|------|
| `conditioning_scale` | 0.7 | 0.9 ~ 1.0 | 실제 seed 엣지를 강하게 따름 |
| `guidance_scale` | 7.5 | 3.0 ~ 5.0 | 범용 SD 모델의 텍스트 오해석 감소 |

두 파라미터는 **반비례로 조정**하는 것이 효과적이다.

### C. ★ Grayscale 후처리 제거 또는 도메인 조건화

현재 코드:
```python
result_pil = result_pil.convert("L").convert("RGB")
```

MVTec bottle처럼 컬러 정보가 중요한 도메인에서는 이 처리를 제거해야 한다.
ISP(라인스캔, 본래 흑백) 도메인에는 유지가 적절할 수 있으므로
**도메인별 조건 적용**이 필요하다:

```python
# 개선안: 도메인이 컬러인 경우 후처리 생략
if domain in ("mvtec", "visa"):
    pass  # 컬러 보존
else:
    result_pil = result_pil.convert("L").convert("RGB")
```

---

## 권장 실험 구성 (bottle 기준)

### 실험 1: A + B 조합

```python
strength           = 0.5    # 0.7 → 0.5
conditioning_scale = 1.0    # 0.7 → 1.0
guidance_scale     = 4.0    # 7.5 → 4.0
```

### 실험 2: C 단독 (가장 빠른 검증)

```python
# grayscale 후처리 제거만으로 성능 회복 여부 확인
strength           = 0.7    # 현재 유지
conditioning_scale = 0.7    # 현재 유지
guidance_scale     = 7.5    # 현재 유지
# → result_pil.convert("L").convert("RGB") 제거
```

### 실험 3: A + B + C 풀 적용

```python
strength           = 0.5
conditioning_scale = 1.0
guidance_scale     = 4.0
# → grayscale 후처리 제거
```

---

## 우선순위

1. **C (grayscale 후처리 제거)** — 코드 한 줄 수정, 즉각 검증 가능, 효과가 클 것으로 예상
2. **B (conditioning_scale ↑, guidance_scale ↓)** — 실제 결함 형태 재현 개선
3. **A (strength ↓)** — B와 조합 시 배경 보존 효과

---

## 관련 파일

| 파일 | 변경 대상 |
|------|---------|
| `stage4_diffusion_synthesis.py:191` | grayscale 후처리 도메인 조건화 |
| `stage4_diffusion_synthesis.py:17` | 기본 파라미터 주석 업데이트 |
| `docs/작업일지/phase2_execute_oneclick.md` 셀 1 | `STRENGTH`, `CONDITIONING_SCALE`, `GUIDANCE_SCALE` 조정 |
| `configs/benchmark_experiment_phase2.yaml` | 실험 파라미터 기록 (선택) |
