# 1차 연구 — 지능적 배치의 유효성 검증

## 연구 개요

**연구명:** AROMA Phase 1 — 도메인별 ROI 기반 지능적 결함 배치의 유효성 검증  
**핵심 파이프라인:** Stage 3 (레이아웃 로직) + Stage 4 (MPB 합성)  
**검증 도메인:** MVTec AD · VisA · ISP-AD  

---

## 연구 가설

> **지능적 배치(Stage 3 적합도 기반 ROI 선택)가 무작위 배치보다 우월하다.**

- `aroma_pruned` (지능적 배치 + quality filtering) vs `baseline` (pretrained 특징 거리)
- 3개 도메인 × 2개 모델에서 AUROC 유의미한 상승으로 검증

---

## 실험 설계

### 비교 그룹

| 그룹 | 구성 | 의미 |
|------|------|------|
| `baseline` | 원본 good 이미지만 | 합성 없는 pretrained 기준선 |
| `aroma_full` | 원본 + 전체 Stage 4 합성 | 배치 전략 효과 (quality 필터링 없음) |
| `aroma_pruned` | 원본 + quality_score ≥ threshold 합성 | 배치 + 품질 필터링 최종 효과 |

### 모델

| 모델 | 역할 |
|------|------|
| YOLO11n-cls | 경량 분류 모델 |
| EfficientDet-D0 | 표준 분류 모델 (EfficientNet-B0 백본) |

### 평가 지표

| 지표 | 도메인 | 설명 |
|------|--------|------|
| `image_auroc` | 전 도메인 | 이미지 수준 이상 탐지 AUROC |
| `image_f1` | 전 도메인 | 이미지 수준 F1 |
| `pixel_auroc` | mvtec, visa | 픽셀 수준 이상 탐지 AUROC |

### 실험 규모

```
2 models × 3 groups × 24 categories = 144 runs
  - ISP:   3개 카테고리 (ASM, LSM_1, LSM_2 제외)
  - MVTec: 15개 카테고리
  - VisA:  8개 카테고리 (macaroni2, pcb2~4 제외)
```

---

## 도메인별 특성 및 AROMA 적용 전략

### MVTec AD

| 항목 | 내용 |
|------|------|
| 태스크 | Segmentation (픽셀 수준 이상 탐지) |
| 표면 다양성 | 높음 — 금속·직물·나무·타일 등 15종 |
| 증강 비율 | `aroma_full` 2:1 / `aroma_pruned` 1.5:1 |
| pruning_threshold | 0.6 |
| 핵심 배치 전략 | directional 배경에 linear_scratch 정렬, compact_blob을 smooth 표면에 배치 |

### VisA

| 항목 | 내용 |
|------|------|
| 태스크 | Segmentation |
| 특이사항 | 고해상도 이미지(~3032×2016), candle 왁스 표면 hf_ratio 과대평가 |
| 증강 비율 | `aroma_full` 2:1 / `aroma_pruned` 1.5:1 |
| pruning_threshold | **0.4** (smooth 표면에서 artifact_score 보정) |
| PNG 압축 | `compression=1` (쓰기 속도 최적화) |

### ISP-AD

| 항목 | 내용 |
|------|------|
| 태스크 | Classification (이미지 수준 불량 판정) |
| 표면 | 라인스캔 카메라 — directional/periodic 배경 우세 |
| 증강 비율 | `aroma_full` 1:1 / `aroma_pruned` 0.5:1 |
| pruning_threshold | 0.6 |
| 핵심 배치 전략 | linear_scratch → directional 배경 dominant_angle 정렬 |

---

## Stage 3 — 지능적 배치 핵심 메커니즘

### 적합도 점수 계산

```
suitability = 0.4 × matching_score(subtype, bg_type)
            + 0.3 × continuity_score
            + 0.2 × stability_score
            + 0.1 × gram_similarity
```

### 배치 전략의 차별점

| 항목 | 무작위 배치 | AROMA 지능적 배치 |
|------|-----------|----------------|
| 위치 선택 | 랜덤 | 적합도 최고 ROI 선택 |
| 회전 | 랜덤 | directional 배경 결 정렬 |
| 결함-배경 매칭 | 없음 | subtype × background_type 점수표 |
| 품질 필터링 | 없음 | quality_score 임계값 pruning |

---

## 결과 요약

> 수치는 실험 완료 후 기입

### AUROC 비교표

| 도메인 | 모델 | baseline | aroma_full | aroma_pruned | Δ (pruned - baseline) |
|--------|------|---------|-----------|-------------|----------------------|
| MVTec | YOLO11 | — | — | — | — |
| MVTec | EfficientDet | — | — | — | — |
| VisA | YOLO11 | — | — | — | — |
| VisA | EfficientDet | — | — | — | — |
| ISP | YOLO11 | — | — | — | — |
| ISP | EfficientDet | — | — | — | — |

### 검증 기준

| 조건 | 해석 |
|------|------|
| `aroma_pruned` > `baseline` (Δ > 0) | 지능적 배치 유효 |
| `aroma_pruned` > `aroma_full` | quality filtering 추가 효과 있음 |
| 3개 도메인 모두 유의미한 상승 | 도메인 범용성 확인 |

---

## 결론 구조

1. **Stage 3 유효성:** 적합도 기반 ROI 선택이 무작위 대비 AUROC 상승에 기여
2. **Stage 4 유효성:** MPB 합성 이미지가 실제 결함 패턴을 충분히 모사
3. **도메인 범용성:** MVTec(segmentation) · VisA(고해상도) · ISP(classification) 3개 도메인 모두에서 유효
4. **quality pruning 효과:** `aroma_full` → `aroma_pruned` 추가 상승 → quality_score 기반 필터링의 독립적 기여 확인

---

## 한계 및 2차 연구 연결

| 한계 | 2차 연구 방향 |
|------|-------------|
| 이진 분류(good/defect)만 검증 | 다중 결함 유형 분류 (multi-class) |
| MPB 합성 품질 한계 | CASDA 적대적 합성으로 교체·비교 |
| 평가 지표: AUROC/F1 | macro-F1, per-class accuracy 추가 |

→ [[13-Research-Phase2]]
