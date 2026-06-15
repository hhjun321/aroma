# AROMA — PPT 슬라이드 초안

> 대상: SCI 저널 투고 준비용 연구 소개 발표  
> 구성: 7슬라이드 | Exp 1 제외 (미확립)

---

## Slide 1 — Title

**AROMA: Adaptive ROI Modeling for Anomaly Synthesis in Industrial Inspection**

- 산업 검사 이상 탐지를 위한 적응형 ROI 모델링
- 핵심 주장: *WHERE* 결함을 배치하느냐가 *HOW* 합성하느냐보다 중요하다

---

## Slide 2 — Problem

**산업 검사 데이터 희소성 문제**

- 결함 샘플 부족 → 합성 데이터 증강 필요
- 기존 방법의 한계:
  - 딥러닝 생성 모델: 대량 학습 데이터 필요, 도메인 종속
  - 고정 ROI 선택: 도메인 분포 무시, 희귀 패턴 누락
  - 도메인별 수동 설계(CASDA): 매 도메인 전문가 재작업 필요
- **핵심 문제**: 특정 (형태, 배경) 조합이 합성 데이터에서 반복 누락됨

---

## Slide 3 — Proposed Approach

**AROMA 파이프라인 (생성 모델 학습 불필요)**

```
Normal + Defect Images
    ↓ Phase 0: Distribution Profiling
    ↓ Step 1: MCI / CCI 계산 + Deficit 산출
    ↓ Step 2: Prompt 생성
    ↓ Step 3: Pair-Aware ROI 선택 (2단계 할당)
    ↓ Step 4: Copy-Paste 합성
→ Synthetic Defect Images
```

- 동일 파이프라인, 도메인별 설정 변경 없음

---

## Slide 4 — Key Concepts

**3가지 핵심 개념**

| 개념 | 정의 | 역할 |
|------|------|------|
| **MCI** (Morphology Complexity Index) | 결함 형태 다양성 지수 | 형태 클러스터 분류 기준 |
| **CCI** (Context Complexity Index) | 배경 맥락 다양성 지수 | 배경 셀 분류 기준 |
| **Deficit** | P(배경\|정상) − P(배경\|결함 근처) | 과소표현 쌍 정량화 |

**Pair-Aware 2단계 할당**
- Phase 1: 모든 (형태, 배경) 쌍에 기본 쿼타 1 보장 + Hamilton 방법으로 deficit 비례 배분
- Phase 2: 미충족 슬롯 → roi_score 상위 K개로 보충

---

## Slide 5 — Experiments

**Exp 2: Cross-Domain Generalization**

| 데이터셋 | 도메인 |
|---------|--------|
| isp_LSM_1 | ISP 센서 검사 |
| MVTec Cable | 산업용 케이블 |
| VisA Cashew | 농산물 |
| VisA PCB | PCB 기판 |

- 비교 방법: **Random ROI** vs **AROMA** (합성 방식 동일: copy-paste)
- 평가 지표:
  - ROI 품질: morphology/context coverage, rare_pair_coverage, entropy, gini
  - 합성 품질: FID (낮을수록 좋음)
  - 이상 탐지: Image AUROC, Pixel AUROC (PaDiM, ResNet-18)

---

## Slide 6 — Expected Results

**4가지 예상 주장**

| Claim | 내용 | 근거 |
|-------|------|------|
| **1** | AROMA > Random: rare_pair_coverage | Deficit 기반 선택 → 소수 쌍 보장 |
| **3** | AROMA < Random: FID | 맥락 적합 ROI → 현실적 합성 |
| **4** | AROMA > Random: Image/Pixel AUROC | 다양한 학습 데이터 → 탐지 성능↑ |
| **5** | 4개 도메인 모두 동일 파이프라인 적용 | 도메인 설정 변경 없음 |

---

## Slide 7 — Summary

**기여 및 한계**

**기여 4가지**
1. MCI/CCI: 정상 이미지만으로 도메인 자동 특성화
2. Deficit: (형태, 배경) 쌍 단위 과소표현 정량화
3. Pair-Aware 할당: 커버리지 보장 + 품질 보충 2단계
4. 도메인 무관: 동일 파이프라인, 수동 재설계 불필요

**한계 (명시 필요)**
> Copy-paste 합성은 기존 결함 외관을 그대로 보존하므로 새로운 결함 형태를 생성할 수 없다.
> 본 연구의 목적은 합성 모델 개선이 아닌, ROI 모델링이 합성 데이터 품질에 미치는 영향 평가다.

---

## 발표 메모

- Slide 3 파이프라인 다이어그램: 화살표 플로우차트로 시각화 권장
- Slide 4 Deficit 수식: `deficit = max(0, P(C|normal) − P(C|defect))` 추가 가능
- Slide 6 수치 placeholder: 실험 완료 후 `[X.X]` 채우기
- 총 발표 시간 목표: 10-15분 (슬라이드당 1.5-2분)
