# AROMA Exp 4 - 하류 이상 탐지 성능 평가

## (사용할 skills: feature-dev)

## 개요

AROMA ROI selection으로 생성된 합성 결함 이미지를 학습 augmentation으로 사용했을 때
하류 이상 탐지(AD) 성능이 개선되는지 검증한다.

비교 조건:
- baseline: 정상 이미지만 학습
- random: 정상 + Random ROI 합성 결함
- aroma: 정상 + AROMA ROI 합성 결함

AD 모델 4종 (anomalib 1.x):
- PatchCore (wide_resnet50_2 backbone)
- SimpleNet
- EfficientAD (small)
- RD++ (Reverse Distillation++)

데이터셋: isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb

## 영향도 분석

### 이 기능이 변경하는 상태
- $AROMA_OUT/exp4/exp4_results.json
- $AROMA_OUT/exp4/exp4_summary.md

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (평가 전용)

### 선행 조건
Step 4 완료 (synthetic_aroma + synthetic_random) -> Exp 3 완료 -> Exp 4

## 수정 내용

### 1. scripts/aroma/experiments/exp4_downstream_ad.py - 신규 작성

Exp 3의 AD 모드(PaDiM 단독)를 4개 모델로 확장.

핵심 설계 원칙 (Exp 3에서 상속):
- Train = 실제 정상 + 합성 결함 (조건별)
- Test = 실제 결함 + 실제 정상 (합성 절대 포함 금지)
- seed=42 고정

CLI 인터페이스:
--model {patchcore|simplenet|efficient_ad|rd_plus_plus|all}
--condition {baseline|random|aroma|all}
--dataset_keys {isp_LSM_1|mvtec_cable|...}

출력 형식:
{
  "isp_LSM_1": {
    "patchcore": {
      "baseline": {"image_auroc": 0.72, "pixel_auroc": 0.68},
      "random":   {"image_auroc": 0.75, "pixel_auroc": 0.71},
      "aroma":    {"image_auroc": 0.81, "pixel_auroc": 0.77}
    },
    "simplenet": {...},
    "efficient_ad": {...},
    "rd_plus_plus": {...}
  }
}

### 2. AROMA연구분析/colab_execute/exp4_execute.md - 신규 작성

Colab Pro A100 기준 예상: PatchCore 1모델 3조건 4데이터셋 = 60-90분

## 수정 대상 파일
- scripts/aroma/experiments/exp4_downstream_ad.py (신규)
- AROMA연구분析/colab_execute/exp4_execute.md (신규)

## 설계 미결 사항

| 항목 | 옵션 | 제안 |
|------|------|------|
| PatchCore backbone | ResNet-18 vs wide_resnet50_2 | wide_resnet50_2 (Exp 3 PaDiM과 다른 backbone) |
| PRO Score | anomalib 지원 여부 | 지원 시 포함, 미지원 시 null |
| PaDiM(Exp 3) 대체 여부 | Exp 3 PaDiM이 Exp 4를 대체? | 아님. 별도 실험 |

## 엣지 케이스

| 상황 | 처리 |
|------|------|
| 특정 모델 학습 실패 | 해당 모델 skip + WARNING |
| GPU 없음 | 에러 + 종료 코드 1 |

## 테스트
CLAUDE.md: pytest 금지. Colab 직접 검증.
PatchCore 단일 + baseline 조건 + isp_LSM_1 단일로 smoke test 먼저.
