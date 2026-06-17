# AROMA Exp 5 - 크로스 도메인 일반화 검증

## (사용할 skills: feature-dev)

## 개요

AROMA ROI selection 전략이 다양한 도메인(ISP, MVTec, VisA)에서
일관된 성능 향상을 제공하는지 검증한다.

핵심 질문: 도메인이 달라져도 AROMA Coverage + Deficit-aware ROI selection이
Random 대비 우위를 유지하는가?

도메인 구성:
- ISP 도메인: isp_LSM_1, isp_LSM_2, isp_ASM (복수 공정 라인)
- MVTec 대표: mvtec_cable
- VisA: visa_cashew, visa_pcb

## 영향도 분석

### 이 기능이 변경하는 상태
- $AROMA_OUT/exp5/exp5_results.json
- $AROMA_OUT/exp5/exp5_summary.md

### 선행 조건
Exp 4 완료 후 착수 (Exp 4 스크립트 재사용 가능성 높음)

## 설계 미결 사항 (확정 필요)

| 항목 | 선택지 | 현재 상태 |
|------|--------|----------|
| "크로스 도메인" 정의 | 각 도메인 독립 학습 후 비교 vs retraining 없이 타 도메인 평가 | 미결 |
| isp_LSM_2, isp_ASM 데이터 존재 여부 | AROMA_DATA 내 확인 필요 | 미확인 |
| Exp 4 재사용 | exp4 데이터셋 확장 방식 vs 별도 스크립트 | 미결 |
| AD 모델 범위 | 4종 전부 vs PatchCore 단독 | 미결 |

> TODO: "크로스 도메인 일반화"의 정확한 실험 설계를 논문 저자와 확정 필요.
> 가장 현실적인 해석: 동일 파이프라인을 여러 도메인에 적용했을 때
> 도메인 전반에 걸쳐 AROMA가 Random 대비 일관되게 우위를 보이는지 확인.

## 수정 내용 (실험 설계 확정 후 구체화)

### 1. scripts/aroma/experiments/exp5_crossdomain.py - 신규 작성 예정
Exp 4의 exp4_downstream_ad.py 확장 또는 래핑 방식 검토.

### 2. AROMA연구분析/colab_execute/exp5_execute.md - 신규 작성 예정

## 우선순위
Exp 4 구현/실행 완료 후 착수. 현재 설계 확정 단계.

## 테스트
CLAUDE.md: pytest 금지. Colab 직접 검증.
