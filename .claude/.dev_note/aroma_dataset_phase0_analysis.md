# AROMA 데이터셋 Phase 0 1차 분석 문서

## 작업 유형
문서화 — 분석 결과 문서 작성

## 목적
isp_LSM_1 / mvtec_cable / visa_cashew 3종 데이터셋에 대해 Phase 0 (`distribution_profiling.py`) 출력 결과를 비교 분석하고, 주요 발견 사항 및 개선 포인트를 정리한다.

## 작성 위치
`D:\project\aroma\AROMA연구분석\phase0_analysis.md`

## 분석 대상 데이터

| 데이터셋 | 경로 |
|---------|------|
| isp_LSM_1 | `.claude/.etc/isp_LSM_1/` |
| mvtec_cable | `.claude/.etc/mvtec_cable/` |
| visa_cashew | `.claude/.etc/visa_cashew/` |

## 작업 항목

1. **기본 지표 비교표 작성**
   - defect 수, defect 종류 수, context patches, fallback masks 비율

2. **Valley Count 분석**
   - feature별 valley 수 3종 비교표
   - expected_range [0,18] 초과 현황 → 전 데이터셋 norm=1.0 천장 문제
   - 과감지 의심 feature 식별 (circularity, eccentricity)
   - expected_range 재보정 제안 ([0,45] 가정 시 norm 비교)

3. **클러스터 품질 비교**
   - 클러스터 라벨 및 centroid 특성
   - 클러스터 균형도 (최대 점유율)
   - isp_LSM_1 Otsu fallback 영향 분석 (area/points 혼재, circularity 왜곡)
   - visa_cashew linear_scratch 명확 분리 특이사항

4. **Fallback Mask 영향 분석**
   - isp_LSM_1: 100% Otsu → points 결함 circularity≈0 왜곡 근거
   - GT mask 보유 셋(cable, cashew) vs Otsu 셋(isp) 형태학 신뢰도 차이

5. **MCI 예측 및 개선 방향**
   - valley_count ceiling 해소 전/후 예상 MCI 순서
   - expected_range 재보정 우선순위
   - valley 과감지 알고리즘 검토 필요성

## 참고 — 핵심 수치 요약

### Valley Count
| feature | isp_LSM_1 | mvtec_cable | visa_cashew |
|---------|----------|------------|------------|
| linearity | 5 | 7 | 9 |
| solidity | 3 | 3 | 3 |
| extent | 7 | 6 | 6 |
| aspect_ratio | 3 | 4 | 3 |
| eccentricity | 3 | 7 | **11** |
| circularity | 6 | **12** | 5 |
| **합계** | **27** | **39** | **37** |

### 클러스터
| 데이터셋 | 최대 점유율 | 특이 클러스터 |
|---------|-----------|------------|
| isp_LSM_1 | 54% (Otsu 노이즈) | cluster3: points인데 circularity=0.046 |
| mvtec_cable | 39% | elongated / compact_blob 명확 |
| visa_cashew | 35% | linear_scratch AR=10.05 완전 분리 |

### Expected Range 재보정 시뮬레이션
| 데이터셋 | [0,18] 현재 | [0,45] 제안 |
|---------|-----------|-----------|
| isp_LSM_1 | 1.000 | **0.600** |
| mvtec_cable | 1.000 | **0.867** |
| visa_cashew | 1.000 | **0.822** |
