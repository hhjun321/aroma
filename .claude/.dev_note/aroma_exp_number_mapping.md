# AROMA 논문 Exp 번호 vs 구현 Roadmap Exp 번호 매핑

## 배경
논문 Experiment 번호(E1, E2, E3, E4)와 개발 Roadmap의 Experiment 번호(Exp 1~5)가 달라 혼동 발생.

## 매핑 테이블

| 논문 Exp | 내용 | Roadmap Exp | 스크립트 | 상태 |
|---------|------|------------|---------|------|
| 논문 E1-1 | Cluster Recoverability (MCI 유효성) | 없음 | 없음 | 미구현 |
| 논문 E1-2 | Complexity Ranking Consistency (CCI 유효성) | 없음 | 없음 | 미구현 |
| 논문 E2-1 | Policy Selection Analysis | 없음 | 없음 | 미구현 |
| 논문 E2-2 | Alternative Policy Comparison | 없음 | 없음 | 미구현 |
| 논문 E3-1~3-3 | Morphology/Context/Rare Pair Coverage | Roadmap Exp 2 | exp2_roi_quality.py | 실측 미완료 |
| 논문 E3-4 | Distribution Balance | Roadmap Exp 2 | exp2_roi_quality.py | 실측 미완료 |
| 논문 E4 | Synthetic Quality (FID/KID/LPIPS) | Roadmap Exp 3 | exp3_generation_quality.py | FID만 구현, KID/LPIPS 없음 |
| 없음 (암묵) | 하류 이상 탐지 | Roadmap Exp 4 | exp4_downstream_ad.py | 미구현 |
| 없음 (암묵) | 크로스 도메인 일반화 | Roadmap Exp 5 | exp5_crossdomain.py | 미구현 |

## 번호 불일치 핵심

1. 논문 Exp 1/2(MCI/CCI 유효성, Policy 검증)는 Roadmap에 없다 — 미구현
2. 논문 Exp 3 = Roadmap Exp 2 (번호 다름)
3. 논문 Exp 4 일부 = Roadmap Exp 3 (FID만, KID/LPIPS/ControlNet 누락)
4. Roadmap Exp 4/5는 논문에 번호 없음 (C8 효과 검증으로 암묵적 포함)

## 논문 Exp 1/2 대응 방안 (미결)

- E1-1: 합성 known-cluster 데이터로 Cluster Recoverability 대체 가능한지 검토
- E1-2: 3-dataset 복잡도 순서(isp > cable > cashew)로 Ranking Consistency 대체 가능한지 검토
- E2-2: exp2_roi_quality.py에 Percentile/Otsu/GMM Silhouette 비교 ablation 추가 가능

> TODO: 논문 저자와 Exp 1/2 대응 방안 확정 필요
