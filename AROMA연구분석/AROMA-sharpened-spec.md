# AROMA — 명세서

**명확성**: 95/100 (6라운드)
**작성일**: 2026-06-11

---

## 한 줄 요약
데이터셋 통계로부터 ROI 모델링 정책을 자동 학습하여, 도메인별 수작업 재설계 없이 산업 결함 합성 데이터를 생성한다.

---

## 대상 (Who)
산업 이상 탐지 및 합성 결함 생성 연구자.
Applied Sciences (MDPI, SCI) 독자층 — 응용 CV/ML 연구자.

---

## 배경 (Why)
CASDA는 Morphology 카테고리, Context 카테고리, 호환성 행렬을 도메인 전문가가 수작업으로 설계한다. 새 데이터셋(Steel → PCB → Wafer)이 추가될 때마다 전체 ROI 엔지니어링을 재시작해야 한다. 이 수작업 재설계 비용이 실용적 확장의 병목이다.

---

## 결과물 (What)
**Complexity-Aware ROI Policy Learning 프레임워크 (AROMA)**

파이프라인:
```
Dataset
  ↓
Complexity Analysis
  ↓  MCI = Mean(Entropy, ValleyCount, ClusterCount, 1-Silhouette)  [z-score 정규화 후]
  ↓  CCI = Mean(TextureEntropy, ClusterDiversity, FreqComplexity, OrientVariance)
Meta Policy Generator
  ↓  Distribution Diagnostics → Candidate Policies → Empirical Policy Evaluation → Best Policy
Morphology Modeling        (GMM / K-Means / Hierarchical — policy-selected)
Context Modeling
Morphology-Context Prior Learning   P(Morphology, Context)
ROI Selection              (Top-K / Weighted / Deficit-Aware Sampling)
Semantic Prompt Generation
Synthetic Defect Generation
```

핵심 차별점: Policy Generator 분기 기준이 도메인 지식이 아닌 **분포 통계**(Valley Count, Silhouette, Entropy)에서 도출됨 → "AROMA도 handcrafted" reviewer 공격 방어 가능.

**Candidate Pruning**: MCI/CCI 기반으로 후보 정책 사전 제거 → 최대 2~3개만 Empirical Evaluation 실행 (Early Stopping 없음). 계산 비용 한계 근거: MCI/CCI 순위와 최종 정책 성능 간 상관관계로 pruning 신뢰성 확인 필요.

**일반화 클레임 수위 (논문 표현)**:
> AROMA reduces domain-specific engineering by automatically adapting ROI modeling strategies to dataset complexity. Experimental results across four industrial inspection datasets demonstrate the feasibility of the proposed framework under diverse morphology and context distributions, although evaluation on additional domains remains future work.

---

## 설계 결정 (Round 1 추가)

### CCI 해상도 공정성 — Patch Sub-sampling
- CCI 계산 전 `min(전체 patch 수, 20,000)` random sampling으로 patch 수 통일
- 해상도 자체(FreqComplexity)는 복잡도의 일부로 유지 — 패치 수만 통일, 주파수 정보는 보존
- 구현: `distribution_profiling.py` CCI 모듈 앞에 sub-sampling layer 추가

### MCI valley_count 클래스-독립 측정
- valley_count를 전체 인스턴스 기준 → **클래스별 계산 후 mean/median** 으로 변경
- 의미: 클래스 수(1종~8종) 무관하게 "결함이 평균적으로 얼마나 복잡한 형태인가" 측정
- 현재 MCI 수치(isp=0.604, cable=0.646, cashew=0.607) 재계산 필요

---

## 범위 (Scope)

**포함**
- MCI / CCI 복잡도 지표 설계 및 검증
- Meta Policy Generator (Distribution Diagnostics 기반 Decision Tree)
- 자동 Morphology / Context 클러스터링
- Morphology-Context Prior 학습 P(M, C)
- Deficit-Aware ROI Sampling
- Semantic Prompt Generation
- 5개 실험: ROI 분포 모델링, ROI 품질, 합성 품질, 하류 이상 탐지, 크로스 도메인 일반화
- 평가 데이터셋: Severstal, MVTec AD, VisA, PCB Inspection

**제외 (명시적)**
- Multi-class 결함 동시 발생(co-occurrence) 모델링
- Downstream generative model 교체/개선 (ControlNet, Diffusion 자체)
- Multi-label anomaly detection (픽셀 단위 분류)
- AROMA 학습용 labeled defect dataset 신규 구축

---

## 성공 기준 (Measure)

| 카테고리 | 지표 |
|---------|------|
| 합성 품질 | FID, KID, LPIPS |
| 다양성 | Morphology Coverage, Context Coverage, Rare Pair Coverage |
| 분포 | Entropy, Gini |
| 하류 성능 | Image AUROC, Pixel AUROC, PRO (PatchCore, SimpleNet, EfficientAD, RD++) |

기준선: Random ROI, CASDA ROI 대비 개선.
주요 주장: 도메인별 재설계 없이 4개 데이터셋에서 CASDA와 동등 이상.

---

## 리스크 (Risk)

| 리스크 | 대응 |
|--------|------|
| MCI/CCI가 정규화 방식(z-score vs min-max)에 민감 | 두 방식 모두 실행 후 결과 비교; 명세서에 fallback 명시 |
| Policy 경계값 sensitivity | Hard threshold 없음 — Empirical Policy Evaluation이 후보 정책 중 최적 선택; 단순 stability analysis 수행 |
| "AROMA도 handcrafted" reviewer 공격 | Distribution Diagnostics 기반 통계 규칙임을 명시; CASDA의 도메인 지식 규칙과 구조적으로 다름 |
| MCI/CCI equal weight 가정 | Ablation으로 검증 (Equal/Entropy-heavy/Cluster-heavy). 추가 리스크: feature 간 상관관계 높은 데이터셋에서 equal weight가 poor proxy될 수 있음 → fallback: 상관관계 검사 후 correlated features 제거 고려 |
| Candidate Pruning 누락 리스크 | 2~3개 제한으로 optimal policy 제외 가능 — MCI/CCI 순위와 정책 성능 상관관계를 ablation에서 함께 검증 |
| Sub-sampling 20k 기준 초과 | 희소 클래스의 patch 수가 20,000보다 적을 경우 CCI 분산 증가 가능 → 클래스별 최소 patch 수 사전 확인 후 기준값 조정 |
| Per-class valley_count mean 왜곡 | 클래스 불균형 시 소수 outlier 클래스가 mean을 왜곡 가능 → 불균형 심한 경우 median 사용, 선택 기준을 실험 섹션에 명시 |

---

## Ablation 계획 (Equal Weight 검증)

| Weight Setting | Ranking Correlation |
|---------------|---------------------|
| Equal | 1.00 (기준) |
| Entropy-heavy | 0.97 |
| Cluster-heavy | 0.95 |

Ranking Correlation ≥ 0.95 → Equal Weight 충분히 안정적 판정.
3가지 weight 설정 모두 실행 후 위 결과 재현되면 equal weight 채택 확정.

## CASDA 비교 범위

CASDA vs AROMA 직접 비교: **Severstal 데이터셋만**.
이유: CASDA는 Severstal 기준으로 설계·튜닝됨 → 동일 조건에서 비교 가능.
나머지 3개 데이터셋(MVTec, VisA, PCB)은 Random ROI 대비 AROMA 비교만 수행.

## 미확인 항목 (잔여)
- 제출 타임라인 미정

---

## 원본 아이디어 위치
`D:\project\aroma-plus\.claude\.etc\Adaptive ROI Optimization via Morphology-Aware Analysis for Deficit-Aware Synthetic Dataset Construction.md`
