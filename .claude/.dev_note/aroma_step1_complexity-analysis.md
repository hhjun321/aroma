# AROMA Step 1 — MCI/CCI 복잡도 분석 및 ROI 정책 라우터 구현

---

## (사용할 skills: feature-dev)

## 개요

CASDA는 수작업 형태학 카테고리, 고정 임계값, 전문가 정의 호환성 매트릭스에 의존하는 규칙 기반 ROI 선택 프레임워크다. AROMA는 이 한계를 데이터 구동 방식으로 극복하기 위해, 결함 분포에서 형태학적·컨텍스트 복잡도를 정량화(MCI/CCI)하고 그 결과로 ROI 모델링 정책을 자동 선택한다.

**아키텍처 변경 (2026-06-09)**: `D:\project\aroma\scripts\distribution_profiling.py`가
형태학/컨텍스트 특징 추출·분포 분석·클러스터링·호환성 행렬·Deficit 분석을 이미 완전 구현함.
Step 1은 이 스크립트를 Phase 0으로 호출한 후, 출력을 읽어 **MCI/CCI 스칼라 + Meta Policy Generator**만 추가 구현하는 `compute_complexity.py`로 범위를 축소한다.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `scripts/aroma/compute_complexity.py` 신규 생성
- Phase 0(`distribution_profiling.py`) 출력 디렉토리를 입력으로 읽음

### 재활용하는 기존 구현
- `D:\project\aroma\scripts\distribution_profiling.py` — morphology_features.csv, context_features.csv, distribution_analysis.json, morphology_clusters.json 생성
- `D:\project\aroma\utils\defect_characterization.py` — DefectCharacterizer (형태 지표 계산)
- `D:\project\aroma\stage1b_seed_characterization.py` — extract_seed_mask (SAM/Otsu fallback)

---

## 수정 내용

### Phase 0 (기존 스크립트 — 직접 호출)

`distribution_profiling.py`를 dataset_key별로 먼저 실행. compute_complexity.py의 전제 조건.

```python
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    isp_LSM_1 \
    --output_dir     $AROMA_OUT/profiling/isp_LSM_1 \
    --num_workers    8
```

출력 파일 (compute_complexity.py 입력으로 사용):
- `morphology_features.csv` — 형태학 특징 (linearity, solidity, extent, aspect_ratio, eccentricity, circularity)
- `context_features.csv` — 컨텍스트 특징 64px 패치
- `distribution_analysis.json` — feature별 분포 유형 + 정책 (percentile/otsu/gmm)
- `morphology_clusters.json` — GMM/BIC 클러스터 (n_clusters, silhouette 포함)

---

### 1 (구현 제거됨) — 형태학 특징 추출

**Phase 0이 커버함.** `morphology_features.py` 별도 구현 불필요.
distribution_profiling.py의 `_morph_worker` + `DefectCharacterizer` 재활용.

> ⚠️ 차이점: distribution_profiling.py는 `Log Aspect Ratio`, `Fill Ratio`를 미포함.
> MCI 공식이 이 두 feature를 요구하지 않으므로 영향 없음.

### 2 (구현 제거됨) — 컨텍스트 특징 추출

**Phase 0이 커버함.** `context_features.py` 별도 구현 불필요.
64px 패치 기반 5개 feature: local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency.

> Context 특징 계산 범위 결정됨: 마스크 외부 패치(배경)에서 추출 + 마스크 내부 패치 skip. distribution_profiling.py의 `_context_worker` 구현 확인됨.

### 3 (구현 제거됨) — 분포 분석

**Phase 0이 커버함.** `distribution_analysis.py` 별도 구현 불필요.
distribution_profiling.py의 `_detect_valleys` + Step 4가 동일 역할 수행.

### 1. `scripts/aroma/features/morphology_features.py` — [제거됨]

### 2. `scripts/aroma/features/context_features.py` — 컨텍스트 특징 추출

`extract_context_features(image, mask) -> dict` 구현:
- Local Variance, Texture Entropy, Edge Density, Frequency Energy, Orientation Consistency
- TODO: 계산 스케일 — 마스크 내부만 vs 마스크 주변 패치 포함 범위 결정 필요

### 3. `scripts/aroma/complexity/distribution_analysis.py` — 분포 검출

- `analyze_distribution(values) -> dict`: Unimodal/Bimodal/Multimodal 검출, Valley Count, Entropy, Skewness, Kurtosis, Heavy-Tail 판정
- `estimate_cluster_separability(values) -> float`
- KDE/히스토그램 valley 검출 실패 시 → 안전한 기본 분포 유형 반환
- 최소 표본 미달(N < 임계값) → Unimodal fallback, TODO: 최소 표본 수 결정 필요
- TODO: "Distribution Complexity" 정확한 수식 정의 필요 (현재 개념어만)

### 4. `scripts/aroma/complexity/mci.py` — MCI 계산

```
정규화: 각 feature를 z-score (또는 min-max) 정규화
MCI = Mean(Entropy, ValleyCount, ClusterCount, 1 - Silhouette)
```

- `compute_mci(features, normalization='zscore') -> float`
- log(0) 방지, NaN/Inf 가드
- 정규화 방식(zscore/minmax)은 config에서 선택 가능하게 노출
- Ablation 실행용 weight override 파라미터 지원:
  - `equal`: [0.25, 0.25, 0.25, 0.25] (기본)
  - `entropy_heavy`: Entropy 가중치 상향
  - `cluster_heavy`: ClusterCount 가중치 상향

### 5. `scripts/aroma/complexity/cci.py` — CCI 계산

```
정규화: 각 feature를 z-score (또는 min-max) 정규화
CCI = Mean(TextureEntropy, ClusterCount, FrequencyComplexity, OrientationVariance)
```

- `compute_cci(features, normalization='zscore') -> float`
- MCI와 동일한 equal weight + normalization 구조 적용
- Ablation weight override 동일하게 지원
- ClusterCount: 컨텍스트 클러스터링 결과의 클러스터 수

### 6. `scripts/aroma/policy/policy_router.py` — 정책 라우팅

Distribution Diagnostics → Candidate Policies → Empirical Policy Evaluation → Best Policy

```
MCI / CCI
  ↓
Complexity Characterization
  ↓
Distribution Diagnostics (ValleyCount, ClusterCount, Silhouette, Entropy)
  ↓
Candidate Policy Set
  ↓
Empirical Policy Evaluation  ← 후보 정책 모두 실행 후 평가 지표로 최적 선택
  ↓
Best Policy
```

후보 정책 목록:

| 조건 | 후보 정책 |
|------|----------|
| Unimodal (ValleyCount=0) | Percentile Partition |
| Bimodal (ValleyCount=1) | Otsu Thresholding |
| Multimodal (ValleyCount≥2) | GMM |
| High Complexity (MCI 고값) | Hierarchical Clustering |
| Heavy-Tailed (Kurtosis 고값) | Log Transform + GMM |

- `get_candidate_policies(distribution_diagnostics: dict) -> list[str]`
- `evaluate_policy(policy, features) -> float` — 평가 지표(실루엣 등)로 점수 반환
- `select_best_policy(candidate_policies, features) -> str`
- Simple stability analysis: 상위 1·2위 정책 점수 차이 < threshold면 경계 케이스 로깅
- TODO: Empirical Policy Evaluation 평가 지표 확정 (Silhouette Score 우선 사용)

### 7. `scripts/aroma/analyze_complexity.py` — Step 1 엔트리포인트 (CLI)

```python
!python $AROMA_SCRIPTS/analyze_complexity.py \
    --image_dir $TRAIN_IMAGES \
    --train_csv $TRAIN_CSV \
    --config $AROMA_CONFIG \
    --output_dir $AROMA_OUT
```

출력: MCI(float), CCI(float), morphology_policy, context_policy, complexity_report(dict)
- 입력 이미지/CSV 로드 실패 → 컨텍스트(파일 경로 포함) 로그 후 해당 샘플 skip, 전체 파이프라인 중단 금지
- 대량 이미지 처리 시 배열 무한 push 금지 — 샘플 단위 순회
- TODO: 출력 리포트 포맷 결정 — per-sample JSON vs 집계 CSV vs 둘 다
- TODO: 입력 마스크 포맷 — Severstal train.csv는 RLE 추정, 디코딩 책임 위치 확정 필요

### 8. `scripts/aroma/config/aroma_step1.yaml` — 설정 파일

```yaml
mci:
  normalization: zscore   # zscore | minmax
  weights: equal          # equal | entropy_heavy | cluster_heavy (ablation용)

cci:
  normalization: zscore
  weights: equal

distribution:
  min_samples: 10         # TODO: 최소 표본 수 확정 필요
  valley_threshold: ???   # TODO: Bimodal/Multimodal 경계 확정 필요
  kurtosis_heavy_tail: ??? # TODO: Heavy-tail 판정 임계값 확정 필요

policy:
  stability_margin: 0.05  # 상위 1·2위 점수 차이 < margin → 경계 케이스 로깅
```

---

## Step 1 완료 기준

아래 3가지 모두 충족 시 Step 1 완료:

1. **기능 실행**: 전체 파이프라인 NaN/Error 없이 완료
2. **데이터셋 판별력**: Severstal(단순) vs VisA PCB(복잡)에서 MCI/CCI 값이 구별 가능한 차이 발생
3. **정책 라우팅 정합성**: 합성 단순 분포(Unimodal) → Percentile, 합성 bimodal → Otsu, multimodal → GMM 선택 확인

---

## 리스크

| 리스크 | 대응 |
|--------|------|
| z-score vs min-max 결과 차이 | config로 두 방식 모두 실행 후 비교; 발산 시 z-score 우선 |
| Policy 경계 케이스 | Empirical Evaluation + stability_margin 로깅으로 감지 |
| RLE 디코딩 실패 | TODO 해결 전까지 디코딩 예외 시 해당 샘플 skip + 경고 로그 |

---

## 참조 문서

- 전체 프레임워크 명세: `D:\project\aroma-plus\AROMA-sharpened-spec.md` (명확성 83/100)
- 원본 아이디어: `.claude\.etc\Adaptive ROI Optimization via Morphology-Aware Analysis...md`

---

## 수정 대상 파일 (신규 생성)

```
scripts/aroma/
├── analyze_complexity.py
├── config/
│   └── aroma_step1.yaml
├── features/
│   ├── morphology_features.py
│   └── context_features.py
├── complexity/
│   ├── distribution_analysis.py
│   ├── mci.py
│   └── cci.py
└── policy/
    └── policy_router.py
```

---

## 테스트 (Colab 셀)

환경변수 설정:
```python
import os
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_CONFIG']  = "/content/AROMA/scripts/aroma/config/aroma_step1.yaml"
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma/step1"
```

정상 실행:
```python
!python $AROMA_SCRIPTS/analyze_complexity.py \
    --image_dir $TRAIN_IMAGES \
    --train_csv $TRAIN_CSV \
    --config $AROMA_CONFIG \
    --output_dir $AROMA_OUT
```

단위 검증 (~10건 fixture):
- 빈 마스크 → `status: empty_mask`, 예외 없이 통과
- 단일 결함 → Unimodal fallback, Valley Count=0
- 동일 형태 N개 → Entropy≈0, NaN 없음
- 합성 bimodal 분포 → distribution_type=Bimodal, 정책=Otsu
- 합성 heavy-tailed → 정책=Log Transform + GMM
- 출력 MCI/CCI: `np.isfinite` 통과, NaN/Inf 없음

---

## 미확정 사항 (TODO 요약)

### 확정됨 (샤펀 세션 2026-06-09)
- ~~MCI/CCI 가중치~~ → Equal weight + z-score 정규화 확정
- ~~"Distribution Complexity" 수식~~ → ClusterCount 사용
- ~~"Cluster Separability" 수식~~ → 1 - Silhouette 사용
- ~~"Cluster Diversity" (CCI) 수식~~ → ClusterCount 사용
- ~~Policy Generator 방식~~ → Empirical Policy Evaluation 확정

### 미확정 (구현 중 결정 필요)
- Empirical Policy Evaluation 평가 지표 (Silhouette Score 우선 후보)
- Multimodal vs Bimodal 판정 경계 (ValleyCount 임계값, 최소 표본 수)
- 컨텍스트 특징 계산 스케일 (마스크 내부만 vs 주변 패치 포함)
- Heavy-Tailed 판정 기준 (Kurtosis 임계값 vs 통계 검정)
- 입력 마스크 포맷 (RLE 디코딩 위치 — Severstal train.csv 기준)
- 출력 리포트 영속화 포맷 (per-sample JSON vs 집계 CSV)

### 신규 추가 (샤펀 세션)
- Ablation 계획: Equal / Entropy-heavy / Cluster-heavy 3가지 weight 설정 실행
  - 성공 기준: Ranking Correlation ≥ 0.95 → equal weight 채택 확정
- CASDA 비교 범위: Severstal 데이터셋만 직접 비교
  - 나머지 3개 데이터셋(MVTec, VisA, PCB)은 Random ROI 대비만 수행
