# AROMA Phase 0 분석 — Data-Driven Anchor 설계 문서화

> WARNING: anchor.json entropy/FreqComplexity/OrientVariance raw_max fields are pending measured values. Needs Phase 0 valley validation first.

## (사용할 skills: micro-fix)

## 개요

`aroma_step1.yaml`의 `expected_range`는 현재 수작업으로 고정된 값(예: `valley_count: [0, 18]`)을 사용한다. 3종 데이터셋 분석 결과 valley_count가 27~39로 모두 상한을 초과해 norm=1.0 고정, MCI 판별력 소실이 확인됐다. 이 문서는 unbounded feature의 expected_range 상한을 데이터로부터 통계적으로 산출하는 **Global Anchor** 설계를 `phase0_analysis.md`에 추가 기록한다.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `AROMA연구분석/phase0_analysis.md` 문서 — 새 섹션 추가 (코드 변경 없음)

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (문서 전용 작업)

---

## 수정 내용

### 1. `AROMA연구분석/phase0_analysis.md` — 섹션 6 추가

현재 파일은 섹션 5(MCI 예측 및 개선 방향)와 요약 테이블로 끝난다.  
**요약 테이블 뒤에 섹션 6을 신규 추가**한다.

#### 6.1 Feature 분류 — Bounded vs Unbounded

| 분류 | Feature | 현재 expected_range | 처리 방법 |
|------|---------|-------------------|---------|
| **Bounded** | inv_silhouette | [0, 1] | 고정 유지 |
| **Bounded** | cluster_count | [2, 10] | 고정 유지 |
| **Unbounded** | valley_count | [0, 18] ← **문제** | Data-driven anchor |
| **Unbounded** | entropy | [0, ?] | Data-driven anchor |
| **Unbounded** | FreqComplexity (CCI) | [0, ?] | Data-driven anchor |
| **Unbounded** | OrientVariance (CCI) | [0, ?] | Data-driven anchor |

Bounded feature는 수학적·설정 기반 상한이 존재하므로 anchor 관리 불필요.  
Unbounded feature는 해상도·defect 종류·텍스처에 따라 값이 달라져 고정 범위 유지 불가.

#### 6.2 Anchor 산출 공식

```
n_datasets < 100:  anchor = max(dataset_values) × 1.2
n_datasets ≥ 100:  anchor = percentile(dataset_values, 95)
```

- 소규모(n<100): 관측 최댓값에 여유 20% 부여
- 대규모(n≥100): 극단값 무시, p95로 robust 상한 산출

#### 6.3 anchor.json 스키마 제안

```json
{
  "version": "1.0",
  "created_from": ["isp_LSM_1", "mvtec_cable", "visa_cashew"],
  "n_datasets": 3,
  "anchors": {
    "valley_count": {
      "method": "max_x1.2",
      "value": 46.8,
      "raw_max": 39.0
    },
    "entropy": {
      "method": "max_x1.2",
      "value": "<TODO: 실측 후 기입>",
      "raw_max": "<TODO>"
    }
  }
}
```

#### 6.4 시뮬레이션 결과 (valley_count 기준)

입력: valley_counts = [27, 39, 37] (isp / cable / cashew)  
anchor = max(39) × 1.2 = **46.8**

| 데이터셋 | valley_count | norm (÷46.8) | 기존 norm (÷18) |
|---------|-------------|-------------|---------------|
| isp_LSM_1 | 27 | **0.577** | 1.000 |
| mvtec_cable | 39 | **0.833** | 1.000 |
| visa_cashew | 37 | **0.791** | 1.000 |

→ 기존 판별력 0 → 3종 간 상대 순위 복원.  
→ 8종 defect cable이 2종 isp보다 높게 측정 — 직관과 일치.

#### 6.5 미해결 이슈 (TODO)

1. **histogram bins fix 선행 조건**: valley_count anchor 산출 전 과감지 버그(Sturges' rule bins 수정) 완료 필요. 현재 39가 아닌 실제 값으로 anchor 재계산 예정.
2. **threshold_n=100 현실성**: 현재 보유 데이터셋 3종 → n<100 경로만 사용. p95 경로는 향후 대규모 실험 후 검증 필요.
3. **entropy/FreqComplexity 단위**: 실측값 범위 미확인. Step 1 실행 후 `distribution_analysis.json`에서 entropy 분포 확인 후 anchor 추가 기입 필요.

---

## 수정 대상 파일

- `AROMA연구분석/phase0_analysis.md`

---

## 테스트

문서 검토 항목:
- [ ] 섹션 번호가 기존 1~5 및 요약과 충돌하지 않는지 확인
- [ ] 시뮬레이션 수치 (anchor=46.8, norm 값) 계산 재확인
- [ ] JSON 스키마 들여쓰기 및 문법 정상 여부
