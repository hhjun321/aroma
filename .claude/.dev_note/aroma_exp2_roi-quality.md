# AROMA Exp 2 — ROI 품질 분석 (Random ROI 기준선 + AROMA vs Random 비교)

---

## (사용할 skills: feature-dev)

## 개요

AROMA Steps 1-4가 4개 데이터셋(isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb)에서 완료됨.
AROMA의 deficit-aware ROI 선택이 무작위 선택 대비 더 나은 형태학적/컨텍스트 커버리지를 
만드는지 정량적으로 입증하기 위해 Random ROI 기준선과 비교 실험 스크립트를 구현한다.

논문 근거: AROMA-sharpened-spec.md §성공 기준 — Morphology Coverage, Context Coverage, 
Rare Pair Coverage, Entropy, Gini.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `roi_selection.py` — `--sampling_strategy random` 옵션 추가
- `scripts/aroma/experiments/` 디렉터리 신규 생성
- `exp2_roi_quality.py` 신규 생성 — `roi_selected.json` / `roi_candidates.json` 읽기만, 쓰기는 `exp2/` 디렉터리에만

### 그 상태를 전제로 동작하는 기존 로직
- `roi_selection.py`의 `select_rois()` — 기존 3개 전략(deficit_aware, top_k, weighted) 동작 불변
- 기존 `roi_selected.json` 형식 불변 (random도 동일 candidate 스키마 출력)

---

## 수정 내용

### 1. `scripts/aroma/roi_selection.py` — `--sampling_strategy random` 추가

`select_rois()` 함수에 `strategy == 'random'` 분기 추가:

```python
if strategy == "random":
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=top_k, replace=False)
    return [candidates[i] for i in sorted(indices.tolist())]
```

- 클램프 `top_k = max(1, min(top_k, len(candidates)))` 이후에 위치 (기존 패턴 동일)
- `_parse_args()`의 `--sampling_strategy choices`에 `'random'` 추가
- docstring 전략 목록에 random 추가

---

### 2. `scripts/aroma/experiments/exp2_roi_quality.py` — 신규 생성

**지표 정의:**

| 지표 | 정의 | 방향 |
|------|------|------|
| morphology_coverage | unique cluster_id in selected / unique cluster_id in candidates | 높을수록 좋음 |
| context_coverage | unique cell_key in selected / unique cell_key in candidates | 높을수록 좋음 |
| rare_pair_coverage | deficit≥p75 unique (cluster,cell_key) in selected / total deficit≥p75 in candidates | 높을수록 좋음 |
| entropy | Shannon entropy of cluster_id distribution in selected | 높을수록 균등 |
| gini | Gini coefficient of cluster_id distribution in selected | 낮을수록 균등 |

**입력 인터페이스:**
```
--aroma_roi_dir   $AROMA_OUT/roi           # {dataset}/roi_selected.json + roi_candidates.json
--random_roi_dir  $AROMA_OUT/roi_random    # {dataset}/roi_selected.json (random strategy 출력)
--dataset_keys    isp_LSM_1 mvtec_cable visa_cashew visa_pcb
--output_dir      $AROMA_OUT/exp2
```

**출력:**
- `exp2_results.json`: `{ dataset: { aroma: { morph_cov, ctx_cov, rare_pair_cov, entropy, gini }, random: {...} } }`
- `exp2_summary.md`: 데이터셋 × 전략 × 지표 마크다운 테이블

**엣지 케이스 처리:**
- 분모 0 (candidate 없음, 클러스터 1개) → 0.0 반환, 경고 로그
- rare pair threshold p75 = 0 (모든 deficit=0인 경우) → rare_pair_coverage=1.0으로 처리 + 주석
- 데이터셋별 json 없으면 skip + warning (4개 중 일부만 준비된 경우)
- entropy: 클러스터 단일값 → 0.0 (정상)
- random selected가 AROMA candidates와 동일 스키마인지 단순 assert

**코드 패턴 (기존 스크립트 통일):**
- `utils.io` bootstrap + inline fallback (roi_selection.py 패턴 동일)
- `logging.basicConfig` + `logger = logging.getLogger("aroma.exp2")`
- entropy 로그 밑: 자연로그(nats). summary.md에 단위 명시.
- entropy 정규화: `H / log(n_clusters)` (0~1 스케일) — summary.md 가독성 위해 정규화 적용

---

### 3. `AROMA연구분析/colab_execute/exp2_execute.md` — 신규 생성

colab-execution.md 규칙 준수 (CPU, `$VAR` 포맷, `!python` 접두사).

실행 순서 2단계:
1. `roi_selection.py --sampling_strategy random`으로 4개 데이터셋 Random ROI 생성 → `$AROMA_OUT/roi_random/{ds}/`
2. `exp2_roi_quality.py`로 AROMA vs Random 비교 → `$AROMA_OUT/exp2/`

환경변수:
```python
os.environ['RANDOM_ROI_DIR'] = f"{os.environ['AROMA_OUT']}/roi_random"
os.environ['EXP2_OUT']       = f"{os.environ['AROMA_OUT']}/exp2"
```

결과 확인 셀: `exp2_summary.md` 출력 + 지표별 AROMA≥Random 여부 요약.

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` — random 전략 추가
- `scripts/aroma/experiments/exp2_roi_quality.py` — 신규 (experiments/ 디렉터리도 신규)
- `AROMA연구분析/colab_execute/exp2_execute.md` — 신규

---

## 테스트

CLAUDE.md 규칙: pytest 금지. Colab에서 직접 검증.

1. `--sampling_strategy random --seed 42` 재실행 → 동일 roi_selected.json 재현 (seed 고정 확인)
2. random selected의 cluster 분포가 deficit_aware 대비 더 쏠리는지 육안 확인
3. `exp2_roi_quality.py` 실행 → results.json에 4×2×5 지표 모두 채워지는지
4. 기대 방향: AROMA morph/ctx/rare_pair coverage ≥ random, entropy ↑, gini ↓
5. top_k > candidates인 소형 데이터셋에서 클램프 동작 (에러 없음)

---

## TODO

- `--sampling_strategy random` seed 기본값: 42 권장 (기존 weighted seed와 통일) → 확인 필요
- entropy 정규화(H/log n) 적용 확정 — results.json과 summary.md 모두 동일 적용
- Random ROI 출력 디렉터리: `$AROMA_OUT/roi_random/` (기존 AROMA roi 덮어쓰기 방지)
- `scripts/aroma/experiments/__init__.py` 생성 여부: 스탠드얼론 스크립트이므로 불필요 (현재 판단)
