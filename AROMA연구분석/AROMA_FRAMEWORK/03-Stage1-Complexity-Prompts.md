# 03 — Stage 1: 복잡도 분석 & 프롬프트 생성

> **Claude 요약:** Stage 0(phase0) 프로파일링 산출을 읽어 **데이터셋 수준 복잡도 스칼라(MCI/CCI)**를 계산하고, 형태학·컨텍스트 **모델링 정책**을 경험적으로 선택한다(step1: `compute_complexity.py`). 이어 형태학 클러스터 × 컨텍스트 빈 조합마다 **고정 템플릿 기반 자연어 프롬프트**를 생성한다(step2: `prompt_generation.py`, LLM 불요). 두 단계 모두 **CPU** 실행이며, 복잡도 계산은 `--local_staging`으로 Drive I/O를 로컬로 스테이징한다. 산출은 다음 스테이지(ROI 선택·ControlNet 학습)의 정책·프롬프트 입력이 된다.

---

## 목적

- **step1 (compute_complexity)**: phase0 프로파일링을 읽어
  - `MCI` (Morphology Complexity Index) — 결함 형태의 다양성/복잡도
  - `CCI` (Context Complexity Index) — 배경 텍스처/컨텍스트의 복잡도
  - `morphology_policy` / `context_policy` — 클러스터링 정책을 실루엣 기반 경험 평가로 선택
- **step2 (prompt_generation)**: 클러스터 centroid(형태학) + 컨텍스트 빈 + compat matrix(prior)를 결합해 조합별 생성 프롬프트 문자열을 만든다. 다음 단계(ROI 선택 / ControlNet 조건)의 semantic 입력.

두 스크립트는 `scripts/aroma/`에 있으므로 Colab에서 `$AROMA_SCRIPTS/`로 호출한다.

---

## 입력 / 출력

`S(stage, ds)` = `sym_final/{stage}/{ds}` (stage-first 규약, `_SPEC §2`).

| 단계 | 입력 | 출력 | 산출 파일 |
|------|------|------|-----------|
| step1 | `S('profiling', ds)` | `S('complexity', ds)` = `sym_final/complexity/{ds}` | `complexity_report.json` |
| step2 | `S('profiling', ds)` + `S('complexity', ds)` | `S('prompts', ds)` = `sym_final/prompts/{ds}` | `prompts.json`, `prompts_summary.md` |

phase0 산출 소비 파일:
- step1 → `distribution_analysis.json`, `morphology_clusters.json`, `morphology_features.csv`, `context_features.csv`
- step2 → `morphology_clusters.json`, `compatibility_matrix.json`, (선택) `deficit_analysis.json` + step1의 `complexity_report.json`

> ⚠️ **연쇄 재실행**: phase0 재실행은 GMM 클러스터링을 재계산해 `cluster_id`가 바뀐다. step1의 MCI/CCI·정책, step2의 `{cluster_id}_{cell_key}` 키가 모두 클러스터에 종속되므로 phase0를 다시 돌렸으면 **step1 → step2 → step3까지 함께 재실행**한다(오류 없이 다운스트림 mAP로만 드러나는 조용한 불일치 방지).

---

## step1: `compute_complexity.py`

```python
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $PROF \
    --output_dir    $CPLX \
    --weight_mode   equal \
    --local_staging
```
(`$PROF = S('profiling', DS)`, `$CPLX = S('complexity', DS)`, DATASETS 4종 루프)

### 파라미터

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--profiling_dir` | ✔ | — | phase0 출력 디렉터리 (distribution_analysis.json 등) |
| `--output_dir` | ✔ | — | `complexity_report.json` 기록 위치 |
| `--config` | | `scripts/aroma/config/aroma_step1.yaml` | 없으면 내장 `_DEFAULT_CONFIG` 사용 |
| `--weight_mode` | | config값(`equal`) | MCI/CCI 가중 프리셋: `equal` / `entropy_heavy` / `diversity_heavy` |
| `--local_staging` | | off | phase0 파일을 `/content/tmp/aroma_staging/`로 복사 후 처리 (Drive I/O 단축, CPU 단계 허용) |
| `--staging_root` | | `/content/tmp/aroma_staging` | 스테이징 베이스 override |

### 복잡도 점수의 의미 / 계산 방식

- **MCI** = 4개 정규화 성분의 가중 평균(`equal` = 각 0.25):
  - `entropy` — 형태학 **클러스터 라벨 분포의 Shannon 엔트로피(bits)**. minmax 정규화 범위 `[0, 4]`.
  - `valley_count` — 6개 형태학 특징(`linearity, solidity, extent, aspect_ratio, eccentricity, circularity`)의 `n_valleys` 합. 범위 `[0, 41]`.
  - `class_diversity` — `defect_type` 카운트의 **Hill 다양성 지수** `Neff = exp(H)`. 정규화는 `log(max(Neff,1)) / log(n_max=8)`로 clamp. 단일 클래스면 `Neff = 1`.
  - `inv_silhouette` — `1 − silhouette(형태학 z-score 특징, 클러스터 라벨)`. 클러스터가 잘 분리될수록(높은 실루엣) MCI 기여 감소. sklearn 없으면 실루엣 0.0.
- **CCI** = 4개 정규화 성분의 가중 평균(`equal`):
  - `texture_entropy` (평균, 범위 `[0, 8]`), `cluster_count_ctx` (컨텍스트 특징 GMM+BIC 클러스터 수, 범위 `[1, 8]`), `freq_complexity` (`frequency_energy` 분산, `[0, 1]`), `orient_variance` (`orientation_consistency` 분산, `[0, 1.6]`).
  - 컨텍스트 특징은 `image_type ∈ {good, normal}` 행 우선 사용, 최대 20000 패치로 서브샘플(부족 시 bootstrap).
- 값이 클수록 형태/컨텍스트가 복잡·다양함을 의미. 정규화는 minmax(config `expected_range`)로 `[0,1]` clamp.
- **가중 프리셋**: `equal (.25×4)` / `entropy_heavy (.40,.20,.20,.20)` / `diversity_heavy (.20,.20,.40,.20)` — ablation용.

---

## step2: `prompt_generation.py`

```python
!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir  $PROF \
    --complexity_dir $CPLX \
    --output_dir     $PROMPTS
```
(`$PROMPTS = S('prompts', DS)`)

### 파라미터

| 인자 | 필수 | 설명 |
|------|------|------|
| `--profiling_dir` | ✔ | phase0 출력 (`morphology_clusters.json`, `compatibility_matrix.json`, 선택 `deficit_analysis.json`) |
| `--complexity_dir` | ✔ | step1 출력 (`complexity_report.json`) — 요약(MCI/CCI/policy) 표기에만 사용 |
| `--output_dir` | ✔ | `prompts.json` + `prompts_summary.md` 기록 위치 |

> `compute_complexity.py`와 달리 `--config`/`--weight_mode`/`--local_staging` 인자는 없다. 고정 템플릿이므로 LLM·외부 모델 불요.

### 프롬프트 구성 방식 (결함 유형 + 컨텍스트)

각 `(cluster_id × cell_key)` 조합마다 3개 서술어를 조합한다:

1. **형태학 서술어** (`generate_morphology_descriptor`, 클러스터 centroid 기반) — 임계 라벨링 후 `"{shape} {linearity} defect with {solidity} boundary"`:
   - `aspect_ratio`: `>3.0` highly elongated / `>1.5` elongated / else compact
   - `linearity`: `>0.65` linear / `>0.30` irregular / else scattered
   - `solidity`: `>0.80` solid / `>0.50` semi-solid / else fragmented
2. **컨텍스트 서술어** (`generate_context_descriptor`, `cell_key` = `'0_1_2_0_1'` 형식) — 5개 컨텍스트 특징 중 빈 값 상위 2개를 골라 각 특징의 `_CTX_LABELS`(low/med/high 3단계) 라벨을 붙이고 `"... surface background"` suffix 부착. 예: `"directional complex texture surface background"`.
3. **prior modifier** (`generate_prior_modifier`, `compatibility_matrix.json`의 `matrix[cluster][cell]` 확률) — `≥0.40` "predominantly occurring on this surface type" / `≥0.20` "commonly found ..." / else "rarely observed ...".

최종 프롬프트 = `assemble_prompt` = `"{morph_desc}, {ctx_desc}, {prior_mod}"`.

- 클러스터에 compat matrix 행이 없으면 컨텍스트 없는 단일 프롬프트(`{cid}_none`, "surface background" / "observed on this surface type")를 emit.
- 각 엔트리에 `prior_prob`(=cell 확률), `deficit`(`deficit_analysis.json` 조인), `cluster_id`, `phase0_label`, `cell_key`, `n_cluster_samples` 동봉.

### 출력 파일

| 파일 | 내용 |
|------|------|
| `prompts.json` | `"{cluster_id}_{cell_key}"` 키 → {morphology_descriptor, context_descriptor, prior_modifier, prompt, cluster_id, phase0_label, cell_key, prior_prob, deficit, n_cluster_samples} |
| `prompts_summary.md` | MCI/CCI/policy 헤더 + 조합별 (Key / Prompt / P(ctx\|cluster) / Deficit) 마크다운 테이블 |

---

## 핵심 계산 로직

### MCI / CCI 공식 (equal 가중 예)

```
MCI = 0.25·norm(entropy) + 0.25·norm(valley_count)
    + 0.25·norm(class_diversity) + 0.25·(1 − silhouette)

CCI = 0.25·norm(texture_entropy) + 0.25·norm(cluster_count_ctx)
    + 0.25·norm(freq_complexity) + 0.25·norm(orient_variance)
```
- `entropy = −Σ p·log2(p)` (클러스터 라벨 분포), `class_diversity = exp(−Σ p·ln p)` (defect_type 분포, Hill Neff).
- 정규화: minmax `(v − lo)/(hi − lo)` → `[0,1]` clamp. `class_diversity`만 log 스케일 정규화.
- 실루엣: sample_size 5000 서브샘플, 클래스 <2면 0.0.

### Meta Policy Generator (정책 선택)

1. **형태학 후보**(`_get_candidate_policies`) — valley 합으로 분기: `0` → `[percentile]`, `≤1` → `[otsu, percentile]`, 그 외 → `[gmm, otsu]`. `MCI ≥ 0.6`이면 `hierarchical` 추가, `MCI < 0.3`이면 `hierarchical`/`log_gmm` 제거. 최대 3개.
2. **컨텍스트 후보** — 고정 `[gmm, percentile]` (형태학 valley와 독립).
3. **경험적 평가**(`select_best_policy`) — 각 후보를 실제 적용해 실루엣으로 랭크. 1·2위 차 `|Δ| < stability_margin(0.05)`이면 **stability tie-break**(n_bootstrap=5 seed 평균 실루엣 최대 선택).
4. `_apply_policy`: percentile(aspect_ratio 33/66 분위 digitize), otsu(중앙값 이진), gmm/log_gmm(GMM+BIC), hierarchical(Agglomerative k=min(3,n)).

### 프롬프트 템플릿 (LLM 불요)

- 임계 기반 이산 라벨 테이블(`_AR_*`, `_LIN_*`, `_SOL_*`, `_CTX_LABELS`, `_PRIOR_*`)만으로 결정적 생성 — 동일 입력 → 동일 출력, 재현성 보장.

---

## 주의사항

- **경로 규약**: 출력은 반드시 stage-first `S('complexity', ds)` / `S('prompts', ds)` (`sym_final/{stage}/{ds}`). ds-first 금지(exp3/5/6 루트 규약 깨짐).
- **`--local_staging`**: complexity(CPU) 단계에만 사용. ControlNet **생성** 단계는 sidecar 캐시 Drive 직결이 필요하므로 미사용(`_SPEC §5`).
- **연쇄 재실행 필수**: phase0 재실행 시 `cluster_id` 변경 → step1·step2를 반드시 함께 재실행(구/신 혼용 금지).
- **sklearn 부재 시 폴백**: 실루엣 0.0, GMM→percentile 폴백. Colab에는 sklearn 존재하므로 정상 경로.
- **aitex(tiled, single-class)**: 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효. 단일 클래스면 `class_diversity = 1`(Neff=1).
- **사후 튜닝 금지**: `--weight_mode` 등 파라미터를 결과 보고 후 변경 금지. 테스트 코드 신규 작성·pytest 금지(CLAUDE.md) — 검증은 Colab 셀 실행으로.
- step2는 `compatibility_matrix.json` 부재 시 빈 프롬프트를 낼 수 있음 — phase0 산출 존재를 STEP 1 루프에서 assert.

---

## 관련 노트

[[00-INDEX]] | [[02-Stage0-Prepare-Profiling]] | [[04-Stage2-ROI-Selection]] | [[07-Scripts-Reference]]
