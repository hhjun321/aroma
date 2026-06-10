# AROMA Step 2 — Prompt Generation

## 사용할 skills: feature-dev

## 실행 환경

| 연산 | 자원 | 비고 |
|------|------|------|
| 프롬프트 조합 생성 | **CPU** | 순수 Python dict/string 연산 |
| JSON 파일 I/O | **CPU** | 입출력 파일 크기 소규모 |

> Workers 불필요 — cluster × cell 조합 수가 수십~수백 건이므로 단일 프로세스로 충분.

---

## 개요

Phase 0 + Step 1 출력을 읽어 형태학 클러스터 × 컨텍스트 빈 조합별 자연어 프롬프트를 생성한다.
LLM 없이 고정 템플릿(fixed template) 방식으로 구현.

---

## 입력

| 파일 | 위치 | 내용 |
|------|------|------|
| `morphology_clusters.json` | Phase 0 output | 클러스터별 centroid (aspect_ratio, linearity, solidity 등 6개) |
| `compatibility_matrix.json` | Phase 0 output | P(ctx_bin \| cluster), bin_edges |
| `deficit_analysis.json` | Phase 0 output | cluster prior, 셀별 deficit |
| `complexity_report.json` | Step 1 output | MCI, CCI, policy (선택 참조) |

### compatibility_matrix 구조
```json
{
  "n_clusters": 2,
  "n_context_bins": 3,
  "context_features": ["local_variance", "edge_density", "texture_entropy", "frequency_energy", "orientation_consistency"],
  "bin_edges": { "feature": [p33, p66] },
  "matrix": { "cluster_id_str": { "cell_key": probability } }
}
```
cell_key 형식: `"0_1_2_0_1"` (5개 feature × bin 0-2)

---

## 구현 항목

1. `generate_morphology_descriptor(centroid) -> str`
   - aspect_ratio (>3.0=highly elongated, >1.5=elongated, else=compact)
   - linearity (>0.65=linear, >0.30=irregular, else=scattered)
   - solidity (>0.80=solid, >0.50=semi-solid, else=fragmented)
   - 출력: `"{shape} {linearity} defect with {solidity} boundary"`

2. `generate_context_descriptor(cell_key) -> str`
   - 5개 feature bin → 의미 레이블 (low/medium/high 대응)
   - 상위 2개 dominant feature 조합
   - 출력: `"{label1} {label2} surface background"`

3. `generate_prior_modifier(cluster_id, cell_key, matrix) -> str`
   - P >= 0.40: "predominantly occurring on this surface type"
   - P >= 0.20: "commonly found on this surface type"
   - else: "rarely observed on this surface type"

4. `assemble_prompt(morph_desc, ctx_desc, prior_mod) -> str`
   - `"{morph}, {ctx}, {prior_mod}"`

5. `generate_prompts(data) -> dict`
   - 모든 cluster × cell 조합 enumerate
   - cluster 없는 cell → "none" key로 neutral prompt

6. CLI: `--profiling_dir`, `--complexity_dir`, `--output_dir`

---

## 출력

```
{output_dir}/
  prompts.json         {"{cluster_id}_{cell_key}": {prompt, morphology_descriptor, context_descriptor, prior_modifier, prior_prob, deficit}}
  prompts_summary.md   마크다운 테이블
```

---

## 완료 기준

- `prompts.json` 생성 확인
- 모든 cluster × cell 조합 커버
- 빈 prompt 없음

---

## 아키텍처 노트

- 고정 템플릿만 사용 (LLM 호출 없음)
- `compatibility_matrix.json`의 matrix가 비어있으면 cluster당 "none" 셀 1개로 fallback
- utils.io import-or-inline 패턴 동일 적용

---

## 구현 완료 (2026-06-09)

- `scripts/aroma/prompt_generation.py` 생성
- `tests/aroma/test_prompt_generation.py` — 22/22 pass
