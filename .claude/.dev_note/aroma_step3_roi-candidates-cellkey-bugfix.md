# AROMA Step 3 — ROI Candidates cell_key 고정 버그 수정

## 사용할 skills: micro-fix

---

## 개요

`build_candidates()`에서 `_context_cell_key(row, ...)` 호출 시 `row`가
`morphology_features.csv` 행(context feature 없음)이라 cell_key가
항상 `"0_0_0_0_0"` 으로 고정됨.
→ `prompts.json` 키와 불일치 → 대부분 cluster에서 prompt 공백 출력.

수정: morphology row 1개당 `cluster_row.items()` 전체를 순회해
(image × cell_key) 조합별 candidate를 생성한다.

---

## 영향도 분석

### 변경되는 상태
- `roi_candidates.json` candidate 수: `n_images × 1` → `n_images × n_occupied_bins`
- 각 candidate의 `cell_key`, `prompt`, `ctx_label` 필드가 실제 값으로 채워짐

### 그 상태를 전제로 동작하는 기존 로직
- `select_rois()` — candidates 리스트를 입력으로 받음. candidate 수 증가는 OK (top_k 클램핑으로 처리됨)
- `generate_defects.py` — `roi_selected.json`의 `prompt` 필드를 사용. 현재 공백이라 사실상 미동작 중

---

## 수정 내용

### 1. `scripts/aroma/roi_selection.py` — `build_candidates()` 내부 루프 교체

**현재 (버그)**:
```python
cell_key  = _context_cell_key(row, bin_edges, context_features, n_bins)
# row는 morphology_features.csv 행 → context feature 없음 → 항상 "0_0_0_0_0"
ctx_prior = float(cluster_row.get(cell_key, 0.0))
deficit   = float(deficit_rows.get(cid_str, {}).get(cell_key, 0.0))
roi_score = score_roi(morph_prior, ctx_prior, deficit)
prompt_key   = f"{cluster_id}_{cell_key}"
prompt_entry = prompts.get(prompt_key, {})
candidates.append({...})  # 이미지당 1개
```

**수정 후**:
```python
# cluster_row.items()로 점유된 context bin 전체 순회 → 이미지당 N개 생성
bin_iter = cluster_row.items() if cluster_row else {"none": 0.0}.items()
for cell_key, ctx_prior in bin_iter:
    deficit      = float(deficit_rows.get(cid_str, {}).get(cell_key, 0.0))
    roi_score    = score_roi(morph_prior, float(ctx_prior), deficit)
    prompt_key   = f"{cluster_id}_{cell_key}"
    prompt_entry = prompts.get(prompt_key, {})
    candidates.append({
        "image_id":    image_id,
        "image_path":  image_path,
        "cluster_id":  cluster_id,
        "cell_key":    cell_key,
        "roi_score":   round(roi_score, 6),
        "morph_prior": round(morph_prior, 6),
        "ctx_prior":   round(float(ctx_prior), 6),
        "deficit":     round(deficit, 6),
        "prompt":      prompt_entry.get("prompt", ""),
        "morph_label": morph_label,
        "ctx_label":   prompt_entry.get("context_descriptor", ""),
    })
```

**엣지 케이스**:
- `cluster_row`가 None/빈 dict → fallback `{"none": 0.0}` 으로 candidate 최소 1개 보장
- `cell_key`가 `deficit_rows`에 없으면 → `deficit = 0.0` (기존 동작 유지)
- `prompt_entry`가 없으면 → `prompt = ""`, `ctx_label = ""`

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` — `build_candidates()` 함수 내부 루프만

---

## 테스트

CLAUDE.md 규칙: pytest 실행 금지. Colab에서 직접 검증.

**검증 절차**:
1. 수정 후 Step 3 재실행 (`roi_selection.py`)
2. `roi_candidates.json` 열어 candidate 수 확인 (`n_images × n_occupied_bins` 규모)
3. `cell_key` 분포 확인 — `"0_0_0_0_0"` 단일 집중 → 다양한 키로 분산
4. Step 4 시각화 (step4_visualize.md) Section 5 재실행
   - cluster별 prompt 출력에 실제 텍스트 표시 확인
   - (이전) `prompt :` 공백 → (이후) 실제 prompt 문자열

---

## TODO

- candidates 폭증(`n_images × n_occupied_bins`)으로 대규모 dataset에서 메모리 영향 있을 수 있음 — 필요 시 `--max_candidates_per_image` 파라미터 추가 고려
- `cluster_row` 비어있을 때 fallback `"none"` cell_key에 대응하는 prompt가 `prompts.json`에 없으면 prompt 공백 — 허용 범위로 판단 (edge case)
