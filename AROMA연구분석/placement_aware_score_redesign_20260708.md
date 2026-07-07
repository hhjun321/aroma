I have all the verification I need. Now producing the final report.

---

# AROMA 배치-인지(Placement-aware) 전환 최종 보고서

## 요약 (TL;DR)

- **deficit 폐기 확정**: `score_roi = 0.4·morph + 0.4·ctx + 0.2·deficit`(`roi_selection.py:242,249,171-173`)에서 deficit 항을 0으로 제거. 실증 반증됨.
- **살아남은 신호 = ctx_prior(배경type↔결함type 호환) + quality_score(사실성)**, 단 **baseline 헤드룸이 있을 때만**(aitex 0.372).
- **핵심 미시도 레버 = placement**: `generate_defects.py`의 `_place_on`(line 845)은 배경 type을 전혀 계산하지 않는 **placement-blind**임을 코드로 확인. ctx_prior는 선택 스칼라(0.4wt)에만 쓰이고 픽셀 배치에는 한 번도 반영 안 됨.
- **★ 조사가 발견한 결정적 사실 (제안들의 기대를 뒤집음)**: 데이터셋별 clean-patch coverage 실측 결과 —
  - **leather(천장 0.83): 4.7%** — exact-match 배치는 ~95% legacy fallback → **placement-aware가 사실상 no-op**
  - **mtd(천장 0.92): 67.2%**
  - **aitex(헤드룸 0.372): 77.8%** — 매트릭스도 가장 풍부(25/9/20/12/8 cells)
  - 즉 **placement 레버가 물리적으로 실현 가능한 곳은 aitex 하나**뿐. leather에서 "placement-aware"를 켜면 대부분 fallback되어 legacy와 구분 불가.

---

## 1. 새 roi_score / 배치 objective 스펙

### 1.1 새 선택 score (deficit 제거, 사실성 승격)

**공식 (score_mode='realism'):**
```
roi_score = 0.5 · ctx_prior + 0.3 · morph_prior + 0.2 · quality_score
```

- `quality_score`는 `build_candidates`가 이미 후보마다 계산해 둔 값(`roi_selection.py:326-331, 362`, `SuitabilityEvaluator.matching_score(subtype, background_type)`). 현재는 `apply_quality_gate`의 **하드 게이트로만** 소비되고 **정렬에는 전혀 안 쓰임** → 이를 graded 항으로 승격하는 것이 유일한 신규 신호.
- deficit는 provenance용으로 JSON에는 남기되 weight=0.

**변경 위치:**
- `roi_selection.py:236 score_roi(...)` — `score_mode` 파라미터 추가. `'realism'`일 때 위 공식, `'legacy'`(기본)면 `0.4·morph+0.4·ctx+0.2·deficit` byte-identical.
- `roi_selection.py:326-331` — 이미 계산된 `quality_score`를 `score_roi`로 전달(현재는 인자에 없음).
- `moderated_score`/`_order_key`(line 451, 494)는 여전히 `roi_score`를 읽으므로 무변경.
- CLI `--score_mode {legacy,realism}` 추가, `run()`에 배선.

**★ 정직성 경고 (검증 verdict에서 확정)**: 이 재가중은 `genuinely_new=false`다 — ctx_prior·morph_prior·quality는 이미 존재하던 항이다. **더 근본적으로, 아래 §2에서 밝혀진 대로 leather에서 ctx_prior는 값 자체가 살아있으나(0.18~0.77) 결함이 좁은 context에 몰려 있어 정렬 변별력이 낮다.** aitex는 매트릭스가 풍부해 재가중 효과가 실재할 여지가 있다. 가중치(0.5/0.3/0.2)는 test 수치 보기 전 커밋하고 사후 튜닝 금지.

### 1.2 새 배치 objective (닫힌 루프) — 채택 형태: **위치-조건화 compat 게이트**

선택이 약속한 문맥을 배치가 물리적으로 이행하도록, `_place_on`의 재샘플 루프에 술어 하나를 추가한다. **배경 이미지 선택(`rng.choice`)은 건드리지 않고 이미지 내 위치만 조건화** — 이것이 confound 최소 형태다(§4).

**공식 (배치 게이트, texture 게이트와 동형):**
```
위치 (px,py) 수락 조건:
   compat_row.get( cell_key(nrgb[py:py+ch, px:px+cw]) , 0.5 ) >= τ
   # compat_row = compatibility_matrix["matrix"][cluster_id]
   # cell_key = distribution_profiling._context_cell_key(feat, bin_edges)  (재사용)
max_bg_tries 소진 시 마지막 후보 유지 (silent drop 금지 — 기존 게이트 계약)
τ = 0.0 → 완전 OFF, rng 스트림 byte-identical (legacy)
```

**변경 위치 (`generate_defects.py`):**
- `_paste_and_finalize`(line 775) 시그니처에 `compat_row: dict`, `bin_edges`, `features`, `compat_threshold` 추가.
- `_place_on`(line 845) 내부 `_tex_ok`(line 859) 옆에 `_compat_ok(pos)` 헬퍼 추가 — patch 5-feature(`distribution_profiling._extract_context_features` 재사용, `GRID_SIZE=64`) → `_context_cell_key` → `compat_row.get(cell,0.5) >= threshold`. random-fallback 루프(line 903-917) 및 foreground 루프(line 885-894)의 `clean and close` 조건에 `and compat_ok`를 결합.
- `run()`에서 `roi_entry['cluster_id']`로 `matrix[cid]` row를 뽑아 전달. `matrix`/`bin_edges`는 `--config`(recommended_config.yaml)에서 로드(ControlNet 경로가 이미 `_load_bin_edges` 사용).
- `texture_on` 가드(line 839, 879)가 이미 정확히 이 "threshold=0 → rng 무변경" 스위치 패턴이므로 그대로 모방.

**★ leather 데이터 기반 필수 보정 — exact-match 대신 `compat>=τ` (soft)**: exact cell match를 쓰면 leather는 매칭 좌표가 clean-patch의 4.7%뿐이라 ~95% fallback한다. 반드시 `compat>=τ`(row에 있는 여러 cell을 모두 수용) 모드를 1차로 쓰고, **fallback률을 `gate_stats`에 집계**해 임계(>50%) 초과 시 그 실행을 "placement-aware"로 보고 금지.

---

## 2. ctx_prior "null" 진단의 실측 정정 (검증 6안이 요구한 확인)

6안(row-pull)의 전제 "ctx_prior가 null인 건 스칼라라서지 matrix가 비어서가 아니다"를 실제 데이터로 검증했다. **결과: 전제가 부분적으로 틀렸다.**

- leather 매트릭스는 **비어있지 않다**: cluster0={0.18, 0.18, 0.46, 0.18}, cluster2={0.16, 0.08, **0.77**}. `build_candidates`(line 348-349)는 **점유된 bin만** 순회하므로 방출되는 후보의 ctx_prior는 오히려 **0이 아니다**(0.08~0.77).
- 진짜 문제는 null이 아니라 **집중성**: leather 결함은 전부 `2_2_*`(고variance·고edge) context에만 몰려 있고, cluster1은 **완전히 비어있다**(`"1": {}`). 즉 3개 클러스터 중 1개는 후보를 못 만들고, 나머지는 좁은 context 밴드에 압축 → ctx_prior가 값은 있으나 **선택 정렬 변별력이 약하다**.
- **결정적**: 그 `2_2_*` 결함 context는 clean-background에서 **희소**(전체 good patch의 4.7%). 지배적 clean cell은 `0_0_*`/`1_1_*`(저·중 variance)인데 이들은 어떤 클러스터의 matrix에도 없다.

→ **6안(row-pull placement)은 leather에서 무의미**: row argmax cell(`2_2_0_0_1`)을 배경에서 찾아도 매칭 clean patch가 1.43%뿐이라 대부분 fallback. **aitex(coverage 77.8%)에서만 유효.** 6안은 5안과 목표가 겹치고 진단 전제가 데이터로 흔들리므로 **1차 채택 비권장**(5안으로 대체).

---

## 3. 가장 결정적인 단일 실험 (최저 비용, confound 분리)

**Placement × Selection 2×2 ablation harness** (검증에서 만장일치로 "먼저 배포"):

|  | selection = random | selection = AROMA(realism) |
|---|---|---|
| **placement = random(legacy)** | A (완전 baseline) | B (선택 전략만) |
| **placement = compat-gate(τ)** | C (배치 게이트만) | D (풀 파이프라인) |

- **모든 arm이 동일 후보 풀·동일 copy-paste 엔진(`_paste_and_finalize`)·단일 seed rng·동일 budget(top_k, n_per_roi)**. 차이는 CLI 플래그 2개(`--score_mode`, `--compat_threshold`)뿐.
- **판정 논리:**
  - **D>B and D>C**: 선택+배치 상승 효과 = 진짜 전략 이득.
  - **C≈D (C도 이득)**: 이득이 **배치 게이트가 부여하는 문맥 필터 자체**에서 옴(pool-agnostic) → AROMA 선택에 귀속 금지, 정직히 규정.
  - **B≈A and D≈C**: 선택 무효, 배치만이 레버.
  - **A≈B≈C≈D**: 무효(천장 or 저-exposure) → "expected null" 선언.
- **필수 telemetry(사전 등록, 학습 전 파일로 freeze)**: 데이터셋별 (a) baseline 헤드룸, (b) **placement fallback률**(§1.2), (c) 실제 paste 위치의 mean texture-distance. 이들을 `placement_report.json`에 기록(annotations.json 아님).
- **비용**: 새 score 수학 없음. 오케스트레이션 + 로깅만. 검증에서 지목한 **최저 비용·최고 판정력** 실험. 이것이 나머지 제안을 falsifiable하게 만든다.

**symmetric filter (전략 vs 후보풀 분리) 추가 arm**: clean-patch에 실현 가능한 cell을 가진 후보만 남기는 realizability gate를 **전략 분기 전(`run()` 최상단, random 경로도 `_pair_aware_allocation`을 우회하므로 여기서 주입)** 양 arm에 대칭 적용. 필터 생존율을 데이터셋별 보고(leather는 대부분 필터링될 것 → 게이트가 no-op임이 드러남).

---

## 4. 채택 / 기각 판정

### 채택 (adopt / adopt_with_fix)

| 제안 | 판정 | 핵심 |
|---|---|---|
| **Compat 임계 배경 게이트 (5안)** | **adopt (1차)** | 배경 이미지 marginal 불변·위치만 조건화 → confound 최소. `texture_on`과 동일 rng-discipline(threshold=0 OFF). §1.2 채택 형태. |
| **Honest ablation harness (8안)** | **adopt (먼저 배포)** | §3. 단 'heatmap' arm은 뒷받침 구현 없으니 `{random, compat-gate}`로 축소. 예측 사전등록 강제. |
| **Ceiling-aware diagnostic (mech 3안)** | **adopt** | 순수 계측. 천장 null을 "메커니즘 확인"으로 전환. 모든 제안 강화. |
| **P2 realism roi_score** | **adopt_with_fix** | §1.1. `genuinely_new=false`이나 저비용·confound 깨끗(random은 roi_score 무시). 가중치 사전 고정. |
| **Clean-bg cell inventory (2안)** | **adopt_with_fix** | 5안의 런타임 재계산을 오프라인 인덱스로 대체(재계산 0). 단 normal 풀 커버리지 리포트 필수, 큰 crop은 다수결 cell. |
| **Placement cell-match (1안/P1)** | **adopt_with_fix, 단 leather 무효** | exact 대신 compat>=τ 권장. **fallback률 실측 보고 필수 — leather 95% fallback.** |

### 기각

| 제안 | 기각 사유 |
|---|---|
| **닫힌 루프 배경+위치 동시 softmax (4안)** | **rigging_risk high + confound 최대**. 배경 이미지 marginal 분포까지 바꿔 "차이는 전략만" 원칙 붕괴 — random은 균일 배경, AROMA는 compat 편향 배경. 5안이 동일 목표를 더 낮은 confound로 달성. |
| **Placement-by-cell-key (mech 중복)** | **P1/1안과 substantive 중복(genuinely_new=false)**. nearest-cell L1 fallback + pool-restricted arm만 harness로 흡수. 이중 구현 위험. |
| **P2를 "새 레버"로 주장** | **ctx_prior 재포장 경계 위반**. ctx 가중을 0.4→0.5로 올려도 변별력 문제(§2)는 안 풀림. 유일 신규 신호는 0.2 quality 항 — 이 사실을 명시하고 "선택-전용 ablation, placement gap 미해결"로 정직히 규정. |
| **row-pull placement (6안)** | **1차 기각**. 진단 전제(matrix가 비어서가 아니다)가 leather에서 데이터로 반증(§2). 5안과 목표 중복 — 동시 스윕 금지. |
| **Appearance heatmap (InstaBoost 연속, 7안)** | **1차 기각**. 신호 축이 다름 — 'source-surround 텍스처 유사성'이지 'compat-matrix 배경type↔결함type'가 아님. 비용 최고(sliding-window). 3-bin이 too coarse로 판명될 때만 fallback. |
| **directional realizability gate (mech 2안)** | **P1 위에 태워 흡수**(별도 JSON 금지). 유용하나 독립 제안 아님. |

---

## 5. 구조적 한계 (조건부 주장만 허용)

1. **천장(mtd 0.92, leather 0.83)에서 어떤 배치 전략도 random을 못 이길 수 있다** — 검출기가 이미 포화. 이는 실패가 아니라 예측된 null로 사전등록.

2. **★ leather는 이중 무효**: (a) 천장 + (b) **exact-match 배치가 4.7% coverage로 ~95% fallback** → placement 레버가 물리적으로 작동 안 함. leather에서 "placement-aware 우위" 주장은 fallback률 로그로 반박 가능하므로 **주장 금지**.

3. **placement이 실현 가능한 유일 데이터셋 = aitex**(coverage 77.8%, 헤드룸 0.372, 매트릭스 풍부). 조건부 주장: *"aitex처럼 baseline 헤드룸이 있고 clean-patch coverage가 높은 도메인에서만 배치-인지가 이득을 낼 수 있다."* aitex 단독 승리는 과적합 위험 — 반드시 harness의 C arm(배치 게이트만)과 비교해 pool-effect를 차감.

4. **severstal(체커플레이트) 미검증**: `_place_on`의 texture 게이트 주석이 severstal 24-34% foreground를 언급 — foreground가 큰 full-frame 객체 데이터셋에서는 clean-bg 게이트가 무의미. severstal profiling 데이터가 없어 coverage 미측정 → **severstal 결과는 별도 검증 전 주장 금지**.

5. **quality_score 축퇴 가능성**: 데이터셋에서 subtype이 단일하면 quality_score가 상수에 수렴 → P2 재가중이 no-op. 각 데이터셋에서 quality_score 분산을 학습 전 확인.

6. **"무조건 우위" 불가능**: 정직 제약대로, 어떤 score/배치도 모든 데이터셋에서 random을 이기지 못할 수 있다. harness가 A≈B≈C≈D를 내면 그것이 정직한 최종 결론이다.

---

## 참고 (검증된 코드/데이터 좌표)

- `roi_selection.py:236,242,249,171-173` — score_roi 공식·가중치 (deficit 0.2 확인)
- `roi_selection.py:326-331,362,407` — quality_score 계산·게이트-전용 소비 확인
- `roi_selection.py:345-351` — build_candidates가 점유 bin만 순회(ctx_prior≠0 방출) 확인
- `generate_defects.py:845-918` — `_place_on` placement-blind 확인, texture 게이트 rng-discipline(line 839,879)
- `distribution_profiling.py:471,488,207,84` — `_context_cell_key`/`_compute_bin_edges`/`_extract_context_features`/`GRID_SIZE=64` 재사용 가능 확인
- `distribution_profiling.py:877` — step6 매트릭스가 `image_type=='defect'`만 소비, **good 패치 미사용** 확인
- **실측 (본 조사)**: clean-patch in-matrix coverage — leather **4.7%**, mtd **67.2%**, aitex **77.8%**; leather matrix cluster1 비어있음, 결함 context는 `2_2_*`에 집중, clean 지배 cell은 `0_0_*`/`1_1_*`