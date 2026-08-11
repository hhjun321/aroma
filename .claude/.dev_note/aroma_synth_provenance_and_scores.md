# synth 샘플 출처·배치점수 전파 (`position_source` + `site_score`)

## (사용할 skills: feature-dev)

## (성격: 결함 수정 + 기록 확장 — **구현·검증·Colab 재실행·재실험 완료 (2026-08-11).** 종결 — 결론은 §10)

exp4v2(20260807, 5종 × 3 arm × 3 seed) 완료 후 `ring_sgm`+`k_fit`의 다운스트림 기여가 **미입증**으로 결론났다. 원인 분석 중, 합성 샘플의 **출처(ring 배치인가 폴백인가)와 배치 점수가 파이프라인 단계마다 유실**되어 학습셋 구성·사후 분석 어느 쪽에서도 사용 불가능한 상태임을 코드에서 확인했다.

본 노트는 그 유실 사슬 중 **① 출처 식별·배제**와 **② ring 점수 저장·전파** 두 건만 다룬다. 점수 기반 synth 선별(③)은 **범위 밖**이다.

관련: [[aroma_adjacent_context_bg_selection]](배치 개선 본체 · §4 위험표에 항목 15 추가 필요), [[aroma_exp4v2_quality-gate-fairness]](대칭 게이트 선례), [[aroma_cleanbg_gate_cv2_dtype_failopen]](런타임 void 게이트 fail-open)

---

## 1. 유실 사슬 (코드 근거)

| 단계 | 점수/정보 | 상태 |
|---|---|---|
| 1. ring 자리 선택 | `Σ min(hist_c(ring), tgt[k])` | **계산 후 폐기** |
| 2. 배경 선택 | `score` · `k_fit` · `class_fit` · `hist_intersection` | JSON 저장됨 → **annotation 전파 안 됨** |
| 3. 합성 | `roi_score` · `deficit` | annotation 기록됨 |
| 4. exp4v2 학습셋 | — | **아무 점수도 안 읽음. 균일 무작위 표본** |

**1단계** — `scripts/aroma/clean_bg_selection.py:611-614`

```python
score = sum(min(v, tgt[c]) for c, v in hist.items() if c in tgt)
if score > best:
    best, best_xy = score, (si * tile, sj * tile)
return best_xy          # ← score 폐기
```

ring 가설의 핵심 지표가 저장조차 되지 않는다. 배치 품질로 순위를 매길 수단이 없다.

**2단계** — `clean_bg_selection.py:933-951`이 `score`/`k_fit` 등을 `clean_bg_selected*.json`에 기록하나, `generate_defects.py:3321-3341`의 annotation 레코드에는 없다.

**4단계** — `scripts/aroma/experiments/exp4_v2_supervised_detection.py:2460-2461`

```python
rng_sub = random.Random(seed)
synth_by_cond[cond] = rng_sub.sample(anns, cap)
```

`roi_score` grep 결과 exp4v2에서 **0건**. 동일 패턴이 `:2367-2368`(`max_synth_per_ds` 경로)에도 있다.

---

## 2. 실제 피해 (정량)

annotations.json에 **ring 배치본과 `_positive_place` 폴백본을 구분할 필드가 없다.** [[aroma_adjacent_context_bg_selection]] §3-5(a)가 좌표 대조로 사후 검출해야 했던 이유다.

| ds | 폴백(위치없음) | 비율 | pool → cap | exp4v2 A−R |
|---|---|---|---|---|
| mvtec_leather | 0/400 | 0% | 400→64 | −0.254 (붕괴 오염) |
| aitex | 3/400 | 0.75% | 400→246 | +0.0146 |
| severstal | 31/2000 | 1.55% | ?→2534 | **+0.0225** |
| kolektor | 8/400 | 2.0% | 400→36 | −0.346 (붕괴 오염) |
| **mtd** | **75/400** | **18.75%** | **400→272** | **−0.0110 (0/3 seed, p=0.071)** |

mtd: pool 400 → cap 272 균일추출 → **기대 폴백 잔존 ≈ 51장**. aroma arm의 **약 19%가 ring 배치가 아니다.**

**그리고 제거할 여유가 있었다** — 폴백 75장을 전부 빼도 325 ≥ 272. **개수 손실 0으로 100% ring arm 구성이 가능했다.** 균일 표본이라 그러지 않았을 뿐이다.

mtd는 5종 중 유일하게 random에 진 데이터셋이고, 폴백률이 나머지의 12~25배다. [[aroma_adjacent_context_bg_selection]] §4 위험 4·4b(mtd 격자 절단 24.5% · build/runtime 격자 불일치)와도 정합한다.

### ⚠ 이 수정이 전체 결론을 뒤집지 않는다

폴백률 0%인 leather(붕괴 제외 시 aroma 0.843 vs random 0.854)와 0.75%인 aitex(A−R +0.0146, 1/3 seed)에서도 AROMA가 random을 이기지 못했다. **본 수정은 mtd의 패배를 설명하지만 전반적 null을 설명하지 못한다.** 결론 문서에 이 단서를 함께 남긴다.

---

## 3. 영향도 분석

### 이 기능이 변경하는 상태

- `clean_bg_selected*.json` 레코드에 필드 추가 (`site_score`, `topk_site_scores`)
- `annotations.json` 레코드에 필드 추가 (`position_source`, `site_score`, `bg_score`, `bg_k_fit`)
- exp4v2 synth cap의 **표본 추출 순서** — 균일 무작위 → ring 우선 + 부족분 폴백

### 그 상태를 전제로 동작하는 기존 로직

| 소비처 | 전제 | 영향 |
|---|---|---|
| `generate_defects.py:3228-3235` | `topk_pool`/`topk_positions`가 index 정렬 | `topk_site_scores`도 **동일 index 정렬** 필수 |
| `generate_defects.py:3247` | `_cbg_pairs[rep_idx % len(_cbg_pairs)]` | pair → triple 확장 시 인덱싱 유지 |
| exp4v2 `_load_synth_annotations` | annotation dict 키 집합 | 필드 **추가만** (제거·개명 없음) → 무영향 |
| [[aroma_adjacent_context_bg_selection]] §3-5(a) 좌표대조 검증 | 좌표 일치로 폴백 판정 | `position_source`가 이를 **대체**. 교차검증 재료로 1회 대조할 것 |

### 0개 / N개 경계

- **ring 샘플 0개**(전량 폴백): cap 전량을 폴백에서 채우고 **경고 로그**. 무음 금지
- **ring 샘플 < cap**: ring 전량 + 부족분 폴백. 소비 내역 로그
- **ring 샘플 ≥ cap**: ring에서만 추출 (현행 폴백 오염 제거)
- **`position_source` 결측**(구 annotations.json): 전량 균일 무작위로 폴백 = **기존 동작 그대로**

---

## 4. 수정 내용

### 4-1. `scripts/aroma/clean_bg_selection.py` — ring 점수 반환·저장 (개선 ②)

**(a) `_best_ring_site` 반환 확장** (`:594-614`)

```python
# 현행:  return best_xy
# 변경:  return best_xy, (best if best_xy is not None else None)
```
- 조기 반환 `if not tgt or bw <= 0 ...: return None` → `return None, None`
- 후보 0건(`best_xy is None`)일 때 점수도 `None`

**(b) 호출부 `_pos_for` / `_positions_for` 갱신** (`:903-928`)

- `_pos_for`가 `(xy, score)` 튜플 반환하도록 통일. ring 외 경로(`geometry_prior` / `off`)는 `(xy, None)`
- `_positions_for`는 `(positions, site_scores)` 두 리스트를 **동일 index 정렬**로 산출
- ring cap(`site_pool_cap`) 밖 항목은 기존대로 `None`, 점수도 `None`

**(c) 출력 레코드 필드 추가** (`:933-951`)

```python
topk_site_scores=site_scores,                     # topk_positions 와 index 정렬
site_score=(site_scores[0] if site_scores else None),
```
- else 분기(후보 없음)는 `topk_site_scores=[]`, `site_score=None`
- `derived`에 분포 요약 추가: `site_score_mean` · `site_score_p10` · `site_score_p90` (임계 판단 근거용)

**(d) legacy 불변 확인**
- `--site_selection off` 경로는 `_pos_for`가 `(None, None)` → `topk_site_scores`가 전부 `None`. **좌표·선택 결과 byte-identical**

### 4-2. `scripts/aroma/generate_defects.py` — 출처·점수 전파 (개선 ①②)

**(a) `_cbg_pairs`를 triple로 확장** (`:3227-3235`)

```python
_poss   = _cbg_entry.get("topk_positions") or []
_scores = _cbg_entry.get("topk_site_scores") or []       # 신규 (결측 → 전부 None)
for _i, _id in enumerate(_ids):
    _rp = _resolve_bg(_id)
    if _rp:
        _cbg_pairs.append((_rp,
                           _poss[_i]   if _i < len(_poss)   else None,
                           _scores[_i] if _i < len(_scores) else None))
```
- `_cbg_pool = [p for p, _, _ in _cbg_pairs]` 로 동반 수정
- `:3247` 언패킹 `_pick_path, _pick_pos, _pick_score = _cbg_pairs[rep_idx % len(_cbg_pairs)]`

**(b) 배치 모드 라벨 확보 — 구현으로 확정**

`_forced_xy`가 설정됐다는 사실만으로는 ring인지 `--geometry_prior`인지 구분되지 않는다.

~~`derived.site_selection`을 로드 시 읽는다~~ → **불가 확인 (2026-08-07)**: `clean_bg_selected*.json` 최상위는 **list** (레코드 배열). `derived`는 summary md로만 나가 JSON 소비자가 접근 불가. **대안 채택**: `clean_bg_selection.py`가 레코드마다 `site_mode` 필드(`"ring"` | `"geometry_prior"` | `"off"`)를 직접 기록하고, generate가 `(_cbg_entry or {}).get("site_mode")`로 읽는다. 구 JSON(필드 없음) + `_forced_xy` 설정 시에는 모드 미상이므로 **`"precomputed"`** 라벨 (non-fallback으로 취급, 도메인에 5번째 값 추가).

**(c) annotation 필드 추가** (`:3321-3341` 본 경로, `:3176-3194` dry-run 경로 **양쪽**)

```python
"position_source": (_cbg_site_mode if roi_entry.get("_forced_xy") else "fallback"),
"site_score":      _pick_score if roi_entry.get("_forced_xy") else None,
"bg_score":        (_cbg_entry or {}).get("score"),
"bg_k_fit":        (_cbg_entry or {}).get("k_fit"),
```

- `position_source` 값 도메인: `"ring"` | `"geometry_prior"` | `"off"` | `"fallback"`
- `_forced_xy`는 `:3224`에서 매 (roi, rep)마다 `pop`되고 `:3249`에서만 재설정되므로, annotation 시점 조회로 정확히 판정된다
- dry-run 경로는 좌표가 없으므로 `position_source`만 기록, `site_score=None`

**(d) 요약 로그 확장** (`:3358-3368`)

기존 `clean_bg resolve: used=%d fallback=%d mismatch=%d` 에 이어:

```
position_source: ring=%d geometry_prior=%d fallback=%d / %d
```

### 4-3. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — ring 우선 소비 (개선 ①)

**(a) 헬퍼 신규**

```python
def _ring_first_sample(anns, cap, seed):
    """position_source == 'fallback' 이 아닌 샘플을 우선 소비. 필드 결측 시
    전량 균일 무작위(= 기존 동작)."""
```
- 필드가 하나도 없으면 → `rng.sample(anns, cap)` 그대로 (**하위호환**)
- ring 풀 ≥ cap → ring 풀에서만 균일 추출
- ring 풀 < cap → ring 전량 + 폴백 풀에서 부족분 균일 추출
- 두 하위 추출 모두 **동일 `random.Random(seed)`** 사용 — seed 재현성 유지
- 반환과 함께 `(n_ring, n_fallback)` 소비 내역 리턴 → 호출부가 로그

**(b) 적용 지점 2곳**
- `:2454-2470` `synth_ratio` 경로
- `:2362-2368` `max_synth_per_ds` 경로

**(c) arm 대칭성**

`random` arm의 annotation에는 `position_source`가 없거나 전부 동일하다 → 헬퍼가 자동으로 기존 균일추출로 폴백한다. **개수(`n_synth_train`)는 cap으로 동일하게 유지되므로 공정성 훼손 없음.** aroma arm만 "AROMA가 배치하지 않은 샘플"을 배제하는 것이며, 이는 선별(③)이 아니라 **arm 정의의 정합성 회복**이다.

**(d) 로그 (무음 금지)**

```
[SynthRatio] mtd/aroma: 1.00 x 272 real_train -> cap=272  (400 -> 272 synth)
[Provenance] mtd/aroma: ring=272 fallback=0  (pool ring=325 fallback=75)
```

ring 풀이 cap에 미달해 폴백을 섞은 경우 `logger.warning`.

---

## 5. 수정 대상 파일

- `scripts/aroma/clean_bg_selection.py` — `_best_ring_site` · `_pos_for` · `_positions_for` · 출력 레코드 · `derived`
- `scripts/aroma/generate_defects.py` — `_cbg_pairs` triple · `_cbg_site_mode` · annotation 2곳 · 요약 로그
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `_ring_first_sample` 신규 · 적용 2곳 · 로그
- `.claude/.dev_note/aroma_adjacent_context_bg_selection.md` — §4 위험표에 **항목 15**(유실 사슬) 추가, §5 항목 6 갱신
- 실행 가이드 (`step3_5_*` / `step5_execute.md` / `exp4v2_execute.md`) — 재실행 절차 반영

---

## 6. 재실행 범위 (중요)

| 개선 | 필요한 재실행 | 이유 |
|---|---|---|
| ② ring score | **step3.5 → step5 전량** | `clean_bg_selected*.json` 재생성 후 annotations 재생성 |
| ① position_source | **step5 전량** (step3.5 산출물 재사용 가능) | annotations 재생성만 필요 |
| exp4v2 반영 | 재학습 | 학습셋 구성이 바뀜 |

②가 step3.5를 요구하므로 **실질적으로 step3.5 → step5 → exp4v2 전 구간 재실행**이다.

⚠️ **좌표 불변 확인 필수** — ②는 `_best_ring_site`의 **반환 형태만** 바꾸고 선택 로직(`if score > best`)은 건드리지 않는다. 재실행 후 `position` 좌표가 기존 `clean_bg_selected_ring.json`과 **완전 일치**해야 한다. 불일치 시 리팩터링이 선택 결과를 바꾼 것이므로 즉시 중단.

> **TODO(사용자 결정)** — exp4v2 재학습을 이번에 수행할지. "모든 성능실험 완료" 상태이므로, 코드·기록 수정만 하고 재학습은 보류하는 선택지도 있다. 보류 시 mtd 19% 오염은 알려진 한계로 문서에 남는다.

---

## 7. 테스트

**CLAUDE.md 정책상 새 테스트 코드 작성·pytest 실행 금지. 검증은 실측으로 수행한다.**

### 7-1. 로컬 실측 — **완료 (2026-08-07, `D:\project\aroma_dataset` kolektor)**

- [x] **좌표 불변** — `--k_fit --site_selection ring` 재실행, 200 레코드 전량에서 `position` · `topk_positions` · `assigned_normal_id` · `score` · `k_fit` · `hist_intersection` 기존 `_ring` 산출물과 **완전 일치 (diff 0)**. 실행 로그도 재현: `3116 positions, fallback 84 (2.6%)`, `w_k=0.2469` = §3-3 기존 실측과 일치
- [x] `topk_site_scores` 길이 == `topk_positions` 길이, `site_score` == `topk_site_scores[0]`, position↔score None 정렬 위반 **0건**. `site_score` 비-null 196/200 — 나머지 4 ROI = §3-5(a) kolektor 위치없음 8/400(rep×2)과 정확 일치
- [x] `site_score` 분포: min 0.109 / max 0.253. `derived`에 `site_score_mean/p10/p90` 기록
- [x] **legacy off 재현** — `--site_selection off` 실행, baseline `clean_bg_selected.json`이 가진 **전체 키에서 0/200 diff**. (baseline은 `k_fit` 필드 도입 이전 산출물이라 신규 필드 비교 제외)
- [x] **폴백률 5종 대조** — rep 단위 재계산: leather 0/400·kolektor 8/400 **정확 일치**, aitex 4(기대 3)·severstal 44(기대 31)·mtd 82(기대 75) 근사 일치. 오차 원인 = 실제 run의 `_resolve_bg` 실패에 따른 modulo 매핑 변동 + mtd skip 59 분모 차이. 정확값은 재합성 시 `position_source` 로그가 직접 산출
- [x] **`_ring_first_sample` 실데이터 검증** (실제 `synth_aroma_tobe/mtd/annotations.json` 400건):
  - T1 legacy(필드 없음) → `random.Random(seed).sample(anns, cap)`과 **원소 단위 동일**
  - T2 mtd 실측 주입(ring 325/fallback 75, cap 272) → **fallback 0건 선택**, 개수 272 유지
  - T3 ring 100 < cap 272 → ring 전량 100 + fallback 172
  - T4 동일 seed 재현 확인
  - T5 전량 fallback(random/casda arm 시나리오) → 균일추출, provenance 로그·경고 미발생

### 7-1b. Colab 재실행 시 확인 (레거시 체크리스트)

```python
!python $AROMA_SCRIPTS/aroma/clean_bg_selection.py \
    --profiling_dir $AROMA_OUT/profiling/severstal \
    --roi_dir       $AROMA_OUT/roi/severstal \
    --output_dir    $AROMA_OUT/roi/severstal \
    --k_fit --site_selection ring \
    --emit_random_arm --output_tag _ring2
```

### 7-2. Colab (step5 합성)

- [ ] `annotations.json`에 `position_source` 기록. 값 분포가 §2 폴백률과 **일치** (mtd fallback ≈ 18.75%, severstal ≈ 1.55%)
- [ ] `position_source == "ring"` 인 항목의 `bbox`가 `clean_bg_selected` `position`과 일치 — [[aroma_adjacent_context_bg_selection]] §3-5(a) 좌표대조를 **1회 교차검증**으로 재수행 (새 필드가 기존 판정과 같은 답을 내는지)
- [ ] `position_source == "fallback"` 항목의 `site_score`가 `None`
- [ ] 요약 로그에 `position_source: ring=N geometry_prior=N fallback=N` 출력

### 7-3. exp4v2 (재학습 수행 시)

- [ ] `[Provenance]` 로그 출력. mtd에서 `ring=272 fallback=0`
- [ ] `n_synth_train`이 arm 간 동일 (공정성 parity 유지)
- [ ] 구 `annotations.json`(필드 없음)으로 실행 시 **기존과 동일한 표본** 선택 (하위호환)

---

## 8. 미확정 사항

| # | 항목 | 상태 |
|---|---|---|
| 1 | ~~`clean_bg_selected*.json` 최상위 구조~~ | **해소 (2026-08-07)** — 최상위 = list, `derived`는 JSON에 없음 → 레코드별 `site_mode` 기록으로 대체 (§4-2b) |
| 2 | ~~exp4v2 재학습 수행 여부~~ | **수행됨 (2026-08-11)** — 결과·결론은 §10 |
| 3 | `position_source == "fallback"` 을 **완전 배제**할지, 부족 시 채울지 | 현 설계 = 부족 시 채움(개수 parity 우선). 전량 배제가 필요하면 cap을 ring 풀 크기로 낮춰야 하는데 arm 간 `n_synth_train` 불일치 발생 → **개수 parity 우선으로 확정** |
| 4 | ③ 점수 기반 within-class 선별 | **범위 밖.** ②로 `site_score`가 확보된 뒤 별도 노트에서 판단 |

## 9. 구현이 사양(§4)에서 벗어난 지점 — 전부 의도적

| # | 편차 | 이유 |
|---|---|---|
| 1 | `position_source` 도메인에 **`"precomputed"` 추가** (5번째 값) | 구 clean_bg JSON은 `site_mode`가 없어 ring/geometry_prior 구분 불가. 추측 라벨 대신 정직한 "모드 미상·non-fallback". exp4v2는 `!= "fallback"`만 보므로 소비 측 영향 없음 |
| 2 | dry-run 레코드의 `position_source = None` (문자열 아님) | dry-run은 배치 자체가 없다. exp4v2 로드가 dry-run을 걸러내므로 샘플링 무영향 |
| 3 | `_ring_first_sample`에 **전량-fallback 균일추출 단락** 추가 | `generate_random.py`/`generate_casda.py`도 `generate_defects.run()` 위임이라 재합성 후 random/casda annotations에도 필드가 생긴다(전량 fallback). 단락 없으면 arm마다 경고 오발. 동작은 균일추출로 동일 |
| 4 | `bg_score`/`bg_k_fit`에 `_cbg_consumed` 가드 | `_cbg_entry`는 있으나 `_resolve_bg` 전멸로 legacy 배경을 쓴 경우, 미사용 할당의 점수를 기록하면 오라벨 → None |
| 5 | exp4v2 `_load_synth_annotations` 화이트리스트에 `position_source` 추가 | 이 함수가 키 화이트리스트로 레코드를 재구성한다 — 추가하지 않으면 필드가 샘플러 도달 전에 유실 (사양에 없던 필수 지점) |

## 10. Colab 재실행·재실험 결과 (2026-08-10 ~ 08-11) — 종결

### 10-1. 파이프라인 발효 확인 (step3.5 → step5)

- **step3.5 5종 재실행** — `site_score` 개수 == `ROI with position` 개수 5/5 정확 일치(986/986 · 200/200 · 159/159 · 198/198 · 196/196). kolektor는 08-03 산출물과 **완전 동일**(positions 3116, fallback 2.6%). 나머지 4종은 positions +1~+99 소폭 드리프트 — 원인은 입력 측(08-03 이후 phase0 `image_w/image_h`·void-floor 재산출), 총 자리수는 보존(16000/2600/3200/3200). mtd fallback 18.9% → **17.7%**
- **step5 재합성 (n_per_roi 3)** — `position_source` 로그가 step3.5 site-level 비율과 정확 일치. aitex `fallback=5 ring=595 / 600` = 0.83% ≈ 0.8%. severstal pool `ring=2954 fallback=46` = 1.53% ≈ 1.5%
- **exp4v2 ring-우선 소비** — `[Provenance] severstal/aroma: ring=2534 fallback=0` 등 5종 전부 **fallback=0** 달성. §2의 오염(구판 mtd ≈19%)이 완전 제거된 학습셋

### 10-2. exp4v2 재실험 (20260811) — baseline·random은 resume 재사용(20260807과 bit 동일), aroma만 재학습

| ds | A−R mean | wins | p (paired t, df=2) | 20260807 대비 |
|---|---|---|---|---|
| **severstal** | **+0.0254** | **3/3** | **0.029** | +0.0225 (2/3, p=0.269) → 유의 도달 |
| mtd | −0.0130 | 0/3 | 0.268 | −0.0110 (0/3) → 열위 지속 |
| aitex | −0.0320 | 0/3 | 0.160 | +0.0146 (1/3) → 반전 |
| leather | −0.497 | 1/3 | — | aroma 붕괴 2 seed (0.140/0.055) — 판정 불가 |
| kolektor | −0.104 | 2/3 | — | aroma 붕괴 1 seed (0.035) — 판정 불가 |

### 10-3. 결론 3건

1. **severstal — 전 run 통틀어 첫 유의 결과.** A−R 3/3·p=0.029, aroma std 최소(0.0130). alloc은 run 간 거의 동일(645/550/721/618 vs 636/545/732/621)이라 **run 간 개선분은 배분 변화가 아니다**(오염 제거 + 재학습). 단 ①배분 confound 자체(random alloc `{374,84,1827,249}` 대 aroma 균형)는 미해소 ②15개 검정 중 유일한 p<0.05라 다중비교 생존 못 함 — "3/3 일관 + 단일 데이터셋 유의"가 서술 상한.
2. **mtd — 오염 가설 기각 ★.** fallback=0으로 학습해도 0/3 열위(−0.0130). §2의 "mtd 패배 = 폴백 19% 오염" 가설은 **반증됐다**. 오염은 실재했으나 패배 원인이 아님. 잔여 용의자 = [[aroma_adjacent_context_bg_selection]] §4 위험 4(격자 절단 24.5%)·4b(build/runtime 격자 불일치→중립 0.5)·ring 배치 자체의 mtd 비유효(P1↔다운스트림 역상관과 정합).
3. **aitex — 노이즈 재확인.** +0.0146 → −0.0320 반전. aitex 폴백은 0.8%뿐이라 오염 제거로 설명 불가 — 합성 재생성(n_per_roi 2→3, 포지션 드리프트)+재학습 노이즈. run 간 swing ≈ std. **aitex A−R은 노이즈 안**.

사용 3종 평균 A−R = −0.0065 (전 run +0.0087). **전반 null 유지, ring 배치의 다운스트림 기여는 severstal 특이적.**

### 10-4. 논문 서술 확정

배치 개선(`ring_sgm`+`k_fit`)은 **산출물 품질 주장**으로 위치: P1 개선 5/5(JS −2.9~−16.9%) · void 침범 제거(20.4%→1.2%) · provenance 실측(fallback=0 학습셋). 다운스트림은 "severstal 3/3·p=0.029, 5종 일반화 미확인"이 상한. mtd 체인(오염 발견→제거→재실험→가설 기각)은 threats-to-validity 절의 성실성 자산으로 활용.
