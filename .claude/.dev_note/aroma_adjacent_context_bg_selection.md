# 배경 선택 · 자리 선택 개선 — `ring_sgm` + `k_fit`

## (성격: 채택 확정 · 구현 완료 · **실제 산출물 검증 완료** (2026-08-04). 다운스트림은 범위 밖)

논문의 `(k, c)` 구조와 수식 `ctx_prior(k,c) ∝ √(P_def(k,c)·P_clean(c))`을 **그대로 유지**한 채, 두 지점을 바꾼다.

```
1단계  배경 이미지 선택 : 기존 3 cue 에 형태 군집 축(k_fit) 을 더해 4 cue
2단계  자리 선택        : footprint 평균 argmax  →  링의 셀 분포 매칭 argmax
```

논문 갱신은 별건 — `aroma_paper_gaps_placement.md`.

---

## 1. 결정

### 1-1. 채택안

| | 현행 | 채택 |
|---|---|---|
| `c`를 읽는 위치 | 자리 **내부**(footprint) — 결함이 덮어써 사라질 픽셀 | 자리 **둘레**(ring) — 합성 후 남아 결함과 접하는 픽셀 |
| score 계산 | 셀별 점수의 **평균** | 셀 **분포 매칭** (히스토그램 교집합) |
| 수식 `√(P_def·P_clean)` | 사용 | **그대로 사용** |
| `P_def` 집계 범위 | 이미지 전역 배경 | **그대로 전역** |
| `k` 조건화 | 사용 | **그대로 사용** |
| 배경 이미지 cue | src / class / size | + **k_fit** (형태 군집 축) |

### 1-2. 기각한 것

- **인접 문맥으로 배경 이미지 선택** — 전역 질의가 MRR 6.7배 우월 (부록 R1)
- **인스턴스 질의 링 매칭** — P1 5/5 초과로 가장 강하지만 `k`를 버려 논문 기여 구조를 벗어난다. **의도적 미채택** (부록 R5)
- **`matrix_symmetric` 정의 개선**(`P_def` 국소화 · `P_clean` 제거) — 수식 유지가 5종 전부에서 우월 (부록 R4)

### 1-3. 평가 틀

clean 이미지에는 "결함이 있어야 할 자리"의 **정답이 없다.** 정확도 지표는 성립하지 않으므로 **분포 정합(P1)**을 주지표로 삼는다.

```
P_real(k,c)  = cluster k 결함의 실제 인접 문맥 셀 분포     (TRAIN 결함)
P_synth(k,c) = 알고리즘이 고른 자리의 인접 문맥 셀 분포     (TEST 결함)
D = mean_k JS( P_synth(k,·) ‖ P_real(k,·) )               낮을수록 타당
개선폭 = D(random) − D(method)
```

random이 기준선인 이유 — 결함 풀과 정상 풀은 애초에 문맥 분포가 다르다(도메인 갭). random의 divergence가 그 갭이고, 방법이 얼마나 좁히는지가 실질 성능이다.

**지표 사용 규칙**

- 정규화 lift 단독 사용 금지 — 희소성만으로도 오른다(부록 R3). 반드시 동수 랜덤 대조군과 함께
- 배경 표본 5장으로 판정 금지 — 부호가 뒤집혔다(§3-1 c)
- 자리 벤치(§3-2)는 보조 — 정답이 "다른 문제의 정답"이므로 파라미터를 맞추지 않는다

---

## 2. 설계 · 구현

### 2-1. 2단계 — `ring_sgm` 자리 선택

```
목표 분포 (cluster 당 1개, 프로파일링 산출물 재사용)
    tgt[k] = L1norm( matrix_symmetric[k] )
    ※ matrix_symmetric 은 행 max 정규화라 확률분포가 아니다. L1 재정규화는
      단조 rescale 이라 행의 상대 형태가 바뀌지 않는다.
    ※ 목표는 지지집합 S_k 밖에서 0 → 미관측 셀은 교집합 기여 0.
      런타임 _positive_place 의 .get(cell, 0.5) 중립 처리와 다르며,
      neutral 0.5 편향이 자연 해소된다.

후보 자리 (정상 이미지)
    bw×bh 창을 1타일 stride 로 슬라이딩   bw,bh = 결함 bbox 의 타일 크기
                                          (fit-rescale 후 EFFECTIVE 크기 기준)
    링 = 창 사각형의 8이웃 타일 (void 제외)
    footprint 에 void/결측 타일이 하나라도 있으면 자리 자체를 버린다  ← 2-3

점수
    score(site) = Σ_c min( hist_c(ring(site))[c], tgt[k][c] )
선택 = argmax
```

**왜 ring인가** — footprint는 결함이 덮어써 사라지는 영역이다. 합성 후 남아 결함과 접하고 경계 연속성을 결정하는 것은 ring이다.

**왜 분포 매칭인가** — 평균 argmax는 "높은 점수 셀이 많은 자리"를 고른다. 행은 양쪽 풀 모두에서 흔한 셀에서 높으므로 이 목적함수는 **데이터셋에서 가장 평범한 표면**으로 수렴한다. 실제 결함은 특정 문맥에 편중돼 있어 선택 분포가 멀어진다. 분포 매칭은 구성을 비교하므로 편향이 상쇄된다. **§3-1 (b)가 이 기여를 분리 측정한다.**

### 2-2. 1단계 — `k_fit` 배경 이미지 cue

```
k_fit(g) = hist∩( hist_c(정상 이미지 g 의 비-void 타일),  tgt[k] )
comb = w_src·src_fit + w_cls·class_fit + w_size·size_fit + w_k·k_fit
w = mean(best − median)  — 기존 lift 자동 산출이 배분, 하드코딩 상수 없음
```

*"이 배경이 cluster k 의 문맥 프로파일을 얼마나 지원하는가"*.

`class_fit`(도메인 라벨 축)과 축이 다르다 — severstal class 4 vs cluster 5, mvtec_leather class 5 vs cluster **3**. 같은 class 안에서 형태가 갈리거나 그 반대인 경우가 실재한다.

`tgt[k]`에 `P_clean`이 섞여 있는데 **1단계에서는 정당하다** — 어떤 배경을 고를지 정하는 시점에는 "clean 풀에 그런 표면이 실제로 존재하는가"가 의미 있다. 부당했던 곳은 자리 선택이다(이미 이미지가 정해져 타일이 손에 있으므로).

### 2-3. ⚠ `forced_xy`는 void·τ 게이트를 우회한다 (코드 확인)

`generate_defects.py:1415-1421`

```python
if forced_xy is not None:
    fx = max(0, min(int(forced_xy[0]), max(0, nrgb.shape[1] - crop_w)))
    fy = max(0, min(int(forced_xy[1]), max(0, nrgb.shape[0] - crop_h)))
    return nrgb, (fx, fy), True          # ← gate_ok = True 무조건
```

`_positive_place` **앞에서 단락**한다. 오프라인 위치는

- ✅ generate 측 **무변경**으로 적용된다
- ⚠️ **void-straddle 배제 · τ 게이트 · stage-2 re-pick**을 전부 건너뛴다

런타임 `_is_clean_background`는 cv2 dtype fail-open으로 무력하므로(`aroma_cleanbg_gate_cv2_dtype_failopen.md`), **오프라인 `_patch_void`가 유일하게 동작하는 void 게이트**가 된다. 따라서 자리 후보 열거에서 footprint void 배제를 반드시 수행한다.

### 2-4. CLI

`clean_bg_selection.py`. **기본 OFF로 legacy 정확 재현 확인** (kolektor `w_src=0.6613 lift_src=0.285` — 기존 summary와 일치). `generate_defects.py` **무변경**.

| 옵션 | 기본 | 내용 |
|---|---|---|
| `--k_fit` | off | 1단계 형태 군집 축 cue 추가 |
| `--site_selection {off,ring}` | off | 2단계 자리 오프라인 확정. `--geometry_prior`와 **배타 — 동시 지정 시 에러 종료** |
| `--site_pool_cap N` | 16 | ring 자리를 산출할 pool 상위 개수. generate가 `rep_idx % len(pool)`로 인덱싱하므로 앞쪽만 소비 |
| `--output_tag S` | `""` | 산출 파일명 접미사. `_ring` → `clean_bg_selected_ring.json` 등. 기존 산출물 보존용 |

신규 헬퍼: `_target_by_cluster` · `_tile_grid`(void 제외 격자, 이미지당 캐시) · `_ring_keys` · `_best_ring_site`.

### 2-5. 실행 (Colab)

```python
!python $AROMA_SCRIPTS/clean_bg_selection.py \
    --profiling_dir $AROMA_OUT/profiling/severstal \
    --roi_dir       $AROMA_OUT/roi/severstal \
    --output_dir    $AROMA_OUT/roi/severstal \
    --k_fit --site_selection ring \
    --emit_random_arm --output_tag _ring

!python $AROMA_SCRIPTS/generate_defects.py \
    ... \
    --clean_bg_json $AROMA_OUT/roi/severstal/clean_bg_selected_ring.json
```

로컬은 5종 전부 `roi/<ds>/`에 `_ring` 접미사로 기록 완료. 기존 `clean_bg_selected.json` 보존됨.

### 2-6. `defect_tiles.py` — 현재 채택 경로에서는 미사용

마스크를 1회 판독해 결함 타일 점유도를 산출하는 보조 스크립트. 산출물 `profiling/<ds>/defect_tiles.json`.

```jsonc
{ "meta": { "tile": 64, "defect_tile_cut": 0.5, "radii": [1,2],
            "grid_policy": "truncated (W//64 x H//64) — matches _context_worker (F1)",
            "key_format": "patch_xy string (x_y, top-left pixel)" },
  "core_empty_ids": [...],
  "tiles": { "<image_id>": { "grid": [gw,gh], "core": ["256_64",...],
                             "adjacent_r1": [...], "adjacent_r2": [...],
                             "csv_grid_mismatch": 0 } } }
```

- `core` = mask_frac > 0, `adjacent_rN` = mask_frac == 0 & N회 8-인접 & CSV 존재
- `core`를 질의에서 제외하는 이유: 프로파일링이 mask 비율 **>0.5**만 빼므로, core 타일의 특징은 최대 50% 결함 텍스처로 오염돼 있다(severstal 평균 15.5%). clean 풀 타일은 0%다

**주의 — `ring_sgm`·`k_fit`은 이 파일을 읽지 않는다.** 질의가 cluster 집계 `matrix_symmetric[k]`이고 `P_def`도 전역 집계이므로 결함 측 인접 정보가 필요 없다. 현재 이 산출물을 소비하는 것은 `--adjacent_radius`(기본 OFF, 부록 R1에서 기각) 뿐이다. 검증 자산·후속 실험용으로 보존한다.

---

## 3. 판정 근거

### 3-1. P1 분포 정합 — 주지표

배경 30장/모양 × 분할 시드 3 · radius 1 · void footprint 배제 · 5종 전 결함 · TRAIN/TEST 분리

| ds | random JS | `ctx_prior`(현행) | **`ring_sgm`**(채택) | `ring_sgm_mean` | `ring_hist`(인스턴스) | `ring_cont` |
|---|---|---|---|---|---|---|
| **severstal** | 0.130±.011 | −0.270±.011 | **−0.059±.008** | **−0.502±.013** | +0.068±.009 | +0.069±.010 |
| **mtd** | 0.435±.026 | −0.119±.011 | −0.017±.010 | −0.142±.009 | +0.025±.013 | +0.020±.024 |
| **aitex** | 0.501±.027 | −0.009±.046 | +0.032±.018 | −0.065±.013 | +0.253±.026 | +0.157±.026 |
| **kolektor** | 0.619±.018 | −0.053±.043 | **+0.062±.056** | −0.089±.037 | +0.040±.024 | +0.047±.004 |
| **leather** | 0.588±.042 | +0.196±.033 | +0.294±.037 | +0.229±.057 | +0.320±.045 | +0.301±.051 |

(random 대비 개선폭, + = 개선)

#### (a) `ring_sgm`이 `ctx_prior`를 5/5에서 이긴다

| ds | `ctx_prior` → `ring_sgm` | 개선 |
|---|---|---|
| severstal | −0.270 → −0.059 | **+0.211** |
| kolektor | −0.053 → +0.062 | +0.115 |
| mtd | −0.119 → −0.017 | +0.102 |
| leather | +0.196 → +0.294 | +0.098 |
| aitex | −0.009 → +0.032 | +0.041 |

**논문 수식 무수정.** severstal에서 개선폭이 가장 크다.

#### (b) 기여 분리 — 개선을 만드는 것은 분포 매칭이다 ★

`ring_sgm_mean`은 링으로 옮기되 평균 방식을 유지한 대조 arm이다.

| 변경 | 단독 효과 |
|---|---|
| footprint → 링 | **악화 4/5** (severstal −0.270 → **−0.502**, 두 배 나빠짐) |
| 평균 → 분포 매칭 | **개선 5/5** |

링은 footprint보다 타일이 적어 흔한-셀 편향이 **더** 세게 먹는다. ⇒ 논문 §3.2.4 수정 지점은 위치가 아니라 ***score*의 정의**다.

#### (c) 배경 표본 확대가 필요했다

| | bg 5 × seed 1 | bg 30 × seed 3 |
|---|---|---|
| aitex `ring_hist` | +0.022 | **+0.229** (10배) |
| mtd `ctx_prior` | −0.011 | **−0.110** (10배 악화) |
| aitex `ctx_prior` | +0.011 | **−0.032** (**부호 반전**) |

### 3-2. 자리 선택 벤치 (보조)

같은 이미지·같은 후보·같은 정답에서 **질의만 교체**. 체커보드 2중 차단(질의는 링 parity 0, 채점은 자리 링 parity 1).

| ds | 결함 | 자리 | chance | `ctx_prior` | `adj_own_cont` | `adj_oth_cont` |
|---|---|---|---|---|---|---|
| severstal | 3423 | 29.4 | 0.520 | **0.617** | **0.168** (top1 41.4%) | 0.461 |
| leather | 92 | 193.5 | 0.531 | **0.583** | **0.077** (16.3%) | 0.470 |
| kolektor | 51 | 63.0 | 0.535 | 0.562 | 0.263 (5.9%) | 0.443 |
| mtd | 248 | 16.7 | 0.569 | 0.563 | 0.454 (11.8%) | 0.580 |
| aitex | 290 | 10.2 | 0.590 | **0.638** | 0.525 (17.9%) | 0.583 |

(평균 백분위, 낮을수록 좋음)

- own ≫ other **5/5** — 인접 문맥이 자리를 특정한다
- **`ctx_prior`가 5종 중 4종에서 chance보다 나쁘다** — P1과 독립적으로 같은 결론

### 3-3. 실행 결과 — `k_fit`이 5/5에서 가중치를 벌었다

| ds | w_src | w_class | w_size | **w_k** | lift_k |
|---|---|---|---|---|---|
| kolektor | 0.498 | 0.255 | 0.0 | **0.247** | 0.141 |
| severstal | 0.449 | 0.332 | 0.0 | **0.219** | 0.200 |
| mtd | 0.481 | 0.243 | 0.069 | **0.206** | 0.183 |
| leather | 0.407 | 0.394 | 0.0 | **0.199** | 0.197 |
| aitex | 0.576 | 0.258 | 0.0 | **0.166** | 0.266 |

`w_size`가 4/5에서 0.0으로 자동 소거되는 것과 대비된다 — lift 필터가 무용 신호를 실제로 걸러내고, `k_fit`은 5/5 통과했다.

`w_class`가 밀려난 폭(severstal 0.425→0.332, leather 0.492→0.394)이 `w_k`보다 작다 ⇒ **완전 중복이 아니라 새 정보를 더한다.**

**ring 자리 산출**

| ds | positions | fallback | 비율 |
|---|---|---|---|
| leather | 2,598 | 2 | 0.1% |
| aitex | 3,170 | 30 | 0.9% |
| severstal | 15,655 | 345 | 2.2% |
| kolektor | 3,116 | 84 | 2.6% |
| **mtd** | 2,596 | **604** | **18.9%** |

severstal ROI 1,000건 중 **978건 위치 확정**. fallback은 `position=None` → generate가 `_positive_place`로 자연 폴백.

### 3-4. 육안 검증 — severstal

합성 없이 자리만 그렸다. cluster별 10건 무작위, 위=결함 원본+실제 bbox(빨강) / 아래=선택된 clean bg+`ring_sgm` 자리(초록).

산출물: `.claude/.etc/new_clean_bg_selection/severstal_k{0..4}.png` + `severstal_report.txt`
생성 스크립트: scratchpad `viz_ring_sites.py`, 대조 시트 `viz_compare_sites.py`

**확인된 것**

| # | 관찰 |
|---|---|
| 1 | **형태가 반영된다** — k=0은 폭 14~47px 세로 박스, k=4는 가로로 넓은 박스 |
| 2 | **검은 void 밴드 위반 0건** — footprint void 배제 작동. 실제 결함(빨강)은 void 인접 사례가 여럿 |
| 3 | **강판 본체 내부에 놓인다** — 상/하단 경계 밴드를 밟지 않는다 |

**`ctx_prior` 대조 시트** — 동일 자리 **1/30 (3%)**. `ctx_prior`(파랑)가 백색 침전물·오염 영역에 자주 붙고 `ring_sgm`(초록)은 피한다. 침전물은 `P_clean`·`P_def` 양쪽에서 어느 정도 흔해 기하평균이 높기 때문이며, 분포 매칭은 국소 이상치 하나로 점수가 오르지 않는다. **§3-1 (b)의 기전이 육안으로 같은 모습이다.**

### 3-5. 실제 합성 산출물 검증 — 파이프라인 끝까지 확인 (2026-08-04) ★

지금까지의 P1(§3-1)은 "알고리즘이 고를 자리"를 계산한 것이었다. 여기서는 **실제로 붙은 좌표**로 다시 잰다 — `--min-bg-quality` fail-open · 스테이징 · 폴백까지 전부 통과한 산출물이다.

**실행 조건**: step3.5 `--k_fit --site_selection ring` → step5 `copy_paste`, `n_per_roi 2`, 출력 `synth_aroma_tobe`. 5종 전량.

#### (a) 배치가 파이프라인 끝까지 살아남았다

`annotations.json`의 `bbox` 좌표를 `clean_bg_selected.json`의 `position`과 대조:

| ds | 좌표 일치 | 불일치 | 위치없음 | step3.5 폴백 예측 |
|---|---|---|---|---|
| mvtec_leather | 400 | **0** | 0 (0%) | 0.1% |
| aitex | 397 | **0** | 3 (0.75%) | 0.9% |
| severstal | 1969 | **0** | 31 (1.55%) | 2.2% |
| kolektor | 392 | **0** | 8 (2.0%) | 2.6% |
| mtd | 325 | **0** | 75 (18.75%) | 18.9% |

**불일치 0** — 오프라인 `_effective_wh` 계산이 generate 측 clamp와 정확히 맞물린다. `위치없음`이 예측 폴백률과 일치.

#### (b) 배치 규칙만 고립해 대조

동일 결함 · **동일 배경**에서 배치 규칙만 갈랐다. `old`는 같은 배경에서 footprint 평균 argmax를 오프라인 재산출한 것이므로, ring과의 차이는 **score 계산 방식 하나**다.

참조 분포 `P_real` = cluster k 결함의 실제 인접 문맥 셀 분포(`defect_tiles.json` `adjacent_r1`). ring이 최적화한 대상은 `matrix_symmetric`이므로 자기참조가 아니다.

**P1 (JS divergence, 낮을수록 실제 결함 문맥에 근접)**

| ds | **ring** | `old` (footprint 평균) | ring 개선 | `rand`(동일 배경 무작위) | `rand_arm`(실제 random arm) |
|---|---|---|---|---|---|
| severstal | 0.0917 | 0.0944 | −2.9% | 0.0626 | 0.1131 |
| mvtec_leather | 0.2093 | 0.2507 | **−16.5%** | 0.1895 | 0.4817 |
| mtd | 0.2397 | 0.2883 | **−16.9%** | 0.2702 | 0.3637 |
| aitex | 0.1457 | 0.1667 | −12.6% | 0.1592 | 0.4523 |
| kolektor | 0.4107 | 0.4297 | −4.4% | 0.3460 | 0.4211 |

**void 침범률** (footprint에 void/결측 타일)

| ds | **ring** | `old` | `rand` | `rand_arm` |
|---|---|---|---|---|
| severstal | **1.2%** | 20.4% | 7.5% | 23.3% |
| mvtec_leather | **0.0%** | 0.0% | 0.5% | 22.5% |
| mtd | **4.1%** | 8.5% | 10.3% | 35.8% |
| aitex | **0.8%** | 3.2% | 3.5% | 13.8% |
| kolektor | **0.0%** | 22.2% | 20.9% | 36.8% |

#### (c) 확인된 것 3건

| # | 주장 | 근거 |
|---|---|---|
| 1 | **ring이 구방식(footprint 평균)을 5/5에서 이긴다** | JS −2.9~−16.9%. 동일 배경 고립 대조이므로 score 계산 방식의 기여 |
| 2 | **오프라인 void 배제가 런타임 fail-open을 대체한다** | severstal 20.4% → **1.2%**, kolektor 22.2% → **0.0%**. 런타임 `_is_clean_background`가 무력한데도 ring 경로가 이를 메웠다 |
| 3 | **배경 선정(4 cue)이 무작위 배경보다 낫다** | `rand_arm`이 5/5 최악 (JS 0.113~0.482, void 13.8~36.8%). 프레임워크 수준 대조 |

`rand`(동일 배경 균등 무작위)는 배경 선정 기여를 제거한 참고 열이다 — 프레임워크 수준 대조는 `rand_arm`이 맡는다.

#### (d) 부수 관찰

- **배경 다양성이 낮다** — leather 16 vs `rand_arm` 202, kolektor 53 vs 249. pool top-1만 쓰는 구조(§4 위험 7)
- **mtd skip 59** — `cluster_id`/배경 해석 실패. 폴백 ROI(위치없음 75)와 겹칠 가능성

#### (e) 착수 중 고친 파이프라인 버그 2건

| # | 버그 | 조치 |
|---|---|---|
| 1 | **normal 스테이징 재실행 개명** — `_stage_inputs`가 `dst.exists()`면 `{stem}_{n}`으로 개명한다. 재실행에서는 이전 실행분과 자기 자신이 충돌해 **풀 전체가 개명** → 스테이징 basename이 `assigned_normal_id` stem과 어긋나 **clean_bg 해석이 0%로 붕괴**(1회차 `used=3000` → 2회차 `used=0`, 5종 동일). `mismatch` 가드는 `image_id`만 보므로 못 잡는다 | 동일 크기 dst 재사용 분기를 suffix 앞에 추가 (`generate_defects.py`) |
| 2 | **aitex `TEX_T=None` argparse 사망** — 선결 체크가 `USE_CN`일 때만 prescan 부재를 잡아, copy_paste에서 `--texture-dist-threshold None`이 그대로 전달됐다 | `TEX_T is not None` 가드 + copy_paste 경로 경고 (`step5_execute.md`) |

`placement-gate stats`의 `active`로는 ring 소비 여부를 알 수 없다는 것도 확인했다 — `texture_on or compat_on`(CLI 인자 on/off)일 뿐이다. 판정은 좌표 대조(a)로 한다.

### 3-6. `defect_tiles.py` 검증

| ds | 마스크 해결 | **csv/grid 불일치** | core 없음 | R1 타일 | R1 <4 |
|---|---|---|---|---|---|
| severstal | 3620/3620 | **0** | 0 | 15.1 | 0.1% |
| aitex | 352/352 | **0** | 0 | 5.7 | 10.8% |
| mtd | 388/388 | **0** | **56 (14.4%)** | 4.2 | 46.1% |
| kolektor | 52/52 | **0** | 0 | 13.6 | 0% |
| leather | 92/92 | **0** | 0 | 15.8 | 0% |

- `csv_grid_mismatch` = 0 (5종) — 마스크에서 재도출한 `{frac ≤ 0.5}` 집합이 `context_features.csv`와 **완전 일치**. F1 절단 격자 정책이 `_context_worker`와 동일함을 실증
- 불변식 전수 통과: `adjacent_rN` 반경 위반 0 · `core` 중복 0 · `r1 ⊆ r2` 위반 0
- 비용: severstal 3,620장 ~14초. Pillow만 사용 (cv2 dtype 지뢰 회피)

**마스크 해결은 class-aware여야 한다** — stem만으로 매칭하면 다른 결함의 마스크를 집는다. leather는 `000_mask.png`가 5개 클래스 전부에, severstal은 flat union과 class별 마스크가 둘 다 존재. 수정 전 `csv_grid_mismatch` severstal 162 / leather 50 → 수정 후 **둘 다 0**. 이 지표가 잘못된 마스크를 잡아냈다 — 산출물 신뢰의 1차 관문으로 계속 감시할 것.

---

## 4. 알려진 한계 · 위험

| # | 항목 | 상태 |
|---|---|---|
| **1** | **`ring_sgm`이 severstal −0.059 / mtd −0.017로 random 미달** — random 초과는 3/5. 논문 구조 유지의 대가로 **감수 결정**(2026-08-03). 인스턴스 질의는 5/5 초과하나 `k`를 버린다 | 감수 |
| **2** | **radius 2 P1 미측정** — R1만 쟀다. 채택안에서 링 반경 영향 미확인 | 감수 |
| ~~3~~ | ~~합성·다운스트림 미수행~~ → 합성은 **완료·검증**(§3-5). 다운스트림(exp4v2)은 미수행 — 이번 트랙은 **이론적 확인까지**로 범위를 확정했다(2026-08-04) | 범위 밖 |
| **4** | **64px 격자 절단** — `_context_worker`가 우측·하단 나머지(최대 63px)를 버린다. mtd 면적 **24.5%**(max 54.9%), kolektor 13.2%. 나머지 3종 0%. mtd 56/388은 결함이 100% 격자 밖(§3-5 core 없음). **F1 채택** — 절단 격자 기준으로 산출하고 폴백 감수. F2(far-edge 앵커 추가 + 재프로파일링)는 `compatibility_matrix` 변경 → τ 재캘리브레이션이 필요해 보류 | 감수 |
| **4b** | **build/runtime 격자 불일치** — `_context_worker`는 far edge를 버리고 `_normal_tile_cells._anchors`·`_tile_anchors`는 포함한다. 프로파일링이 학습하지 않은 타일을 런타임이 조회 → 전부 중립 0.5. mtd·kolektor 결과 해석 시 감안 | 미해결 |
| **5** | **`_patch_void`의 구조적 한계** — 텍스처 전용이라 밝기를 직접 못 본다. 저분산·저엣지로 대리하므로 **밝고 매끈한 이상**(백색 침전물)을 못 잡는다. §3-4 육안에서 실제로 오염 배경 위에 자리가 잡힌 사례. p15 분위 컷이라 void가 없는 데이터셋에서도 하위 15%를 자른다. **그대로 유지 결정**(2026-08-03) | 감수 |
| **6** | **clean 배경이 깨끗하지 않다** — severstal `train/good`은 "결함 라벨 없음"일 뿐 무결하지 않다. 위험 5와 결합해 오염 배경이 선택될 수 있다 | 데이터셋 특성 |
| **7** | **배경 재사용** — severstal distinct backgrounds 264/1000, 평균 3.8회. pool top-1만 쓰기 때문(`topk_pool`에 14+개 있음) | 개선 여지 |
| **8** | **배경 선택과 자리 선택의 상호작용 미측정** — P1은 배경을 무작위 30장으로 고정했다. 실제로는 `clean_bg_selection`이 고른 배경 위에서 자리를 정한다 | 미측정 |
| **9** | **`ring_hist` vs `ring_cont` 미결** — 인스턴스 질의 두 변형이 P1(hist 우세)과 자리 벤치(cont 압도)에서 반대 답. 채택안이 아니므로 실무 영향 없으나, 인스턴스 arm을 되살릴 경우 선결 | 보류 |
| **10** | **kolektor 표본 부족** — ROI 52, P1 test 26, mean pool 17. 개선폭 부호는 일관되나 신뢰구간이 넓다 | 감수 |

---

## 5. 다음 단계

1. ✅ `defect_tiles.py` 구현·검증·5종 산출
2. ✅ `--k_fit` · `--site_selection ring` 구현·5종 실행·legacy 재현 확인
3. ✅ 육안 검증 (자리만, 합성 전) — §3-4
4. ✅ **Colab 합성 실행** (step3.5 → step5 copy_paste, `n_per_roi 2`, 5종) — 실행 가이드 갱신 완료
5. ✅ **실제 산출물 검증** — §3-5. 좌표 일치·불일치 0, ring이 구방식을 5/5 우세, void 침범 20.4%→1.2%
6. **→ 검증 종료 (2026-08-04).** 다운스트림(exp4v2)은 이번 트랙 범위 밖 — 벤치마크가 아니라 이론적 확인이 목적이었다
7. 논문 반영은 별건 트랙 (`aroma_paper_gaps_placement.md` §4)

---

## 부록. 기각된 가설

탐색 과정에서 세운 뒤 실측으로 기각한 것들. **같은 가설을 다시 시도하지 않기 위해 남긴다.**

| # | 가설 | 기각 근거 | 핵심 수치 |
|---|---|---|---|
| **R1** | 배경 **이미지** 선택 질의를 결함 인접 타일로 국소화하면 개선된다 | self-retrieval(결함 이미지 자신의 배경을 corpus에 넣고 순위 측정, 체커보드 leak 차단, 5종 전 결함)에서 **전역 질의가 5/5 압도**. 인접은 동수 랜덤 질의 대비 4승 6패 | `global` MRR severstal 0.785 vs `adjacent_r1` 0.117 (**6.7배**), leather 0.993 vs 0.589 |
| **R2** | corpus(정상 이미지)도 창 단위로 좁히면 질의와 단위가 맞아 개선된다 | R1이 기각되며 함께 폐기. 근거였던 "이미지 전체 히스토그램이 국소 창을 대표하지 못한다"는 관찰 자체는 유효 | ∩(창, 이미지전체) leather 0.133 · mtd 0.150 · kolektor 0.222 · severstal 0.415 · aitex 0.819 |
| **R3** | 정규화 lift(= lift/ceiling) 상승이 국소화 효과다 | **동수 랜덤 질의 대조군**이 반증. 상승분의 75~100%가 희소성 아티팩트 | gap(인접−랜덤) aitex +0.000 · mtd +0.003 · **leather −0.007** · severstal +0.035 · kolektor +0.048 |
| **R4** | `matrix_symmetric` 정의를 고치면 자리 선택이 개선된다 — `P_def` 국소화(A1~A4) · `P_clean` 곱셈 제거(lift, B1~B4) | P1에서 둘 다 부족. `ctx_local`은 **참조분포 자체**를 행 정규화한 점수표인데도 개선 없음. 병목은 정의가 아니라 footprint 평균 argmax | `ctx_prior`/`ctx_local`/`ctx_local_lift` severstal −0.270/−0.150/−0.099, mtd −0.119/−0.109/**−0.186** |
| **R5** | 인스턴스 질의 링 매칭이 최선이다 | 성능은 실제로 최강(P1 5/5 random 초과). 그러나 `k`를 버려 논문 `(k,c)` 기여 구조를 벗어난다 → **성능이 아니라 구조를 이유로 미채택** | `ring_hist` severstal +0.068 · aitex +0.253 vs `ring_sgm` −0.059 · +0.032 |
| **R6** | 8-슬롯 방위별 테두리 서술자(LT/TC/RT/…)로 방향 정보를 살린다 | 유효 슬롯이 8이 아니라 평균 2.1~6.6. severstal은 결함이 이미지 높이를 관통해 좌/우 2개만 실재(무효 사유 96%가 경계 밖). 방향 정렬 이득도 미미 | 유효 슬롯 severstal 2.49 · mtd 2.07 · aitex 3.11 · kolektor 3.96 · leather 6.57. 정렬(L-L) vs 교차(L-R) 이득 **+0.1~3.4%** |
| **R7** | bbox 접촉 타일을 결함 인접 문맥으로 쓴다 | 그 타일의 **72~96%가 결함 픽셀을 포함**한다. 순수 배경 인접 집합과 Jaccard 0.03~0.14로 사실상 서로소 | Jaccard(bbox접촉, 순수인접) aitex 0.026 · mtd 0.058 · leather 0.079 · severstal 0.122 · kolektor 0.135 |
| **R8** | 밝기 평균을 문맥 특징에 추가한다 (초창기 방법의 신호) | **범위 밖.** AROMA는 텍스처 전용 연구 — 사용자 결정(2026-08-03) | — |

### 부록 주의 — 지표 함정

**낮은 `hist∩(질의, 전역)`을 곧바로 "국소성이 좋다"로 읽으면 안 된다.** R7의 bbox 접촉 타일은 오염 때문에 divergence가 낮게 나온다(severstal B 0.201 vs 순수 인접 C 0.401). 오염이 통제된 조건에서만 유효한 비교다.

---

## 6. 관련 문서

- `.claude/.dev_note/aroma_paper_gaps_placement.md` — **논문 갱신 트랙.** 반영 완료 6건 + 미반영 5건(§3.2.3 `background_type` 값 출처, `MATCHING_RULES` 하드코딩, §4 기전 귀속)
- `.claude/.dev_note/aroma_ctxprior_localization_and_normalization.md` — `matrix_symmetric` 정의 개선안. **부록 R4로 기각**
- `.claude/.dev_note/aroma_compat_gate_clean-grounded_redesign.md` — SGM 도입 경위, `roi_selection` 미전환 근거
- `.claude/.dev_note/aroma_cleanbg_gate_cv2_dtype_failopen.md` — 런타임 void 게이트 fail-open (위험 5·§2-3과 연결)
- `AROMA연구분석/aroma_core_compatibility_model_20260729.md` — `ctx_prior` 전체 모델, 정직성 목록 23~25
