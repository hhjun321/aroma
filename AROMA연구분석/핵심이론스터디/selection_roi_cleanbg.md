# Selection ③ — 그 배경의 어느 자리에 붙일까 (step3.5 2단계 · `ring_sgm` + 자리 quality 필터)

> 스터디 3부작의 3편. ① 결함 = `selection_defect.md`, ② 배경 이미지 = `selection_clean_bg.md`.
> ②가 고른 배경 pool 각각에 대해 **paste 좌표를 오프라인 확정**한다. step5는 그 좌표를 `forced_xy`로 소비만 한다.

## 0. 선택 순서 (2026-08-11 확정판)

```
① void 게이트 (타일 admissibility)          — 격자 자체를 정의
② admissible 자리 열거
③ 자리 quality 분위 필터 (qf run, 기본 OFF)  — 표면이 나쁜 자리 하위 P15 배제
④ 생존 자리 중 ring score argmax            — 문맥이 맞는 자리 확정
폴백: 자리 0개 → position=None → step5 런타임 배치 (position_source="fallback")
```

## 1. 격자와 admissibility — ① (`_tile_grid` · `_best_ring_site:596`)

- 배경을 64px **절단 격자**(W//64 × H//64)로 본다 — `_context_worker`와 동일 정책(F1).
  우/하단 나머지(최대 63px)는 **unobserved** — mtd 면적 24.5%가 여기서 버려진다 (위험 4).
- `_patch_void`(분산·엣지 P15 미만)인 타일은 격자에서 제외.
- 자리 s = bw×bh footprint가 놓일 타일 오프셋. **footprint 전 타일이 관측·비-void**여야 admissible:
  `valid(s) ⟺ ∀t∈F(s): observed ∧ ¬void`.
- footprint 크기는 **effective(fit-rescale 후)** 크기 기준 (`_effective_wh:458`) — 생성이 실제로 붙일 크기와 일치.

⚠️ **왜 오프라인이 void를 직접 배제하나**: step5의 `forced_xy` 분기(`generate_defects.py:1415-1421`)가
`_positive_place` **앞에서 단락**해 런타임 τ·void 게이트를 전부 우회한다. 게다가 런타임
`_is_clean_background`는 cv2 4.13 fail-open으로 no-op (`aroma_cleanbg_gate_cv2_dtype_failopen.md`).
→ **오프라인 admissibility가 유일하게 동작하는 void 게이트**다.

## 2. ring score — ④의 목적함수

```
tgt[k]  = L1norm(matrix_symmetric[k])          — 군집 k의 목표 문맥 분포 (②의 k_fit과 공유)
h_s(c)  = ring R(s)(footprint 사각형의 8이웃, void 제외)의 셀 정규화 히스토그램
score(k,s) = ∩(h_s, tgt[k])                    — 히스토그램 교집합
s* = argmax_{valid s} score(k,s)
```

### 왜 ring인가 (footprint가 아니라)

footprint는 결함이 **덮어써 사라지는** 픽셀이다. 합성 후 남아 결함과 접하고 경계 연속성을 결정하는 건 둘레(ring)다.

### 왜 분포 매칭인가 (평균이 아니라) ★ 문서의 심장

평균 argmax는 "높은 점수 셀이 많은 자리"를 고른다. 행 값은 양쪽 풀에서 흔한 셀에서 높으므로
그 목적함수는 **데이터셋에서 가장 평범한 표면으로 수렴**한다. 분포 매칭은 구성을 비교해 이 편향이 상쇄된다.

**기여 분리 실측** (dev_note §3-1(b)) — ring으로 옮기되 평균을 유지한 대조 arm(`ring_sgm_mean`):
- footprint → ring (평균 유지): **악화 4/5** (severstal −0.270 → −0.502)
- 평균 → 분포 매칭: **개선 5/5**
→ 개선의 원천은 위치가 아니라 **score의 정의**다. "ring이 좋다"로 단순화하면 틀린다.

## 3. 자리 quality 필터 — ③ (2026-08-11 신설, `--site_quality_filter`)

- **단위 = 자리(crop-영역)** — CASDA 원 의미 복원. 타일 단위 배제는 severstal 폴백 +20.2p 폭발로 기각
  (세로 관통 footprint가 타일 1개 배제에 전멸하는 비선형).
- quality = CASDA 4-성분(blur 0.3 + contrast 0.3 + brightness 0.2 + noise 0.2), **numpy 전용 구현**(`_np_quality`).
- **분위 플로어 P15** (데이터셋 자기 자리-분포 기준). 절대 0.7 폐기 — leather 자리 max 0.67로 100% 포화 실측.
- 고유 기여 = **밝기 축**: void 게이트(분산·엣지)가 원리적으로 못 보는 밝은 침전물류 (leather 실측:
  배제 타일의 100%가 밝기 극단). 실측 폴백 영향 ≤+0.2p (aitex만 +10.5p — 한계 59%의 1/5, ring 풀 ≥ cap 유지).
- 픽셀이 필요해 `--image_dir` 필수 — step3.5의 "픽셀 불요" 계약의 유일한 예외.

## 4. 기록·소비 사슬 (provenance — 2026-08-07/11)

```
clean_bg_selected.json:  position, topk_positions
                         topk_site_scores / site_score      (ring 매칭 점수)
                         topk_site_quality / site_quality   (자리 품질, qf run)
                         site_mode ("ring"/"geometry_prior"/"off")
       ↓ step5 (forced_xy 소비)
annotations.json:        position_source ("ring"|"fallback"), site_score, site_quality,
                         bg_score, bg_k_fit
       ↓ exp4v2
_ring_first_sample:      cap 추출 시 fallback 표본 배제 (총량 parity 유지)
```

이 사슬이 생기기 전에는 점수가 단계마다 폐기됐고, mtd aroma arm의 ~19%가 폴백 표본으로 오염돼 있었다
(dev_note `aroma_synth_provenance_and_scores.md` §2 — 수정·재실험 완료, §10).

## 5. 실측 근거 요약

| 검증 | 결과 |
|---|---|
| P1 분포 정합 (bg30×seed3) | `ring_sgm`이 `ctx_prior`(구방식)를 **5/5** 개선 (§3-1a) |
| 실합성 좌표 대조 | 불일치 **0/5종** — 오프라인 좌표가 파이프라인 끝까지 보존 (§3-5a) |
| void 침범 | severstal 20.4%→**1.2%**, kolektor 22.2%→**0.0%** (§3-5b) |
| 육안 (severstal) | void 밴드 위반 0, 형태 반영, `ctx_prior` 대비 동일 자리 1/30 — 침전물 회피 (§3-4) |
| exp4v2 (20260811, fallback=0) | severstal A−R +0.0254 **3/3 p=0.029** (유일 유의) · mtd 0/3 열위 지속 · aitex 노이즈 |

### ★ 최대 경고 — P1↔다운스트림 역상관

ring의 P1 개선폭 순위(mtd 최대→severstal 최소)가 exp4v2 A−R 순위(severstal 최대→mtd 최소)와
**완전 역전** (Spearman −1, n=3). "배치 품질 지표가 좋다 ≠ mAP가 오른다"가 이 프로젝트의 실측 교훈이다.
자리 점수를 성능 대리로 읽지 말 것.

## 6. 기각된 대안 (재시도 금지 목록)

| 안 | 기각 근거 |
|---|---|
| 인스턴스 질의 링 매칭 (R5) | 성능은 최강(P1 5/5)이나 `k`를 버려 논문 (k,c) 기여 구조 이탈 — **구조 이유 미채택** |
| footprint 평균 유지 + ring (ring_sgm_mean) | 악화 4/5 (§2 기여 분리) |
| 타일 단위 quality 배제 | severstal +20.2p 폭발 |
| 절대 임계 0.7 | leather 100% 포화 |
| **score 상위 선별 (synth_final/aroma_sel)** | 전형성 함정(P1 역상관) + 클래스 skew 실측(mtd crack 1.8배·aitex 2.7배) + 다양성 붕괴 + severstal 여지 1.16× — 2026-08-11 사용자 결정 기각. **하위 배제(P15 필터)와 상위 선별은 다르다** |

## 7. 미결

- mtd 열위의 잔여 원인 — 위험 4(격자 절단 24.5%)·4b(build/runtime 격자 불일치 → 런타임 조회 중립 0.5). 오염 가설은 기각됨
- radius 2 P1 미측정 (위험 2), 배경×자리 상호작용 미측정 (위험 8)
- qf(필터 ON) 판의 다운스트림 효과 — exp4v2_qf 재실험 진행 중 (기대: ON ≈ OFF)

## 관련

- dev_note: `aroma_adjacent_context_bg_selection.md`(본체), `aroma_synth_provenance_and_scores.md`, `aroma_site_quality_filter.md`, `aroma_cleanbg_gate_cv2_dtype_failopen.md`
- 논문: §3.2.4 (q_k, valid(s), h_s, score 수식 + Figure 3.2.4-1/2), §3.2.6 (void/quality 게이트)
- 실행: `colab_execute_new/step3_5_execute.md` (qf 셀·세대 체크), `step5_execute.md` §qf run

---

## QnA (스터디 기록)

### Q1. 자리 s와 footprint F(s)의 차이는?

**s = 좌표(점), F(s) = 그 좌표가 결정하는 면적(타일 집합).**

```
s    = 후보 위치 — 타일 오프셋 (si, sj), 좌상단 앵커
F(s) = s에 놓았을 때 덮이는 bw×bh 타일 사각형 (s의 파생물)
R(s) = F(s)의 8이웃 테두리 (역시 s의 파생물)
```

구체 예 (severstal 25×4 격자, 세로 관통 결함 bw=1, bh=4): 후보 s는 25개.
s=(9,0) 선택 → F(s)={(9,0)..(9,3)} → position=(576,0) — annotation `bbox [576,0,…]`과 일치.

| 단계 | 보는 대상 |
|---|---|
| 열거 | s (오프셋 전수 순회) |
| admissibility ① | F(s) — 자격 심사 (하드) |
| quality 필터 ③ | s 앵커의 픽셀 crop-영역 — 표면 심사 (하드) |
| ring score ④ | R(s) — 순위 (소프트) |
| 출력·소비 | s 하나 (픽셀 환산 → step5 forced_xy) |

심사는 F(s)·R(s)에서 하되, 저장·소비되는 것은 s 하나.

### Q2. s / footprint / ring 을 그림으로

**그림 1 — 기본 기하 (bw=2, bh=2), s=(2,1)**

```
    0   1   2   3   4   5
  ┌───┬───┬───┬───┬───┬───┐
0 │   │ R │ R │ R │ R │   │   R = ring R(s) — 합성 후 살아남는 둘레 → ring score가 읽는 곳
  ├───┼───┼───┼───┼───┼───┤
1 │   │ R │ F │ F │ R │   │   F = footprint F(s) — 결함이 덮어쓰는 타일
  ├───┼───┼───┼───┼───┼───┤       → admissibility(①)·quality(③)가 심사하는 곳
2 │   │ R │ F │ F │ R │   │
  ├───┼───┼───┼───┼───┼───┤   s = F의 좌상단 앵커 (2,1) — 픽셀 (128, 64)
3 │   │ R │ R │ R │ R │   │
  └───┴───┴───┴───┴───┴───┘
```

**그림 2 — severstal 실전형 (세로 관통 bw=1, bh=4). ▓=void**

```
     …  7   8   9  10  11 …
  ┌───┬───┬───┬───┬───┬───┐
0 │   │ R │ F │ R │   │▓▓▓│  ← 상단 경계라 위쪽 ring 없음
1 │   │ R │ F │ R │   │▓▓▓│  F가 세로 전체 관통 → ring은 좌·우 기둥만
2 │   │ R │ F │ R │   │▓▓▓│  (부록 R6: severstal 유효 슬롯 평균 2.49의 이유)
3 │   │ R │ F │ R │   │▓▓▓│  ← 하단 경계라 아래쪽 ring 없음
  └───┴───┴───┴───┴───┴───┘  ▓ void 기둥에 F가 겹치는 s는 ① 탈락
```

**그림 3 — 선택 과정**: 후보 s 전수 순회 → F(s) 유효성(①) → crop quality(③) →
∩(R(s), tgt[k]) 계산(④) → argmax s* 하나만 `position=(si·64, sj·64)`로 저장.

**한 줄**: s는 핀 꽂는 점, F는 핀이 덮는 스티커 면적, R은 스티커 둘레 —
심사는 면적과 둘레로, 저장은 핀 하나.

### Q3. F/R 타일에는 무슨 값이 있나? background_type인가?

**아니다 — 타일 값 = context cell key** (텍스처 5특징의 tertile 조합 문자열).

```
CSV(연속값): local_variance, edge_density, texture_entropy,
             frequency_energy, orientation_consistency  (+patch_xy)
   → 각 특징을 데이터셋 자기 P33/P66로 3단 이산화
   → grid[(i,j)] = "2_1_1_1_1"  (cell key — 타일이 갖는 유일한 값)
```

`background_type`(smooth/directional/…)은 **데이터셋당 상수 1개**로 step3 quality_proxy 전용 —
타일 단위 값이 아니며 F/R 어디에도 없다. 완전히 다른 좌표계.

**F와 R의 사용 방식 차이가 핵심**:

| 영역 | cell key 읽나? | 사용 |
|---|---|---|
| F(s) | **안 읽음** | grid 존재 여부만 (관측·비-void) — 어차피 결함이 덮어 지움. qf 필터는 셀이 아닌 F의 픽셀을 봄 |
| R(s) | **읽음** | key들을 세어 히스토그램 h_s 형성 → ∩(h_s, tgt[k]) |

예: R 타일 6개 key {A,A,A,B,B,C} → h_s={A:.5, B:.33, C:.17}.
void 타일은 grid에 없어 F/R 어디에도 등장 불가.

**한 줄**: 타일 값 = 텍스처 5특징 tertile 문자열. **F는 존재만 묻고, R은 구성을 센다.**

### Q4. tgt[k]는 어떻게 만들어지나?

**4단계 사슬** (phase0 3단계 + step3.5 1단계, `_target_by_cluster:525`):

```
1. 세기 (phase0):
   P_clean(c)  = 전체 good 배경 패치의 셀 분포
   P_def(k,c)  = 군집 k 결함 이미지들의 배경 패치 분포
                 — 패치가 이미지의 군집 라벨 상속, 결함 겹침 패치 제외 (= 결함 "둘레" 분포)
2. 스무딩 ε=10⁻³ + 기하평균:  raw = √(P_def·P_clean)
   — "군집이 사는 문맥" AND "clean에 실존" (한쪽 0이면 0)
3. 행 max 정규화 → matrix_symmetric 저장 (최적 셀=1, 확률분포 아님 — 히트맵용)
4. L1 재정규화 → tgt[k] (합=1)  ← step3.5
   — 스칼라 나눗셈이라 상대 형태 불변 (논문 수식 무수정 근거)
   — 행에 없는 셀은 tgt에도 없음 → 교집합 기여 0 (런타임 .get(cell, 0.5) 중립 편향의 자연 해소)
```

미니 예제: raw {A:.02, B:.01, C:.005} → 저장 {1, .5, .25} → tgt {.571, .286, .143}.
ring h_s={A:.5, B:.5} → score = min(.5,.571)+min(.5,.286)+0 = 0.786 —
**B의 초과분(0.5>0.286)은 기여하지 않는다 = "흔한 셀 몰빵" 차단 기전.**

같은 tgt[k]를 ②(k_fit — 배경 전체 히스토그램과)와 ③(ring — 자리 둘레와) 공유.

### Q5. k_fit(1단계)과 R(2단계)의 연결 상세

**같은 tgt[k]에 두 번 조준 — 전역(공급 확인) → 국소(위치 특정).**

```
k_fit = ∩(h_g, tgt[k])   h_g = 이미지 전체 비-void 타일 히스토그램 (수백~수천 타일)
ring  = ∩(h_s, tgt[k])   h_s = 자리 둘레 6~20타일 히스토그램
```

- **ring 혼자 안 되는 이유**: argmax는 랭킹이라 절대 실패하지 않는다 — 나쁜 배경에서도 "그중 최선"을
  내놓는다. **2단계는 1단계가 공급한 것 이상을 만들 수 없다** (k_fit이 천장, ring이 그 아래 위치 특정).
- **k_fit 혼자 안 되는 이유**: h_g는 전체 집계라 "어디에"를 모른다 — 목표 혼합이 흩어진 경우와
  한 구역에 모인 경우를 구분 못 함.
- **분산 차이**: h_g 안정 / h_s 거침(타일 1개가 5~17%p) — 1단계는 미세 랭킹, 2단계는 구성 매칭 수준.

미묘한 지점 2개:
1. **P_clean 정당성이 단계마다 다르다** (§2-2): 1단계 정당(가용성 질문 유효), 2단계는 어색하나 **유지** —
   R4 실측(P_clean 제거 재정의)이 개선 실패 + 수식 무수정이 논문 구조 보존.
2. **greedy, joint 아님** (위험 8): 1단계 확정 후 2단계 argmax — 버려진 이미지 속 완벽한 자리는 못 찾는다.
   P1 검증도 배경 무작위 고정으로 쟀으므로 **k_fit 선정 배경 위 ring 성능(상호작용)은 미측정**.

예: tgt={A:.57,B:.29,C:.14}. g1(A20%+B10%+C5%, k_fit=.35)이 pool에 오르고,
2단계가 g1 안에서 A·B가 실제로 모인 구역의 자리를 특정 — 흩어져 있으면 최고 ring도 낮다(1단계는 몰랐던 정보).
