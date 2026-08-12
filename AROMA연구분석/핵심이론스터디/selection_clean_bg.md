# Selection ② — 어느 배경 이미지에 붙일까 (step3.5 1단계 · `clean_bg_selection.py`)

> 스터디 3부작의 2편. ① 결함 선택 = `selection_defect.md`, ③ 자리 선택 = `selection_roi_cleanbg.md`.
> 체인 위치: step3 → **step3.5 1단계(본 문서) → 2단계(③)** → step5. CPU 전용.

## 0. 한 줄 정의

step3이 선발한 결함 ROI 각각에, **clean(good) 배경 이미지의 랭킹 pool(top ~16장)**을 배정한다.
픽셀을 다시 안 본다 — 전부 phase0 `context_features.csv`의 셀 히스토그램 연산이다
(예외: qf run의 자리 quality — ③에서 다룸).

## 1. 유효 배경 풀 — `valid_bg_pool` (`:298`)

배정 전에 good 이미지 자체를 거른다:

```
void floor:  타일 특징 분포의 P15 (var_floor, edge_floor) — 데이터 유도, no-hardcoding
void_frac_max = 0.5:  과반-void 이미지 제외 (부분 강판/검은 밴드 — "48% black 합성" 사고의 수정)
```

severstal 실측: 5,902 → 5,350장 (2026-08-12 Colab). 근거: dev_note `aroma_cleanbg_void-floor-fix.md`.

## 2. 4 cue — 후보 배경 g의 점수 재료 (`build_and_rank:714`)

전부 context-cell 공간의 히스토그램 교집합 `∩(a,b) = Σ_c min(a_c, b_c)`:

| cue | 정의 | 묻는 것 |
|---|---|---|
| `src_fit` | ∩(h_g, p_src(v)) — ROI 소스 이미지의 배경 분포 | "결함이 원래 살던 표면과 닮았나" (E1 계보) |
| `class_fit` | ∩(h_g, p_cls) — 클래스 집계 배경 분포 | "이 클래스 결함들의 표면과 닮았나" |
| `k_fit` | ∩(h_g, tgt[k]) — **형태 군집의 목표 분포** (2026-08-04 신설) | "군집 k의 문맥 프로파일을 지원하나" |
| `size_fit` | min(1, 0.95·W/w, 0.95·H/h) — fit-rescale 배율 | "왜곡 없이 들어가나" (1=무왜곡) |

- `tgt[k]` = `L1norm(matrix_symmetric[k])` — ③의 ring 매칭과 **같은 분포를 공유**한다 (§5-4 설계).
- `k_fit ≠ class_fit`: 축이 다르다 — severstal class 4 vs cluster 5, leather class 5 vs cluster 3.
  실측 `w_class` 감소폭 < `w_k` → 완전 중복 아닌 새 정보 (dev_note §3-3).

## 3. 가중 — lift 자동 유도 (하드코딩 없음)

```
λ_j = E_ROI[ max_g u_j(g) − median_g u_j(g) ]     (cue의 변별력 = best−median 갭)
w_j = λ_j / Σλ                                     → U(g) = Σ w_j·u_j(g)
```

평탄한 cue는 스스로 0으로 소거된다. 5종 실측 (dev_note `aroma_adjacent_context_bg_selection.md` §3-3):

| ds | w_src | w_class | w_size | w_k |
|---|---|---|---|---|
| kolektor | 0.498 | 0.255 | 0.0 | 0.247 |
| severstal | 0.449 | 0.332 | 0.0 | 0.219 |
| mtd | 0.481 | 0.243 | 0.069 | 0.206 |
| leather | 0.407 | 0.394 | 0.0 | 0.199 |
| aitex | 0.576 | 0.258 | 0.0 | 0.166 |

`w_size`가 4/5에서 자동 소거(배경 크기 균일)되는 것과 `w_k` 5/5 생존의 대비가 lift 필터의 작동 증거.

## 4. pool 확정과 소비 방식

- 후보 전체를 `U(g) + 결정적 jitter`로 정렬 → **top pool** (`--pool_k` 또는 데이터 유도 P95, 실측 mean ~16)
- 레코드에 `topk_pool`(id 배열) + top-1 `assigned_normal_id` 기록
- **step5 소비**: `rep_idx % len(pool)` — n_per_roi=3이면 pool의 앞 3장만 실제 사용
- `--emit_random_arm`: 동일 ROI 집합에 균등 무작위 배경 배정 (`clean_bg_random_arm.json`) —
  "배경 정체성만 다른" 대칭 대조군 (논문 §3.3.4의 계측기)

## 5. 실측 근거 (채택/기각)

- **전역 질의 채택** — 배경 이미지 선택을 결함 인접 타일로 국소화하는 안(R1)은 self-retrieval MRR에서
  전역이 5/5 압도 (severstal 0.785 vs 0.117, **6.7배**) → 기각. "이미지 선택은 전역, 자리 선택은 국소"가 결론.
- **배경 선정 자체의 기여** — 실합성 검증(§3-5c): `rand_arm`(무작위 배경)이 JS·void 침범 모두 5/5 최악.
- **Mann-Whitney** (논문 §4.1, Figure 4.1-3): AROMA 배정 배경의 텍스처 유사도가 4/5 데이터셋에서 유의 우위 (mtd만 n.s.).

## 6. 알려진 한계

| # | 내용 | 상태 |
|---|---|---|
| 배경 다양성 | top-1 위주 소비로 severstal distinct 264/1000(평균 3.8회 재사용), leather 16 vs rand_arm 202 | 위험 7, 개선 여지 |
| clean이 깨끗하지 않음 | severstal `train/good` = "라벨 없음"일 뿐. 오염 배경 선택 가능 | 위험 6, 데이터셋 특성 |
| 히스토그램 변별력의 도메인 조건부 | aitex 강신호, severstal/mtd는 랜덤 배경과 사실상 구분 약함 (E1) | step3_5 가이드 정직성 절 |
| 세대 계약 | `roi_selected.json`과 같은 세대에서만 유효 — image_id 가드 (2026-08-11 mismatch 85% 사고) | 가이드에 선결 체크 명문화 |

## 관련

- dev_note: `aroma_adjacent_context_bg_selection.md` §2-2·§3-3, `aroma_cleanbg_void-floor-fix.md`
- 논문: §3.2.4 (u_src/u_cls/u_mor/u_siz, U(g), λ_j 유도), §3.3.4, Figure 3.2.4-3
- 다음 편: `selection_roi_cleanbg.md` (같은 스크립트의 2단계 — 자리)

---

## QnA (스터디 기록)

### Q1. src_fit은 "defect_type 별 속해있는 배경과 유사한가"인가?

**✗ 단위가 틀렸다 — 클래스가 아니라 결함 개체 1개 단위다.**
src_fit = ∩(후보 배경, **이 ROI가 잘려나온 바로 그 원본 이미지**의 배경 분포). 같은 클래스라도 결함마다
소스 이미지가 다르므로 기준 분포가 ROI마다 다르다. "defect_type별 배경"은 class_fit의 정의다.

### Q2. class_fit은 "class label 별 전체 이미지들의 통계치와 유사한가"인가?

**≈ 방향은 맞다. 정밀화 2개:**
- "전체 이미지"가 아니라 **그 클래스 결함이 있는 이미지들의, 결함-제외 배경 패치들** (프로파일링이 결함 겹침 패치를 배제)
- "통계치" = **cell 히스토그램** (평균·분산류 요약값이 아님)

### Q3. k_fit 상세 설명

**"이 배경이, 이 결함의 형태 군집이 사는 문맥을 공급하는가."**

```
tgt[k] = L1norm(matrix_symmetric[k]),   matrix_symmetric[k][c] ∝ √(P_def(k,c)·P_clean(c))
k_fit  = ∩(h_g, tgt[k])
```

- 기하평균이라 tgt[k]가 높은 셀 = "군집 k 결함이 실제로 그 옆에서 관찰되는 문맥" **AND** "clean 풀에 실존하는 표면".
  한쪽이 0이면 목표가 못 된다.
- **class_fit과의 차이가 존재 이유**: k는 클래스 라벨이 아니라 GMM **형태 군집**. 분할 자체가 다르다
  (severstal class 4 vs cluster 5, leather class 5 vs cluster 3). 같은 클래스 안에서 형태가 갈리면 사는 문맥도
  갈린다 — class_fit이 못 보는 축. 실측 w_class 감소폭 < w_k → 중복 아닌 새 정보.
- ②의 k_fit과 ③의 ring 매칭이 **같은 tgt[k] 공유**: ② = "이 배경 어딘가에 그런 표면이 있나"(이미지 전체 히스토그램),
  ③ = "정확히 이 자리 둘레가 그런 표면인가"(ring 히스토그램). **전역→국소 2단 조준.**

### Q4. size_fit은 "bbox가 들어가는 사이즈인가"인가?

**절반 — 이진(yes/no)이 아니라 등급값이다.**
`size_fit = min(1, 0.95·W_g/w, 0.95·H_g/h)` — 1.0=무축소, 0.6=40% 축소 필요(생성의 fit-rescale이 실제 적용할 배율).
yes/no는 별도 필드 `size_ok`. 등급값이기에 배경 크기 균일 셋에서 전 후보 동일값 → lift 0 → **w_size 자동 소거**
(4/5 데이터셋 실측 0.0)와 맞물린다.

### 암기 구조

```
src_fit  : 개체 단위   — 이 결함이 살던 그 이미지의 표면과 닮았나
class_fit: 클래스 단위 — 이 클래스 결함들 주변 배경 집계와 닮았나
k_fit    : 형태군집 단위 — 군집이 사는·clean에 실존하는 문맥을 공급하나
size_fit : 기하        — 축소 없이 들어가는 정도 (등급값)
```

**개체 → 클래스 → 형태군집으로 집계 반경이 넓어지는 3개 + 기하 1개.**

### Q5. "이 수식으로 paste할 clean 이미지 후보들을 거른다"가 결론인가?

**거의 맞다 — 동사 교정: "거른다"(필터)가 아니라 "순위 매긴다"(랭킹).**

| 동작 | 어디서 | 방식 |
|---|---|---|
| 필터 (탈락) | §1 `valid_bg_pool` — U(g) **이전** | void floor + 과반-void 컷 (severstal 5,902→5,350) |
| 랭킹 (정렬) | §2~4 U(g) | 임계 없음 — 유효 후보 전체를 정렬해 **상위 pool ~16장 채택**, 탈락 개념 없음 |

정정판 결론: ① void 게이트가 못 쓸 배경을 거르고 → ② U(g)가 나머지를 정렬해 ROI별 상위 pool을 뽑고 →
③ step5가 `rep_idx % len(pool)`로 소비. 랭킹은 최악의 풀에서도 항상 상위 16을 채운다 —
위험 6("clean이 깨끗하지 않다")이 구조적으로 완전히 안 막히는 이유.
