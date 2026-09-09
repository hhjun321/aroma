# Selection ① — 어떤 결함을 쓸까 (step3 · `roi_selection.py`)

> 스터디 3부작의 1편. ② 배경 이미지 선택 = `selection_clean_bg.md`, ③ 자리 선택 = `selection_roi_cleanbg.md`.
> 체인 위치: phase0(profiling) → step1 → step2 → **step3(본 문서)** → step3.5 → step5 → exp*.

## 0. 한 줄 정의

프로파일링된 실제 결함 인스턴스 전체에서, **synth 합성의 소스가 될 결함 크롭 top_k개**를 점수·quota·다양성 규칙으로 선발한다. **배경은 여기서 등장하지 않는다** — 출력은 순수하게 "결함 목록"이다.

## 1. 입력 (phase0 산출물)

| 파일 | 제공 정보 |
|---|---|
| `morphology_features.csv` / `morphology_clusters.json` | 결함별 형태 특징(AR, solidity …) + GMM 군집 k, prior P(k) = n_k/N |
| `context_features.csv` | 결함 주변 배경의 context cell (`cell_key`, 예 `2_1_1_1_1` — 5특징 tertile) |
| `compatibility_matrix.json` (`matrix_symmetric`) | `ctx_prior(k, c) ∝ √(P_def(k,c)·P_clean(c))`, 행 max=1 정규화 |
| `deficit_analysis.json` | (k, c) 조합 희소도 — **realism 모드에서는 미사용** (아래 §3) |
| step2 prompts | 결함별 프롬프트 문자열 (annotation 전파용) |

## 2. 후보 생성 — `build_candidates` (`roi_selection.py:376`)

결함 인스턴스마다 후보 레코드 1개. 핵심 파생 4개:

1. **subtype** — percentile 임계 cascade (`_percentile_subtype:206`, 논문 Table 4/4b):
   `AR > P66 → linear_scratch` → `AR < P33 AND Sol > P66 → compact_blob` → `Sol < P33 → irregular` → `general`.
   임계는 데이터셋 자기 결함 모집단의 P33/P66 (kolektor solidity만 균질성 세이프가드로 고정값 폴백).
2. **quality_score** — `quality_proxy(:224)` = `SuitabilityEvaluator.matching_score(subtype, background_type)`.
   `background_type`은 **데이터셋 단위 도메인 라벨 1개**(severstal=directional, mtd=smooth …)다.
   특정 배경 이미지 선택이 아니다 — "이 결함 형태가 이 표면 도메인에 어울리는가"의 상수 조건. ★혼동 주의
3. **morph_prior** = P(k) — 군집 크기 비율.
4. **ctx_prior** = `matrix_symmetric[k][cell_key]` — 결함 자신의 소스 문맥 셀에서 읽은 호환성 **스칼라**.
   (step3.5의 자리 선택은 같은 행을 **분포 전체**로 소비한다 — 3부작 ③의 핵심 차이.)

## 3. 점수 — `score_roi` (`:331`)

```
legacy  (기본):  ROI_score = 0.4·morph_prior + 0.4·ctx_prior + 0.2·deficit
realism (정본):  ROI_score = 0.5·ctx_prior  + 0.3·morph_prior + 0.2·quality_score
```

- **정본 CLI는 `--score_mode realism`** (step3_execute.md §공통). deficit 항은 **경험적으로 반증되어 폐기**
  (severstal flat = H1 재조합 무정보, dev_note `aroma_step4_h1-recombination-no-info.md`) — 값은 provenance로만 JSON에 남는다.
- quality_score가 hard gate에서 **graded 항**으로 승격된 것이 realism 모드의 두 번째 특징.

### ★ 논문-코드 불일치 (스터디 중 발견, 미해결)

논문 §3.2.4: `ROI_score = 0.6·ctx_prior + 0.4·morph_prior` — 이는 코드의 **`compatibility` 전략**(별도 branch) 식이다.
정본 산출물은 **realism(0.5/0.3/0.2)**로 생성됐다. frozen table 산출 당시 모드 확인 + 논문 §3.2.4 수식 정정 필요 여부가 열려 있다.

## 4. 사전 필터 — `apply_quality_gate` (`:507`)

`--min_quality X` 지정 시 quality_score < X 후보 제거. 기본 0 = OFF.
켤 경우 **random arm도 동일 필터를 통과한 풀에서 샘플해야 공정** — 대칭 설계는 dev_note `aroma_exp4v2_quality-gate-fairness.md`.

## 5. 최종 선발 — `select_rois` (`:1470`)

정본 전략 `deficit_aware`의 실체 = **pair-aware 할당** (이름과 달리 realism 모드에서는 deficit 값을 안 쓴다. 순서 키는 moderated_score):

| 규칙 | 코드 | 목적 |
|---|---|---|
| (k, c) pair 단위 quota | `_pair_aware_allocation:643` | 희소 조합 커버리지 보장 (논문 §4.1 rare-pair coverage의 원천) |
| 클래스 대칭 floor + per-pair cap 5% | `_stratified_pair_aware:976` (`--class_floor --per_pair_cap_frac 0.05`, multi 3종) | 클래스 기아 방지 — **exp4v2에서 aroma arm의 균형 배분**(severstal c2 545 vs random 84)이 여기서 나온다 |
| `img_diversity_cap=1` | Fix4 | 같은 소스 크롭(image_path, bbox) 최대 1회 — 다양성 붕괴 confound 제거. 논문 §3.2.4 "each source defect crop is selected at most once" |
| 결정적 jitter | `_img_jitter:615` (blake2b) | 동점 붕괴 방지, rng 없이 재현 |

기타 전략: `compatibility`(0.6/0.4 rank), `top_k`, `weighted`, `random`(exp2 baseline).

### top_k — 데이터셋별 (`synth_pool_sizing.md` §2 정본)

**severstal 1000 / 나머지 4종 200.** severstal은 real_train=2534라 ratio 1.0 증강에 top_k 200×3=600으로는 부족.
후보 26.6만 중 1000 = 상위 0.4%라 품질 꼬리 유입 미미.

## 6. 출력 — `roi_selected.json`

레코드(roi_idx 0..top_k−1): `image_id`(class-prefix 고유키), `image_path`, `defect_bbox [x,y,w,h]`,
`cluster_id`, `class_key`, `cell_key`, `roi_score`, `deficit`, `prompt`, subtype 등.

**소비 계약**: step3.5가 **roi_idx 위치 조인 + image_id 가드**로 읽는다.
⚠️ `roi_selected.json`과 `clean_bg_selected.json`은 **같은 세대**여야 한다 — 다른 세대를 섞으면 staleness 가드가
mismatch로 legacy 폴백시킨다 (2026-08-11 실측: 로컬 구세대 json 업로드 → mismatch 85% → 합성 무효).

## 7. 알려진 논점

- deficit 폐기 경위와 "deficit_aware"라는 전략명이 남아 있는 것 (이름 ≠ 동작)
- §3의 논문-코드 가중 불일치
- quality_proxy의 `background_type`이 Table 3의 ROI-단위 분류와 이름이 겹쳐 오독 여지 (여기선 데이터셋 상수)

## 관련

- dev_note: `aroma_step4_h1-recombination-no-info.md`(deficit 반증), `aroma_subtype_percentile_thresholds.md`, `aroma_exp4v2_quality-gate-fairness.md`
- 논문: §3.2.3(subtype·Table 4), §3.2.4 첫 문단(ROI_score)
- 다음 편: `selection_clean_bg.md`
