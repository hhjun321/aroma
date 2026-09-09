# CASDA Stage 3 — 문맥 기반 결함 생성·합성 (논문 §3.2.3)

> CASDA 스터디의 1편. 대응 AROMA 노트: ① 결함 `AROMA/selection_defect.md`,
> ② 배경 이미지 `AROMA/selection_clean_bg.md`, ③ 자리 `AROMA/selection_roi_cleanbg.md`.
> **읽는 목적**: AROMA 논문이 "handcrafted compatibility matrix를 data-driven으로 재도출했다"고
> 주장하는 그 **원본 handcrafted matrix가 실제로 무엇이었는지**를 코드 수준에서 확정하는 것.
>
> 원문: `AROMA연구분석/Article/CASDA.txt:70-71` (§3.2.3, 단 4문장)
> 코드: `D:\project\CASDA\CASDA\` — `scripts/compose_casda_images.py`(1723줄), `src/preprocessing/background_library.py`,
> `src/analysis/{defect_characterization,roi_suitability,background_characterization}.py`
> 문서: `D:\project\CASDA\CASDA\CASDA\04-Pipeline-StageC.md` (Stage C = 논문 Stage 3~4)

## 0. 논문 4문장 → 코드 대응

| 논문 주장 (§3.2.3) | 코드 실체 |
|---|---|
| "정상 이미지에서 배경 패치 추출" | `find_clean_images:148` — train.csv에 없는 이미지 = clean. **패치 아니라 이미지 전장(1600×256)** |
| "결함타입×배경텍스처 공존 확률 matrix 구성" | `COMPATIBILITY_MATRIX` `compose_casda_images.py:89-106` — **손으로 적은 4×5 상수표**. 확률 아님, 정규화 없음 |
| "matrix 기반 배경 확률적 선택" | `BackgroundPool.get_compatible_background:562` — `rng.choices(names, weights=호환점수)` |
| "Poisson blending으로 경계 불연속 제거" | `PoissonBlender` `src/preprocessing/poisson_blender.py:39` — `cv2.seamlessClone`, dilation 8px, mask=hint R채널 thr 127 |

실행: Stage C Step 6, `--compositions-per-roi 5`. ablation은 동일 스크립트 `--no-blend`(직접 paste, `:811-905`).

## 1. matrix 도출 = 3단 수작업 체인 (통계 0회 참조)

```
단계 1: 행 인덱스   결함 4지표 → 손임계값 5개 → subtype 5종
단계 2: 열 인덱스   64px 격자 → 손임계값 6개 → bg type 5종
단계 3: 셀 값       "결함이 잘 보이는가" 가시성 논증 → {1.0, 0.8, 0.5, 0.2}
```

**단계 1** (`defect_characterization.py:213-229`)
```
HIGH_LINEARITY=0.85  HIGH_ASPECT_RATIO=5.0  LOW_ASPECT_RATIO=2.0
HIGH_SOLIDITY=0.9    LOW_SOLIDITY=0.7
linearity>0.85 ∧ aspect>5.0      → linear_scratch
solidity<0.7                     → irregular
aspect>5.0 ∧ linearity>0.6       → elongated
aspect<2.0 ∧ solidity>0.9        → compact_blob
else                             → general
```

**단계 2** (`compose_casda_images.py:193-224`)
```
var<200 ∧ edge<0.02       → smooth
x_ratio>0.65 ∧ edge>0.02  → vertical_stripe
y_ratio>0.65 ∧ edge>0.02  → horizontal_stripe
edge>0.15                 → complex_pattern
var>500 ∨ edge>0.05       → textured
else                      → smooth
```

**단계 3** (`background_library.py:50-80`) — 도출 근거는 **주석이 전부**
```python
'compact_blob': {
    'smooth': 1.0,            # Perfect for isolated blobs
    'vertical_stripe': 0.8,   # Good contrast
    'textured': 0.5,          # May blend in
    'complex_pattern': 0.2,   # Poor visibility
}
```

★ **의미론 오류**: 이 값의 의미는 "**보이면 좋다**"(detectability desideratum)인데,
논문은 "**coexistence probability**"(공존 확률)로 서술. 두 개념이 다르며 실측은 서로 **반대 방향**(§4).

## 2. 단위(granularity) 지도 — CASDA 내부 불일치

| | 행 (defect subtype) | 열 (background type) |
|---|---|---|
| **Stage A 집계** | region = connected component 1개 = bbox 1개 (`defect_characterization.py:174-180`) | **결함 centroid가 든 64px 격자셀 1개** (`roi_suitability.py:117-120` → `background_characterization.py:200-232`, `--grid_size 64`) |
| **Stage 3 적용** | 동일 (roi_meta의 subtype) | **이미지 전체 1라벨** — 1600×256의 중앙 256열만 crop→128×128 축소 (`compose_casda_images.py:172-181`) |

즉 통계는 **국소 64px** 단위로 잡히는데 적용은 **이미지 단위**로 조회된다. 좌우 1344열(84%)은 미검사.
`get_compatible_background(roi_x_center=...)` 인자가 존재하나 주석이 **"미사용, 향후 위치 기반 매칭 확장용"**(`:579`) —
Stage A는 위치 단위인데 Stage 3이 위치를 버린 미봉합 지점. **AROMA가 채운 축이 정확히 여기다.**

참고: CASDA도 64px 격자를 쓴다(AROMA와 동일 크기). 차이는 격자 사용 여부가 아니라 **적용 단계에서 격자를 유지했는가**.

## 3. Stage 3 배경 선택 순서 (`compose_all` Step 4~5)

### 준비 (1회, 메인 프로세스)
```
P1 clean pool  = train_images/*.jpg − train.csv의 ImageId, 정렬               (:148-157)
P2 상위 5000장만 분석  clean_names[:max_analyze]  ← 정렬순이라 파일명 앞쪽 편향  (:438)
P3 BG_CACHE(v2 JSON) 히트 확인 → 미스분만 분석                                 (:441-447)
P4 이미지당 중앙256열 crop → 128×128 → type 1개 + mean_brightness 1개          (:160-227)
P5 type_index[bg_type] = [filename,...] 역색인                                (:382)
P6 생성 결함 이미지 전량 평균밝기 사전계산 gen_brightness                       (Step 4.5, :1100-1138)
```

### 생성 이미지 1장당 (`rng = random.Random(seed)` 전 루프 공유, `:1025`)
```
class_id ← 파일명 파싱 (실패 skip)
roi_meta ← roi_lookup[sample_name] (없으면 skip) → roi_bbox, defect_subtype, background_type
hint 존재 확인 (없으면 skip)
target_brightness = gen_brightness[filename]
roi_x_center = (x1+x2)//2          ← 계산만 하고 전달 후 미사용
for comp_idx in range(5):          ← compositions_per_roi, 매 회 재추첨
    bg_name = get_compatible_background(defect_subtype, target_brightness, rng)
```
★ `roi_meta['background_type']`(Stage A가 알아낸 **원 결함의 배경 타입**)은 선택에 **전혀 안 쓰인다**.
메타데이터로만 기록(`:903`). 선택 입력은 `defect_subtype` + `target_brightness` 둘뿐.

### `get_compatible_background()` 내부 (`:562-673`)
```
1. compat_scores = COMPATIBILITY_MATRIX.get(defect_subtype, {})
   └ 빈 dict → [폴백 X] 전체 배경 밝기매칭 후 균등 랜덤 → return   (:648-655)
2. score 내림차순 정렬
3. candidates = []
   for bg_type, score in sorted:
       if score < 0.3: continue        ← complex_pattern(0.2) 전량 배제 (min_compatibility)
       candidates += [(name, score) for name in type_index[bg_type]]
4. candidates 비면 → [폴백 Y] 전체 밝기매칭 후 균등 랜덤 → return
5. _select_with_brightness:
   a. 밝기 ±30 필터
   b. 남은 수 < 5 → ±60 완화
   c. 그래도 < 5  → 밝기 필터 해제
   d. rng.choices(names, weights=호환점수, k=1)
```

**실효 선택 확률** = 호환점수 × **그 type의 파일 개수**.
```
P(특정 vertical_stripe 파일) = 1.0 / Σ_type (score_type × count_type)
```
→ 호환성은 "**어떤 type 그룹을 후보에 넣을까**"만 정하고, 그 안에서는 **밝기와 파일 개수가 실제 선택을 지배**.

## 4. 실측 근거 (`D:\project\CASDA\CASDA\morphological_features.csv`, 19,958 rows / 6,666 images)

### 4-1. subtype 분포 — matrix 키 불일치
| subtype | 개수 | 비율 | matrix 행 존재 |
|---|---|---|---|
| linear_scratch | 8,187 | 41.0% | ✅ |
| **general** | 6,610 | 33.1% | ❌ → 폴백 X |
| compact_blob | 4,756 | 23.8% | ✅ |
| **irregular** | 405 | 2.0% | ❌ → 폴백 X |
| scattered_defects | **0** | — | ✅ (죽은 행) |
| elongated_region | **0** | — | ✅ (죽은 행) |

→ **결함의 35.1%가 matrix를 우회**. matrix 4행 중 2행은 한 번도 조회되지 않는다.

### 4-2. 실측 P(bg | subtype) vs hand matrix
| subtype | smooth | vertical | horizontal | textured | **complex** |
|---|---|---|---|---|---|
| linear_scratch | 0.132 | 0.221 | 0.067 | 0.001 | **0.579** |
| general | 0.093 | 0.078 | 0.086 | 0.008 | **0.735** |
| compact_blob | 0.129 | 0.038 | 0.042 | 0.009 | **0.783** |
| irregular | 0.072 | 0.027 | 0.175 | 0.005 | **0.721** |

배경 전체 분포: complex_pattern 13,618 / vertical 2,514 / smooth 2,333 / horizontal 1,385 / **textured 108**

★ hand matrix가 **최하점 0.2를 준 complex_pattern이 실제 결함의 68%가 사는 곳**이며,
`min_compatibility=0.3`에 걸려 **후보에서 완전 배제**된다. 반대로 0.5를 받은 textured는 실측 0.5%(108장)뿐이라
`type_index`가 거의 비어 가중치가 무의미. → matrix가 실측 분포와 **역상관**.

### 4-3. matrix가 2벌, 값 충돌
| | `roi_suitability.MATCHING_RULES:24-60` | `background_library.COMPATIBILITY_MATRIX:50-80` |
|---|---|---|
| 키 집합 | linear_scratch/elongated/compact_blob/**irregular/general** (분류기 출력과 일치) | linear_scratch/compact_blob/**scattered_defects/elongated_region** |
| compact_blob × textured | **0.7** | **0.5** |
| compact_blob × complex | **0.6** | **0.2** |
| irregular × complex | **1.0 (최적)** | 행 없음 |
| 미지 키 폴백 | — | `background_library:185`는 `compact_blob`으로 대체 / `compose:648`은 균등 랜덤 — **경로별로 다름** |

Stage A의 suitability_score는 A표로, Stage 3의 배경 선택은 B표로 — 같은 파이프라인이 서로 다른 두 matrix를 쓴다.

## 5. AROMA가 겨냥한 지점 (대조표)

| 축 | CASDA | AROMA |
|---|---|---|
| 행 k | 규칙 subtype 5종, 임계값 5개 손설정 | GMM+BIC 클러스터, 개수·경계 데이터 결정. **잔여 버킷(general) 없음 → 폴백 없음** |
| 열 c | 규칙 bg type 5종, 임계값 6개 손설정 | 5특징 P33/P66 tertile → 최대 3⁵=243 cell key (`'2_2_1_2_0'`) |
| 셀 값 | 손으로 적은 {1.0, .8, .5, .2}, 가시성 논증 | `ctx_prior(k,c) ∝ √(P_def(k,c)·P_clean(c))`, ε=1e-3, row-max=1 |
| 대칭성 | 비대칭 (결함→배경 일방) | 대칭 (결함측·배경측 기하평균) |
| 배경 단위 | 이미지 1장 = 라벨 1개 → **이미지 선택** | 64px 타일 → **이미지 안 좌표 선택** |
| 위치 | `roi_x_center` 계산 후 폐기 | ring R(s) 히스토그램 매칭으로 s* 확정 |
| 데이터셋 이식 | 임계값 11개 + 셀 20개 재설계 필요 | 재설계 0 |

AROMA 초록의 "automatic, data-driven re-derivation of the compatibility matrix"가 정확히
**§1의 3단계 전부**를 대체한다는 뜻. "matrix 값만 자동화"가 아니라 **행·열 인덱스 정의까지** 자동화.

## 6. 미결 / 후속 스터디

- Stage 3 나머지 절반(Poisson blending 파라미터: dilation 8px, NORMAL vs MIXED_CLONE, mask thr 127) 미정리
- Tier-1 증강(`jitter_x`, `scale_factor`, `use_smooth_mask`) — 논문 §3.2.3에 **미기재**. AROMA와 비교 시 조건 동등성 확인 필요
- 밝기 매칭(±30→±60→해제) — 논문 미기재. AROMA `clean_bg_selection`의 히스토그램 매칭 cue와 역할 대응 관계 미확인
- `max_analyze=5000` 상한이 clean pool을 얼마나 자르는지 (Severstal clean 이미지 총량 미확인)
- Stage 4 품질 게이트(Q≥0.7, 4성분)와 AROMA `_np_quality` P15 분위 필터의 계보 관계 → `AROMA/selection_roi_cleanbg.md §3` 참조

## 관련

- 논문: `Article/CASDA.txt:70` (§3.2.3), `Article/AROMA.txt:134-138` (ctx_prior 수식), `AROMA.txt:38` (related work의 CASDA 서술)
- AROMA 노트: `AROMA/selection_defect.md`(k), `AROMA/selection_clean_bg.md`(배경 이미지), `AROMA/selection_roi_cleanbg.md`(자리)
- CASDA 문서: `D:\project\CASDA\CASDA\CASDA\04-Pipeline-StageC.md`, `06-Scripts-Reference.md`
- memory: `project_paper_324_simplification.md` (논문 §3.2.4 의도적 단순화 — 감사 시 결함 아님)

---

## QnA (스터디 기록 · 2026-09-02)

### Q1. AROMA의 k·c에 대응하는 CASDA 개념은?

AROMA는 k=결함, c=배경이고 k는 결함 bbox, c는 64px 조각 — 맞지만 **한 단계 추상화가 더 있다.
k·c는 영역이 아니라 이산 인덱스다.**

| | AROMA 원 단위 | AROMA 인덱스 | CASDA 대응 |
|---|---|---|---|
| 결함 | defect_bbox → 6 morph feature | **k = GMM cluster id** (BIC로 개수 자동) | subtype 5종 (규칙) |
| 배경 | 64px patch → 5 context feature | **c = tertile cell key** `'0_1_2_0_1'`, 최대 243종 | bg type 5종 (규칙) |

`distribution_profiling.py:77-84` — `MORPH_FEATURES` 6개, `CONTEXT_FEATURES` 5개, `GRID_SIZE=64`, `N_CONTEXT_BINS=3`.
`_context_cell_key:480`이 연속값을 문자열 키로 이산화.

**한 줄**: AROMA의 k·c는 bbox·타일 그 자체가 아니라 **그것을 데이터가 정한 경계로 이산화한 인덱스**다.

### Q2. CASDA matrix는 어떻게 도출되었나? (예시로)

→ §1 참조. 실제 행 하나로 추적:

`0002cc93b.jpg` class 1 — `linearity 0.8035, solidity 0.9509, extent 0.8658, aspect_ratio 2.2558`
```
1. 0.8035 > 0.85 ?  No  → linear_scratch 탈락
2. 0.9509 < 0.7  ?  No  → irregular 탈락
3. 2.2558 > 5.0  ?  No  → elongated 탈락
4. 2.2558 < 2.0  ?  No  (0.26 초과)  → compact_blob 탈락
5. → general
배경: edge_density > 0.15 → complex_pattern
조회: COMPATIBILITY_MATRIX.get('general', {}) → {}
→ 폴백: 밝기 ±30 매칭 후 균등 랜덤. matrix 영향 0.
```
**한 줄**: 도출 = 임계값·임계값·가시성 주석의 3단 수작업. 데이터 카운트는 한 번도 세지 않는다.

### Q3. type은 이미지 1개당 값이면, subtype은 bbox 당건인가?

**subtype은 맞다 — 정확히는 connected component 단위** (component당 bbox 1개이므로 실질 동일).
`defect_characterization.py:174-180`이 RLE 마스크 → skimage label → component별 loop.

실측: 6,666 images → 19,958 rows (이미지당 평균 3.0개, 최대 21개).
1행 = (image_id, class_id, region_id) = 1 component = 1 bbox.

**type은 "이미지 1개당"이 단계에 따라 달라진다** — Stage A는 결함 centroid의 64px 셀,
Stage 3은 이미지 중앙 crop 1라벨. → §2 단위 지도 참조. 이 불일치가 CASDA의 구조적 미봉합점.

### Q4. Stage 3 배경 선택 순서는?

→ §3 참조. 요약 흐름:
```
train.csv 없는 이미지 → 상위 5000 → 중앙crop 1라벨 + 밝기
                                       ↓ type_index
결함 subtype ─→ matrix 행 ─→ score≥0.3 type들 ─→ 해당 type의 파일 전부 후보
                 │ (키 없음 35%)                        ↓
                 └──────────────→ 균등랜덤     밝기±30 → ±60 → 해제
                                                       ↓
                                       rng.choices(weights=호환점수) × 5회
```
**한 줄**: 호환성은 후보 **집합만** 정하고, 최종 1장은 **밝기와 파일 개수**가 고른다. 위치는 계산 후 버려진다.
