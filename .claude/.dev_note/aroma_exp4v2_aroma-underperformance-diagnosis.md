# AROMA exp4v2 하위성능 진단 및 대응 계획 — 전략 노트

---

## (성격: 연구 진단·계획 노트 — 코드 패치 노트 아님)

이 문서는 구현 패치 노트가 아니다. exp4v2 Severstal 4조건(baseline/random/casda/aroma)에서 **AROMA가 random/casda보다 하위**인 현상을 코드 추적으로 진단하고, 단계적 대응 분기와 **공정성 가드**를 기록한다. 모든 코드 근거는 `file:line`으로 추적 가능한 것만 사용한다.

> 관련 노트: [[aroma_exp4v2_source-diversity-fairness]] (source-concentration fix), [[aroma_exp4v2_clean-background-gate]] (배경 게이트), [[aroma_exp4v2_roi-quality-gate]] (ROI 품질 게이트), [[aroma_severstal_flat_diagnosis_and_direction]] (평탄 결과 진단).
> 검증 가이드: `AROMA연구분석/colab_execute/severstal_exp4v2_synth_inspect.md`, `.../clean_background_gate_verify.md`.

---

## 1. 관찰 (육안 검사)

`severstal_exp4v2_synth_inspect.md` 결과: 증강 3조건 모두 baseline보다 낮고, **random의 c1 합성에 큰+오목(concave) bbox가 빈번**. AROMA c2는 별도 붕괴(이 노트의 1차 대상은 c1 random 관찰을 출발점으로 한 ROI 선택 분포 문제).

---

## 2. 진단 — 코드 추적 결론

### 2.1 합성 버그 아님 (paste 엔진은 충실)

- `generate_defects.copy_paste_synthesis`: `use_real_mask`일 때 소스 ROI의 `defect_bbox`로 크롭 + **실제 GT 마스크** 그대로 paste (`generate_defects.py:479-483`), 반환 bbox = `[x_paste, y_paste, crop_w, crop_h]` (`:545`).
- → 붙은 bbox 크기·오목 형태 = **선택된 소스 결함의 속성**. 합성이 확대·왜곡하지 않음. 라벨 정확.

### 2.2 random = un-gated full 중복 풀의 균등 샘플

- `roi_selection.py:1341`이 **게이트 이전** `candidates`(전체 중복 풀)를 `roi_candidates.json`으로 저장. 후보 풀은 source 1장당 (cluster, cell_key) bin 수만큼 중복(`:446-451`).
- `generate_random.py`는 `roi_candidates.json`을 읽어 `select_random`(`rng.choice`) **균등 추출** — 품질게이트·점수선택·`img_diversity_cap` **전부 미적용**.
- → random의 c1 선택 분포 ≈ 후보 풀 c1 분포 원형. 풀에 있는 큰/오목 결함을 그대로 통과시킴.

### 2.3 aroma = 풀을 "가공"해 큰/오목에서 멀어짐

- `roi_selected.json`은 `roi_score` top_k + `img_diversity_cap=1`(`:1140, :1280`)로 선택.
- **정밀 교정**: `morph_prior`는 **클러스터 단위 prior**(`roi_selection.py:322`)이지 개별 결함의 형태 전형성이 아니다. `score_roi`는 image-blind → 같은 (cluster,cell) bin 내 전 이미지 동일 score(상세: [[aroma_exp4v2_source-diversity-fairness]] `:69-74`). 따라서 aroma의 실제 분리는 roi_score가 아니라 **img_diversity_cap + pair allocation**에서 옴.
- → `synth_inspect.md:271`의 *"morph_prior(형태 전형성)가 대형 비정형을 간접 거른다"* 표현은 부정확. 위로 교정.

### 2.4 solidity 품질게이트는 이번 run에서 OFF

- `apply_quality_gate`의 `min_quality` 기본 `0.0` = OFF (`roi_selection.py:377, :1404`). 즉 solidity 기반 ROI 필터는 양쪽 다 작동 안 함. random↔aroma 차이는 **순수 선택(점수+cap) 차이**.

### 2.5 정량 확정은 미완 (Colab 실측 대상)

"c1 풀이 실제로 큰/오목 쪽으로 치우쳤는가"는 풀 데이터 의존 → `synth_inspect.md` **Cell 7**(조건별 area/solidity 히스토그램) / **Cell 8**(roi_metadata.csv)로 실측 확정 필요. 기대: random `area median↑ / solidity median↓`.

---

## 3. clean-bg 게이트의 위치 (현재 검증 단계)

- `--reject-clean-bg` / `--min-bg-quality` / `--bg-blur-threshold` (`clean_background_gate_verify.md`)는 **배경(normal 이미지) + paste 위치**만 거른다. ROI/합성 개수는 보존: all-reject 폴백(`generate_defects.py:875-881`), position 폴백 무드롭(`:521-535`), Cell 3 `n_generated` ON/OFF ratio≥0.8 검증.
- 세 조건 동일 적용(**대칭**) → `synth_ratio`/`per_class_cap` 무력화 안 됨.
- **함의**: 대칭 개입이라 *공유* 라벨노이즈(검은 배경 위 결함)는 줄이지만 **AROMA의 *상대* 열위는 못 풀 가능성이 큼**. 상대 격차 원인은 §2의 ROI 선택 분포 쪽.

### 3.1 검증 결과 — clean-bg 게이트는 severstal에서 무효 (Colab 실측 완료)

`clean_background_gate_verify.md` Cell 2~4B 실측(severstal, aroma/random/casda):

- **pool 게이트 R=0** (3조건 공통, log `kept 5902/5902, 0 rejected`). severstal `train/good`에 전역-검은 normal 없음 → no-op.
- **Cell 4: OFF==ON 검은배경 비율·개수 동일** (aroma 35/385, random 35/400). 게이트가 합성을 안 바꿈.
- **Cell 4B Part A** (실제 `_foreground_mask` import, N=600): non-None **92.2%**(=foreground 배치 경로 → position 게이트 우회), 그중 **어두운 전경(void 오검출) 26.8%**. 전경 밝기 median 118 > 배경 88(다수는 정상, ~27% 역전). 전경비율 median 0.33(void margin 큼).
- **Cell 4B Part B** (실제 검은 샘플 경로 추적): 검은배경 35장 중 **via_fg(우회) 33장(94%)**, via_fallback 2장. aroma==random 동일 → **공유 paste 엔진/`_foreground_mask` 레벨 문제, ROI 전략 무관.**

### 3.2 ★확정된 근본 원인 — `_foreground_mask` void 오검출

> severstal 검은배경 결함의 **~94%**는 `_foreground_mask`의 corner-vote가 검은 void를 "전경"으로 오분류(`generate_defects.py:206-217`, 강판 corner ≥2 → `bright_corners>=2` → 전경=어두운 클래스=void)하고, `_foreground_paste_position`이 그 void에 결함을 배치(`:366-372`)하기 때문. clean-bg 게이트는 이중으로 무력: (1) pool 단은 거대-void 이미지를 통과시킴(품질공식의 noise항이 평탄 void를 "깨끗"으로 보상 → R=0), (2) position 단은 폴백 한정이라 92% 전경경로를 우회.

**수정 우선순위** (정량 근거):
1. **[1순위, ~94% 해소] `_foreground_mask` void 오검출 수정** — 두 Otsu 클래스 중 품질(`_is_clean_background`) 높은 쪽(강판)을 전경으로, 또는 선택 전경이 void면 스왑/None. 단 진짜 dark-object 데이터셋과 충돌 안 하도록 가드.
2. **[2순위, ~6% 해소] pool 게이트 유효-면적 기준** — 거대-void normal 제외(whole-image 품질론 못 거름).

모드 A(전경 오검출)가 94%라 **1순위 단독으로 대부분 해결.** 이 수정은 3조건 공통 paste 엔진이므로 대칭 적용(공정성 영향 없음). 코드 변경 → `/feature-dev` 별도 패치.

---

## 4. ROI quality 게이트의 위험 — budget parity 붕괴 (코드로 확정)

`--min_quality > 0`를 **그냥 켜면** 공정 비교가 깨진다:

1. 게이트는 `roi_selection.py` main()에서만 동작 → `roi_selected.json`(**aroma만**)을 줄인다. `select_rois`는 `passing`에만 작동, backfill 없음 (`:1308, :1315`).
2. random은 un-gated `roi_candidates.json` 사용(`:1341`) → **면역**. casda는 `min_suitability`만 사용(별개 필터), `min_quality` 호출 경로 없음 → **면역**.
3. 학습 cap은 조건 무관 동일값 `cap = max(1, int(n_real_train * synth_ratio))` (`exp4_v2_supervised_detection.py:2048`). aroma만 게이트로 줄어 `len(anns) ≤ cap`이면 **aroma만 "no trim" 분기**(`:2056-2060`, 전량 사용 < cap), random/casda는 cap으로 trim.
4. → **같은 synth_ratio인데 aroma만 더 적은 synth로 학습** = budget parity 붕괴 → "ROI 질 효과"와 "표본량 효과"가 confound. 희소 클래스 전량 게이트아웃 시 per-class 심각(`roi_selection.py:409-414` ZERO-pass warning).

이것이 ROI quality 게이트가 **기본 OFF**인 구조적 이유. [[aroma_exp4v2_source-diversity-fairness]] `:40` 불변식("4조건은 선택 전략만 달라야, cap budget 동일") 위반.

---

## 5. 단계적 대응 분기 (결정 트리)

1. **(현재) clean-bg 게이트 검증** — `clean_background_gate_verify.md` 실행. 단 §3대로 대칭 개입이라 AROMA 상대 열위 해소는 기대 낮음.
2. **clean-bg ON 후에도 AROMA가 하위면 → ROI quality 게이트 고려.** 단 §4 때문에 **그냥 켜면 안 됨 — 공정성 보정 필수.**

### 공정성 보정 선택지 (게이트 도입 시)

- **(권장) 선택지 1**: random도 게이트 통과(`passing`) 풀에서 샘플하도록 입력 변경 → 세 조건 동일 quality 기준 → "selection 전략만 다르다" 불변식 유지, quality 게이트 효과를 깨끗이 측정.
- **선택지 2**: post-gate 풀 ≥ 공유 cap 보장(ZERO-pass warning 점검), 안 되면 cap 하향.
- **선택지 3**: 게이트 후 realized-count 최솟값으로 cap 재정의 → 셋 다 동일 trim.

---

## 6. TODO / 미확정

- `TODO`: §2.5 — Cell 7/8 Colab 실측으로 random `area↑/solidity↓` 정량 확정 (현재 가설).
- `TODO`: §5.2 — clean-bg 검증 수치 확인 후 ROI quality 게이트 진행 여부 결정.
- `<결정 필요>`: ROI quality 게이트 도입 시 공정성 보정 선택지(1/2/3) 확정 — 구현은 별도 패치 노트로 분리.

---

## 7. 수정 대상 (해당 시 — 현재는 분석/계획만, 코드 변경 없음)

- (보정 선택지 1 채택 시) `generate_random.py` 또는 `roi_selection.py` — random 입력을 `passing` 기반으로 전환.
- (선택지 2/3 채택 시) `exp4_v2_supervised_detection.py` cap 로직 (`:2043-2060`).

> 본 노트는 **진단/계획**까지만. 실제 코드 변경은 결정 확정 후 `feature-dev` 경유 별도 패치 노트로 진행.
