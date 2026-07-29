# Table 3 배경 텍스처 지표 정의 추가 + 구현 불일치 기록

## (성격: 문서 작업 + 불일치 증거 기록 — 코드 패치 아님)

`Article/text/section3_2.txt`의 Table 3에 등장하는 4개 지표(`LocalVariance`, `OrientationEntropy`, `FreqComplexity`, `TextureEntropy`)에 정의·수식이 없어 추가했다. 작업 중 **Table 3 서술과 실제 구현이 어긋나는 지점 3건 + 이름 충돌 1건**을 발견했다. 사용자 결정(2026-07-29)에 따라 **코드는 고치지 않고, Table 3의 판정 조건도 그대로 두고, 불일치는 본 노트로만 기록**한다.

관련: [[aroma_step3_5_clean-bg-selection]](배경 선정), [[project_synth_visual_verification]]

---

## 1. 구현 위치 (확정)

Table 3의 5개 카테고리를 실제로 산출하는 코드:

- `utils/background_characterization.py` — `BackgroundAnalyzer.classify_patch` (lazy cascade, 64px 패치)
- 호출부: `stage1_roi_extraction.py:164` (`BackgroundAnalyzer(grid_size=effective_grid, variance_threshold=100.0)`)
- 소비처: `roi_metadata`에 `background_type`/`stability_score`/`continuity_score` 저장 → `utils/suitability.py`의 `MATCHING_RULES`(subtype × background_type)에서 quality proxy 입력으로만 사용

`grep -rn "BackgroundAnalyzer"` 결과 호출부는 `stage1_roi_extraction.py` 단 하나.

## 2. 실제 판정식 vs Table 3 서술

| 순서 | 실제 코드 (`classify_patch`) | Table 3 서술 (수정 전) |
|---|---|---|
| 1 smooth | `np.var(patch) < 100.0` | `LocalVariance ≤ P25` |
| 2 directional | 그라디언트 방향 엔트로피 `< 1.0` | `OrientationEntropy ≤ P25 AND FreqComplexity ≤ P25` |
| 3 periodic | off-origin 자기상관 peak `> 0.15` | `FreqComplexity ≥ P75` |
| 4 organic | LBP 엔트로피 `> 2.5` | `TextureEntropy ≥ P50 AND LocalVariance ≥ P75` |
| 5 complex | else | else ✓ |

### 불일치 A — 임계가 고정 상수다 (percentile 아님)

`BackgroundAnalyzer.__init__` 기본값: `variance_threshold=100.0`, `direction_entropy_threshold=1.0`, `periodic_threshold=0.15`, `organic_entropy_threshold=2.5`. `stage1_roi_extraction.py:166`은 `variance_threshold=100.0`만 명시 전달(기본값과 동일)하고 나머지는 기본값 그대로. **코드베이스 전체에 이 4개 임계에 대한 percentile/quantile 유도 로직이 없다**(`grep P25|percentile|quantile` on `utils/suitability.py`, `roi_selection.py` → 0건).

→ Table 3의 `P25`/`P50`/`P75` 표기와 §3.2.3 본문의 "by the per-dataset percentile cascade of Table 3" 둘 다 구현과 불일치. **이번 작업에서 수정하지 않음**(사용자 결정: 현상만 기록).

`direction_entropy_threshold`는 코드 주석에 조정 근거가 남아 있다 — 초기 2.0에서 1.0으로 하향, 이유는 "entropy≈1.67의 dot grid가 directional로 오분류"됨. 즉 이 상수는 데이터로 유도된 값이 아니라 **관측 기반 수동 튜닝값**이다. AROMA의 "no hand-set constants" 서술(§3.2.2·§3.2.4)과 긴장 관계에 있으나, 이 분류는 core placement 경로가 아니라 optional quality proxy에만 쓰이므로 헤드라인 주장을 직접 훼손하지는 않는다.

### 불일치 B — `FreqComplexity`(periodic 기준)의 실체는 자기상관이다

`_compute_autocorrelation_peak`:
```python
f = np.fft.fft2(patch); ac = np.real(np.fft.ifft2(np.abs(f) ** 2))
ac = ac / (ac[0, 0] + 1e-6); ac[0, 0] = 0.0
return float(np.max(ac))
```
Wiener–Khinchin 경로로 FFT를 쓰지만 산출량은 **공간 자기상관 peak**다. 주파수 에너지 측도가 아니다.

### 불일치 C — 판정이 단일 조건이다 (AND 결합 아님)

directional·organic 모두 코드상 조건 1개. Table 3의 AND 표기는 cascade 순서상 앞 단계 부정이 암묵 포함된다는 점을 조건으로 풀어 쓴 것으로 볼 수 있으나(organic 도달 시 이미 `var ≥ 100` 통과), directional의 `AND AutocorrPeak ≤ P25`는 코드에 대응물이 없다 — periodic 검사는 directional **다음**에 오므로 directional 판정 시점에 자기상관은 계산조차 되지 않는다.

## 3. 이름 충돌 — Table 2(CCI) vs Table 3

Table 2의 CCI 성분과 Table 3의 지표가 동명이인이면서 다른 양이다.

| Table 3 (ROI extraction, 패치 단위) | Table 2 / CCI (데이터셋 단위) |
|---|---|
| `TextureEntropy` = LBP 엔트로피 (P=8,R=1,uniform,10bin) — 패치값 | `Texture Entropy` = `Mean(texture_entropy)` — 같은 LBP 식의 **패치 평균** |
| `FreqComplexity` = 자기상관 peak | `Frequency Complexity` = `Var(frequency_energy)` — FFT 고주파 에너지비의 **패치간 분산** |
| `OrientationEntropy` = 8 bins × 22.5°, `[0,180)`, magnitude 가중 | `Orientation Variance` = `Var(orientation_consistency)`, 그쪽은 18 bins, `[−π,π]`, **비가중** |

근거: `scripts/distribution_profiling.py:213 _extract_context_features`(5개 context feature 정의), `scripts/aroma/compute_complexity.py:625-628`(CCI 성분 = mean/var 집계).

## 4. 수정 내용 (문서만)

`AROMA연구분석/Article/text/section3_2.txt` §3.2.3:

1. **Table 3 지표 개명** (충돌 해소):
   - `OrientationEntropy` → `GradOrientEntropy`
   - `FreqComplexity` → `AutocorrPeak`
   - `TextureEntropy` → `LBPEntropy`
   - `LocalVariance` — 유지(Table 2 열에 동명 없음)
2. **Table 3 직후에 정의 문단 신설** — 산문 + display 수식 **2개**. 수식은 **코드 기준으로 정확**하게 작성.

   최종 형태 (232 words / 5 단락·수식줄):
   - `LocalVariance` — 산문 인라인("the intensity variance of p"). 순수 분산이라 display 수식·인용 불요.
   - 두 엔트로피를 **공통 연산자로 factoring** — `Entropy(q) = − Σ_k q_k · log₂ q_k` 1개로 처리하고, 어떤 히스토그램에 적용하는지만 산문으로 구분:
     - `GradOrientEntropy` = Sobel 방향 히스토그램, 8×22.5° bins, orientation mod 180°, magnitude 가중
     - `LBPEntropy` = 10-bin rotation-invariant uniform LBP (P=8, R=1)
   - `AutocorrPeak(p) = max_{(u,v) ≠ (0,0)} R(u,v) / R(0,0), where R = ℱ⁻¹(|ℱ(p)|²)` — CCI의 frequency-energy 성분과 다른 양임을 명시
   - 마무리 문장에 lazy cascade 의미(첫 매치 승 → 선행 조건 부정 암묵 결합)

   **간소화 이력** (사용자 요청 "간략하게 표기"):
   | 버전 | 형태 | 분량 |
   |---|---|---|
   | 1차 | 지표별 산문 4단락 + display 수식 4개 | 381 words / 27줄 |
   | 2차 | Table 3b 4행 표(수식을 표 안에) | 330 words / 11줄 |
   | **최종** | 산문 + display 수식 2개 (표 폐기) | **232 words / 5줄** |

   2차의 표 형태는 **반려**됨 — 수식을 표 안에 넣지 않는다(사용자, 2026-07-29). Table 3b는 삭제.

   표기 정책(사용자 결정): 보조 기호는 **최소한** — `R`(자기상관)·`ℱ`(푸리에)·`q`(Entropy 연산자의 dummy)만 도입하고 `p̄`·`g_x/g_y`·`m`·`q_orient`·`q_LBP`는 쓰지 않고 산문으로 서술. **값역 표기 생략**(`∈ [0,3] bits` 등). 지표명은 기존 §3.2 수식 스타일(풀네임, 번호 없음)에 맞춤.

3. **레퍼런스 4종 추가 — 임시 번호 `[36]`–`[39]`**:
   | 번호 | 레퍼런스 | 대응 |
   |---|---|---|
   | [36] | Shannon, BSTJ 1948 | `Entropy(q)` 공통 연산자 |
   | [37] | Dalal & Triggs, HOG, CVPR 2005 | GradOrientEntropy — magnitude 가중 + unsigned(mod 180°) orientation binning |
   | [38] | Ojala·Pietikäinen·Mäenpää, TPAMI 2002 | LBPEntropy — LBP^riu2, P/R 파라미터화 = skimage `method="uniform"` |
   | [39] | Haralick, Proc. IEEE 1979 | AutocorrPeak — 자기상관을 텍스처 주기성 측도로 |

   번호는 **문단 내 등장순**으로 부여(Shannon → Dalal&Triggs → Ojala → Haralick). 엔트로피 공통화로 Shannon이 맨 앞에 오게 되어 초기 배정을 한 번 재배정했다.

   **번호는 임시다.** 전역 등장순 규칙(Intro [1-4,8] → §2 [9-26] → §3.1 [27-30] → §3.2 [31] → §3.3 [32] → §5 [33-35])상 본래 `[31]`(SAM) 앞에 와야 하지만, 개선 중 레퍼런스 추가가 반복될 것으로 예상되어 **말미 임시 부여 후 추후 일괄 재정렬**로 결정(사용자, 2026-07-29). `Reference.txt`에 PROVISIONAL 블록 주석으로 명시.
   - 구현 라이브러리 인용(scikit-image / OpenCV)은 **미포함** — 다른 곳에서도 sklearn/ultralytics를 무인용하므로 일관성 유지.
4. **판정 임계는 percentile 표기 유지** — 불일치 A/C는 본 노트로만 기록(사용자 결정). Table 3b에는 수치 상수를 넣지 않아 Table 3와 모순되는 서술이 생기지 않게 했다.
5. **core placement 미사용 고지 생략** — 사용자 결정. `table_background_categories_spec.md`의 캡션 초안에는 이 고지가 있으나 본문에는 넣지 않음.
6. **Otsu·BIC 미인용 갭은 이번 범위 밖** — 사용자 결정(§5 TODO로 이관).

## 5. 후속 TODO (차단 아님)

- [ ] **레퍼런스 등장순 일괄 재정렬** — `[36]`–`[39]`가 임시 번호. 재정렬 시 `[31]`(SAM)~`[35]`(LPIPS)가 밀림. 동시 갱신 대상: `section3_2.txt`, `section3_3.txt`, `section5.txt`, `Reference.txt`, `AROMA.txt`.
- [ ] **`AROMA.txt` 재동기화 필요** — 통합 원고 `AROMA.txt:99-107`에 구버전 Table 3가 남아 있음(개명·Table 3b 없음). 직전 커밋 `57b5f8f`("AROMA.txt를 개별 section 파일 최신본으로 재동기화")와 동일 절차 재수행.
- [ ] **미인용 갭 보완** — `Otsu thresholding`(§3.2.3, 2회 등장 / Otsu, IEEE TSMC 1979), `Bayesian Information Criterion`(§3.2.2 / Schwarz, Ann. Statist. 1978). 위 재정렬 작업과 묶어 처리하면 renumber 1회로 끝남.
- [ ] **불일치 A 처리 방향 결정** — 택1: (a) Table 3·§3.2.3 본문을 고정 임계로 정정, (b) `background_characterization.py`를 percentile 기반으로 개조 후 재프로파일링. reviewer가 "percentile cascade"의 데이터 유도 근거를 물으면 (a)가 최소 비용.
- [ ] 불일치 C의 directional `AND AutocorrPeak ≤ P25` — 코드에 대응물 없음(periodic 검사는 directional 다음 순서라 그 시점에 자기상관 미계산). A 처리 시 같이 제거 검토.
- [ ] **DOI 최종 검증** — `[36]`–`[39]` DOI는 표준 서지 기준으로 기재. 제출 전 발행처 페이지에서 확인 권장.
