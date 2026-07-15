# Figure Pattern Analysis — Reference Papers
> 9편 참조 논문의 그림(Figure) 관련 문법·서식 패턴 분석
> CASDA 논문 그림 캡션·콜아웃·서식 점검 시 이 파일을 기준으로 적용한다.
> 문장 문법 전반은 [grammar_patterns.md](grammar_patterns.md)를 참고한다.

---

## 1. 캡션(Caption) 문장 구조

- **길이:** 8–25 단어 (한 문장 원칙, 필요 시 두 번째 보조 문장)
- **구조:** 명사구 주도(주어·동사 생략 가능) + 필요 시 부연 절
- **패턴:** "[대상 명사구]. [부연: 조건/방법/색상 의미]"

```
"Overview of the proposed pipeline for bacterial microcolony detection."

"Placement distribution across compatibility cells. Warmer colors indicate
 higher context-compatibility scores."

"Qualitative comparison of augmented samples. (a) baseline, (b) random paste,
 (c) proposed AROMA placement."
```

### 시제·태
| 요소 | 규칙 |
|------|------|
| 캡션 본문 | 명사구 우선, 동사 사용 시 현재형 |
| 방법 설명 부연 | 현재형 ("indicates," "shows," "denotes") |
| 결과 그림 부연 | 현재형 해석 ("higher values correspond to ...") |

---

## 2. 본문 콜아웃 + 인용 순서

- **모든 그림은 본문 서술에서 명시적 콜아웃으로 최소 1회 인용**한다. 캡션·`\includegraphics`만으로는 부족하며, 본문 문장이 그림을 직접 지목해야 한다.
- 콜아웃 문형(권장): **"As shown in Figure N, ..."**, "Figure N illustrates/depicts/reports ...", "..., as visualized in Figure N", "(Figure N)". LaTeX면 `Figure~\ref{fig:...}`.
- 각 그림의 **최초 콜아웃이 번호 오름차순**. 콜아웃 없는 그림은 삽입(예: "As shown in Figure 8, ...").

```
좋음:  "As shown in Figure 8, the placement distribution shifts toward high-compatibility cells."
나쁨:  그림만 존재하고 본문이 그 그림을 전혀 지목하지 않음.
점검:  본문에서 "Figure 1" → "Figure 2" → … 순으로 첫 콜아웃이 등장하는가?
```

### 콜아웃 동사
- **illustrate, depict, show, present** → 그림 내용 지목
- **visualize, plot, display** → 데이터 시각화 지목
- **compare, contrast** → 비교 그림
- **summarize, overview** → 파이프라인/개요 그림

---

## 3. 섹션-지역성

- 그림은 **그 그림이 배치된 섹션의 본문에서만 인용**한다. 이전/이후 섹션에서의 참조는 지양하고, 교차 참조가 필요하면 그림 대신 해당 **섹션**을 가리킨다.

```
좋음:  §4.1 본문에서 §4.1에 놓인 Figure 4를 인용.
나쁨:  §5(Discussion)에서 §4.1의 Figure 4를 인용 → 대신 "(see §4.1)".
점검:  각 그림의 콜아웃이 그 그림이 속한 섹션 안에 있는가?
```

---

## 4. 그림 해상도

- 그림 내용은 **판독 가능**해야 한다. 저해상도 이미지는 **최소 가로/세로 1000픽셀 이상, 또는 300 dpi 이상**의 고해상도 이미지로 교체한다.

---

## 5. 캡션·라벨 내 수치 서식

- 천단위 콤마는 **5자리 이상** 숫자에만 사용한다. **4자리 숫자에서는 콤마 제거**. 그림 캡션·그림 내부 라벨 모두 적용.
- 상세 규칙은 [grammar_patterns.md §9.4](grammar_patterns.md)를 정본으로 따른다.

```
올바름:  1200      12,000     114,174
잘못됨:  1,200  →  1200
```

---

## 6. 패널·서브그림(Subfigure) 라벨

- 다중 패널은 **소문자 괄호 라벨** 사용: `(a)`, `(b)`, `(c)`.
- 캡션에서 각 패널을 순서대로 지목: "(a) baseline, (b) random paste, (c) proposed."
- 본문 콜아웃 시 패널 지정 가능: "Figure 6(c) shows ..."

```
"Figure 6. Qualitative results. (a) input, (b) ground truth, (c) prediction."
"As shown in Figure 6(c), the model recovers fine boundary details."
```

---

## 7. 성능 비교 표(Table) 내용 표본

성능 비교 표는 아래 구조를 표준으로 한다. (참고: 표 콜아웃·순서 규칙은 [grammar_patterns.md §9.1](grammar_patterns.md), 그림은 §2–§3.)

**캡션 문형:** `Table N. [모델] [지표] on the [데이터셋] test set (mean ± std, n = K seeds)`

### 예시 1: 단일 지표 (Single-class 또는 전체 mAP)

```
Table 8. YOLOv8n detection performance on AITeX test set 
         (mean ± std, n = 3 seeds). Single-class tiled-pattern dataset.
```

| Method | mAP@0.5 |
|--------|---------|
| Baseline | 0.2039 ± 0.1138 |
| Random | 0.4187 ± 0.0514 |
| AROMA | **0.4683 ± 0.0461** |
| Random vs. Baseline | +21.48 pp |
| AROMA vs. Baseline | +26.44 pp |
| AROMA vs. Random | +4.96 pp |

### 예시 2: 다중 지표 (클래스별 성능)

```
Table 9. YOLOv8n detection performance on Kolektor test set 
         (mean ± std, n = 3 seeds). Two-class plastic defects.
```

| Method | mAP@0.5 |
|--------|---------|
| Baseline | 0.9740 ± 0.0225 |
| Random | 0.9938 ± 0.0020 |
| AROMA | 0.9870 ± 0.0139 |
| Random vs. Baseline | +1.98 pp |
| AROMA vs. Baseline | +1.30 pp |
| AROMA vs. Random | −0.68 pp |

> 위 표본은 **구조·표현 방식** 참조용이다. 구체적 예시는 AITeX(low-headroom 폭증) vs. Kolektor(high-headroom 수렴)을 대조한다.

### 규칙 (구조·표현)
- **레이아웃:** 1열 = Method, 이후 열 = 지표(전체 mAP 또는 mAP + 클래스별 AP). 상단 = 절대 성능 행 (3행 이상), 하단 = 델타(비교) 행.
- **절대 성능 행:** `Method` 라벨은 데이터 구분 기준(예: "Baseline", "Random", "AROMA"). 값은 `평균 ± 표준편차` (소수 자릿수 열 내 통일). n(=seed 수)은 캡션에 명시.
- **델타 행:** `Method A vs. Method B` 형식 라벨, 퍼센트포인트(`pp`) 단위, 부호 명시. 증가 `+`, 감소 `−`(U+2212 마이너스, 하이픈 아님).
- **최우수 강조(선택):** 절대 성능 열 중 최고값을 **볼드** 처리 가능 (예시 1 AROMA 행).
- **천단위 콤마:** 5자리 이상만(§5). 예시는 전부 4자리 미만 → 콤마 없음.

---

## 8. 데이터셋 구성 설명 표본

증강 전/후 데이터 규모를 클래스별로 제시하는 표. 실험에 쓰인 데이터셋 설명에 사용.

**캡션 문형:** `Table N. Dataset composition before and after CASDA augmentation (instances per class)`

```
Table N. Dataset composition of the Severstal training set before and after
         CASDA augmentation (instances per class).
```

| Class | Original | CASDA Added | Total | Increase (%) |
|-------|----------|-------------|-------|--------------|
| Class 1 | 2494 | 756 | 3250 | 30.3 |
| Class 2 | 247 | 519 | 766 | 110.1 |
| Class 3 | 1836 | 501 | 2337 | 27.3 |
| Class 4 | 660 | 462 | 1122 | 70.0 |
| **Total** | **5237** | **2238** | **7475** | **42.7** |

> 구조·표현 참조용. 셀 값은 표본이다.

### 규칙 (구조·표현)
- **레이아웃:** 1열 = Class(범주), 이후 열 = `Original → Added → Total → Increase(%)` 순(원본→증가분→합계→증가율).
- **합계 행:** 최하단 `Total` 행으로 열별 합산, **볼드** 강조.
- **증가율:** `Increase(%) = Added / Original × 100`, 소수 1자리로 통일(예: `70` → `70.0`).
- **정수 카운트:** `평균 ± std` 없이 정수. 천단위 콤마는 5자리 이상만(§5) → 위 표본은 전부 4자리 이하라 콤마 없음(`2,494` → `2494`).

---

## 9. CASDA 논문 적용 요약

| 상황 | 권장 패턴 |
|------|-----------|
| 파이프라인 개요 | "Overview of the proposed [명사구]." + 본문 "Figure 1 illustrates ..." |
| 정성 비교 그림 | "(a) baseline, (b) random, (c) AROMA." + 본문 "As shown in Figure N(c), ..." |
| 분포/히트맵 | "Warmer colors indicate higher ..." + 본문 "Figure N visualizes ..." |
| 색상 의미 명시 | 캡션에 "[색] denotes/indicates [의미]" |
| 저해상도 그림 | 1000px+ 또는 300 dpi로 교체 (§4) |
| 캡션 내 4자리 수 | 콤마 제거 (§5) |

---

*분석 대상 논문 9편은 [grammar_patterns.md](grammar_patterns.md) 하단 목록과 동일.*
