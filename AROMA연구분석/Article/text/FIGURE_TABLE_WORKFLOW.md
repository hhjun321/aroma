# AROMA 논문 — Table / Figure 작업 규칙

이 문서는 `AROMA연구분석/Article/text/` 하위 **모든 section 문서**에 table·figure를 생성·배치할 때 공통으로 적용하는 규칙이다. section별 작업 시 반드시 이 규칙을 따른다.

---

## 1. 목적

각 section의 내용에 맞는 **table**과 **figure**를 생성하고, 논문에 바로 배치할 수 있는 형태로 산출한다.

---

## 2. 산출물 형식

### 2.1 Table

- **대상 section 문서에 직접 작성**한다 (예: `section3_1.txt`).
- section 텍스트가 참조하는 표 번호(예: "Table 1")와 일치시킨다.
- 표의 모든 수치는 **실측 데이터**만 사용한다 (§4 참조).

### 2.2 Figure

3단계로 진행한다:

1. **Spec 작성** — 필요한 내용을 정리한 `.md` 파일을 `AROMA연구분석/Article/figure/script/`에 작성한다.
   - 무엇을 보여줄지, 데이터 출처, 서브플롯 구성, 축·범례·색상, 참조할 section 문장을 명시.
   - **Caption(초안) 포함** — Table과 동일하게 "**Figure N.** [caption]" 형식으로 기록하여 이미지 첨부 시 참고.
2. **스크립트 작성** — spec을 바탕으로 figure 생성 스크립트를 `figure/script/`에 작성한다.
3. **이미지 저장** — 생성된 이미지를 **figure 번호와 함께** `AROMA연구분석/Article/figure/image/`에 저장한다.
   - 파일명 규칙: 기존 `[figure1] aroma_pipeline.png` 형식을 유지 → `[figureN] <설명>.png`
4. **Section 문서에 Caption 추가** — 이미지 직전에 `**Figure N.** [caption]` 형식으로 section 텍스트 안에 기록한다. 캡션은 spec의 "Caption" 섹션을 참고한다.

---

## 3. 데이터 출처

- 실 데이터는 `D:\project\aroma_dataset` 에서 참조한다.
- AROMA에서 사용한 **5개 데이터셋**(AITeX, Kolektor, Severstal, MTD, MVTec Leather)을 기준으로 한다.
- 복잡도 지표: `aroma_dataset/complexity/<dataset>/complexity_report.json`
- 프로파일링: `aroma_dataset/profiling/profiling/<dataset>/` (morphology_clusters, compatibility_matrix, threshold_policies 등)

---

## 4. 데이터 진실성 (필수)

- 표·그림의 모든 수치는 **실측값**만 사용한다. 추정·창작 금지.
- section 텍스트에 이미 인용된 수치가 있으면 **교차 검증**하여 일치시킨다 (불일치 시 사용자에게 보고).
- 데이터가 없어 채울 수 없는 항목은 비워두거나 사용자에게 확인한다.

---

## 5. 작업 범위

- 작업 범위는 **현재 작업 중인 section 문서** 단위이며, 진행 순서·대상은 **사용자의 선택**을 따른다.
- 한 번에 전 section을 훑지 않는다. 사용자가 지정한 section만 처리한다.

---

## 6. Figure 규칙 (figure_patterns.md 준용)

다음 규칙은 **모든 figure**에 적용한다. 상세는 `figure_patterns.md`를 정본으로 참고.

### 6.1 캡션(Caption) 문장 구조
- **길이**: 8–25 단어 (한 문장 원칙, 필요시 부연 문장)
- **구조**: 명사구 주도 + 필요시 부연절
- **패턴**: `**Figure N.** [대상 명사구]. [부연: 조건/방법/색상 의미]`

### 6.2 본문 콜아웃 + 인용 순서
- 모든 figure는 본문 서술에서 명시적으로 최소 1회 인용 필수 (캡션만으로 부족)
- 콜아웃 문형: **"As shown in Figure N, ..."**, "Figure N illustrates/depicts ..."
- 각 figure의 **최초 콜아웃이 번호 오름차순** (Figure 1 → Figure 2 → ...)

### 6.3 섹션-지역성
- Figure는 **그 figure가 배치된 section의 본문에서만 인용**
- 다른 section에서 참조 필요 시 figure 대신 **해당 section 명시** (예: "see §3.1")

### 6.4 그림 해상도
- 모든 figure는 **최소 가로/세로 1000픽셀 이상, 또는 300 dpi 이상** 해상도로 저장
- `matplotlib`, `PIL` 등으로 생성 시 `dpi=300` 명시

### 6.5 캡션·라벨 내 수치 서식
- 천단위 콤마는 **5자리 이상** 숫자에만 사용
- **4자리 숫자에서는 콤마 제거** (예: `1,200` → `1200`, `12,000` 유지)
- 캡션·그림 내부 라벨 모두 적용

---

## 7. 암묵적 필요성 (Implicit Requirements)

텍스트에서 직접 참조하지는 않지만, 내용 이해를 위해 table/figure가 필요한 경우가 있다. 다음 기준으로 판단한다:

### 7.1 분류·범주 설명 뒤
- **조건**: 섹션이 N개 카테고리/타입을 정의하고 각각 설명하는 경우
- **필요성**: 정의만으로는 직관성 부족 → table 또는 시각화 추천
- **예시**:
  - §3.2.3 "five-stage background analysis pipeline classified the surrounding texture into one of five categories: smooth, directional, periodic, organic, or complex"
    → Table (5 categories + description + visual example) 고려
  - §3.2.4 "assigned to one of five subtypes—linear_scratch, elongated, compact_blob, irregular, or general"
    → **Table 2** (subtype별 defect variant warping strategy) 필수

### 7.2 절차·알고리즘 설명 뒤
- **조건**: 복잡한 pipeline/단계를 텍스트로만 설명하는 경우
- **필요성**: flow diagram 또는 절차 표로 명확화
- **예시**:
  - §3.2.5 "TF-IDG that produces subtype-specific variants of each seed defect by displacement ... with brightness/contrast jitter" 
    → Figure (elastic warping 시각화: before/after, displacement field) 고려
  - §3.2.6 "two-phase coverage-first allocator ... PairDeficit ... Hamilton's method"
    → Figure (allocation phase 1 / phase 1b / phase 2 흐름도) 고려

### 7.3 데이터-공학적 설명 뒤
- **조건**: 게이트·필터·스코어 공식을 텍스트로 정의하는 경우
- **필요성**: 공식의 동작을 실제 데이터로 검증한 결과 시각화
- **예시**:
  - §3.2.6 "symmetric compatibility gate ... compat = sqrt((P_def+ε)·(P_clean+ε)) ... accepted only if its compatibility exceeds a dataset-specific threshold τ"
    → Figure (compat score 분포 + τ threshold 시각화, 데이터셋별) 고려
  - §3.2.8 "artifact score ... blur score ... composite quality score Q"
    → Figure (low Q vs high Q 샘플 비교, 정성) 고려

### 7.4 우선순위
암묵적 figure/table이 많은 경우, 다음 우선순위로 진행:
1. **분류표** (명확성, 논문 기본) — Table 필수
2. **파이프라인/절차 흐름** — Figure 추천
3. **정성 비교** (before/after, good/bad 샘플) — Figure 고려

---

## 8. 표기·환경

- Colab 실행 규칙(`.claude/rules/colab-execution.md`)을 따른다: `!python`, `$VAR` 형식.
- 새 테스트 코드 작성·pytest 실행 금지 (`CLAUDE.md`). 검증은 실제 실행 결과로 확인.

## 9. 명확성과 간결성 규칙

### 9.1 참고(committed) 및 부연(NOTE) 표기 제거

본문에 `[committed: ...]` 또는 `[NOTE — ...]` 표기는 포함하지 않는다.

**근거**: 이러한 표기는 편집 과정 및 추적용이며, 최종 논문에는 부적절함.

**처리 방식**:
1. **Committed 참고** (`[committed: path/to/doc]`): 제거. 해당 문서는 작성자가 별도로 참조.
2. **NOTE 부연** (`[NOTE — previously drafted...]`): 제거. 중요한 내용만 요약.

### 9.2 NOTE 내용 중 중요 정보 처리

NOTE에 포함된 정보 중 reader가 알아야 할 내용이 있으면:

- **경우 A**: 본문 문맥상 이미 명확하면 → 제거
- **경우 B**: 독자가 알아야 할 중요 내용이면 → 새 문장으로 통합
  - 예: "one-class AD methods (PatchCore, EfficientAD) are not used in the committed evaluation" → "The evaluation is limited to supervised YOLOv8n detection; alternative one-class anomaly detection methods are not assessed here."

### 9.3 적용 대상

- 방법론/설계 결정이 완료된 섹션 (§3.2, §3.3 등)
- 초안 단계의 부연은 제거 (최종본만 유지)

---
