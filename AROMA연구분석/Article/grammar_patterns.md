# Grammar Pattern Analysis — Reference Papers
> 9편 참조 논문의 섹션별 문법 패턴 분석  
> CASDA 논문 영문 개선 시 이 파일을 기준으로 적용한다.

---

## 1. 섹션별 문장 구조

### Abstract
- **길이:** 20–35 단어
- **구조:** 참여구(Participial phrase) + 주절 + 종속절
- **핵심:** 문제 → 방법 → 결과를 단일 문단으로 압축

```
"In this study, we conducted a controlled architectural evaluation of YOLOv8
 for detecting bacterial microcolonies in high-resolution microscopy images."

"The final configuration achieved a recall of 0.943, surpassing the original
 high-resolution baseline (0.926) and several recent state-of-the-art detectors."
```

### Introduction
- **길이:** 15–30 단어(기본) / 복합 구조 35–45 단어
- **구조:** 배경 확립 → 문제 진술 → 연구 공백(gap) 식별

```
"Detection of brain tumors is of great importance in clinical diagnostics and
 treatment planning, as early localization of neoplastic lesions directly affects
 prognosis and therapeutic efficacy."

"Despite progress in adapting detectors to specialized domains, two practical
 knowledge gaps remain for microscopy-based bacterial screening systems."
```

### Methods
- **길이:** 18–32 단어
- **구조:** 수동태 중심의 절차 서술; 번호 붙은 하위 섹션
- **패턴:** "[수행 내용] to [목적/보장]"

```
"All models were trained using the Ultralytics YOLOv8 optimization pipeline,
 which performs end-to-end learning for bounding box regression, classification,
 and confidence estimation."

"To ensure a controlled comparison, the original configuration was trained and
 evaluated on 480×480 input images with consistent architecture and hyperparameters."
```

### Results
- **길이:** 16–28 단어
- **구조:** 주제문(topic sentence) + 수치 근거 + 비교 주장
- **패턴:** "[지표] achieved [수치], surpassing/outperforming [baseline]"

```
"The best-performing configuration improves performance to an mAP@50 of 0.987,
 precision of 0.976, and recall of 0.968."

"Across all backbone sizes, architectures retaining a feature fusion neck
 significantly outperform no-neck variants."
```

### Conclusion
- **길이:** 20–35 단어
- **구조:** 역피라미드 — 구체 발견 → 일반적 기여
- **패턴:** "These findings demonstrate/confirm that ..."

```
"Overall, the proposed framework highlights that sensitivity-oriented detection
 can be achieved efficiently at lower resolution."

"These findings confirm that progressive architectural scaling and regularization
 collectively enhance detection sensitivity and localization reliability."
```

---

## 2. 시제 규칙

| 섹션 | 시제 | 용도 |
|------|------|------|
| Abstract | 단순 과거 | 수행한 것 기술 |
| Abstract | 현재 | 결과가 보여주는 것 |
| Introduction | 현재 | 확립된 사실, 연구 공백 |
| Introduction | 현재 완료 | 연구 트렌드 ("have enhanced", "have shown") |
| Methods | 단순 과거 수동 | 구현 절차 전반 |
| Results | 단순 과거 | 관찰된 것 |
| Results | 현재 | 결과가 시사하는 것 |
| Conclusion | 현재 | 함의, 일반화 |
| Conclusion | 조동사(could/should) | 미래 방향 |

```
# Abstract 혼용 예시
"We conducted [과거] ... The results demonstrate [현재] ..."

# Introduction 현재 완료
"Recent studies have enhanced the YOLOv8 architecture to increase detection
 accuracy, mainly through attention mechanisms and multi-scale feature fusion."

# Methods 수동 과거
"The dataset was collected, curated, and verified over approximately one year
 through repeated cycles."
```

---

## 3. 헤징(Hedging) 언어

### 섹션별 헤징 강도

| 섹션 | 헤징 강도 | 주요 표현 |
|------|-----------|-----------|
| Abstract | 약 | "may," "can," "support," "enable" |
| Introduction | 중강 | "remain insufficiently explored," "limited attention has been given" |
| Methods | 최약 | 절차는 단정적으로 서술; 설계 선택에만 사용 |
| Results | 중 | "demonstrate," "indicate," "suggest" (입증된 것에는 단정) |
| Conclusion | 강 | "may be applicable," "further research required," "could" |

```
# Introduction — gap 외교적 표현
"Despite these advancements, most existing approaches predominantly rely on fixed
 input resolutions and frequently overlook the performance implications."

# Results — 해석에만 헤징
"These results indicate that optimal detection performance arises from a trade-off
 between precision and recall governed by backbone capacity."

# Conclusion — 일반화 헤징
"The underlying optimization strategy is not task-specific and may be applicable
 to other detection tasks, particularly those involving small targets."
```

---

## 4. 기술적 주장(Claim) 패턴

### 성능 수치 제시
```
# 괄호 안에 수치
"surpassing the original baseline (0.926 recall and 0.653 mAP50–95)"

# 전치사구로 비교
"outperforming YOLOv8x (0.881), YOLOv9e (0.869), YOLOv10x (0.808)"
```

### 비교급 동사 목록
- **outperform**, surpass, exceed → 우위 주장
- **demonstrate**, indicate, suggest → 측정 결과 해석
- **achieve**, attain, reach → 수치 달성 서술
- **confirm**, validate, establish → 가설 검증

### Introduction gap 패턴
```
"[X] have [verb], [however/but] [gap remains]"

"Recent studies have enhanced detection accuracy; however, the specific roles
 of scaling strategies under resolution constraints remain scarcely investigated."
```

---

## 5. 전환 표현(Transitions)

### Introduction
```
문제 심화:  "However," / "Despite," / "While," / "Although,"
추가:       "Furthermore," / "Moreover," / "In addition,"
대조:       "In contrast," / "Conversely,"
```

### Methods
```
순서:       "First," / "Second," / "Finally,"
목적:       "To ensure," / "To enable," / "In order to,"
```

### Results
```
비교:       "Relative to," / "Compared to," / "Unlike,"
증거 제시:  "As shown in Table X," / "The results indicate,"
추가:       "Additionally," / "Furthermore,"
```

### Conclusion
```
종합:       "Overall," / "In summary," / "These findings,"
한계:       "Nevertheless," / "Although," / "Despite,"
미래:       "Future work should," / "Further research is required,"
```

---

## 6. 명사구(Noun Phrase) 구성 패턴

### 패턴 1: [형용사] + [수식어구] + [기술 명사]
```
"single-stage object detection framework"
"high-resolution microscopy images"
"density-aware self-supervised vision transformers"
```

### 패턴 2: [개념] + "for/of/with" + [목적/맥락]
```
"automated system for endpoint detection"
"framework for brain tumor detection"
"pipeline for bacterial microcolony detection in microscopy images"
```

### 패턴 3: 하이픈 복합 기술어
```
"feature-extraction capability"        (명사 수식)
"multi-scale feature fusion"           (형용사구)
"confidence-threshold robustness"      (명사 수식)
"end-to-end learning framework"        (형용사구)
```

### 패턴 4: 중첩 수식어 (Multiple Adjectives)
```
"early-stage bacterial microcolonies"
"resource-constrained environments"
"controlled experimental evaluation"
```

---

## 7. 인용 통합(Citation Integration)

### Introduction — 사실 확립
```
"[진술] [Citation]."
"Detection of brain tumors is of great importance in clinical diagnostics [1]."
```

### Introduction — 선행 연구 귀속
```
"[저자] [동사] [발견/방법] [Citation]"
"Boppana et al. [18] developed a low-cost automated titration system using
 colorimetric endpoint detection."
```

### Introduction — 대조절 내 인용
```
"Although [X] [Citation], [gap remains] [Citation]"
```

### Introduction — 합의/사실 다중 인용
```
"YOLO-family detectors are widely adopted due to their single-stage design [16, 20, 21]."
```

### Results — 비교 시 인용
```
"outperforming [Method A] [Citation] and [Method B] [Citation]"
```

### Conclusion — 선행 연구 비교
```
"Prior work [verb phrase] [Citation], [대조/유사 서술]"
"Previous studies have mainly analyzed indicator color using pH strips [40, 41],
 while recent work focused on endpoint detection [18, 20, 21]."
```

---

## 8. CASDA 논문 적용 요약

| 상황 | 권장 패턴 |
|------|-----------|
| 증강 효과 주장 | "achieves X improvement over Y, demonstrating [해석]" |
| 방법 서술 | 수동 과거: "was generated," "was composed," "were filtered" |
| 실험 조건 | "To ensure fair comparison, all groups were evaluated under identical..." |
| 결과 해석 | "These results indicate that ... whereas ... suggests ..." |
| 한계 인정 | "Although the dataset is domain-specific, the proposed framework may be applicable to..." |
| 기여 서술 | "This work presents / proposes / introduces [명사구]" |
| 미래 연구 | "Future work should investigate / extend / validate ..." |

---

*분석 대상 논문 9편:*
1. A Comprehensive Review of Position and Movement Visual Monitoring Systems
2. Addressing the Impact of Resolution Scaling on YOLO Performance for Brain Tumor Detection
3. Backbone and Feature Fusion Design for YOLOv8-Based Bacterial Microcolony Detection
4. Integration of Computer Vision and Machine Learning for Automated pH Prediction
5. LDA-YOLO: A YOLO-Based Rotated Object Detection Method for Remote Sensing
6. Mislabel Detection in Multi-Label Chest X-Rays via Prototype-Weighted Neighborhood Consistency
7. Small-Data Neural Computing Outperforms RSM Low-Cost Smart Optimization
8. Unsupervised Hierarchical Visual Taxonomy of Marble Natural Stone
9. VMMedSAM-X: A State-Enhanced Dual-Branch Encoder for Medical Image Segmentation
