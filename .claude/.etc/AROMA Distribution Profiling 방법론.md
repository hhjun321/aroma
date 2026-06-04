# AROMA: Adaptive ROI Optimization via Morphology-Aware Analysis

## 1. 연구 배경

CASDA는 결함 특성과 배경 특성을 분석하여 적합한 ROI를 선택하고, 이를 기반으로 합성 데이터를 생성하는 데이터 증강 프레임워크이다.

그러나 CASDA는 다음과 같은 한계를 가진다.

### 1.1 수동 임계값 의존성

현재 CASDA는 다음과 같은 파라미터를 전문가 경험에 기반하여 수동 설정한다.

* Aspect Ratio Threshold
* Linearity Threshold
* Solidity Threshold
* Background Variance Threshold
* Edge Strength Threshold
* Compatibility Matrix

이러한 값들은 특정 도메인(Severstal Steel Defect Dataset)에 최적화되어 있으며 일반화 가능성이 제한적이다.

### 1.2 도메인 종속성

현재 배경 분류 체계는 다음과 같이 철강 표면 특성에 강하게 의존한다.

* Smooth
* Vertical Stripe
* Horizontal Stripe
* Textured
* Complex

따라서 PCB, 반도체 웨이퍼, 섬유, 의료영상 등 다른 도메인으로의 확장이 어렵다.

### 1.3 전문가 기반 Compatibility Rule

현재 ROI 적합도는 다음과 같이 정의된다.

```text
Linear Defect → Stripe Background
Blob Defect → Smooth Background
Irregular Defect → Complex Background
```

이는 전문가 지식 기반 가설이며 실제 데이터 분포에 의해 검증되지 않았다.

---

## 2. 연구 목표

AROMA의 목표는 다음과 같다.

> 수동으로 정의된 도메인 특화 규칙을 데이터 기반 적응형 정책으로 대체하여 다양한 산업 도메인에 적용 가능한 ROI 최적화 프레임워크를 구축한다.

---

## 3. 핵심 개념

### CASDA

```text
Feature Extraction
        ↓
Manual Threshold
        ↓
Manual Compatibility
        ↓
ROI Selection
```

### AROMA

```text
Feature Extraction
        ↓
Distribution Analysis
        ↓
Adaptive Threshold Selection
        ↓
Compatibility Learning
        ↓
ROI Selection
```

---

## 4. Stage 0: Distribution Profiling

CASDA 개선 문서의 analyze_dataset.py는 AROMA의 Stage 0 프로토타입으로 활용한다.

### 입력

* Defect Mask
* Original Image

### 출력

#### Morphology Distribution

* Aspect Ratio
* Linearity
* Solidity
* Fill Ratio

#### Context Distribution

* Local Variance
* Edge Density
* Frequency Energy
* Texture Entropy

#### Co-occurrence Distribution

P(Context | Morphology)

#### Threshold Recommendation

* P10
* P25
* P50
* P75
* Valley Threshold

---

## 5. Morphology Category Derivation

기존 CASDA

```text
Linear
Compact Blob
Irregular
General
```

은 연구자가 직접 정의하였다.

AROMA에서는 Morphology Feature Space를 구성한다.

### Feature Set

* Aspect Ratio
* Circularity
* Solidity
* Fill Ratio
* Eccentricity

### Category Derivation Strategy

#### Case 1: Bimodal Distribution

Otsu Threshold

#### Case 2: Multimodal Distribution

Gaussian Mixture Model

#### Case 3: Unimodal Distribution

Percentile Partition

### 결과

```text
Morphology Cluster 1
Morphology Cluster 2
Morphology Cluster 3
...
```

도출 후 의미를 부여한다.

---

## 6. Context Representation Generalization

CASDA

```text
Smooth
Vertical Stripe
Horizontal Stripe
Textured
Complex
```

AROMA

```text
Local Variance
Edge Density
Texture Entropy
Frequency Energy
Orientation Consistency
```

즉 Semantic Category가 아닌 Physical Descriptor를 사용한다.

이를 통해

* Steel
* PCB
* Semiconductor
* Fabric
* Medical

도메인에 공통 적용 가능하도록 한다.

---

## 7. Adaptive Threshold Selection

CASDA

```text
Threshold = Constant
```

AROMA

```text
Threshold = f(Distribution)
```

### 정책

#### Bimodal

Otsu

#### Multimodal

GMM Boundary

#### Unimodal

Percentile-based

### 목적

Threshold 자체를 저장하는 것이 아니라

Threshold Derivation Policy를 저장한다.

---

## 8. Compatibility Learning

### 기존 CASDA

```text
Compatibility
=
Human Prior
```

### AROMA

실제 데이터에서

P(Context | Morphology)

를 계산한다.

```text
Linear + Context A
Linear + Context B
Blob + Context A
Blob + Context B
```

공출현 빈도를 기반으로 Compatibility Score를 정의한다.

### 장점

* 전문가 편향 제거
* 도메인 적응 가능
* 재현성 확보

---

## 9. Deficit-aware Synthesis Strategy

CASDA는 적합한 ROI를 선택하는 것에 집중하였다.

AROMA는 데이터셋 균형까지 고려한다.

### 계산

Global Context Distribution

```text
P(Context)
```

Class-wise Context Distribution

```text
P(Context | Class)
```

Deficit

```text
Deficit
=
P(Context)
-
P(Context|Class)
```

### 활용

부족한 조합에 우선적으로 합성 데이터를 생성한다.

---

## 10. 제안 시스템 구조

```text
Dataset
    ↓
Morphology Analysis
    ↓
Context Analysis
    ↓
Distribution Profiling
    ↓
Adaptive Threshold Selection
    ↓
Compatibility Learning
    ↓
Deficit Analysis
    ↓
ROI Selection
    ↓
Synthetic Data Generation
```

---

## 11. 예상 연구 기여

### Contribution 1

수동 임계값 기반 ROI 선택을 데이터 기반 적응형 ROI 선택으로 전환

### Contribution 2

도메인 특화 Context Category를 도메인 독립 Physical Descriptor로 일반화

### Contribution 3

전문가 기반 Compatibility Matrix를 데이터 기반 Compatibility Learning으로 대체

### Contribution 4

데이터 불균형을 고려한 Deficit-aware Augmentation Framework 제안

### Contribution 5

Steel Domain에 국한된 CASDA를 Multi-domain Augmentation Framework로 확장

---

## 최종 정의

AROMA는 단순한 CASDA 개선 버전이 아니다.

AROMA는 ROI 선택 규칙을 전문가 경험으로부터 분리하고, 데이터 분포로부터 적응적으로 학습하는 Domain-Adaptive Data Augmentation Framework이다.
