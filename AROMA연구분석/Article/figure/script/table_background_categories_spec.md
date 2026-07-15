# Background Category Classification Table — §3.2.3 Support

## 목적

§3.2.3에서 "five-stage background analysis pipeline classified the surrounding texture into one of five categories: smooth, directional, periodic, organic, or complex"라고 정의한 5개 category를 정형화.

텍스트에서는 나열만 있고 정의/특성 없으므로, table로 명확화.

## Table 위치 및 번호

- 위치: §3.2.3 ROI Extraction 섹션, "five categories" 나열 직후
- 번호: Table X (배치 시 결정, 잠정 Table A 또는 별도 참조 숫자)

## Table 구조

| Category | Definition | Characteristics | Examples |
|----------|-----------|-----------------|----------|
| **Smooth** | 균질하고 변동 적은 배경 | Low texture entropy, uniform gradient | 평탄 금속판, 균일 색상 영역 |
| **Directional** | 일방향 또는 선형 구조 | High linearity, oriented edges, directional gradient | 브러시 마크, 스트라이프, 결 방향 |
| **Periodic** | 반복되는 패턴 또는 격자 | Regular frequency content, peak in spectral domain | 타일 패턴, 체크무늬, 주기적 텍스처 |
| **Organic** | 자연스러운 불규칙 구조 | High entropy, variable orientation, no dominant frequency | 가죽, 천, 나무 결리스 |
| **Complex** | 다중 특성 또는 고도로 불규칙 | Multiple orientations, high entropy, mixed features | 배경-결함 경계, 복합 표면 |

## 기술 출처

- 텍스트 정의: §3.2.3 직접 인용
- 특성 설명: profiling stage에서 사용하는 feature들 기반
  - texture entropy (TextureEntropy)
  - orientation variance (OrientVariance)
  - frequency complexity (FreqComplexity)
  - linearity (from morphology analysis)
- Examples: 데이터셋 시각 확인 (aroma_dataset/profiling에서 background type 분포 파악 후 추가)

## 역할 (Context for Table)

이 분류는 roi_metadata.json에 저장되지만, **core placement scoring에는 미사용** — optional quality proxy에만 입력(`--min_quality > 0`일 때만 활성화).

Core placement는 morphology clusters와 64-pixel context-cell grid에 의존.

## Caption (초안)

**Table X.** Background texture categories used in ROI extraction (§3.2.3). Each category is identified by a five-stage pipeline analysis based on texture entropy, orientation variance, and frequency content. While the category label is stored in roi_metadata.json, it serves only as an optional input to the subtype-matching quality proxy (active when `--min_quality > 0`); core placement scoring instead operates on morphology clusters and a 64-pixel context-cell grid (§3.2.6).

## 저장

- 파일: Article/text/section3_2.txt 내 Table X로 직접 삽입
- 위치: §3.2.3 "five categories" 문단 직후
