**Considerations**
1. 학습 및 실험의 수행은 google colab 환경에서 진행한다.
2. 그러므로 연구의 단계별 .py script 로 분리하여 작성하여야 한다.
3. 단계별 참조하는 파일의 경로는 무조건 cli 를 통해 입력받도록 준수한다.


1. Executive Summary (연구 요약)
본 연구는 산업용 시각 검사 시스템의 고질적 문제인 **데이터 희소성(Data Scarcity)**을 해결하기 위해, 관심 영역(ROI) 기반의 맞춤형 결함 합성 및 증강 프레임워크를 제안한다.
(AROMA: Adaptive ROI-Oriented Multi-domain Augmentation via Structured ROI Decomposition)
기존의 강철 결함 도메인을 넘어 ISP-AD(스크린 인쇄) 및 MVTec AD(객체) 도메인으로의 확장을 목표로 하며, 핵심 기술로 
**Modified Poisson Blending (MPB)**과 **Training-Free Industrial Defect Generation (TF-IDG)**을 결합한다.  
이를 통해 데이터 수집 비용을 획기적으로 낮추면서도 SCI급 논문의 독창성과 모델 성능의 강건성을 동시에 확보한다.

2. Theoretical Background & Novelty (이론적 배경 및 독창성)
2.1 Context-Aware Data Augmentation (CASDA)
전통적인 증강 기법과 달리, CASDA는 결함의 물리적 발생 맥락을 고려한다. 결함 유형(Defect Type)과 배경(Background)의 논리적 결합을 통해 생성된 합성 데이터가 실제 분포(In-distribution)를 따르도록 유도한다.

2.2 Modified Poisson Blending (MPB)의 기술적 우위본 연구의 핵심 차별화 요소는 단순한 합성(Cut-and-paste)이 아닌 그래디언트 영역의 최적화이다. MPB는 타겟 이미지뿐만 아니라 소스 이미지의 경계 픽셀에 대한 의존성을 동시에 강화하여 기존 포아송 블렌딩의 한계인 '색상 번짐(Color Bleeding)'을 방지한다. 
수학적 정의: $$\Delta f = \text{div} \mathbf{v} \text{ over } \Omega, \text{ with } f|_{\partial\Omega} = f^*|_{\partial\Omega}$$
여기서 $f$는 합성될 결함 영역의 픽셀값이며, $f^*$는 배경 이미지의 경계 조건이다. 

3. Recommended Datasets (연구 대상 데이터셋)
연구의 구현 난이도와 학술적 가치를 고려하여 다음 데이터셋을 벤치마크 대상으로 선정한다.

Dataset	Domain	Characteristics	Complexity	Key Contribution
ISP-AD	Screen Printing	Structured patterns	Medium	Context-aware matching rules
MVTec AD	Object categories	Component-level ROI 	Low	Semantic-guided inpainting
DAGM 2007	Textures	Homogeneous textures	Very Low	Baseline verification


4. Proposed Framework Architecture (제안 시스템 구조)
Step 1: Automated ROI Extraction
Method: 기존 ROI extract 방식을 참조하여, SAM(Segment Anything Model) 또는 SURF를 사용하여 도메인별 배경 구성 요소를 분리한다.
Constraint: 배경의 기하학적 구조(예: 인쇄 선로, 제품 리드선)를 보존하는 마스크를 생성한다.

Step 2: One-Shot Defect Seed Generation
Model: TF-IDG (Training-Free Industrial Defect Generation).
Logic: 단일 결함 샘플로부터 특징 정렬(Feature Alignment)을 통해 다양한 변이체를 생성하며, 추가 학습 과정이 필요 없어 도메인 확장에 매우 유용하다.

Step 3: Context-Aware Layout Logic
Logic: '배경 질감-결함 형태' 매칭 매트릭스를 기반으로 물리적으로 타당한 ROI 내에 결함을 배치한다. 

Step 4: Seamless Synthesis via MPB
Implementation: 생성된 결함 패치를 MPB 기법으로 배경에 삽입하여 경계선 아티팩트를 제거하고 조명 조건을 일치시킨다.

5. Downstream Benchmarking Models (성능 평가 모델)
증강 데이터셋의 효용성을 입증하기 위해 최신 SOTA(State-of-the-Art) 모델을 활용한다.
Detection: YOLO11 (최신 C3k2 블록 기반 파라미터 효율성 검증), RT-DETRv2 (트랜스포머 기반의 전역 맥락 이해도 평가).
Classification Baseline: ResNet-50, Swin Transformer (기존 연구와의 정량적 비교 지표로 활용). 
Evaluation Metrics: mAP@0.5, F1-score, FID (Fréchet Inception Distance)를 통한 생성 품질의 통계적 입증.

6. SCI Publication Strategy (논문 투고 전략)
6.1 Technical Contributions
제조 현장의 초기 데이터 부재 문제를 해결하는 Training-Free 확장 프레임워크 제시.
결함 경계면의 품질을 획기적으로 개선한 MPB 기반의 정밀 합성 로직 공식화.
다양한 산업 도메인(ISP-AD, MVTec 등)에 대한 범용적 ROI 매칭 규칙 입증.

6.2 Essential Ablation Studies (필수 소거 연구)
Baseline vs Simple Copy-Paste vs Proposed MPB Synthesis.
Random Placement vs Context-Aware ROI Matching.