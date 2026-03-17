**Considerations**
1. Training and experiments are conducted in the Google Colab environment.
2. Therefore, the research must be written by separating it into .py scripts for each stage.
3. The paths of files referenced at each stage must always be entered via the CLI.

1. Executive Summary
This study proposes a relationship (ROI)-based clustering and framework to address the chronic **data scarcity** of problematic inspection systems. 
(AROMA: Adaptive ROI-Aware Augmentation Framework)
It aims to extend beyond original steel defects to ISP-AD (screen printing) and MVTec AD (object) touch, combining **Modified Poisson Blending (MPB)** and **Training-Free Industrial Defect Generation (TF-IDG)** as core technologies. Through this, it simultaneously guarantees the originality of SCI-level papers and the robustness of model performance by drastically reducing data collection reports.

2. Theoretical Background & Novelty
2.1 Context-Aware Data Augmentation (CASDA)
Unlike traditional augmentation techniques, CASDA considers the physical context of defect occurrence. It ensures that the synthetic data generated through the logical combination of defect type and background follows the actual distribution.

2.2 Modified Poisson Blending 
Technical Advantage of (MPB): The key differentiating factor of this research is the optimization of the gradient region, rather than simple cut-and-paste. MPB prevents 'color bleeding,' a limitation of conventional Poisson blending, by simultaneously reinforcing dependence on boundary pixels of both the target image and the source image.

Mathematical definition: $$\Delta f = \text{div} \mathbf{v} \text{ over } \Omega, \text{ with } f|_{\partial\Omega} = f^*|_{\partial\Omega}$$
Here, $f$ is the pixel value of the defect area to be composited, and $f^*$ is the boundary condition of the background image.

3. Recommended Datasets 
The following datasets are selected as benchmarks, taking into account the difficulty of implementation and academic value of the research.

Dataset	Domain	Characteristics	Complexity	Key Contribution
ISP-AD	Screen Printing	Structured patterns	Medium	Context-aware matching rules
MVTec AD	Object categories	Component-level ROI 	Low	Semantic-guided inpainting
DAGM 2007	Textures	Homogeneous textures	Very Low	Baseline verification


4. Proposed Framework Architecture
Step 1: Automated ROI Extraction
Method: Referencing existing ROI extraction methods, domain-specific background components are separated using SAM (Segment Anything Model) or SURF.
Constraint: Creates a mask that preserves the geometric structure of the background (e.g., print lines, product lead lines).

Step 2: One-Shot Defect Seed Generation
Model: TF-IDG (Training-Free Industrial Defect Generation).
Logic: It generates various variants through feature alignment from a single defect sample, and is very useful for domain expansion as it does not require an additional training process.

Step 3: Context-Aware Layout Logic
Logic: Place defects within a physically valid ROI based on the 'background texture' matching matrix.

Step 4: Seamless Synthesis via MPB
Implementation: Insert the generated defect patch into the background using the MPB technique to remove boundary artifacts and match lighting conditions.

5. Downstream Benchmarking Models
To demonstrate the utility of augmented datasets, the latest SOTA (State-of-the-Art) models are utilized.Detection: YOLO11 (최신 C3k2 블록 기반 파라미터 효율성 검증), RT-DETRv2 (트랜스포머 기반의 전역 맥락 이해도 평가).
Classification Baseline: ResNet-50, Swin Transformer (used as a quantitative comparison metric with existing research).
Evaluation Metrics: Statistical verification of generation quality via mAP@0.5, F1-score, and FID (Fréchet Inception Distance).

6. SCI Publication Strategy
6.1 Technical Contributions
Presentation of a Training-Free extension framework to address the issue of initial data scarcity in manufacturing sites.
MPB-based precision synthesis logic formulation that dramatically improves defect interface quality.
Verification of universal ROI matching rules for various industry domains (ISP-AD, MVTec, etc.).

6.2 Essential Ablation Studies
Baseline vs Simple Copy-Paste vs Proposed MPB Synthesis.
Random Placement vs Context-Aware ROI Matching.