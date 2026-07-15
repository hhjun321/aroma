# Figure 1 — AROMA Pipeline Architecture

## 목적 (연결 section)

§3.2 opening: "The overall architecture and the data flow between the eight stages are illustrated in Figure~\ref{fig:pipeline}."

파이프라인의 입력(Multi-domain Industrial Datasets) → 출력(Downstream Anomaly Detection Model)까지 6개 stage(+Phase 0 Meta Policy Generator)의 **data flow**와 각 stage의 **output artifacts**를 시각화.

## 이미지 출처

기존 artifact: `D:\project\aroma\AROMA연구분석\Article\figure\image\[figure1] aroma_pipeline.png`

(원본은 제시된 이미지. 로컬에 이미 존재.)

## 이미지 내용 확인 (제시 이미지 기준)

- **입력**: "Multi-domain Industrial Datasets" (isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb4)
- **Phase 0**: Dataset Complexity Analysis (MCI, CCI 계산)
- **Meta Policy Generator**: auto-selects clustering policy
- **Stage 1**: ROI Extraction → roi_metadata.json
- **Stage 1b**: Seed Defect Classification → seed_profile.json
- **Stage 2**: TF-IDG Elastic Warping (Training-Free, No GAN, No Diffusion)
- **Stage 3**: Defect-Aware Placement Optimization → placement_map.json
- **Stage 4**: Poisson Blending Synthesis → 512×512 synthetic image + ROI mask
- **Stage 5**: Quality Gate (artifact_score + blur_score, threshold Q ≥ 0.7)
- **Stage 6**: Augmented Training Dataset (600 synthetic images per condition)
- **출력**: "Downstream Anomaly Detection Model" (PatchCore, SuperSimpleNet, EfficientAD, ReverseDistillation++)

수치는 섹션 텍스트와 정합.

## Caption 작성

**Figure 1.** AROMA pipeline architecture and data flow. Inputs are multi-domain industrial datasets; the Phase 0 complexity analysis quantifies dataset characteristics (MCI, CCI) that drive Meta Policy Generator selection of the clustering and placement strategies. The six stages, annotated with stage-specific operations and output artifacts, transform seed defects and ROI candidates into synthetic defect composites; a quality gate filters low-fidelity samples before the augmented dataset (baseline + random-ROI + AROMA-ROI conditions, 600 synthetic images per condition) feeds the downstream anomaly detector. The right-side boxes list the per-stage artifacts (JSON metadata, image outputs) committed for auditability.

---

## 저장

- Figure: 기존 `[figure1] aroma_pipeline.png` 재사용 (해상도 확인 필요)
- Caption: section3_2.txt에 추가 (##3.2 소개 문단 뒤)
