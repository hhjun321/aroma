# Figure 2 — AROMA Pipeline Architecture

## 목적 (연결 section)

§3.2 opening (현재 `section3_2.txt` 실제 문장): "The AROMA pipeline progresses from dataset profiling and complexity analysis to defect synthesis and quality control."

파이프라인의 입력(**5종** 산업 데이터셋) → 출력(YOLOv8n 다운스트림 검출)까지의 **data flow**를 시각화한다. 다이어그램의 스테이지 박스는 현행 §3.2의 7개 하위절(3.2.1~3.2.7)에 **1:1 정렬**한다. 데이터셋 로스터는 `dataset_config.json`(정본), 생성 엔진은 **copy-paste**(ControlNet 아님)를 정본으로 한다.

> ⚠️ **정렬 갱신(2026-07-20, 사용자 확인)**
> 1. **스테이지 재편**: 그림 박스를 현행 §3.2 하위절 구조에 맞춰 재정의. 기존 "Complexity + Meta Policy(auto-select modeling policy)" 스테이지의 **Meta Policy Generator / 정책 자동선택 서사를 제거** — symmetric compatibility gate가 단독 spine/novelty ([[aroma-compat-gate-spine-reframe]] 메모리).
> 2. **ROI 스코어 정정**: 그림 detail을 현행 `ROI_score = 0.6·ctx_prior + 0.4·morph_prior`로 수정 (quality 항 없음; 구 spec의 `0.5·ctx + 0.3·morph + 0.2·quality`는 stale).
> 3. **데이터셋 개수**: 실제 **5종**(severstal / mvtec_leather / mtd / aitex / kolektor).

## 이미지 출처

기존 artifact: `D:\project\aroma\AROMA연구분석\Article\figure\image\[figure 3.2] aroma_pipeline.png`

**재생성 완료** (`[figure 3.2] aroma_pipeline.py`, 2026-07-20 — §3.2 7-하위절 정렬 + Meta Policy 서사 제거 + ROI_score 0.6/0.4 정정 반영).

## 다이어그램 스테이지 (§3.2.1~3.2.7 정렬)

| # | §    | 박스 제목 | 한 줄 설명 |
|---|------|-----------|-----------|
| 1 | 3.2.1 | Dataset Complexity Analysis | MCI / CCI from patch profiling |
| 2 | 3.2.2 | Morphology & Context Modeling | Data-driven clusters (GMM+BIC) + tertile context cells |
| 3 | 3.2.3 | ROI Extraction | Otsu + connected-components; texture categories |
| 4 | 3.2.4 | Seed Defect Classification | SAM masks → 5 morphology subtypes |
| 5 | 3.2.5 | ROI Selection & Placement | ROI_score = 0.6·ctx_prior + 0.4·morph_prior; symmetric compat gate |
| 6 | 3.2.6 | Blending Synthesis | Seamless copy-paste; AROMA vs Random arm |
| 7 | 3.2.7 | Quality Gate | Composite Q ≥ 0.7 filter |

- **입력**: 데이터셋 **5종** — `severstal` · `mvtec_leather` · `mtd` · `aitex` · `kolektor`
- **출력**: YOLOv8n 지도학습 검출 헤드라인(baseline / random / aroma 3조건)

## 다이어그램 라벨링 규칙 (단순화 유지)

가독성을 위해 다이어그램 라벨은 다음을 **생략**한다(문서 텍스트·caption에는 유지, 그림 라벨에서만 제거):

- 스테이지 순번 배지
- 우측 참조 JSON/artifact 박스 전체
- 데이터셋 버전 표기, 실험 코드명(exp4v2 등)
- 상세 산출물명·절차

각 박스는 제목(스테이지 이름)과 한 줄 설명만 유지. 상세 산출물명·절차는 spec 문서(본 파일)와 §3.2 본문에서만 서술.

## Caption 작성

**Figure 2.** AROMA pipeline architecture and data flow, aligned to §3.2. Inputs are the five industrial datasets (severstal, mvtec_leather, mtd, aitex, kolektor). The pipeline profiles dataset complexity (MCI, CCI; §3.2.1), builds data-driven morphology clusters and context cells (§3.2.2), extracts candidate ROIs (§3.2.3), classifies seed defects into morphology subtypes (§3.2.4), and ranks defect–context pairs by ROI_score = 0.6·ctx_prior + 0.4·morph_prior under a symmetric compatibility gate with offline clean-background assignment (§3.2.5). The selected crops are composited by seamless copy-paste for the AROMA arm and by naive placement for the random arm (§3.2.6), and a composite quality gate (Q ≥ 0.7) filters the synthesized samples (§3.2.7) that feed the downstream detector.

---

## 저장

- Figure: `[figure 3.2] aroma_pipeline.png` **재생성 완료**(`[figure 3.2] aroma_pipeline.py`).
- Caption: `section3_2.txt`의 `###Figure2. AROMA Pipeline` 캡션에 반영 시 위 Figure 2 caption 사용.
