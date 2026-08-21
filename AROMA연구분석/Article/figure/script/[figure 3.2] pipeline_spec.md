# Figure 2 — AROMA Pipeline Architecture

## 목적 (연결 section)

§3.2 opening (현재 `section3_2.txt` 실제 문장): "The AROMA pipeline progresses from dataset profiling and complexity analysis to defect synthesis."

파이프라인의 입력(**5종** 산업 데이터셋) → 출력(YOLOv8n 다운스트림 검출)까지의 **data flow**를 시각화한다. 다이어그램의 스테이지 박스는 현행 §3.2의 **5개** 하위절(3.2.1~3.2.5)에 **1:1 정렬**한다.

> 🔴 **정렬 갱신(2026-08-21)** — §3.2.6(Quality Gate) 본문 삭제(T4)에 따라 **Quality Gate 스테이지 박스 제거**(6→5 스테이지). Eq.2 상수항 제거에 따라 stage 4 수식을 **무가중 합**으로 갱신: `ROI_score = ctx_prior + morph_prior`. 아래 2026-07-27 노트의 6-하위절·Quality Gate 서술은 구판 이력. 데이터셋 로스터는 `dataset_config.json`(정본), 생성 엔진은 **copy-paste**(ControlNet 아님)를 정본으로 한다.

> ⚠️ **정렬 갱신(2026-07-27, 사용자 확인)**
> 1. **§ 번호 재정렬**: 구 spec은 §3.2.3(ROI Extraction)·§3.2.4(Seed Defect Classification)를 분리했으나, 실제 `section3_2.txt`는 이 둘을 **§3.2.3 하나로 병합**(Background Categories and Defect Subtypes). 이후 ROI Selection→**§3.2.4**, Blending Synthesis→**§3.2.5**, Quality Gate→**§3.2.6**으로 한 칸씩 당겨짐(§3.2.7 없음, 총 **6개** 하위절).
> 2. **용어 통일**: `compat_sym` → `matrix_symmetric`.
> 3. **Quality Gate 내용 정정**: "합성 후 composite Q=0.5·artifact+0.5·blur 랭킹·최하위 fraction pruning" 서사는 코드에 없음(실측 확인, §3.2.6 대조). 실제는 **배경 patch quality gate** — blur/contrast/brightness/noise 4항 가중합(0.30/0.30/0.20/0.20), threshold 0.7 이상이면 accept. clean-bg 승인과 foreground-void(빈 배경) 거부에 쓰이며, 합성 완료본을 사후 필터링하지 않음.
> 4. **다이어그램 라벨 단순화**: 박스 설명(한 줄 detail) 제거, **제목만** 표시. 단순한 framework-pipeline 다이어그램으로 전환(상세 설명은 본 spec 표·§3.2 본문에서만 서술).

## 이미지 출처

기존 artifact: `D:\project\aroma\AROMA연구분석\Article\figure\image\[figure 3.2] aroma_pipeline.png`

**재생성 완료** (`[figure 3.2] aroma_pipeline.py`, 2026-07-27 — §3.2 6-하위절 정렬 + matrix_symmetric 용어 통일 + Quality Gate 내용 정정 + 라벨 제목-only 단순화 반영).

## 다이어그램 스테이지 (§3.2.1~3.2.5 정렬)

| # | §    | 박스 제목 | 한 줄 설명 (spec·본문 참고용 — 다이어그램 라벨에는 미표시) |
|---|------|-----------|-----------|
| 1 | 3.2.1 | Dataset Complexity Analysis | MCI / CCI from patch profiling |
| 2 | 3.2.2 | Morphology & Context Modeling | Data-driven clusters (GMM+BIC) + tertile context cells |
| 3 | 3.2.3 | ROI Extraction & Defect Subtype Classification | Otsu + connected-components ROI extraction; SAM seed masks → 5 morphology subtypes |
| 4 | 3.2.4 | ROI Selection & Compatibility-Aware Placement | ROI_score = ctx_prior + morph_prior (unweighted); symmetric compatibility gate (matrix_symmetric) |
| 5 | 3.2.5 | Blending Synthesis | Same blend operator for AROMA and Random arms; mask + bbox co-saved |

- **입력**: 데이터셋 **5종** — `severstal` · `mvtec_leather` · `mtd` · `aitex` · `kolektor`
- **출력**: YOLOv8n 지도학습 검출 헤드라인(baseline / random / aroma 3조건)

## 다이어그램 라벨링 규칙 (단순화 강화)

가독성을 위해 다이어그램 라벨은 다음을 **생략**한다(문서 텍스트·caption·본 표에는 유지, 그림 라벨에서만 제거):

- 스테이지 순번 배지
- **박스 한 줄 설명(detail)** — 제목만 표시 (2026-07-27 변경)
- 우측 참조 JSON/artifact 박스 전체
- 데이터셋 버전 표기, 실험 코드명(exp4v2 등)
- 상세 산출물명·절차

각 박스는 **제목(스테이지 이름)만** 유지. 상세 산출물명·절차는 spec 문서(본 파일)와 §3.2 본문에서만 서술.

## Caption 작성

**Figure 2.** AROMA pipeline architecture and data flow, aligned to §3.2. Inputs are the five industrial datasets (severstal, mvtec_leather, mtd, aitex, kolektor). The pipeline profiles dataset complexity (MCI, CCI; §3.2.1), builds data-driven morphology clusters and context cells (§3.2.2), extracts candidate ROIs and classifies seed defects into morphology subtypes (§3.2.3), and ranks defect–context pairs by ROI_score = ctx_prior + morph_prior under a symmetric compatibility gate (matrix_symmetric) with offline clean-background assignment (§3.2.4). The selected crops are composited by the same blend operator for the AROMA and random arms (§3.2.5) before the synthesized samples feed the downstream detector.

---

## 저장

- Figure: `[figure 3.2] aroma_pipeline.png` **재생성 완료**(`[figure 3.2] aroma_pipeline.py`).
- Caption: `section3_2.txt`의 `###Figure2. AROMA Pipeline` 캡션에 반영 시 위 Figure 2 caption 사용.
