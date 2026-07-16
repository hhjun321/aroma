# Figure 1 — AROMA Pipeline Architecture

## 목적 (연결 section)

§3.2 opening (현재 `section3_2.txt` 실제 문장): "The AROMA pipeline was organized as a sequence of stages that progress from dataset-level complexity quantification, through automated policy selection, to defect synthesis and quality control."

파이프라인의 입력(v2-1 **5종** 산업 데이터셋) → 출력(YOLOv8n 다운스트림 검출)까지의 **data flow**를 시각화한다. 데이터셋 로스터는 `dataset_config.json`(정본) + 사용자 확인 기준, 생성 엔진은 사용자 확인 기준 **copy-paste**(ControlNet 아님)를 정본으로 한다.

> ⚠️ **정정 이력(2026-07-16, 사용자 확인)**
> 1. **데이터셋 개수**: `AROMA_FRAMEWORK/08-Datasets.md`·`01-Overview.md`는 "v2-1 4종(severstal/mvtec_leather/mtd/aitex)"이라 서술하지만 stale. 실제는 **5종**, `kolektor`(single-class, 금속 commutator) 포함.
> 2. **생성 엔진**: `AROMA_FRAMEWORK/05-Stage3-ControlNet-Generation.md`는 ControlNet(SD v1.5 fine-tune)을 주 엔진으로 서술하지만, 사용자 확인 결과 **공식 흐름은 copy-paste 방식**(ControlNet 미사용)이다. 이에 따라 **Prompt Generation 스테이지를 제거**(ControlNet conditioning 전용 산출물이라 copy-paste 흐름에 역할 없음)하고, ControlNet 학습 스테이지를 copy-paste 변형 생성(elastic-warp) + compat 게이트 τ 사전스캔으로 교체했다.
>
> `AROMA_FRAMEWORK/` 노트는 두 항목 모두에서 실제와 어긋났다 — 노트를 정본으로 맹신하지 말 것([[feedback_framework_notes_stale]] 메모리 참고).

## 이미지 출처

기존 artifact: `D:\project\aroma\AROMA연구분석\Article\figure\image\[figure1] aroma_pipeline.png`

**재생성 완료** (`figure1_aroma_pipeline.py`, 2026-07-16, 4차 갱신 — 5종 데이터셋 + copy-paste 엔진 + prompt generation 제거 + 라벨 추가 단순화 반영).

## 이미지 내용 확인 (dataset_config.json + 사용자 확인 기준 현행 파이프라인)

- **입력**: v2-1 데이터셋 **5종** — `severstal`(multi, 4 class) · `mvtec_leather`(multi, 5 class) · `mtd`(multi, 5 class) · `aitex`(single, 256×256/stride128 tiled) · `kolektor`(single, 금속 commutator)
- **Prepare Datasets**: 5종 원본을 AROMA/MVTec 레이아웃으로 정규화 + `dataset_config.json` 등록 (`prepare_severstal.py` / `prepare_mtd.py` / `prepare_aitex.py` / `prepare_kolektor.py`; `mvtec_leather`는 준비 불요)
- **Distribution Profiling**: 결함 형태(morphology)·배경 맥락(context) 분포 프로파일링 → `compatibility_matrix.json`(legacy `matrix` + clean-grounded `matrix_symmetric`/`P_def_patch`/`clean_dist`), `morphology_clusters.json`, `deficit_analysis.json`
- **Complexity + Meta Policy**: MCI(Morphology Complexity Index) / CCI(Context Complexity Index) 산출 + Meta Policy Generator가 morphology/context 모델링 정책 자동 선택 → `complexity_report.json`
- **ROI Selection**: `deficit_aware` 2-phase 할당 + `realism` 스코어(`0.5·ctx_prior + 0.3·morph_prior + 0.2·quality`)로 top-k=200 ROI 선별 → `roi_candidates.json` / `roi_selected.json`. multi 3종(severstal/mvtec_leather/mtd)은 class-stratified 게이트 추가, single 2종(aitex/kolektor)은 게이트 제거
- **Clean-BG Selection**: 프로파일링 파생 신호(hist-matching + void 필터)로 ROI별 clean 배경 오프라인 선정 → `clean_bg_selected.json` / `clean_bg_random_arm.json`
- **Elastic-Warp Variants + tau Prescan**: 실 결함 crop을 seed로 subtype별 elastic-warp(`cv2.remap`) 변형 생성(copy-paste, training-free) + CPU로 compat 게이트 τ 사전확정(+ aitex AR/텍스처 임계) → `compat_tau_prescan_{ds}.json`
- **Generate: AROMA / Random arm**:
  - **AROMA arm**: copy-paste 결함(원본 crop + elastic-warp 변형) + symmetric compat 게이트 + clean-bg 게이트 + seamless(`cv2.seamlessClone` + Reinhard 색 이전) 블렌딩 → `synth_aroma/{ds}`
  - **random arm**: naive baseline(placement grounding·게이트 전부 우회, 기본 ON) → `synth_random/{ds}` — AROMA smart-placement 기여를 격리하는 대조군
- **출력**: YOLOv8n 지도학습 검출 헤드라인(baseline/random/aroma 3조건 fresh 학습)

## 이전 버전 대비 변경 요약

| 항목 | 최초 spec (stale) | 1차 현행화(잘못됨: ControlNet) | 현행(3차, copy-paste 확정) |
| --- | --- | --- | --- |
| Stage 구성 | 6 stage + Phase0 Meta Policy | 8 stage(prompt generation 포함) | Prompt Generation 제거, ControlNet 학습 스테이지 → Elastic-Warp Variants + τ Prescan |
| 생성 엔진 | Training-free elastic warping (TF-IDG), No GAN/Diffusion | ControlNet(SD v1.5 fine-tune) 주 엔진 | **copy-paste**(elastic-warp 변형, training-free) — ControlNet 미사용 |
| 입력 데이터셋 | isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb4 | v2-1 4종 | v2-1 **5종**: severstal · mvtec_leather · mtd · aitex · kolektor |
| 다운스트림 | 이상탐지(PatchCore/SuperSimpleNet/EfficientAD/RD++) | YOLOv8n 지도학습 검출 | YOLOv8n 지도학습 검출(변경 없음) |

## Caption 작성

**Figure 1.** AROMA pipeline architecture and data flow. Inputs are the five v2-1 industrial datasets (severstal, mvtec_leather, mtd, aitex, kolektor); profiling and complexity analysis quantify dataset characteristics (MCI, CCI) that drive the Meta Policy Generator's selection of morphology/context modeling policy. ROI selection follows a deficit-aware realism score, with an offline step assigning clean backgrounds. Real defect crops are then expanded into subtype-specific elastic-warp variants (training-free copy-paste, no learned generator) and the symmetric compatibility threshold τ is fixed by CPU pre-scan. The final stage synthesizes the AROMA arm (copy-paste composition, symmetric compatibility gate, clean-background gate, seamless blending) and the random arm (naive placement baseline, grounding and gates bypassed by design), which together feed the downstream detector.

## 다이어그램 라벨링 규칙 (2026-07-16 사용자 지시로 단순화)

가독성을 위해 다이어그램 라벨은 다음을 **생략**한다(문서 텍스트·caption에는 유지, 그림 라벨에서만 제거):

- 스테이지 순번 배지(`-1`, `0`, `1`, ... 뱃지 원)
- 우측 참조 JSON/artifact 박스 전체
- 상단 입력 박스의 "v2-1" 표기
- 하단 출력 박스의 "exp4v2" 표기
- Prepare Datasets 박스의 `(aitex: 256/stride128 tiled)` 괄호 detail
- ROI Selection 박스의 "deficit-aware" 표기(top-k/realism만 유지)
- Elastic-Warp Variants 박스의 tau prescan 서술(박스 제목·detail 모두 τ 언급 제거)

각 박스는 제목(스테이지 이름)과 한 줄 설명만 유지. 상세 산출물명·순번·τ 사전스캔 절차는 spec 문서(본 파일)와 caption에서만 서술.

---

## 저장

- Figure: `[figure1] aroma_pipeline.png` **재생성 완료**(`figure1_aroma_pipeline.py`) — 5종 데이터셋 + copy-paste 엔진 + prompt-generation 제거 반영판
- Caption: `section3_2.txt`에 반영 시 현재 파일의 Figure 1 캡션(구 서술)도 함께 갱신 필요 — 이번 세션은 `section3_2.txt` 미수정(사용자 결정), 후속 작업으로 남김
