# AROMA Exp 3 (논문 Exp 1 + Exp 2) — 생성 품질 평가 명세서

**명확성**: 89/100 (2라운드)  
**작성일**: 2026-06-16

---

## 한 줄 요약

AROMA ROI selection이 Random 대비 더 나은 합성 결함 이미지를 생성하고, 그 결과 anomaly detection 성능을 향상시킨다는 인과 체인을 FID + PaDiM AUROC로 입증한다.

---

## 대상 (Who)

SCI 논문 심사자 및 산업 검사 도메인 연구자.  
"ROI 선택 전략이 합성 데이터 품질에 실제로 영향을 미치는가?"라는 질문에 정량적 근거가 필요한 독자.

---

## 배경 (Why)

AROMA는 결함 합성 시 ROI(배치 위치)를 어디에 선택하느냐가 합성 품질을 결정한다는 가설에 기반한다.  
기존 연구들은 합성 이미지의 시각적 품질(FID) 또는 downstream 성능(AUROC)을 단일 지표로만 평가했다.  
**고품질 시각 합성이 반드시 탐지 성능 향상으로 이어지지 않는다.** 따라서 두 지표 모두 필요하다.

데이터셋 선택 근거: 산업 도메인 다양성 + 형태/맥락 복잡도 범위 커버.

| 데이터셋 | 도메인 | 특성 |
|---------|--------|------|
| isp_LSM_1 | ISP 센서 표면 검사 | 미세 패턴, 다수 정상 샘플 |
| mvtec_cable | 산업 케이블 | 다중 결함 유형, 공개 벤치마크 |
| visa_cashew | 농산물 | 유기적 형태, 맥락 다양성 |
| visa_pcb4 | 전자 PCB | 구조적 결함, 고밀도 패턴 |

---

## 논문 실험 구조 (What)

```
Experiment 0: ROI Modeling Validation
  지표: morphology_coverage, context_coverage, rare_pair_coverage, entropy, gini
  스크립트: exp2_roi_quality.py

       ↓ (ROI 선택이 실제로 다양한가?)

Experiment 1: Synthetic Quality Validation
  지표: FID (real defect patch vs synthetic defect patch)
  스크립트: exp3_generation_quality.py --mode fid

       ↓ (좋은 ROI → 좋은 합성 이미지인가?)

Experiment 2: Downstream Utility Validation
  지표: Image AUROC, Pixel AUROC (PaDiM ResNet-18)
  스크립트: exp3_generation_quality.py --mode ad
```

**인과 체인**:  
`AROMA ROI → Better Coverage → Better Synthetic Defects → Better Anomaly Detection`

---

## 범위 (Scope)

**포함**:
- 4개 데이터셋 (isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb4)
- 비교 조건: Random ROI vs AROMA ROI
- FID: 결함 bbox crop 기반 patch 비교 (full image 아님)
- AD: 3조건 (baseline / random / aroma), PaDiM ResNet-18, seed=42
- 출력: `exp3_results.json` + `exp3_summary.md` (테이블 + delta 섹션 + FID delta)

**제외**:
- CASDA 비교 (Severstal 전용, 별도 Exp 1 devnote)
- Test set에 합성 이미지 포함 (절대 금지)
- 다른 AD 모델 (FastFlow, PatchCore 등 — 후속 연구)
- 통계 검정 (p-value, bootstrap CI) — 현재 범위 외, 추후 추가 가능

---

## 성공 기준 (Measure)

| 지표 | 성공 방향 | 의미 있는 차이 기준 |
|------|----------|-----------------|
| FID | AROMA < Random | 10 포인트 이상 개선 |
| Image AUROC | AROMA > Random > Baseline | Δ ≥ 0.03 |
| Pixel AUROC | AROMA > Random > Baseline | Δ ≥ 0.03 |

> **인과 체인 검증**: Exp 0에서 coverage가 높은 데이터셋에서 Exp 1 FID 개선이 크고, Exp 2 AUROC 개선도 큰 패턴이 관찰되면 인과 주장 강화.

소표본 경고: `fid_unstable: true` (n<50) → 해당 데이터셋 FID는 참고값으로만 보고.

---

## 리스크 (Risk)

| 리스크 | 발생 시나리오 | 대응 |
|--------|------------|------|
| FID 불안정 | real patch n<50 | unstable 플래그 + 논문 footnote |
| Train/Test 누수 | synthetic이 test에 혼입 | `_prepare_ad_dataset_with_masks` 에서 test/defect = real only 강제 |
| 인과 체인 역전 | AROMA FID 개선 但 AUROC 미개선 | Copy-paste limitation 명시로 방어, ROI coverage 분리 기여 강조 |
| PaDiM 재현성 | seed 고정에도 결과 변동 | seed=42 고정, anomalib Engine seed_everything 사용 |
| 도메인 gap 편향 | visa (no disk split) vs isp/mvtec (disk split) | VisA 80/20 seed=42 split 명시, 논문 appendix에 분리 방법 기술 |

**검증 체크리스트** (Colab 실행 전):
- [ ] `synthetic_aroma/{ds}/images/` 이미지 수 ≥ 50 (4개 데이터셋)
- [ ] `synthetic_random/{ds}/images/` 이미지 수 ≥ 50
- [ ] test/defect 에 합성 이미지 경로 없음 (`annotations.json` source 확인)
- [ ] GPU 사용 가능 확인 (`torch.cuda.is_available()`)

---

## 미확인 항목 (TODO)

- 통계 검정 방법 미결정 (bootstrap CI? paired t-test?) — 결과 확보 후 결정
- 논문 figure 형식 미정 (bar chart? table only?)
- 4개 도메인 간 결과 일관성 없을 경우 처리 방침 미결

---

## 원본 devnote

`D:\project\aroma\.claude\.dev_note\aroma_exp3_generation-quality.md`
