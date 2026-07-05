# Exp4v2 — epochs·patience 실측 재산정 파일럿 (L3 공통 절감, 코드 수정 0)

## (사용할 skills: micro-fix)

> 계열: low_compute_validation_plan.md **4순위**. **코드 수정 없음** — Colab 가이드 문서 1건 신설. 이후 모든 GPU run(real_frac 커브 포함)에 30–50% 절감이 곱해지는 horizontal 절감 장치.

## 개요

현행 `--baseline_epochs 300 --patience 50`은 보수적 상한. 실측 수렴점(best_epoch) 기반으로 상한을 재산정한다. 조기종료 기준이 전 조건 동일하므로 비교 공정성 불변.

**파일럿 구성 (workflow 비판 반영)**: severstal 1셀이 아니라 **"큰 것 1종(severstal) + 작은 것 1종(mtd)" 2셀 × 3조건** — 소규모 데이터셋은 epoch당 스텝이 적어 더 늦게 수렴할 수 있어 단일 외삽 위험. 이 파일럿 run은 이후 real_frac 커브의 100% 지점과 **겸용**(순증 GPU ≈ 0).

## 절차 (신규 가이드 `exp4v2_epoch_pilot_execute.md`)

1. **파일럿 실행**: severstal+mtd × 3조건, `--baseline_epochs 300 --patience 50 --seeds 1`(단일 seed — 수렴점 추정 목적), fresh output_dir(`$EXP4V2_OUT/pilot`). 기존 exp4v2 커맨드 그대로(코드 무수정).
2. **수렴 분석 (CPU 셀)**: 각 `{ds}/{model}/{cond}/results.csv`(ultralytics 산출)에서 `argmax(mAP50-95)` epoch 추출 → 조건·데이터셋별 best_epoch 표.
3. **상한 산정**: `new_epochs = ceil(max(best_epoch 전체) × 1.3)`, `--patience 30`. (max는 조건별 — synth 받은 arm이 늦게 수렴할 수 있음.)
4. **사후 게이트 (사전 명시 규칙)**: 이후 본실험에서 어떤 셀의 best_epoch가 `new_epochs × 0.9` 이상이면 그 셀은 **자동 재실행 대상**(상한이 그 셀에 binding — Δ가 조건-비대칭으로 왜곡될 위험). 게이트 위반 확인 셀 포함.
5. 산정값을 이후 run의 `--baseline_epochs/--patience`로 사용 — epochs 변경 시 fresh output_dir(resume skip-cache 규칙).

## 수정 대상 파일
- **신규** `AROMA연구분석/colab_execute/exp4v2_epoch_pilot_execute.md` (문서만)

## 한계 (정직)
- 절감 장치일 뿐 증거를 생산하지 않음. 파일럿 GPU 1–3시간은 real_frac 100% 지점과 겸용해 순증 최소화.
- 수렴점은 seed 종속 분산이 있음 — ×1.3 마진 + 사후 게이트가 방어선.
