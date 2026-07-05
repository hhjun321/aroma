# Exp4v2 결과 분석 — 2026-07-04 run (multi-class 스모크)

원본: `.claude/.etc/20260704/exp4v2_results.json` + `exp4v2_log`
조건: baseline / random / aroma 3조건 (**casda 제외 — 평가에서 제거 확정**)

---

## ⚠️ 판정: 본실험 아님 — epochs=1 스모크 테스트

로그의 모든 `fit start`가 **`epochs=1`** (seed 42 단일, imgsz=640, val_frac=0.3, synth cap=1267≈2534×0.5 → synth_ratio 0.5). 1 epoch YOLO는 수렴 근처도 아니므로 **절대 mAP 수치로 성능 결론을 내릴 수 없다.** 본 문서는 (1) multi-class 일반화 파이프라인 검증, (2) 본실험 전 조치 항목 도출이 목적이다.

| 항목 | 값 |
|------|-----|
| 데이터셋 | severstal, aitex, mtd (**mvtec_leather 부재** — aroma 합성 미생성 상태) |
| epochs / seed | **1** / 42 단일 |
| synth_ratio | 0.5 (severstal cap 1267) |
| casda | 미포함 (제거 방침과 일치) |

---

## 1. 파이프라인 검증 — multi-class 일반화 정상 동작 ✅

이번 run의 실질 성과. 4일 구현한 데이터셋-일반 multi-class가 end-to-end로 동작함을 확인:

| 검증 항목 | 결과 |
|-----------|------|
| per_class 키 | severstal `c1~c4`(표시명 유지 ✓), aitex 결함코드(`002`~`030`), mtd 클래스명(blowhole~uneven) — 설계대로 |
| **budget parity** | `n_synth_train` random==aroma: severstal 1267/1267, aitex 35/35, mtd 136/136 ✓ |
| synth class 해소 | `n_synth_per_class`가 전 데이터셋에서 다중 클래스로 분포 — generic `test/{type}` resolver 동작 ✓ (aitex는 idx 0~10, 즉 nc>4 정상 매핑 = severstal 하드코딩 탈피 확인) |
| real val 라벨 multi-class | severstal/mtd per_class AP·recall이 클래스별로 상이 → val GT가 클래스 분리됨(C1 fix 유효) ✓ |
| val GT check | 3 데이터셋 모두 staged_val_labels 정상 (1086/17/…) |

**aitex per_class에 8개 클래스만 등장**(012종 중 025/027/029/036 부재): val 17장에 미등장 — "GT 없음은 미포함" 설계대로. 희소 클래스는 본실험에서도 대부분 판정 불가할 것(예상된 한계).

## 2. 수치 스냅샷 (참고용 — 1 epoch, 결론 금지)

### map50

| 데이터셋 | baseline | random | aroma | Δ(A−R) |
|----------|----------|--------|-------|--------|
| severstal | 0.056 | 0.064 | **0.113** | +0.049 |
| aitex | 0.000 | 0.000 | 0.000 | 0 |
| mtd | 0.005 | 0.009 | 0.006 | −0.003 |

- **severstal**: aroma가 random·baseline의 ~2배 — 방향은 고무적이나 **1 epoch 단일 seed는 노이즈 지배**. random은 precision 0.81/recall 0.03(거의 무검출), aroma·baseline은 recall형(0.45~0.48/precision<0.01) — 1 epoch에서 학습이 조건별로 전혀 다른 미성숙 상태에 있다는 신호일 뿐.
- severstal per_class: c2=0.0 전 조건(희귀클래스, 1 epoch에선 당연), c3/c4에서 aroma(0.24/0.20)가 baseline·random(≈0.10~0.13) 상회 — 본실험에서 재확인할 관찰 포인트.
- **aitex 전 조건 0.0**: 1 epoch + real_train 50장 + 4096×256 극단 종횡비면 예상 범위 — 단 아래 aitex 상세 분석의 구조적 이슈 2건은 epochs와 무관하므로 본실험 전 점검 필요.

### aitex 상세 분석

로그 정밀 확인 결과, 0.0 자체보다 중요한 **구조적 관찰 3건**:

**(1) real 라벨 34% 손실 — split 71/31 → 라벨 50/17**
`masks_matched=102`(전량 매칭)인데 YOLO 라벨은 67개만 생성 — **35장(34%)이 bbox 추출에서 탈락**. mask는 있으나 `min_area=50` 미달(4096×256에서 실 결함이 수px 폭)이 유력. val이 31→**17장**으로 줄어 per-class 판정력이 더 약화됨. → **조치**: aitex의 defect 스케일 대비 `min_area` 적정성 점검(라벨 빌드 진단 로그 `no_bbox` 카운트 확인), 필요시 min_area 하향 또는 이미지 종횡 스케일 보정.

**(2) synth 라벨 bbox 파편화 (심각) — 이미지당 수십 개 box**
train label instance 카운트:

| 조건 | real만(baseline) | random(+35 synth) | aroma(+35 synth) |
|------|------------------|--------------------|-------------------|
| 총 instance | 58 | 536 | **1,065** |
| idx4(019) | 7 | **467** (synth 17장에서) | **924** (synth 13장에서) |

synth 13~17장이 instance 460~920개를 만들었다 = **이미지당 ~27–70개 bbox**. 합성 결함 1개가 mask/diff 추출에서 수십 조각으로 파편화되고 있다는 뜻 (aroma idx7은 synth 1장 → 69 instance). 이 상태로는 019 클래스가 학습을 지배하고 tiny-box 노이즈가 detector를 오염 — **epochs를 올려도 aitex 결과 신뢰 불가**. → **조치(본실험 전 필수)**: aitex synth 라벨 `.txt` 몇 개와 해당 mask를 육안 확인, 파편화 원인(mask 다중 성분 vs diff 노이즈) 규명 후 synth bbox 추출에 largest-component 또는 min_area 상향/모폴로지 병합 적용 검토.

**(3) 정상 동작 확인 항목**
- nc=12 열거·synth class 해소 정상: train instance가 idx 0~11에 분포, synth per-class **미해소(-1) = 0건**(100% 파싱).
- parity 정상: cap 35 = int(71×0.5), random==aroma.
- val 17장에 8/12 클래스만 등장(희소 4종 부재) — per-class는 aggregate 보조로만 해석.
- **mtd 전 조건 ≈0.01**: 동일 — 1 epoch 미성숙.

## 3. 본실험 전 조치 항목

1. **mvtec_leather aroma 합성 생성** (step4) — 현재 결과에 leather 자체가 없음. 4종 완성 필수.
2. **본실험 파라미터로 재실행**: `--baseline_epochs 300 --patience 50 --seeds 1 2 43` (seed 규약) + `--batch 64 --cache ram --rect` + fresh output_dir(현재 `exp4v2/test`는 스모크 전용으로 격리).
3. **severstal synth_ratio**: 이번 cap 1267(ratio 0.5)은 aroma pool 600 문제가 해소된 상태(재생성 완료 확인 — n_synth_train 1267 도달). ratio 1.0(cap 2534)로 갈 경우 pool ≥ 2534 확인 후 진행.
4. **casda 제거 반영**: 실행 커맨드에서 `--casda_synthetic_dir` 미지정 → `--condition all`이어도 casda는 no_synth로 자연 제외되나, 문서/집계 스크립트에서 3조건 기준으로 정리(별도 작업).
5. ~~aitex 구조 이슈 2건~~ → **원인 규명 + 코드 수정 완료 (2026-07-05)**. Colab 실측으로 확정: mask 유효픽셀 median 20px(max 43) < min_area 50 → 210/600이 mask 0-box → **diff 폴백이 직물 텍스처에서 mean 40.4(max 394) 파편 bbox 양산**. 수정(`exp4_v2_supervised_detection.py`): ① `_mask_to_bboxes`에 union-bbox 폴백(전 성분 미달 시 유효픽셀 합집합 1-box — real 35장·synth 210장 회수), ② mask 존재 시 diff/template 폴백 금지(mask=authoritative), ③ real 캐시 schema v2→v3(자동 재빌드). 재실행 시 로그에서 aitex `train label instances per class`가 정상 규모(수십)인지 확인할 것.
6. aitex는 해결 후에도 val 17장·희소클래스 다수로 per-class 판정력 제한 — aggregate 위주 해석.

## 4. 결론

- **파이프라인**: multi-class 4종(현재 3종 실측) 일반화·parity·per-class 산출 모두 정상 — 구현 검증 완료.
- **성능**: 판정 보류(epochs=1). severstal의 aroma 우위 신호는 본실험(300ep × 3seed)에서 검증할 가설로만 기록.
