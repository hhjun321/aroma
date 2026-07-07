# H1 — copy-paste 재조합 무정보 가설: 증거 정리 (severstal flat 원인)

> **상태**: 사실상 확정 방향 (결정타 realfrac 곡선만 미실행)
> **조사일**: 2026-07-07 (exp4v2 20260705 결과 기반)
> **용도**: `scripts/aroma/generate_defects.py` 개선 설계의 근거 문서
> **원 데이터**: `.claude/.etc/_investigate_severstal/` (annotations 2종 + results.csv 3종 + 샘플 25쌍), `.claude/.etc/exp4v2/20260705/exp4v2_results.json`, `.claude/.etc/profiling/severstal/morphology_features.csv`

---

## 1. 가설 정의

**H1**: copy-paste 합성은 train 실 결함 crop을 그대로 재사용해 새 배경에 붙이는 **재조합**이다. 결함의 "외형(appearance)"에는 신규성이 0이고, 신규성은 오직 (결함, 배경) **조합**에서만 나온다. 데이터가 충분한 도메인(severstal, real 2534장)에서는 모델이 이미 모든 결함 외형을 학습했으므로 재조합이 주는 추가 정보가 없다. 소데이터 도메인(leather 64장, mtd 272장)에서는 재조합 자체가 유효 다양성이므로 효과가 있다.

**예측**: 합성 효과는 real 데이터 크기에 반비례한다. → 실측과 일치 (leather +7pp / mtd +1.3pp / severstal 0pp).

---

## 2. 증거 (독립 5개 + 보조 2개)

### E1. 재조합 구조 자체 (코드 사실)

- `generate_defects.copy_paste_synthesis`는 `roi_selected.json`의 실 결함 crop(`defect_bbox` + `defect_mask_path`)을 그대로 잘라 normal 배경에 blend. 픽셀 신규성 = blend 경계의 feather뿐.
- 합성 source 결함 종수: aroma 1690종 / random 1339종 (전체 실 결함 3620 중). **모두 train에서 이미 보는 외형** — severstal 2534장 학습이면 결함 외형은 전부 기지(旣知).
- 참고 leak: `roi_selected`는 exp4v2 seed 분할 **이전** 전체 pool에서 선정 → 합성 source에 val 결함 외형 포함(양 arm 동등). 외형 leak이 있는데도 flat = 외형 정보가 이미 포화라는 방증.

### E2. 거대 bbox — 재조합의 실효 다양성 자체가 작음 (정량)

severstal 이미지는 1600×256(=409,600px). 결함 bbox가 이미지 대부분을 차지:

| 분포 | real (n=3620) | synth aroma (n=5070) | synth random (n=5070) |
|---|---|---|---|
| bbox area med | 53,230 | 52,736 | 48,470 |
| p75 | 168,416 | 172,854 | 158,242 |
| max | 409,600 (전면) | 409,600 | 409,600 |

- med만 해도 이미지의 13%, p75는 41%, 검수 샘플엔 86% 짜리(1372×256)도 존재.
- **"새 배경에 붙인다"의 실체**: 이미지 대부분이 원본 결함 crop이고 가장자리만 새 배경 → 합성물 ≈ 원본 결함 이미지의 근소 변형. 조합 다양성조차 구조적으로 미미.
- 클래스별 real bbox med: c1 8,732 / c2 5,181 / c3 66,008 / c4 67,886 — c3/c4(다수 클래스)가 특히 거대.

### E3. 학습 곡선 — 합성은 "쉽고 정보 없음" (seed42 results.csv, 100 epochs)

| 조건 | best mAP50(@ep) | train cls loss(final) | val cls loss(min) | n_train |
|---|---|---|---|---|
| baseline | 0.4952 (@99) | 0.955 | 1.260 | 2,534 |
| random | 0.4938 (@95) | **0.770** | 1.220 | 5,068 |
| aroma | 0.5012 (@76) | **0.784** | 1.204 | 5,068 |

- 합성 arm은 epoch당 **2배 데이터**를 보면서 train loss가 baseline보다 ~20% **낮음** → 합성 샘플을 매우 쉽게 fit (신규 난이도 없음).
- gradient step 2배 투입의 효과 = 수렴 가속뿐 (aroma best @76ep vs baseline @99ep). 도달 천장은 동일 (~0.50).
- val cls loss는 합성 arm이 근소하게 낮음(1.20 vs 1.26)인데 mAP 무변 — 기지 패턴에 대한 확신도만 개선, 검출력 불변.
- 부수 발견: baseline이 100ep에서도 상승 중(0.443@50 → 0.491@100) — "완전 수렴 후 포화"가 아니라 **동일 epoch 예산 내에서 합성 기여 0**. 셋이 lockstep으로 상승.

### E4. 다중 run 일관성 — severstal 합성은 한 번도 도움된 적 없음

| run (로컬 보관) | baseline | random | aroma | 비고 |
|---|---|---|---|---|
| 20260705 (3-seed, 100ep) | 0.4975 | 0.4968 | 0.4970 | flat, std ≤0.010 |
| ratio1.0 구run | 0.3614 | 0.3556 | 0.4126 | 구 설정 |
| clean_bg 1.0 (3-seed) | 0.3900 | 0.3428 | 0.3565 | 합성 **역효과** |
| compounding (1-seed) | 0.3614 | 0.3750 | 0.3154 | casda 0.3259 |

설정이 제각각임에도 "합성이 유의미하게 이긴 적 없음"이 일관.

### E5. 효과-크기 반비례 (20260705 교차 데이터셋)

- leather (real 64): random +7.4pp, aroma +6.5pp
- mtd (real 272): random +1.1pp, aroma +1.4pp
- severstal (real 2534): 0pp
- H1 예측과 정확히 일치. 소데이터에서 재조합=유효 다양성, 대데이터에서 재조합=중복.

### 보조 E6. 라벨/parity 결함 아님 (반증 완료)

- 양 arm 5070/5070 전량 라벨 (mask 100%, bbox 100%, area<50 drop 0) → H4 기각.
- bbox 분포 real과 일치(위 표) → 라벨 기하 왜곡 없음.
- blend=alpha 양 arm 동일, budget parity 완전(n_synth_train 2534 동수).

### 보조 E7. 육안 검수

- 샘플 6장 오버레이: blend 자체는 정상, 경계 위화감 크지 않음 → "합성이 깨져서"가 아니라 "합성이 정상인데 정보가 없어서" flat.

### E8. leather 대조군 — 동일 엔진에서 재조합이 "정보"가 되는 조건 (2026-07-07 추가)

`.claude/.etc/_investigate_leather/` 동일 분석. 합성 파이프라인 상태는 severstal과 완전 동일(600/600 전량 라벨, drop 0, alpha, parity 완전)인데 효과는 +7pp:

| 항목 | severstal (0pp) | leather (+7pp) |
|---|---|---|
| real train | 2,534장 | 64장 |
| 합성 source 외형 | 1,690종 — 전부 기지 | 92종 = 실 결함 **전체** (top 재사용 9회) |
| bbox/이미지 비율 | med **13%**, p75 41%, max 100% | med **0.91%**, p75 2.1%, max 28% |
| 재조합의 실체 | 원본 결함 이미지의 근소 변형 | 진짜 신규 (결함, 배경) 조합 (배경 217종) |
| 클래스별 real bbox med | c1 8.7k / c2 5.2k / c3 66k / c4 68k | color 8.7k / cut 12k / fold 28k / glue 11k / poke 3.6k |

→ 같은 코드·같은 파라미터에서 재조합의 정보량은 (real 크기)⁻¹ × (bbox 비율)⁻¹에 의해 결정. H1의 양방향 검증.

### ⚠️ E9. val-외형 leak — leather/mtd 이득의 부풀림 위험 (미정량, 후속 확인 필요)

- 합성 source는 exp4v2 seed 분할 **이전** 전체 결함 pool에서 선정 (E1). leather는 실 결함이 92개뿐이라 **92개 전부** 합성 source로 사용됨 — val_frac 0.3이면 val ~28개의 "외형"이 새 배경에 붙어 train에 유입.
- 양 arm(random/aroma) 동등 → **arm 간 비교는 공정**. 그러나 "합성 vs baseline +7pp" 주장은 부분적으로 leak 기여 가능.
- severstal은 같은 구조인데 0pp → leak조차 무효 = 포화 방증 (H1 강화). leak이 실제 작동 가능한 곳은 소데이터 leather/mtd.
- **후속 정량법**: seed별 `_split_defects` val 목록 재현 → synth source ∩ val 비율 산출 → val-외형 합성 제외 재학습 delta. ControlNet arm 재비교 전에 확인 권장 — ControlNet arm도 seed mask·hint를 전체 pool에서 취하므로 동일 구조를 상속한다.

---

## 3. 반증된 대안 가설

| 가설 | 판정 | 근거 |
|---|---|---|
| H2 학습 동역학 이상 | 기각(이상 없음) | E3 — 정상 수렴, 곡선 동일 |
| H4 합성 라벨 품질/drop | 기각 | E6 |
| H6 아티팩트 지름길 | 미검(weights 필요) | E3의 train-loss 패턴이 부분 대체 — 어차피 easy-fit이면 지름길 여부와 무관하게 정보 없음 |
| 순수 데이터 포화(더 못 배움) | 부분 기각 | E3 — 100ep에서 아직 상승 중. "합성이 못 돕는 것"이지 "더 배울 게 없는 것" 아님 |

**미확정**: realfrac 곡선(baseline `--real_frac 0.25/0.5/0.75`, 100ep) — real 데이터 절반으로도 ~0.49면 "실데이터 정보 포화" 확정, H1 완결.

---

## 4. generate_defects.py 개선에의 시사점

핵심: **개선 목표는 "합성물의 외형 신규성"이지 배치·블렌딩·라벨 품질이 아니다** (후자는 전부 정상 판정).

1. **외형 신규성 주입이 유일한 레버** — 실 crop 재사용을 벗어나 결함 텍스처를 생성해야 함. → `--method controlnet` (2026-07-06 구현, f45c689)이 정확히 이 지점. severstal에서 aroma-CN > random이 나오면 H1의 개선판 검증이 된다.
2. **거대 bbox 문제 (E2)는 ControlNet arm에도 그대로 상속됨** — GT mask=seed mask 설계라 bbox 기하는 real 재사용. 이미지의 40~86%를 덮는 crop은 "생성물"이어도 배경 조합 다양성이 여전히 낮다. 개선 옵션:
   - (a) ROI 선택/합성 시 bbox area 상한 필터 또는 대형 결함의 부분(sub-region) 합성
   - (b) 대형 결함은 n_per_roi를 낮추고 소형 결함(c1/c2, med 5~9k px)에 합성 예산 집중 — c2(최희소, AP 최저 0.33)가 가장 개선 여지 큰 표적
3. **소형·희소 클래스 표적화**: severstal에서 유일하게 "배울 게 남은" 곳은 c2(recall 0.33~0.40)와 소형 결함. 균등/랜덤 재조합 대신 **약점-표적 합성**(per-class AP 기반 배분)이 재조합으로도 미미하게나마 신호를 만들 수 있는 유일한 경로.
4. **easy-fit 완화 (E3)**: 합성 샘플이 너무 쉬우면 gradient 기여가 조기 소멸. 생성 기반이라도 hard-example 성격(저대비, 소형, 경계 모호)을 유지해야 함 — blank 판정 임계(`_CN_BLANK_*`)를 과도하게 올려 "선명한 결함만 통과"시키면 easy-fit을 재생산한다.
5. **평가 유의**: 개선 실험은 반드시 20260705 이식 + **epochs 100** 고정 (`controlnet_aroma_arm_execute.md` STEP 8-B, 2026-07-07 수정). epoch 예산을 바꾸면 E3의 lockstep 상승 때문에 절대치가 같이 움직여 비교 무효.
6. **train-pool-only source 옵션 (E9)**: 합성 source를 exp4v2 train split 결함으로 제한하는 옵션(`--source_split train` 류)을 검토 — val-외형 leak을 차단해 "합성 vs baseline" 주장을 방어. 단 split은 exp4v2 seed에 종속이라 generate_defects 단독으로는 알 수 없음 → (a) exp4v2 쪽에서 seed별 train image_id 목록을 export하고 generate_defects가 필터로 소비, 또는 (b) 논문에선 leak 정량 보고로 대체. ControlNet arm에도 동일 적용.

---

## 5. 관련 파일

- 합성 엔진: `scripts/aroma/generate_defects.py` (copy_paste + controlnet arm)
- 실행 가이드: `AROMA연구분석/colab_execute/controlnet_aroma_arm_execute.md`
- 원 증거: `.claude/.etc/_investigate_severstal/`, `.claude/.etc/exp4v2/20260705/exp4v2_results.json`
- 관련 dev_note: `aroma_step4_generate-defects.md` (엔진 원 설계)
