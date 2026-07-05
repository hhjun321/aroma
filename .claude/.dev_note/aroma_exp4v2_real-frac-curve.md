# Exp4v2 — `--real_frac` label-efficiency 커브 (최소 downstream, L3)

## (사용할 skills: feature-dev)

> 계열: low_compute_validation_plan.md **5순위**, [[aroma_exp4v2_epoch-pilot]](epochs 산정 선행). exp4v2 스크립트에 `--real_frac` 인자 신설(~30줄) + 실행 가이드.

## 개요

downstream 인과(L3)의 마지막 고리를 **단일 full-budget 점 비교 대신 real 25/50/100% × 3조건 커브**(severstal+mtd 2종)로 재구성:
- "real이 희소할수록 AROMA-선택 합성이 gap을 더 메운다"는 증강 연구 표준 논증 축 획득.
- 25/50% 지점은 데이터가 적어 학습·조기종료가 빨라 **총 GPU가 flat 3seed보다 적음** (추정 5–8h).
- 도메인 폭(aitex/mvtec_leather)은 **100% 지점 single-point로만** 커버 — "커브 2종 + 폭 4종" 역할 분리.

**주 주장 = Δ(aroma−random) 커브로 한정** (workflow 비판): 25/50%에서 제외된 real에 합성 ROI 소스가 포함될 수 있는 leakage는 **양 arm에 대칭**이라 A-vs-R은 유효. vs-baseline "few-shot 해결" 서사는 부풀림 — 각주로만. ROI/합성 pool 구성을 논문에 투명 기술.

**seed 정책 반전** (비판 반영): 분산 작은 severstal은 **1–2 seed + val bootstrap CI 보조**, 분산 큰 소형(mtd)은 **3 seed(1 2 43) 유지** — "절감은 분산이 작은 곳에서만". bootstrap CI는 시각화 보조(학습 확률성 분산 미포착 — 유의성 주 근거 금지).

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `--real_frac` (기본 1.0)

- **삽입 지점**: orchestrator에서 `_split_defects` 직후. `train_defect`를 seeded subsample:
  ```python
  if real_frac < 1.0 and train_defect:
      k = max(1, int(round(len(train_defect) * real_frac)))
      rng_rf = random.Random(seed)          # run seed 종속 — seed별 다른 부분집합
      train_defect = sorted(rng_rf.sample(train_defect, k))
      logger.info("  [RealFrac] %s: train %d -> %d (frac=%.2f)", ds, n_before, k, real_frac)
  ```
- **val 무변경** (load-bearing): 평가셋은 전 frac 지점에서 동일해야 커브가 비교 가능 — subsample은 train만.
- **synth cap 자동 축소**: 기존 `[SynthRatio]` cap이 `len(train_defect)` 기반이므로 subsample 후 자동 재계산(코드 무수정) — 확인만.
- **YOLO real 캐시 격리**: `_build_or_load_real_yolo_dataset`의 `cache_leaf`에 `real_frac<1.0`이면 `_frac{f}` suffix — full 캐시와 상호 무효화 thrash 방지(`n_source_defects` 불일치로 매번 재빌드되는 것 차단).
- **background negatives**: 현행 유지(train_normal 전량, 조건 대칭) — frac과 무관, 문서에 명시.
- **결과/resume 충돌**: results key가 dataset 단위라 frac 지점별 **fresh `--output_dir`**(`frac025`/`frac050`/`frac100`) 강제 — 가이드 규칙.
- argparse: `--real_frac` float default 1.0, 도움말에 Δ(A−R) 커브 목적·val 불변 명시. `main()`→`run_detection` 스레딩.

### 2. 신규 `AROMA연구분석/colab_execute/exp4v2_realfrac_execute.md`

workflow 형식: STEP 0 env → 1 (선행: epoch 파일럿 산정값·leather 합성·aitex schema v3 재빌드) → 2 커브 실행(severstal+mtd × frac{0.25,0.5} — 100%는 파일럿/도메인폭 run 겸용) → 3 도메인 폭(4종 × 1.0) → 4 결과 집계(지점별 Δ(A−R) + 단조성 확인 + 소형 3seed mean±std) → 판정 기준.
- **10% 지점 제외**(mtd 결함 수 한 자릿수 split 불안정), AUBC 단일 스칼라 대신 **지점별 Δ + 단조성**.
- seed 정책 표: severstal `--seeds 1 2` / mtd·aitex·leather `--seeds 1 2 43`.
- strict 변형(25% 지점 1개, ROI 소스를 retained real로 제한해 leakage 크기 실측) = **선택 각주** — 본 구현 범위 밖(TODO).

## 수정 대상 파일
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (`--real_frac` + cache leaf suffix)
- **신규** `AROMA연구분석/colab_execute/exp4v2_realfrac_execute.md`

## 영향도 분석
- **기본 1.0 = byte-identical**: subsample 블록·cache suffix 모두 `real_frac<1.0` 가드 → 기존 동작 불변(회귀 0).
- subsample이 `train_defect`만 바꾸므로 하위 전 경로(라벨 빌드·synth cap·multi-class 열거는 test_defect 기반이라 **불변**)와 정합. ⚠️ 열거(`class_names`)는 test_defect 전체 기반이라 frac과 무관 — id 안정성 유지.
- seed 종속 subsample: `--seeds 1 2 43` 시 seed별 다른 real 부분집합 — 의도된 동작(분산에 subsample 변동 포함).

## 테스트 (Colab, pytest 금지)
1. `py_compile`.
2. 스모크: severstal `--real_frac 0.25 --baseline_epochs 1` → 로그 `[RealFrac] train 2534 -> 634`, `[SynthRatio] cap`이 634 기준으로 재계산, val 수 불변(1086) 확인.
3. `--real_frac 1.0`(기본) → 기존 로그와 동일(subsample 로그 부재, cache leaf 무변경).
4. frac 캐시 격리: 0.25 실행 후 1.0 재실행 시 full 캐시 재빌드 없이 hit.

## 미확정 (TODO)
1. strict leakage 변형(25% 1지점, ROI 소스 제한) — 본 커브 결과 본 뒤 필요성 판단.
2. severstal val bootstrap CI 시각화 셀 — 집계 단계에서 추가(plot 스크립트 범위).
