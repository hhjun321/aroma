# AITEX tiled 재실행 결과 분석 — 2026-07-06 (단일 클래스, 3-seed)

원본: `.claude/.etc/exp4v2/aitex/exp4v2_results.json`
설정: `aitex_tiled_rerun_execute.md` — 4096×256 → 256×256 타일(stride 128, 원본 이미지 단위 split으로 overlap leak 차단), 11클래스→단일 `defect`, seeds **42/1/2**, synth_ratio 1.0(cap 246 = n_real_train).

---

## 결과 (mAP50, mean±std, n_seeds=3)

| 조건 | mAP50 | precision | recall | n_train |
|------|-------|-----------|--------|---------|
| baseline | 0.372 ± 0.024 | 0.474 | 0.393 | 246 |
| random | 0.388 ± 0.026 | 0.406 | 0.433 | 246+246 |
| **aroma** | **0.485 ± 0.054** | **0.649** | **0.450** | 246+246 |

### Paired seed 분석 (같은 seed끼리 — 단순 std 비교보다 강함)

| 비교 | seed별 Δ | mean Δ | paired t(df=2) | 판정 |
|------|----------|--------|----------------|------|
| **aroma − random** | +0.079 / +0.072 / +0.139 | **+0.097** | **4.51** | **3/3 seed 일관 + t > 4.30 (양측 p<0.05)** ✅ |
| aroma − baseline | +0.062 / +0.079 / +0.198 | +0.113 | 2.64 | 3/3 일관, t는 경계(단측 p≈0.06) |
| random − baseline | −0.017 / +0.007 / +0.058 | +0.016 | 0.73 | noise — 방향도 비일관 |

## 핵심 발견

1. **★ aroma > random이 전 seed 일관 + paired t 유의** — 이 세션 최초의 **통계적으로 방어 가능한 downstream 우위**. 소표본(n=3)이지만 paired 방향 3/3 + t=4.51은 단순 2σ 룰(경계)보다 강한 신호.
2. **★ 논지 정합 — "선택의 가치" 분리**: random 합성은 baseline 대비 무효과(+0.016, noise·방향 비일관)인데 **동일 엔진·동일 budget의 aroma 합성은 +0.113** — 효과가 합성 자체가 아니라 **ROI 선택 전략에서** 나온다는 thesis를 aitex가 정확히 보여줌. precision(0.406→0.649)까지 동반 상승 = 노이즈 추가가 아닌 정보 추가.
3. **타일링 개입 성공**: baseline 0.372 — 비타일 시절 0.066(종횡비 붕괴)·스모크 0.0 대비 파이프라인이 정상 판정력 회복. union-bbox fix(점결함 70타일 보존)와 결합.
4. **budget parity 정상**: n_synth_train 246 = n_real_train (양 조건 동일).

## 유의점 (정직)

- **seed 규약 이탈**: 이 run은 42/1/2 — 세션 규약(1/2/43)과 불일치. 타 데이터셋과 통합 비교 시 규약 통일 필요(hygiene; 결과 자체는 paired라 내적으로 유효).
- **n=3 소표본**: aroma의 std(0.054)가 큼(seed2 0.544가 견인). 과대주장 금지 — "aitex(tiled)에서 3/3 seed 일관 + paired 유의" 수준으로 서술.
- **단일 클래스 전환의 대가**: per-class(희소 코드별) 관측은 포기 — aitex는 aggregate 전용. 4종 per-class 서사에서 aitex는 예외로 명시.
- 이 결과는 **비타일 aitex와 비교 불가**(평가셋 자체가 다름) — 커브/폭 실험에서 aitex는 tiled 기준으로 통일.

## 다음 액션

1. **regenerate 없이 즉시**: exp5/exp6를 `aitex`(tiled 경로) 대상으로 실행 — L2 증거(PRDC Recall/Coverage·kNN·rare)가 이 downstream 신호와 정합하는지 교차 확인. 정합하면 "기제→결과" 스토리 완성.
2. severstal·mtd·leather도 동일 프로토콜(3-seed) 결과 확보 → 4종 통합표. seed 규약(42/1/2 vs 1/2/43) 하나로 통일 결정.
3. real_frac 커브(`exp4v2_realfrac_execute.md`)의 소형 커브 데이터셋으로 mtd 대신/병행 **aitex(tiled) 채택 검토** — 판정력이 회복됐고 Δ가 커서 커브 신호가 선명할 가능성.
