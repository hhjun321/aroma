# Exp4v2 소표본 detection — val_frac 재설정 및 synth:real 비율 subsampling 도입

---

## (사용할 skills: feature-dev)

## 개요

mvtec_cable yolov8n 결과에서 AROMA precision=0.19 붕괴 및 random mAP50=0.21 저하가 관측됨.
원인은 두 가지: (1) synth:real = 600:46 = 13:1로 합성 과잉 주입 → 도메인 shift로 precision 붕괴,
(2) val_frac=0.5 → train 46장만으로 소표본 학습 제약.
재합성 없이 `--max_synth_per_ds` subsampling + val_frac 기본값 조정으로 즉시 개선한다.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `synth_by_cond[cond]` 리스트 크기 (annotation 개수)
- `val_frac` 기본값 → `_split_defects()` 호출 시 train/val 분리 비율
- `exp4v2_results.json`의 `n_synth_train` 값

### 그 상태를 전제로 동작하는 기존 로직
- `_local_cache_for_yolo()`: synth annotation 리스트 순회하며 이미지 복사 — subsample 전에 호출되면 전량 복사 후 버리는 낭비 발생
- `_write_yolo_labels()`: annotation 개수만큼 bbox 추출 실행
- resume skip 판정: `n_synth_train` 변경 시 이전 캐시가 다른 비율로 기록되어 있을 수 있음

### delete / remove / revoke / bulk 계열 여부
- subsampling은 in-memory만 변경 (Drive 파일 삭제 없음)
- `--max_synth_per_ds None` (default) → 기존 동작 100% 보존, 회귀 없음

---

## 이론적 근거

### val_frac 최적값 도출

**적용 이론**
1. **Small-sample learning curve**: detector AP는 train 샘플 수에 대해 로그-선형 수확체감. n이 수십 장 수준에서는 train 1장 감소 비용 > val 1장 증가 이득 → train 최대화 방향이 유리.
2. **YOLO 권장 최소 train**: Ultralytics 가이드 클래스당 최소 수십 장 이상. n=92는 최소선 부근 → train을 이 선 위로 끌어올리는 것이 1차 목표.
3. **Val 신뢰도 (표본 복잡도)**: mAP 표준오차 ≈ √(p(1−p)/N). 신뢰구간 ±0.07~0.1 확보에 val ≈ 25~50장 필요.

**n=92 적용**
- val_frac=0.3 → val≈28 (신뢰구간 하한 충족), train≈64 (+39% vs 현재 46)
- val_frac=0.2 → val≈18 (신뢰구간 하한 미달, 평가 노이즈 증가)
- **결론: val_frac = 0.3** (train≈64 / val≈28)

> TODO: 인용 논문 원문 재확인 필요 (learning curve 기울기 수치, Ultralytics 권장 수치). 논문 본문 기재 전 검증.  
> TODO: val_frac ∈ {0.2, 0.3, 0.4} ablation으로 경험적 뒷받침 여부 결정.

### synth:real 비율 최적값 도출

**적용 이론**
1. **Copy-paste augmentation 권장 비율**: 합성 증강 연구(Cut-Paste-and-Learn, Simple Copy-Paste 등)에서 synth:real ≈ 1:1 ~ 3:1 구간이 이득, 이를 초과하면 도메인 shift 손실 전환.
2. **Domain adaptation**: 합성 비율 높을수록 학습 분포가 합성 도메인으로 편향 → real test에서 covariate shift. precision 붕괴(합성 artifact 오탐)는 이 shift의 직접 증상.
3. **Class/domain imbalance theory**: 다수 도메인이 소수를 압도하면 결정경계 편향. 균형 비율로 환원이 표준 처방.

**n_real_train≈64(val_frac=0.3 적용 후) 적용**
- 권장 상한 3:1 → max_synth ≈ 3×64 ≈ **192**
- 균형점 2:1 → ≈ **128**
- 보수적 1:1 → ≈ **64**
- **결론: `--max_synth_per_ds 128~192`** (1차 후보). precision 회복 우선이면 1:1~2:1.

> TODO: 인용 논문 원문 재확인 (copy-paste 권장 비율 수치). 논문 본문 기재 전 검증.  
> TODO: max_synth_per_ds ∈ {64, 128, 192, 600} 스윕으로 precision-recall 곡선 제시 여부 결정.

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `--max_synth_per_ds` 파라미터 추가

**`_parse_args()`에 추가**:
```python
parser.add_argument(
    "--max_synth_per_ds", type=int, default=None,
    help="데이터셋·조건당 synth annotation 최대 사용 수. "
         "None=제한 없음(기본). 지정 시 random.sample로 subsampling (seed 적용)."
)
```

**`_run_detection_mode()` synth_by_cond 구성 직후, local cache 전에 삽입**:
```python
if max_synth_per_ds is not None:
    for cond in ("random", "aroma"):
        anns = synth_by_cond[cond]
        if len(anns) > max_synth_per_ds:
            rng_sub = random.Random(seed)   # 조건별 동일 seed → 공정 비교
            synth_by_cond[cond] = rng_sub.sample(anns, max_synth_per_ds)
            logger.info(
                f"  [SubSample] {ds_key}/{cond}: {len(anns)} → {max_synth_per_ds}"
            )
```

**`run()` / `main()` 시그니처 전파**: `max_synth_per_ds` 인자 추가.

### 2. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `val_frac` 기본값 변경

`_parse_args()`:
```python
# 변경 전
parser.add_argument("--val_frac", type=float, default=0.5, ...)
# 변경 후
parser.add_argument("--val_frac", type=float, default=0.3, ...)
```

### 3. `AROMA연구분석/colab_execute/exp4v2_execute.md` — 신규 파라미터 설명 추가

```
--val_frac 0.3           # train≈64 / val≈28 (n=92 기준)
--max_synth_per_ds 128   # synth:real ≈ 2:1 (권장 구간 1:1~3:1)
```

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
- `AROMA연구분析/colab_execute/exp4v2_execute.md`

---

## 테스트

CLAUDE.md 정책: pytest 금지, Colab에서 직접 검증.

1. **회귀(None)**: `--max_synth_per_ds` 미지정 → 로그에 n_synth=600 확인, 기존 동작 동일.
2. **subsampling**: `--max_synth_per_ds 128` → 로그 `[SubSample] mvtec_cable/random: 600 → 128` / `aroma: 600 → 128` 확인.
3. **재현성**: 동일 seed 2회 실행 → 동일 map50.
4. **경계**: `--max_synth_per_ds 5000` (>600) → no-op, 전량 사용.
5. **val_frac=0.3**: 로그 `real defect split — train=64  val=28` 확인.
6. **성능 회복**: AROMA precision > 0.19, aroma mAP50 > random mAP50 패턴 유지.

---

## 미확정 사항 (TODO)

- [ ] `--max_synth_per_ds` 절대값 vs `--synth_real_ratio` 배수 인자 — 데이터셋마다 real 수 달라도 비율 일정하게 유지하려면 배수가 유리. 현재는 절대값로 구현.
- [ ] subsampling 후 random/aroma 개수 불균형 처리 — 한 조건 annotation이 max 미만이면 비교 비대칭 → 로그에 실제 사용 개수 명시로 대응.
- [ ] 이론 인용 논문 원문 재확인 (§ 이론적 근거 TODO 참조).
- [ ] val_frac / max_synth ablation 스윕 수행 여부 결정.
- [ ] 결과 확정 후 AROMA.txt §4.3 / Table 12 갱신.
