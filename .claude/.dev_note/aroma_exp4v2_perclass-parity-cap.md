# AROMA exp4v2 — per-class stratified synth cap + 라벨화-후 arm 동수 보정

## 사용할 skills: feature-dev

---

## ⏸ 상태: 보류 (reviewer-triggered)

**결정(2026-07-11)**: 본 작업은 **지금 구현하지 않는다.** 논문/리뷰 과정에서 **리뷰어가 "arm 간 클래스 균형·라벨화 수율 차이" 문제로 report/재실험을 요청하는 경우에만** 수행한다.

**방법론적 근거(요약)**: exp4v2의 표준·정당한 통제는 **총 학습 예산 동수(현행 `synth_ratio` 전역 cap)**까지다. 합성 결함의 **클래스 분포·라벨화 수율은 AROMA 선택·배치의 결과(post-treatment)**이므로, 이를 강제로 arm 간 동일화하면 인과추론상 **bad control / post-treatment bias** — AROMA가 정당하게 이득 내는 경로(유용 클래스 선택·깨끗한 배치→라벨 추출 용이)를 스스로 제거해 결과를 null 쪽으로 편향시킨다. 따라서 본 작업은 **primary 비교를 대체하는 강제 parity가 아니라, 리뷰어 반박("aroma가 유리한 클래스 조합을 얻은 것뿐") 방어용 secondary robustness ablation**으로만 의미가 있다.

**당장 권장(구현 불요, 무비용)**: 강제 균등화 대신 **arm별 클래스 분포·라벨화 수율을 계측·보고 + per-class AP 제시**(현행 로그가 이미 상당 부분 제공: `n_synth_per_class`, distinct sources per-class). gain이 클래스 조합 때문인지 배치 품질 때문인지 **분해해서** 보이는 것이 가장 정직하며, 리뷰어 요청 전까지는 이걸로 충분.

**재개 조건**: 리뷰어가 명시적으로 클래스 균형/라벨 수율 통제를 요구 → 아래 설계를 `--label_parity`/`--stratified_cap` **선택 플래그**(default OFF, primary 결과 byte-identical 보존)로 구현하고, primary(총량 동수)와 **나란히** 보고.

아래 설계는 그 시점을 위한 **보존용 스펙**이다.

---

## 개요

exp4v2(YOLOv8n supervised detection)는 `baseline`/`random`/`aroma`(+`casda`) 조건의 mAP50을 비교해 AROMA 파이프라인의 기여를 판정한다. 비교가 공정하려면 arm 간 **합성 학습 데이터 개수가 같아야** 한다. 현행 코드는 `synth_ratio` 지정 시 전역 cap(`int(n_real_train*synth_ratio)`)으로 **전체 개수만** 맞추고, 클래스 무시 uniform 무작위 subsample을 쓴다. 그 결과 (a) multi-class에서 **클래스별 개수가 arm마다 달라질 수 있고**, (b) 라벨화 단계(`_write_yolo_labels`)에서 bbox 추출 실패분이 배경으로 빠지면서 **라벨화-후 실제 학습 라벨 수가 arm 간 불일치**할 수 있다.

본 작업은 (1) cap을 **per-class stratified**로 바꿔 arm 간 클래스별 개수를 동일하게 강제하고, (2) 라벨화-후(bbox 추출 성공 기준) arm 공통 min으로 **2차 보정**하여 실제 학습 라벨 수 parity까지 확보한다. `val` 평가셋은 불변(train만 조정), single-class 모드는 기존 동작 100% 보존.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `synth_by_cond[cond]` (조건별 합성 annotation 리스트)의 **구성·개수** — cap 단계에서 클래스별 층화 subsample로 교체.
- 학습 라벨 디렉토리에 실제 기록되는 합성 이미지/라벨 집합 — 라벨화-후 2차 trim으로 arm 간 동수화.
- 결과 JSON/로그 필드: `n_synth`, `n_train`, `n_synth_per_class`(라벨화-후 기준으로 갱신), `[SynthRatio]` 로그.

### 그 상태를 전제로 동작하는 기존 로직
- **synth_ratio cap** (`:2417-2434`) — 현재 uniform. 본 작업이 이 블록을 대체.
- **`_write_yolo_labels`** (`:1101-1191`) — bbox 추출 실패 시 `continue`(배경). 반환 `n_synth`(라벨화 이미지 수)를 소비하는 `n_train = n_real + n_synth`(`:2022`). 반환 시그니처 확장 시 이 소비처 갱신 필요.
- **`_resolve_synth_class`** (`:2005` 사용) — 클래스 그룹핑 키 재사용(신규 로직의 입력).
- **`--max_synth_per_ds` 절대 cap** (`:2327-2336`) — synth_ratio 미지정 경로. 상호작용 결정 필요(TODO 3).
- **per-seed 반복**(`--seeds`) — 각 seed 내부에서 arm parity가 성립해야 함(seed 교차 금지).

### delete/remove/revoke/bulk 계열 확인
- 파괴적 작업 아님. 단 **합성 학습셋 축소**(2차 trim)로 "더 적은 데이터" 상태를 만들 수 있음 → 그 축소가 arm **공통**으로 일어나야 공정(한 arm만 줄면 역-교란). 축소량은 로깅으로 투명화.

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — per-class stratified cap (`:2417-2434` 대체)

현행 uniform:
```python
if synth_ratio is not None:
    n_real_train = len(train_defect)
    for cond in ("random", "casda", "aroma"):
        anns = synth_by_cond.get(cond, [])
        cap = max(1, int(n_real_train * synth_ratio))
        if len(anns) > cap:
            rng_sub = random.Random(seed)
            synth_by_cond[cond] = rng_sub.sample(anns, cap)
```

개선안 (stratified):
- 전역 예산 `cap_total = max(1, int(n_real_train * synth_ratio))` 유지.
- 각 조건 anns를 `_resolve_synth_class(ann, name_to_id, dataset_key)`로 클래스별 그룹핑 → `avail[cond][cls]` 카운트.
- **arm 공통 `target_per_class[cls]`** 산정 — 후보 규칙(TODO 1에서 확정):
  - 규칙 A(엄격 parity): `target[cls] = min(cap_share[cls], min_{cond} avail[cond][cls])` — 모든 arm이 채울 수 있는 만큼만(가장 굶주린 arm 기준). arm 간 클래스별 **정확히 동수** 보장.
  - `cap_share[cls]`는 real train의 클래스 분포 비례(권장) 또는 균등 분할.
- 각 (cond, cls)에서 `random.Random(seed_derived).sample(group, target[cls])`로 결정적 추출. seed는 `(seed, cond, cls)` 파생으로 재현성 유지(단, arm 간 **개수는 동일**, 표본 정체성은 arm별 상이 — 정상).
- 예산 잔여(합이 cap_total 미만)는 로깅. 초과 배분 금지.
- 신규 헬퍼 함수 도입: `_stratified_cap(synth_by_cond, resolve_fn, cap_total, real_class_hist, seed) -> {cond: [ann...]}`.

### 2. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — 라벨화-후 동수 보정

- `_write_yolo_labels`(`:1101-1191`) 반환 확장: 현재 `n_synth`(int) → **라벨화 성공한 annotation 식별자(또는 ann 리스트)와 클래스별 성공 카운트**를 함께 반환. 최소 변경으로는 "성공한 ann들의 리스트"를 반환(호출부에서 클래스 집계).
- 호출 구조 변경: baseline 제외 조건들에 대해
  1. 1차로 각 arm 라벨화 → 성공 ann을 클래스별로 수집(`labeled[cond][cls]`).
  2. **arm 공통 `final_target[cls] = min_{cond} len(labeled[cond][cls])`** 산정.
  3. 각 arm에서 클래스별로 `final_target[cls]`만 남기고 **초과분의 라벨/이미지를 제거**(또는 애초에 staging을 2-pass로: 성공셋 확정 후 공통 min으로 stage). seed 고정 결정적 선택.
- 이때 `n_synth`/`n_train`/`n_synth_per_class`는 **최종(보정 후)** 값으로 로깅·JSON 기록(TODO 6, 정직성).
- 구현 편의상 2-pass가 어려우면: 1-pass 라벨화 후 초과 라벨파일+이미지를 삭제하는 방식도 가능(파일시스템 정리 주의).

### 3. Fallback / 축퇴 경로

- `class_mode != "multi"`: 기존 uniform cap + 기존 라벨화 경로 **완전 보존**(신규 로직 우회, byte-identical 목표).
- `synth_ratio is None`: 라벨화-후 보정을 적용할지 결정(TODO 3). 잠정 — cap 미적용 시에도 arm 간 라벨화-후 min 보정은 공정성상 유익하나, 기존 동작 변경폭이 커지므로 별도 플래그(`--label_parity`) 뒤로 두는 안 검토.
- casda arm 부재 데이터셋: 존재하는 arm들(random/aroma)만으로 parity.
- 특정 클래스가 어느 arm에서 available 0 → 그 클래스 공통 0(모든 arm에서 제외), 로깅.
- `_resolve_synth_class`가 None(-1 unresolved) 버킷: arm 공통 규칙으로 포함/제외 결정(TODO 1). 잠정 — 별도 클래스처럼 취급해 arm 간 동수화(제외 시 정보 손실).

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
  - cap 블록(`:2417-2434`) 대체 + `_stratified_cap` 헬퍼 신설
  - `_write_yolo_labels`(`:1101-1191`) 반환 확장 + 호출부(`:1978-2022`) 갱신
  - per-class 로깅/JSON 필드(`:1994-2021`, `:2213-2214`)를 라벨화-후 기준으로 갱신
- (문서) `AROMA연구분석/colab_execute_new/exp4v2_execute.md` — parity 서술·판정 셀에 "클래스별·라벨화-후 동수" 반영, 신규 플래그 있으면 사용법 추가

---

## 테스트 / 검증

CLAUDE.md: pytest 금지 — Colab exp4v2 재실행 + 로그/JSON 확인.

1. **multi-class(severstal 4-class)**: `[SynthRatio]` 로그에서 random·aroma의 **클래스별 개수 동일** 확인. `n_synth_per_class`(보정 후)가 arm 간 일치.
2. **라벨화-후 parity**: `n_synth`(최종)가 random==aroma. bbox 추출 성공률 차이가 있어도 공통 min으로 수렴.
3. **single mode(aitex/mtd 등)**: 결과가 **기존과 동일**(byte-identical) — 신규 로직 우회 확인.
4. **casda 부재 데이터셋**: 2-arm parity 정상.
5. **per-seed**: 각 seed 내 arm parity 성립, seed 간 표본은 달라도 무방.
6. **정직성**: 로그/JSON의 개수가 **실제 staged 학습 라벨 수**와 일치(사후 추정·창작 금지).
7. **회귀**: baseline/val 경로 불변, mAP 판정 셀(`Δ(aroma−random)`) 정상 동작.

---

## TODO / 미확정

- **TODO 1 — target_per_class 규칙 확정**: 규칙 A(min-across-arms, 엄격 동수) vs 클래스분포 비례 후 arm-min. -1(unresolved) 버킷 포함/제외. 권장은 규칙 A + real class 분포 비례 예산 + unresolved 포함.
- **TODO 2 — 라벨화-후 보정 구현 방식**: `_write_yolo_labels` 반환 확장(성공 ann 리스트) + 2-pass staging vs 1-pass 후 초과 파일 삭제. 성능·복잡도 트레이드오프 결정.
- **TODO 3 — synth_ratio=None 경로**: cap 미적용 시 라벨-parity 적용 여부. `--max_synth_per_ds`(:2327-2336)와의 상호작용. 신규 플래그(`--label_parity`) 도입할지.
- **TODO 4 — seed 파생 규칙**: `random.Random(seed)` 단일 vs `(seed, cond, cls)` 파생. arm 간 개수 동일·표본 독립 보장 방식 확정.
- **TODO 5 — casda 3-arm vs 2-arm**: 3-arm 동시 parity 시 min이 과하게 줄 수 있음(casda c2/c4 starvation이 전체를 끌어내림). casda 포함/별도 판정 여부.
- **TODO 6 — 공정성 vs 데이터량**: 엄격 동수는 "가장 굶주린 arm/클래스" 기준이라 **총 학습량이 줄 수** 있음 → near-ceiling이 아닌데도 신호가 약해질 위험. 축소량 로깅 + 필요 시 `--n_per_roi` 상향으로 available 늘리는 운영 지침.

---

## 참고 (관련 dev_note)

- `aroma_exp4v2_source-diversity-fairness.md` — "pool·cap budget 동일, selection 전략만 다름" 불변식(본 작업이 클래스·라벨화 층위까지 확장).
- `aroma_exp4v2_quality-gate-fairness.md` — 게이트 대칭(공정성 계열).
- `aroma_exp4v2_per-class-metrics.md` — per-class 지표 산출.
- `aroma_exp4v2_class-key-propagation.md` — class_key/`_resolve_synth_class` 근거.
