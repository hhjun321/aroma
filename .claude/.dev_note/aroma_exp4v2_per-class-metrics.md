# Exp4v2 — multi 모드 per-class metric 결과 기록

## (사용할 skills: micro-fix)

> 단일 관심사(per-class metric 기록), 단일 파일(`exp4_v2_supervised_detection.py`), 새 추상화·의존성 없음, 순수 additive. 결과추출부+집계부만 손대면 ~40줄. summary.md 표 렌더링까지 포함하면 상한(50줄)에 근접 → summary는 **선택 2차 항목**으로 분리(아래 §미확정 TODO).

## 개요

사용자가 `--class_mode multi`(severstal 전용, nc=4, names=`c1~c4`)로 학습했으나 `exp4v2_results.json`에는 클래스 평균(macro-average) 스칼라만 저장된다. 현재 결과추출부(`_run_yolo_condition`, L1567-1575)는 `box.map50`/`box.map`/`box.mp`/`box.mr`(전부 전 클래스 평균)만 기록 → 4개 결함 클래스 AP가 단일 값으로 뭉개진다. severstal multi 모드의 목적인 **클래스별(특히 희귀 Class2) AROMA vs Random 비교**를 결과 파일에서 직접 확인할 수 없다.

Ultralytics `val_results.box`에는 per-class 데이터가 이미 존재(`ap_class_index`, `class_result(i)`, `ap50`/`maps`/`p`/`r` 배열)하므로, 이를 꺼내 `out["per_class"]`로 기록한다. **`class_mode=="multi"` 일 때만** 추가하고 single 모드/타 데이터셋은 byte-identical 유지.

---

## 영향도 분석

### 변경 상태
- `_run_yolo_condition`의 `out` dict에 `per_class` 키 추가 (multi 모드 한정).
- multi-seed 집계(`_aggregate_seeds`)에서 per_class를 보존/집계.

### 그 상태를 전제로 동작하는 기존 로직 (하위호환)
- `plot_exp4v2_results.py` 및 결과 reader는 `results[ds][model][cond]["map50"]` 등 top-level 스칼라만 읽음 → per_class 추가는 **무시되어 안전**(키 추가는 비파괴적).
- `_run_yolo_condition`에 `class_mode`(L1377)·`box`(L1545) 이미 스코프 존재 → 시그니처 변경 불필요.
- single 모드(default)·타 데이터셋: per_class 미추가 → 기존 JSON byte-identical.

### 회귀 위험
- multi 모드에서만 키 1개 추가. single 경로 무변경 → 회귀 0.
- multi-seed 미보강 시 per_class가 top-level 집계에서 **누락**(현 `_aggregate_seeds`는 `_AGG_METRICS` 스칼라만 재구성, agg_cell을 scratch로 작성 L2050-2062). → 집계부 보강 필수(아래 §2).

---

## 수정 내용

### scripts/aroma/experiments/exp4_v2_supervised_detection.py

#### 1. 결과추출부 — `_run_yolo_condition` (L1545 `box` ~ L1575 `out`)

`out` dict 생성 직후, multi 모드일 때 per_class 추출 블록 추가.

```python
out = {
    "map50":        round(float(box.map50), 4),
    "map50_95":     round(float(box.map), 4),
    "precision":    round(float(box.mp), 4),
    "recall":       round(float(box.mr), 4),
    "n_train":      int(n_train),
    "n_real_train": int(n_real),
    "n_synth_train": int(n_synth),
}

# multi 모드: 클래스별 metric 기록 (single/타 데이터셋은 미추가 → byte-identical)
if class_mode == "multi":
    per_class: Dict[str, Dict[str, float]] = {}
    try:
        # val_results.names: {cls_id: name}; fallback ['c1'..'c4']
        names = getattr(val_results, "names", None) or {0:"c1",1:"c2",2:"c3",3:"c4"}
        idxs = list(getattr(box, "ap_class_index", []) or [])
        for i, cls_id in enumerate(idxs):
            cls_id = int(cls_id)
            p, r, ap50, ap = box.class_result(i)   # (P, R, AP@0.5, AP@0.5:0.95)
            name = names.get(cls_id, f"class{cls_id}") if isinstance(names, dict) \
                   else (names[cls_id] if 0 <= cls_id < len(names) else f"class{cls_id}")
            per_class[name] = {
                "map50":     round(float(ap50), 4),
                "map50_95":  round(float(ap), 4),
                "precision": round(float(p), 4),
                "recall":    round(float(r), 4),
            }
    except Exception as _pc_exc:
        logger.warning("    per-class extract failed: %s", _pc_exc)
    out["per_class"] = per_class   # 빈 dict 가능(검출 전무) — 키는 항상 존재
```

- `box.ap_class_index[i]` = i번째 행이 가리키는 **실제 클래스 id**, `box.class_result(i)` = 그 행의 (P, R, AP50, AP50-95). 두 인덱싱은 같은 i로 정합.
- val GT에 등장하지 않은 클래스는 `ap_class_index`에서 제외됨(정상) → per_class에 미포함.

#### 2. multi-seed 집계 — `_aggregate_seeds` (L1947, agg_cell 작성 L2050-2062)

현 `agg_cell`은 scratch로 작성되어 per_class를 버린다. per_class를 **클래스별 mean으로 집계**해 agg_cell에 추가한다(top-level이 mean인 기존 규약과 정합).

```python
# (per-seed 루프 안) 각 seed cell의 per_class 수집
#   pc_samples: {class_name: {metric: [seed별 값,...]}}
# (cells 루프 진입부에서 pc_samples = {} 초기화)
pc = cell.get("per_class")
if isinstance(pc, dict):
    for cname, cmetrics in pc.items():
        slot = pc_samples.setdefault(cname, {m: [] for m in _AGG_METRICS})
        for m in _AGG_METRICS:
            v = cmetrics.get(m)
            if isinstance(v, (int, float)):
                slot[m].append(float(v))

# (agg_cell 작성 직후) per_class mean 부착 — 하나라도 있으면
if pc_samples:
    per_class_mean = {}
    for cname, slot in pc_samples.items():
        per_class_mean[cname] = {
            m: round(float(np.mean(slot[m])), 4) if slot[m] else None
            for m in _AGG_METRICS
        }
    agg_cell["per_class"] = per_class_mean
```

- seed마다 등장 클래스 집합이 다를 수 있음 → 클래스명 **union**, 각 (class, metric)별로 값이 있는 seed만 평균.
- per-seed 원본 per_class는 `_seeds/seed{N}/exp4v2_results.json`에 그대로 보존(개별 실행 결과라 무변경).

---

#### 3. summary.md 렌더링 — `_per_class_block` 신규 + 두 빌더 호출 (후속 반영 완료)

신규 헬퍼 `_per_class_block(m_data, conds)`: 조건별 `per_class`에서 map50을 읽어 **rows=class, cols=조건 + Δ(A-R)** 표 생성. per_class가 어떤 조건에도 없으면(single/타 데이터셋) `[]` 반환 → 표 미출력(byte-identical). 클래스명 union·정렬, 누락 셀 N/A. `_build_summary`·`_build_summary_multiseed` 양쪽에서 delta 패널 직후 호출. 멀티시드는 클래스별 mean 사용(per-class std는 미추적 — 전체 값은 JSON 참조).

## 수정 대상 파일
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
  - §1 결과추출부 `_run_yolo_condition` (L1567 부근)
  - §2 집계부 `_aggregate_seeds` (L1947-2068)
  - §3 summary 빌더 `_per_class_block`(신규) + `_build_summary`·`_build_summary_multiseed` 호출

---

## 암묵적 요구사항 (엣지)

- **검출 전무**: `box.ap_class_index`가 비어 있으면 `per_class = {}` (빈 dict). 키는 multi 모드에서 항상 존재 → reader가 `.get("per_class", {})`로 안전.
- **GT에 없는 클래스**: ap_class_index에서 자동 제외. 강제로 c1~c4 전부 채우지 않음(0으로 메우면 "검출 0"과 "GT 없음" 구분 불가).
- **single/타 데이터셋 byte-identical**: `class_mode != "multi"`면 per_class 키 자체를 추가하지 않음(빈 dict도 X).
- **multi-seed 클래스 집합 불일치**: seed별 등장 클래스가 달라도 union + per-(class,metric) 평균으로 처리. 특정 클래스가 일부 seed에만 있으면 그 seed들만 평균.
- **resume 호환**: 기존 JSON에 per_class 없는 셀과 혼재 가능. per_class는 부가 정보라 skip/resume 판정(`map50` 유효성)에 영향 없음.
- **names 출처**: `val_results.names` 우선, 없으면 `{0:c1..3:c4}` fallback. dict/list 양쪽 방어.

---

## 테스트 (Colab, pytest 금지)

1. severstal `--class_mode multi --condition all --baseline_epochs 1`(스모크) → `exp4v2_results.json` 각 조건에 `per_class: {c1,c2,c3,c4 일부}` 존재 확인. 각 클래스 map50/map50_95/precision/recall 키 정상.
2. **macro 정합성**: per_class map50들의 (등장 클래스) 평균이 top-level `map50`과 근사(반올림 오차 범위)한지 확인.
3. single 모드(타 데이터셋, 예: mvtec_cable) → JSON에 per_class 키 **부재**(byte-identical) 확인.
4. multi-seed `--seeds 42 1 2` → top-level 집계 셀에 `per_class`(클래스별 mean) 존재, `_seeds/seed{N}/exp4v2_results.json`에 per-seed per_class 보존 확인.
5. py_compile.

---

## 미확정 TODO

- ~~**summary.md per-class 렌더링(선택 2차)**~~ → **반영 완료**(§3). `_per_class_block` 헬퍼로 per-class map50 sub-table을 두 빌더에 추가. map50_95/precision/recall 등 나머지 metric은 JSON 참조(표는 map50 + Δ(A-R)만).
- **per-seed per_class를 top-level `per_seed{}`에도 넣을지**: 현재 `per_seed_cell`은 스칼라만(bloat 방지). 필요 시 ps_metrics에 per_class 추가 검토(현재는 `_seeds/*`에 보존되므로 보류).
- severstal class id ↔ 결함 유형명 매핑(c1~c4가 어떤 Severstal defect class인지) 문서화는 별도.
