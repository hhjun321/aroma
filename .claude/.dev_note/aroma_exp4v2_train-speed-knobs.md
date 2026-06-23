# Exp4v2 — YOLO 학습 속도 인자 (batch / cache / rect)

## (사용할 skills: micro-fix)

## 개요

exp4v2 속도 분석 결과(workflow wf_e948fb02), `model.train()`이 batch/cache/rect 미전달 → Ultralytics 기본(batch=16, cache=False, rect=False) 적용. Severstal(1600×256, train 5천+장)에서:
- batch=16 → A100 저활용(step 과다)
- cache=False → 매 epoch 디스크 재디코드
- rect=False → 정사각 letterbox로 ~84% 회색 패딩 연산 낭비(6.25:1 종횡비)

세 인자를 CLI로 노출해 model.train()에 전달. **기본값=현행(Ultralytics 기본)이라 미지정 시 byte-identical**. 전 method(baseline/random/aroma)에 동일 적용 → 비교 공정성·val 셋 불변.

## 영향도 분석

### 변경 상태
- argparse `--batch`, `--cache`, `--rect` 추가.
- `model.train()`에 batch/cache/rect 전달.
- run() / _run_detection_mode / _run_yolo_condition 시그니처에 3 인자 전파.

### 그 상태를 전제로 동작하는 기존 로직 (하위호환)
- 미지정 기본값 = Ultralytics 기본(batch=16, cache=False, rect=False) → 기존 동작 동일.
- 전 조건이 동일 값 사용 → method 비교 공정성 유지. val 셋·split 무관.

### 회귀 위험
- 기본값이 현행과 같으면 회귀 0. rect=True는 Ultralytics가 shuffle 비활성·mosaic 상호작용 — opt-in이라 기본 영향 없음.

## 수정 내용

### scripts/aroma/experiments/exp4_v2_supervised_detection.py

1. **argparse**:
   - `--batch` (type=int, default=16) — Ultralytics 기본. -1=auto(주의: method간 batch 달라질 수 있어 고정값 권장, help에 명시).
   - `--cache` (type=str, default="", choices 자유: ""/"ram"/"disk"/"True") — ""=미사용(False).
   - `--rect` (action="store_true", default False).
2. **시그니처 전파**: main → run(batch, cache, rect) → _run_detection_mode → _run_yolo_condition.
3. **`_do_train`/`model.train()`**: `batch=batch`, `rect=rect`, `cache=(cache or False)` 추가. cache 문자열 → ram/disk, 빈값 → False.
4. 전 조건(_run_yolo_condition은 조건마다 호출되나 동일 인자) → 공정.

## 수정 대상 파일
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`

## 암묵적 요구사항 (엣지)
- **기본값 현행 보존**: batch=16/cache False/rect False → 미지정 시 기존과 동일.
- **cache 파싱**: 빈 문자열/None → False. "ram"/"disk"/"True" 그대로 전달.
- **batch=-1(auto)**: Ultralytics가 VRAM 기준 자동 — method간 batch 달라질 수 있음 → help에 "고정값(64/128) 권장" 명시.
- **rect+augmentation**: rect=True 시 Ultralytics shuffle off. opt-in이며 전 조건 동일 적용이라 공정성 불변. 정확도 보통 중립~소폭.
- **RAM 부족**: cache='ram' OOM 시 'disk' 폴백 — 사용자 판단(문서).

## 테스트 (Colab, pytest 금지)
1. 미지정 실행 → 기존과 동일 동작(batch16/cache off/rect off) 확인.
2. `--batch 64 --cache ram --rect` → 로그 args에 반영, step 수 감소·속도 개선 체감.
3. 전 조건(baseline/random/aroma) 동일 인자 적용 확인(공정성).
4. py_compile.

## 미확정 TODO
- staging 3배 복사(#1 병목)는 별도 — 본 작업은 train 인자만.
- batch 기본값 16 유지 vs 64 상향 — 기본 16(현행 보존), 사용자가 opt-in. (논의 후 기본 상향 가능.)
