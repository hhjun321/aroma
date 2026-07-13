# Exp4v2 — YOLO 학습 속도 인자 확장 (workers / compile) + A100 가이드 상향

## (사용할 skills: feature-dev)

## 개요

exp4v2 학습시간 단축(A100). 기존 `aroma_exp4v2_train-speed-knobs.md`(batch/cache/rect 노출)의 **연장**. 웹 조사(2026-07) 결과 두 신규 레버를 추가한다:

- **`--workers`** (dataloader worker 수): 현재 `model.train()`에 미전달 → Ultralytics 기본 8. CLI로 노출해 A100 런타임 vCPU 여유에 맞춰 상향(가이드 12).
- **`--compile`** (torch.compile inductor): Ultralytics 네이티브 train 인자(최근 8.3.x). ~14% 속도↑. **opt-in**(기본 off).

동시에 실행 가이드(`exp4v2_execute.md`)의 A100 파라미터를 상향: `--batch 64→128`(YOLO26 A100 레시피 일치), `--patience 50→25`(실질 early-stop, 특히 aitex 300ep), `--workers 12`·`--compile` 추가.

**모델 교체 없음**: 웹 확인 결과 YOLOv8 은 2026 Ultralytics 완전 지원 → `yolov8n` 유지(논문 비교실험 정체성 보존).

## 영향도 분석

### 이 기능이 변경하는 상태
- argparse에 `--workers`(int, default=8), `--compile`(store_true, default False) 추가.
- 전파 체인 5지점 시그니처/호출 수정: `_parse_args → main → run → _run_detection_mode → _run_yolo_condition`(closure `_do_train`이 지역변수 캡처).
- `model.train()`에 `workers=` 전달, `compile`은 **조건부** 주입(아래 fallback 참조).
- 가이드 문서 파라미터 값 변경(코드 무관).

### 그 상태를 전제로 동작하는 기존 로직 (하위호환)
- `workers` 기본 8 = Ultralytics 기본 → 미지정 시 **동작 불변**(byte-identical).
- `compile` 기본 off → 미지정 시 train kwargs에 `compile` 키 자체를 넣지 않음 → 기존 동작 불변.
- batch/cache/rect 경로(기존 노트) 그대로 재사용 — 동일 5지점에 인자만 추가.

### delete/remove 계열 아님 — 해당 없음

### 회귀 위험
- 모든 신규 기본값이 현행과 동일 → 미지정 회귀 0.
- **compile 키 검증 하드에러**: Ultralytics `check_dict_alignment`가 미지원 kwargs를 SyntaxError로 abort. 구버전(8.3.x 미만)은 `compile` 키 존재만으로 실패 → **무조건 전달 금지**. `--compile` 지정 시에만 kwargs에 주입하고, 실패 시 compile 제거 후 1회 재시도(fallback)로 방어.

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — 인자 노출·전파

**argparse** (`_parse_args`, `--rect` 뒤 ~L3282 근처):
- `--workers` (type=int, default=8) — "dataloader worker 수. 기본 8=Ultralytics 기본(미지정 시 동작 불변). A100 고vCPU 런타임에서 12 권장."
- `--compile` (action="store_true", default False) — "torch.compile(inductor) 활성. ~14%↑이나 조합마다 warmup. torch>=2.0 + 최근 Ultralytics 필요. 실패 시 자동 fallback."

**전파 5지점** (batch가 지나는 것과 **정확히 동일 경로**, 전부 keyword 전달):
1. `main`(L3322): `run(... workers=args.workers, compile=args.compile)`
2. `run`(L3045): 시그니처 `workers: int = 8, compile: bool = False` 추가 → `_run_detection_mode(... workers=workers, compile=compile)`
3. `_run_detection_mode`(L2256): 시그니처 추가 → `_run_yolo_condition(... workers=workers, compile=compile)`
4. `_run_yolo_condition`(L1898): 시그니처 `workers: int = 8, compile: bool = False` 추가 (patience/rect 근처 L1912-1915). closure `_do_train`이 이 지역변수를 캡처하므로 `_do_train` 자체 시그니처는 무변경.

**`model.train()` 호출부** (`_do_train`, L2116-2132):
- `workers=workers` 추가(batch=batch L2129 옆).
- `compile`은 **조건부 kwargs**로 처리 (직접 `compile=compile` 하드코딩 금지):
  ```python
  train_kwargs = dict(
      data=yaml_path, epochs=epochs, imgsz=imgsz, project=checkpoint_dir,
      name=condition, seed=seed, verbose=False, plots=False, save=do_save,
      device=device, exist_ok=True, patience=patience, batch=batch,
      cache=train_cache, rect=rect, workers=workers,
  )
  if compile:
      train_kwargs["compile"] = True
  try:
      return model.train(**train_kwargs)
  except (SyntaxError, TypeError) as exc:
      if "compile" in train_kwargs:
          logger.warning("compile 미지원 Ultralytics(%s) — compile 제거 후 재시도", exc)
          train_kwargs.pop("compile", None)
          return model.train(**train_kwargs)
      raise
  ```
  (SyntaxError = check_dict_alignment 미지원 키, TypeError = 구 시그니처. compile 없을 땐 재-raise.)

### 2. `AROMA연구분석/colab_execute_new/exp4v2_execute.md` — A100 파라미터 상향

- **STEP 2 (severstal·mvtec_leather·mtd, 그룹 A)**: `--batch 64 → 128`, `--patience 50 → 25`, `--workers 12` 추가, `--compile` 추가. 본문 "그룹 A 파라미터" 문장도 동기화.
- **STEP 3 (aitex, 그룹 B)**: 동일 (`--batch 128`, `--patience 25`, `--workers 12`, `--compile`).
- STEP 4 무결성 섹션의 frozen-param 목록(`--patience`, batch 등)이 등장하면 새 값과 정합되게 갱신. imgsz/epochs/rect/class_mode/seeds/synth_ratio는 **불변**.

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
- `AROMA연구분석/colab_execute_new/exp4v2_execute.md`

## 암묵적 요구사항 (엣지)

- **기본값 현행 보존**: workers=8, compile off → 미지정 실행은 기존과 동일.
- **compile 미지원 방어**: `--compile` 지정 + 구 Ultralytics → SyntaxError/TypeError catch → compile 제거 후 재시도 + warn. compile 없는 실패는 정상 전파(가림 금지).
- **전파 누락 = 조용한 no-op**: 모든 keyword 전달. 5지점 중 하나라도 누락 시 기본값(8/False)이 조용히 train에 도달 → 크래시 아니라 무효화. 4 시그니처 + 3 forwarding 호출 전부 갱신 확인.
- **공정성(비교실험 핵심)**: workers/compile/batch/patience 모두 baseline/random/aroma **3조건 동일** 값. `_run_yolo_condition`은 조건마다 호출되나 동일 인자 → 공정. val 셋·split 불변.
- **frozen-param 재조정**: batch/patience 변경 → 기존 부분결과와 **혼합 불가**. fresh `--output_dir` 또는 기존 삭제 후 재실행(사후튜닝 아님, 실행 전 확정).
- **compile 재컴파일 비용**: 조합(seed×ds×cond)마다 새 YOLO+train → run당 1회 warmup. 100~300ep 장기 run엔 상각되어 net-positive. 짧은 run엔 손해 가능 → opt-in 유지.
- **RAM/워커 과다**: 고workers가 저vCPU 런타임에서 oversubscribe 가능(속도만 영향, 정확도 무관). A100 런타임 vCPU 확인 후 12.

## 테스트 (Colab, pytest 금지 — CLAUDE.md)

1. `python -m py_compile` — 문법.
2. 미지정 실행 → 기존과 동일(workers 8, compile off) 로그 확인.
3. `--workers 12 --compile` → 로그 args 반영, compile warmup 로그 or fallback warn 확인.
4. 구 Ultralytics 환경 시뮬레이션(또는 실제) → compile fallback warn 후 정상 학습 지속 확인.
5. 3조건(baseline/random/aroma) 동일 인자 적용 확인(공정성).

## 미확정 TODO

- Colab 설치 Ultralytics 버전이 `compile` 지원(8.3.x+)인지 실행 시 확인 — 미지원이면 fallback으로 자동 우회(warn만).
- torch.compile 실제 속도 이득(~14%)은 데이터셋별 run 길이에 의존 — 실측은 load-test policy상 자동측정 안 함, 사용자 관찰.
