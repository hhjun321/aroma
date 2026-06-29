# AROMA exp4v2 — config-driven generic `visa_*` detection 로더 추가

---

## (사용할 skills: feature-dev)

## 개요

exp4v2(`exp4_v2_supervised_detection.py`)의 `_load_detection_dataset`는 dataset_key를 자체 하드코딩 분기로 해석한다(`isp_LSM_1` / `mvtec_*` generic / `visa_cashew` / `visa_pcb` / `severstal`). visa는 cashew·pcb4만 하드코딩되어 있고 **generic 핸들러가 없어**, `visa_macaroni`/`visa_fryum` 등은 `else → "Unknown dataset_key" → skip`된다(실행 로그 확인: `Detection skip visa_macaroni — dataset not found`). 그 결과 `--dataset_keys`에 4개를 줬는데 mvtec 2개만 학습되고 `n_datasets=2`로 끝났다.

한편 `dataset_config.json`에는 `visa_macaroni`(→`visa/macaroni1`), `visa_fryum` 등이 이미 정의돼 있으나 **exp4v2는 이 config를 읽지 않는다**(grep: `dataset_config` 참조 파일은 `exp3_generation_quality.py` 단 하나). exp3에는 이미 **config-driven generic 로더**가 있다(`_load_dataset_config` L148, config 기반 `_load_*_dataset` L215~250). 본 작업은 그 로더 패턴을 exp4v2로 이식해 **visa_* 를 config 기반으로 지원**한다. `macaroni1`/`pcb4` 같은 비자명 key→폴더 매핑이 config에 있으므로 하드코딩 alias가 필요 없다.

## 영향도 분석

### 이 기능이 변경하는 상태
- exp4v2가 **`dataset_config.json`에 신규 의존**(읽기 전용). 새 dataset-key 해석 경로(`visa_*`) 추가.
- visa_* 합성을 학습에 투입 가능 → exp4v2 결과셋에 visa 데이터셋 행 신설.

### 그 상태를 전제로 동작하는 기존 로직
- `_load_detection_dataset` 호출부(seed 루프 / synth 로딩): 반환 shape `{train_normal, test_good, test_defect, mask_map}` 동일 유지하면 무영향.
- `visa_cashew`/`visa_pcb` 하드코딩 분기: **그대로 유지**(아래 §수정 4) → 기존 Data/ 동작 보존, 회귀 0.
- `mvtec_*`/`isp_LSM_1`/`severstal` 분기: **무수정**. 특히 severstal은 `class_mode=multi` 특수 처리라 절대 건드리지 않음.

### "없음(0개)" 상태 점검
- config 파일을 못 찾거나(env 미설정) key가 config에 없으면 → **warning + None 반환** = 기존 `Unknown dataset_key` skip과 동일 거동. 신규 크래시 경로 없음.

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `_load_dataset_config()` 이식

exp3의 로더를 그대로 이식(env `DATASET_CONFIG` → `AROMA_REF/dataset_config.json` → `/content/AROMA/dataset_config.json` fallback, 모듈 캐시). 없으면 빈 dict + warning.

### 2. 같은 파일 — `_load_detection_dataset`에 config-driven `visa_*` generic 분기 추가

exp3 로더(L215~250) 패턴 그대로:
```python
elif dataset_key.startswith("visa_"):
    cfg = _load_dataset_config()
    entry = (cfg or {}).get(dataset_key)
    if not entry or not entry.get("image_dir"):
        logger.warning("visa dataset_key not in dataset_config: %s", dataset_key)
        return None
    image_dir = Path(entry["image_dir"])          # .../<cat>/train/good
    if not image_dir.exists():
        logger.warning("%s image_dir missing: %s", dataset_key, image_dir)
        return None
    root = image_dir.parent.parent                # .../<cat>
    train_normal = _glob_images(str(image_dir))
    test_good    = _glob_images(str(root / "test" / "good"))
    seed_dirs = entry.get("seed_dirs")
    if seed_dirs:
        defect_dirs = [Path(d) for d in seed_dirs]
    else:
        troot = root / "test"
        defect_dirs = [p for p in sorted(troot.iterdir())
                       if p.is_dir() and p.name != "good"] if troot.exists() else []
    test_defect = []
    for d in defect_dirs:
        test_defect.extend(_glob_images(str(d)))
    mask_map = _resolve_masks_generic(test_defect, root)
```
> `dataset_config.json`의 visa 항목은 `seed_dir`(단수)만 있고 `seed_dirs`(복수)는 없음 → else 분기로 `root/test/<non-good>`(=`test/anomaly`) 자동 스캔됨. (확인: `test/`에 `anomaly`+`good` 존재.)

### 3. 같은 파일 — mask resolver

exp3의 `_resolve_masks_generic`를 이식(visa prepared `ground_truth/anomaly`에서 검증됨). `TODO`: 이식 vs 기존 `_resolve_mvtec_masks(test_defect, root/"ground_truth")` 재사용 — 구현 시 `_resolve_masks_generic` 우선, ground_truth/anomaly 매칭 동작 확인.

### 4. 분기 순서 — cashew/pcb 보존

기존 `elif dataset_key == "visa_cashew"` / `== "visa_pcb"` 블록은 **삭제하지 않고 그대로 두며, 새 `startswith("visa_")` generic 분기는 그 뒤(아래)에** 배치한다. elif는 위에서부터 매칭되므로 cashew/pcb는 기존 Data/ 경로로, 나머지 visa_*만 generic config 경로로 흐른다. → cashew/pcb 회귀 0.

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (단일 파일)
  - `_load_dataset_config()` 신설 (모듈 캐시 포함)
  - `_resolve_masks_generic()` 이식 (exp3에서)
  - `_load_detection_dataset`에 `visa_*` generic 분기 추가 (cashew/pcb 뒤)

## 테스트 (Colab)

> 프로젝트 규칙: 새 테스트 코드/pytest 금지. Colab 재실행으로 검증.

1. **인식 검증**: `--dataset_keys visa_macaroni visa_fryum`로 exp4v2 실행 → 로그에 `Unknown dataset_key` skip이 **사라지고** `train_normal/test_good/test_defect/masks_matched` 정상 출력 + `n_datasets` 증가.
2. **합성 결합**: `synthetic_mp/visa_macaroni` 등 synth가 `synth_ann>0`로 붙는지(`[SynthRatio]` 로그).
3. **회귀**: `visa_cashew`/`visa_pcb`를 동일 실행에 포함 → 기존과 동일 동작(Unknown 아님, 기존 Data/ 경로) 확인.
4. **무영향**: mvtec_carpet/leather, severstal 분기 결과 불변.

## 미확정 사항 (TODO / 구현 시 결정)

- `TODO`: mask resolver — `_resolve_masks_generic` 이식 vs `_resolve_mvtec_masks(root/ground_truth)` 재사용. (우선 `_resolve_masks_generic`.)
- `<결정 필요>`: cashew/pcb를 장기적으로 generic으로 흡수할지 — **이번엔 보류**(회귀 0 우선).
- `<결정 필요>`: 범위를 `mvtec_*`/`isp_LSM_1`까지 config-driven으로 통합(single source of truth)할지 — **이번엔 `visa_*`만**.
- `<별개 트랙>`: **dark-object 무회귀** — visa는 object-centric. visa 합성이 수정된 `_foreground_mask`(void 가드)로 생성됐다면, 본 로더와 별개로 가드가 정상 객체를 void 오판하지 않았는지 `clean_background_gate_verify.md` Cell 4B Part A를 visa normal에 적용해 확인 권장. [[aroma_exp4v2_foreground-void-rejection]]
