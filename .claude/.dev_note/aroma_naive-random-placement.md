# AROMA exp4v2 — naive random-placement baseline (placement 비대칭 = AROMA 기여 ablation)

---

## (사용할 skills: micro-fix)

> **설계 의도(확정)**: exp4v2에서 **AROMA > random**을 성립시키되, 그 차이가 AROMA의 **smart-placement 프레임워크 전체**(올바른 paste ROI 탐지 → context 분석 → grounded 결정: compat symmetric + clean-bg 게이트 + `clean_bg_selected` 배정 + `_positive_place` void-제외 랭킹)에서 오도록 한다. random은 **어떤 검사도 없이**(compat 없음, foreground 추정 없음, void/clean-bg 게이트 없음) **무작위 위치**에 붙이는 naive copy-paste baseline이다. 결함이 검은/void/edge에 착지할 수 있다. 이는 AROMA 기여(smart placement 전체)의 정당한 ablation이며 **불공정 비교가 아니다** — naive random-position copy-paste는 표준 증강 baseline이다.

## 배경 — 왜 현재 random arm이 naive가 아닌가 (실측)

현재 random arm은 `generate_random.py` → `generate_defects.run(method="copy_paste")` 경로다. `generate_random`은 `compat_mode`/`compat_matrix_json`/`clean_bg_json`을 넘기지 않으므로 `run()`에서 `compat_threshold=0`·`compat_mode="defect"` → `compat_on=False`, `compat_tile=False`. 따라서 `_place_on`의 `if compat_on and compat_tile:`(L1478, `_positive_place`) 분기는 타지 않는다. 대신:

- **`_foreground_mask` 경로(L1502)**: severstal good 이미지에서 `_foreground_mask`가 **45/60(75%)** 에 대해 mask를 반환 → placement가 추정 전경 75%로 **제약**됨(= 부분적으로 grounded).
- **random fallback(L1525-1547)**: 가이드가 `--reject-clean-bg`를 넘기므로 void 재샘플 게이트가 **ON**.

즉 현재 random은 (1) foreground-constrained + (2) void 게이트 ON이라 **naive가 아니다.** 이것이 의도한 대비(AROMA grounded vs random naive)를 무력화한다. 또한 `--reject-clean-bg`는 `load_normal_images()`의 **pool 게이트**(L2837-2842)도 켜서 void normal 자체를 풀에서 제거한다 — 이것도 grounding의 일종이라 naive에서 빼야 한다.

## 옵션 1 — placement 비대칭 (채택)

`generate_defects`에 **opt-in naive placement 모드**를 추가한다. 기본 OFF이라 AROMA/grounded 경로는 무변경. random arm에서만 ON.

### 호출 체인 (구현자 참조 — scope 추적)

`_place_on` 클로저는 **`_paste_and_finalize`(L1296)** 내부에 정의된다. `random_placement`를 `_paste_and_finalize`의 파라미터로 넣으면 클로저가 free variable로 자동 캡처한다. 위로의 스레딩:

```
run() [L2751]  -- random_placement 파라미터 신설
  └─ synthesis_fn(...) 호출 [L3239]  -- random_placement=random_placement 추가
       ├─ copy_paste_synthesis() [L1646]  -- 파라미터 신설 → _paste_and_finalize 호출[L1786]에 전달
       │     └─ _paste_and_finalize() [L1296]  -- 파라미터 신설
       │           └─ _place_on 클로저 [L1387]  -- 캡처, 최상단에 naive 분기
       └─ controlnet_synthesis() [L2116]  -- **kwargs; random_placement는 무시(전달 안 함)
             → controlnet의 _paste_and_finalize는 default False로 무영향 (random arm은 controlnet 미사용)
```

## 수정 내용 — 파일별

### 1) `scripts/aroma/generate_defects.py`

**(a) `_place_on` 클로저 naive 분기 — L1399 직후 / L1401(`forced_xy`) 직전 삽입**

```python
        nrgb = np.asarray(nrm.convert("RGB"))

        if random_placement:
            # Naive baseline (--random-placement): random arm은 의도적으로
            # un-grounded — 모든 placement 결정(forced_xy geometry prior,
            # compat/_positive_place, foreground 제약, clean-bg/void 게이트)을
            # 우회하고 uniform-random 유효 top-left에 붙인다. 이것이 AROMA의
            # ablation: smart-placement 프레임워크(=기여) 전체를 제거해 격리한다.
            # paste당 결정적 rng.randint 2회(seeded 스트림) → 동일 seed 재현.
            # gate_ok=True → stage-2 re-pick 없음.
            return nrgb, _random_paste_position(
                nrm.size, defect_crop.size, rng), True

        if forced_xy is not None:
```

**(b) `_paste_and_finalize` 시그니처(L1296-1318)**: 마지막 파라미터 뒤 `random_placement: bool = False,` 추가.

**(c) `copy_paste_synthesis` 시그니처(L1646-1664)**: `compat_tile` 뒤 `random_placement: bool = False,` 추가. `_paste_and_finalize(...)` 호출(L1786-1805) 마지막에 `random_placement=random_placement,` 추가.

**(d) `run()` 시그니처(L2751-2782)**: `compat_mode` 근처(예: `reject_clean_bg` 뒤)에 `random_placement: bool = False,` 추가 + docstring 한 줄. `synthesis_fn(...)` 호출(L3239-3258)에 `random_placement=random_placement,` 추가. (controlnet은 `**kwargs`라 여분 key 무해; grounded 유지.)

**(e) CLI `_parse_args`(L3397 근처)**: `--random-placement`(store_true, dest=random_placement) 추가. `main()`의 `run(...)`(L3486)에 `random_placement=args.random_placement,` 추가.

### 2) `scripts/aroma/generate_random.py`

random arm은 **naive BY DESIGN**이므로 여기서 기본 ON.

**(a) `run()` 시그니처(L110-124)**: `random_placement: bool = True,` 추가 + docstring. `generate_defects.run(...)` 호출(L197-209)에 `random_placement=random_placement,` 추가.

**(b) CLI `_parse_args`(L218-258)**: `--no-random-placement`(store_false, dest=random_placement, default=True) escape hatch 추가. `main()`의 `run(...)`(L263-277)에 `random_placement=args.random_placement,` 추가.

**(c) clean-bg 게이트 강제 중단**: `reject_clean_bg`는 generate_random에서 이미 default False. 가이드 STEP 4에서 `--reject-clean-bg …`를 **제거**한다(아래 §3). random_placement=True면 naive 분기가 placement 게이트를 우회하지만, `--reject-clean-bg`는 pool 게이트도 켜므로 반드시 빼야 진짜 naive(풀에 void normal 잔존).

ROI-선택 무작위성(`select_random`)은 **무변경**. `clean_bg_random_arm.json` symmetric control 경로(`generate_defects --clean_bg_json`)는 **손대지 않는다** — 이 대비에 쓰이지 않는 별개 메커니즘.

### 3) 가이드 `AROMA연구분석/colab_execute_new/step5_execute.md`

**STEP 4 명령(L273-279)**: `--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0` 3플래그 제거(generate_random가 random_placement 기본 ON).

**narrative supersede**:
- L9 (random arm 개요) "동일 clean-bg 게이트" → "naive 배치(placement grounding·게이트 없음)".
- L264 "positive placement·compat 게이트는 미적용(무변경 통제군)" + "동일 clean-bg 게이트" → naive baseline 설명으로 교체(top_k/n_per_roi/seed는 동일 유지, 게이트·grounding 없음).
- L282 "random은 개선 무관이지만 clean-bg 게이트는 AROMA와 대칭이어야 비교가 공정하다" → **supersede**: 의도된 placement 비대칭. random=naive 표준 baseline, clean-bg/grounding 게이트를 의도적으로 OFF.
- L322 (무결성) "clean-bg 게이트 대칭 … 공정하다" → **supersede**: 동일 프레이밍.
- L11 / L256(무학습 copy_paste vs controlnet, **둘 다 AROMA arm**의 placement 게이트 parity)은 **유지** — random 대비가 아니라 생성 novelty ablation이므로 게이트 맞춤이 옳다.

**프레이밍(가이드+본 노트)**: AROMA framework = 기여, random = naive 표준 baseline. AROMA는 grounded smart placement(올바른 ROI·context·void 제외 랭킹), random은 무검사 무작위 위치. ablation이 정당한 이유: naive random-position copy-paste는 문헌의 표준 증강 baseline이며, placement 프레임워크를 제거해 그 기여를 격리 측정한다. 숨기는 역전이 아니라 **의도된 대비**다.

## 결정론

- **grounded 스트림(random_placement=False) byte-identical**: naive 분기는 `forced_xy` 검사 앞에 추가되지만 flag OFF면 진입하지 않는다. L1401 이후 코드는 무변경 → AROMA copy_paste(STEP 3B)·controlnet(STEP 3) 스트림 불변. **재실행 불요.**
- **random 스트림 변경(의도)**: naive 분기는 paste당 `_random_paste_position` 1회(=`rng.randint` 2 draw)만 소비, foreground/게이트 재샘플 루프 없음 → 기존(foreground_paste 45/60 + void 재샘플)과 draw 수가 달라 스트림 이동. 추가로 `--reject-clean-bg` 제거로 `load_normal_images` 풀 크기가 바뀌어 `rng.choice(normal_images)`(L3234/3238) 선택도 이동. 둘 다 의도. **random arm 재합성 필요.**
- Math.random류 비결정성 도입 없음 — seeded `rng`의 단일 결정적 draw만 사용.

## Colab 재실행 목록

- **재합성(STEP 4, random arm만)**: 모든 DATASETS의 `generate_random.py`를 갱신된 가이드 명령으로 재실행 → `S('synth_random', DS)` 갱신.
- **재실행 불요**: STEP 3B(AROMA copy_paste), STEP 3(AROMA controlnet) — grounded 스트림 byte-identical.
- **다운스트림**: random arm 재합성 후 exp4v2 재평가. exp5(PRDC)·exp6(kNN)은 random 데이터셋 재생성 시 **임베딩 캐시 무효화 필수**(`!rm -rf $EMBED_CACHE_DIR/{ds}` 후 재실행) — 파일명 동일·내용만 변경 시 stale 캐시 재사용(step5 무결성 §).
- 검증(가이드 §STEP 5 parity): random labelable > 0 확인. Colab에서 naive 분기 실제 발동 로그(placement-gate stats 미출력 = 게이트 우회 확인) 대조.

## 수정 대상 파일

- `scripts/aroma/generate_defects.py` — `_place_on` naive 분기 + `_paste_and_finalize`/`copy_paste_synthesis`/`run` 시그니처 스레딩 + CLI `--random-placement`.
- `scripts/aroma/generate_random.py` — `run`/CLI에 `random_placement`(기본 ON, `--no-random-placement` escape) 스레딩.
- `AROMA연구분석/colab_execute_new/step5_execute.md` — STEP 4 명령에서 clean-bg 게이트 제거 + narrative supersede(L9/264/282/322).

## 범위 밖 (건드리지 않음)

- `clean_bg_selection.py`, profiling.
- `generate_defects --clean_bg_json`(clean_bg_random_arm.json symmetric control) — 별개 메커니즘.
- pytest / 신규 테스트 코드(CLAUDE.md) — Colab 검증.
