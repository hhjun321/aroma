# AROMA clean_bg_selection — void-floor 재유도 + 절대 void_frac 컷 (48% black 합성 원인 제거)

## 사용할 skills: micro-fix → post-verify

> **근거**: 단일 파일(`scripts/aroma/clean_bg_selection.py`)만 수정, 단일 관심사(void 탐지 임계 재유도 + 컷 방식 교체), 새 파일·추상화·의존성 없음. 다만 CLI 인자 3종 추가 + 함수 시그니처 스레딩으로 50줄에 근접하므로 구현 후 **post-verify**로 리뷰 소급. Colab 검증 필수(pytest 금지).

---

## 개요 (문제)

`exp4v2` AROMA 합성 class2 composite가 평균 48%(최대 88%) **BLACK**. 로컬 실데이터(`D:/project/aroma_dataset`, severstal) 측정 결과:

- `corr(composite black, normal black) = 1.000` — black은 paste/blend가 아니라 **선정된 normal(good) 배경**에서 100% 유래. 즉 부분/가장자리 강판(partial plate) good 이미지가 배경으로 배정됨.
- 근본 원인은 `scripts/aroma/clean_bg_selection.py`(Step 3.5, OFFLINE, `context_features.csv`만 사용, raw good 픽셀 미접근)의 **void 탐지 임계가 너무 낮아** 어두운 경계/void 패치를 void로 인식하지 못하는 것. void_frac이 사실상 0으로 계산되어 partial plate가 유효 풀에 그대로 잔류.

두 가지 결함:

1. **`_derive_void_floors()`** — void floor를 관측 `local_variance`/`edge_density`의 **1st percentile**로 유도(`var_floor=0.10`, `edge_floor=0.65` 실측). dark-void 클러스터보다 한참 아래라 black 패치의 1%만 잡힘 → `_patch_void`가 black normal을 ~0% void로 보고.
2. **`valid_bg_pool()`** — void_frac 컷을 관측 void_frac의 **90th percentile(상대 컷)**로 유도. 구조상 tail이 아무리 void-heavy해도 **항상 ~90%를 유지** → partial plate를 절대 완전히 제거하지 못함.
3. (확인 항목) 게이트는 `reject_clean_bg=True`일 때만 동작. → §"reject_clean_bg 확인" 참조. **현재 ON**이라 수정은 유효(moot 아님).

---

## 측정 증거 (로컬 severstal 실데이터)

| 항목 | 값 |
|---|---|
| 현행 유도 floor (p1) | `var_floor=0.10`, `edge_floor=0.65` |
| 88%-black normal `_9edf820d2` black-patch median | `local_variance=0.21`, `edge_density=0.98` |
| valid full-plate normal `_7576e7f81` median | `local_variance=61.7` |
| severstal dark border/void 패치 범위 | `local_variance ~0.15–0.25`, `edge_density ~0.6–1.0` |
| pool percentile — local_variance | p5=0.23, p10=0.48 |
| pool percentile — edge_density | p5=1.27, p10=3.21 |
| 절대 floor `(lv<=2.0 AND ed<=3.0)` 분리도 | black normal 패치 **84%** flag, valid full-plate **12%** flag |
| 절대 컷 `void_frac>0.5` drop율 | pool의 **~6.5%**만 drop(= 최악 partial plate만) |
| 절대 컷 하 severstal void_frac 분포 | 대부분 이미지 ~0%, p90 ~36%(raw-pixel 참조 지표) |

**핵심 논리**: black normal의 black 패치는 pool에서 가장 낮은 variance tail에 집중(median 0.21). floor를 dark-void 클러스터(median 0.21)**위**, valid 클러스터(median 61.7)**아래**로 올리면, black normal은 대부분 패치가 flag되어 per-image `void_frac`이 majority(>0.5)를 넘어 제거되고, full-plate normal은 거의 flag되지 않아 유지된다. `edge_density`는 black 패치(0.98)가 pool p5(1.27)보다도 낮아 binding 아님 — 결정 변수는 `local_variance`.

---

## 수정 내용 (function-level, before/after intent)

**대상 파일: `scripts/aroma/clean_bg_selection.py` 단독.** `generate_defects.py`(타 세션 소유)·profiling generator·raw-pixel 스캔 **금지** — "profiling-derived, no raw pixels" 계약 유지.

### (a) `_derive_void_floors()` — p1 → p15 percentile (라인 247–264)

- **before**: `var_floor = float(np.percentile(lv, 1.0))`, `edge_floor = float(np.percentile(ed, 1.0))`. p1은 dark-void 클러스터(var median 0.21) 아래 → black 패치의 1%만 catch.
- **after**: percentile을 인자 `floor_pct`(기본 15.0)로 파라미터화하고 그 percentile로 유도.
  - 시그니처: `_derive_void_floors(good_by_img, floor_pct: float = 15.0)`.
  - 본문: `var_floor = float(np.percentile(lv, floor_pct))`, `edge_floor = float(np.percentile(ed, floor_pct))`.
  - docstring 갱신: "1st percentile" → "raised low percentile (default p15): dark-void 타일 클러스터 위, textured background 아래에 위치. severstal 실측 p10 var=0.48/edge=3.21이 dark cluster(var median 0.21, edge 0.98)를 이미 상회하며, p15는 클러스터 상단(~0.25)에 여유 마진을 둠."
- **선정**: p15 채택. p10(var=0.48)이면 이미 dark cluster median 0.21의 2.3배로 분리 충분하나, dark 경계 패치가 var~0.25까지 퍼지므로 **p15로 클러스터 전체 + 마진** 확보. 순수 percentile이라 **데이터셋별 자동 적응**(매직넘버 아님) — 이것이 크로스-데이터셋 워시아웃을 막는 1차 방어.
- **절대 override 안전밸브**: `run()`에서 CLI `--var_floor`/`--edge_floor`(기본 None)가 주어지면 percentile 유도값을 **대체**. prescan에서 특정 데이터셋의 percentile이 undershoot으로 판명되면(예: severstal에서 var<=2.0가 필요) 코드 수정 없이 그 데이터셋만 절대 floor 고정. `_derive_void_floors` 호출 직후 `run()`에서:
  ```python
  if var_floor_override is not None: var_floor = var_floor_override
  if edge_floor_override is not None: edge_floor = edge_floor_override
  ```

### (b) `valid_bg_pool()` — 상대 p90 → 절대 majority 컷 0.5 (라인 282–286)

- **before**: `void_frac_max = float(np.percentile(vals, 90.0))`. 구조상 항상 ~90% 유지 → partial plate 완전 제거 불가.
- **after**: `void_frac_max = 0.5`(void 패치가 **majority(>50%)** 인 이미지 = 진짜 partial plate만 drop).
  - 라인 282–286 대체:
    ```python
    if void_frac_max is None:
        # Absolute majority-void cut: an image is a "partial plate" only when
        # MORE THAN HALF its patches are void. 0.5 = the majority boundary
        # (semantic, not a dataset-tuned constant), NOT the relative p90 that
        # always kept ~90% regardless of how void-heavy the tail was.
        # severstal ref: cut@0.5 drops ~6.5% (just the worst partial plates).
        void_frac_max = 0.5
    ```
  - `<=` 비교(라인 296) 유지 → void_frac==0.5(정확히 반)는 유지, >0.5만 reject. majority 시맨틱과 정합.
- **매직넘버 방어**: 0.5는 severstal-튜닝 상수가 아니라 **majority(과반) 경계** — 어느 데이터셋에서도 "패치의 과반이 void면 clean background 아님"은 원칙적으로 성립. `--void_frac_max`로 override 가능하고 prescan으로 데이터셋별 검증. floor 수정(a)이 void_frac을 유의미하게 만든 뒤에야 이 컷이 의미를 가짐.
- **all-reject fallback 유지**(라인 301–309): 게이트가 전부 reject하면 full pool로 fallback — **변경 없음**. silent 0-output 금지 계약 준수.

### (c) CLI override 노출 (라인 868–895 `_parse_args`, `run()` 시그니처 라인 801–812, `main()` 라인 898–911)

신규 3종 (모두 데이터-유도 기본값 + override 가능):

| arg | 기본 | 의미 |
|---|---|---|
| `--void_floor_pct` (float) | 15.0 | `_derive_void_floors` percentile. 데이터-유도 기본, 조정용 노출. |
| `--var_floor` (float) | None | percentile 유도값 대체(절대 override). prescan 안전밸브. |
| `--edge_floor` (float) | None | 동상. |

- `--void_frac_max`(기존, 라인 883–885)는 유지하되 help 문구를 "Default: DATA-DRIVEN (P90…)" → "Default: 0.5 (majority-void boundary). Override to pin a value." 로 갱신.
- 스레딩: `_parse_args` 3종 추가 → `main()`이 `run(..., void_floor_pct=, var_floor=, edge_floor=)`로 전달 → `run()`이 `_derive_void_floors(data["good_by_img"], void_floor_pct)` 호출 후 절대 override 적용 → `valid_bg_pool(...)`에 최종 `var_floor`/`edge_floor` 전달(기존 흐름 그대로, 라인 819–822).

### (d) meta/summary auditability (라인 311–317 `derived`, 라인 766–769 `_build_summary`)

- `valid_bg_pool`의 `derived` dict에 `"void_floor_pct"`(사용한 percentile), `"floor_source"`(`"percentile"` | `"override"`)를 추가. 기존 `var_floor`/`edge_floor`/`void_frac_max`/`n_good`/`n_valid`는 유지 → 유도값이 이미 `clean_bg_selected.json` 산출 로그·`clean_bg_summary.md`에 기록됨.
- `_build_summary`의 void 라인(766–769)에 `void_floor_pct=` 와 컷 방식(`void_frac_max=0.5000 (majority)`)을 표기 → 재현/감사 가능.

---

## 결정성 (determinism) 확인

- 변경 지점은 전부 **percentile 계산 + 비교 연산**뿐 — **rng 미사용**. `random_arm()`의 `np.random.default_rng(seed)`는 **불변**. 결정성 완전 보존.
- 단, floor 상향은 **유지되는 normal 집합을 바꾼다** → 배정(selection) 변경 → 이전 실행과 **byte-identical 아님**. 따라서 Step 3.5 재생성 + generate_defects(AROMA arm + random arm) 재실행 **필수**(§재실행).

---

## 크로스-데이터셋 워시아웃 주의 + 필수 Colab prescan

프로젝트 메모리 교훈(GPU 의존 임계는 데이터셋별 발동률 CPU 사전 스캔 후 확정):

- **severstal**(로컬 검증됨): floor 상향으로 black partial plate 제거 — 의도된 동작.
- **leather/mtd**(textured, high variance): p15 floor는 절대값이 커도 **각 데이터셋 percentile**이라 flat한 하위 15% 패치만 flag → 이미지 전반에 분산 → per-image void_frac ~15% << 0.5 컷 → **거의 제거 안 됨**(워시아웃 없음). 절대 void_frac 컷 0.5가 워시아웃 방어.
- **aitex**(bright-but-flat, low variance): 진짜 저-variance라 flag되어 제거됨 — **의도**. 단 good 전부가 flat이면 all-reject → full-pool fallback(안전).

**leather/mtd/aitex good 풀은 로컬 디스크에 없음** → 전역 활성화 **전** Colab CPU prescan 필수:

- 각 데이터셋 `context_features.csv` good 행 로드 → p15 `var_floor`/`edge_floor` 유도 → per-image `void_frac` 분포 산출.
- 확인: (1) textured 셋(leather/mtd)에서 대부분 이미지가 `void_frac<=0.5`로 유지되는가(과제거 아님), (2) severstal에서 black partial plate가 `void_frac>0.5`로 제거되는가, (3) aitex 제거가 의도대로인가.
- undershoot/과제거 데이터셋은 `--var_floor`/`--edge_floor`(절대) 또는 `--void_floor_pct`/`--void_frac_max`로 **그 데이터셋만** 조정. severstal-튜닝 상수를 전역 하드코딩 금지.

---

## reject_clean_bg 확인 (게이트 활성 여부 — 수정이 moot 아님을 입증)

- **두 개의 서로 다른 `reject_clean_bg`가 존재**:
  - `clean_bg_selection.py`(Step 3.5)의 `run(reject_clean_bg=True)` — CLI opt-out은 `--no_reject_clean_bg`. **기본 True(ON)**. 이것이 `valid_bg_pool`의 §"if not reject_clean_bg → keep all"(라인 290) 게이트를 지배.
  - `generate_defects.py`의 `--reject-clean-bg`(기본 **OFF**) — **별개의 생성-시점 fallback 게이트**. `clean_bg_selected.json`이 존재하면 generate_defects는 이를 소비(라인 3113+)하고 풀 필터링은 이미 Step 3.5에서 끝남 → JSON 경로에선 이 플래그는 **moot**.
- **AROMA arm 실제 호출**(`AROMA연구분석/colab_execute_new/step3_5_execute.md` STEP 1): `--no_reject_clean_bg`를 **전달하지 않음** → `reject_clean_bg=True` → **게이트 ACTIVE**. `--emit_random_arm`으로 같은 실행에서 `clean_bg_random_arm.json`도 방출되므로 게이트가 **AROMA arm·random arm 양쪽에 동일 적용**(대칭 대조군 유지).
- **결론**: 게이트는 현재 켜져 있으므로 floor/컷 수정은 유효. **문서화 요구**: Step 3.5 실행 시 `--no_reject_clean_bg`를 붙이면 게이트가 꺼져 수정이 무력화되므로 절대 붙이지 말 것(가이드 STEP 1 기본형 유지).

---

## 부작용 — var_floor/edge_floor의 두 번째 소비자 (리뷰 지적)

`var_floor`/`edge_floor`는 void 게이트(`valid_bg_pool`)뿐 아니라 **랭킹 히스토그램**(`_image_hist` / `_class_bg_hist` → `build_and_rank`)에서도 void 타일 스킵에 쓰인다. 따라서 p1→p15 상향은 배경 **선택(pool 게이트)** 과 **랭킹(cell 히스토그램)** 을 **둘 다** 이동시킨다.

- 영향 제한적: 히스토그램 교차는 source·target에 **대칭** 적용되고 non-void 셀로 정규화되며, `class_fit`/`src_fit` 가중은 데이터 파생이라 신호가 평탄해지면 자동 하향된다 → 워시아웃 blocker 아님.
- 그러나 prescan 체크리스트에 **배경 배정 다양성 분포**도 포함해, floor 변경의 히스토그램 효과를 leather/mtd에서도 관측할 것(void 게이트만 보지 말 것).

---

## 재실행할 Colab 스테이지

프로파일링/roi_selection은 **재실행 불필요**(profiling-derived 계약 유지, 신규 컬럼 없음, `roi_selected.json` 불변):

1. **(선행, 필수) CPU prescan** — leather/mtd/aitex/severstal `context_features.csv` good 행으로 p15 floor + per-image void_frac 분포 확인(§워시아웃). 과제거/undershoot 시 데이터셋별 override 확정.
2. **Step 3.5 — `clean_bg_selection.py`** (`step3_5_execute.md` STEP 1, DATASETS 루프, `--emit_random_arm`, `--no_reject_clean_bg` 미전달) → `clean_bg_selected.json` + `clean_bg_random_arm.json` 재생성. floor/컷 수정이 여기서 발효. STEP 2의 E1 재현 게이트·요약 확인 포함.
3. **Step 5 — `generate_defects.py`** (AROMA arm = `clean_bg_selected.json`, random arm = `--clean_bg_json clean_bg_random_arm.json`) → 새 배경으로 composite 재생성. black% 급감 확인(재생성 이미지 black-fraction 재측정).
4. **(다운스트림, 영향 측정용)** exp4v2 YOLO 학습/평가 — black-composite 제거의 mAP 영향 측정 시. 사용자 요청 시에만(부하/성능 자동 실행 금지 정책).
