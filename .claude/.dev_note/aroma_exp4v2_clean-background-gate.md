# copy-paste 합성에 검은/평탄 배경 거부 품질 게이트 이식 (전 데이터셋 적용)

## (사용할 skills: feature-dev)

## 개요

AROMA 합성 파이프라인은 normal 이미지 풀에서 **균일 랜덤**(`rng.choice(normal_images)`)으로 배경을 선택하며, 풀 적재(`load_normal_images`) 시 **품질 필터가 전무**하다. 검은/평탄(void) 이미지가 그대로 배경 후보가 되고, foreground 추정 실패 시 `_random_paste_position` 폴백으로 결함이 검은 영역에 붙는다. 검은 normal 이미지 자체는 결코 거부되지 않는다.

결과적으로 void 위에 결함이 붙은 **비현실 합성 샘플**이 학습 데이터에 유입 → 다운스트림 detector 학습 품질 저하. exp4v2 Severstal 분석에서 육안 확인된 현상이며, 모든 증강 조건(random/casda/aroma)이 baseline 미달인 원인 중 하나로 의심.

CASDA에는 검증된 메커니즘(`extract_clean_backgrounds.py` → `compute_quality_score`)이 존재. 이를 AROMA로 이식하되 severstal 전용이 아니라 **전 데이터셋(severstal full-frame strip / mvtec full-frame / visa object-centric)** 공통 적용. 선행 dev_note [[aroma_exp4v2_foreground-placement]]가 추가한 foreground 배치 게이트와 **상보적** — 그건 배치 위치를 객체로 제약, 본 건은 검은 배경 자체를 회피.

### 핵심 제약 (사용자 확정)

1. **생성 시점 차단** — 이미 만들어진 `syn_*.jpg`를 사후 제거하는 게 아니라, 합성 **입력단**(배경 풀 + paste 위치)에서 검은 배경을 배제해 **처음부터 검은배경 샘플이 안 생기게** 한다.
2. **aroma / random / casda 모두 동일 적용** — 세 조건 전부 `generate_defects.run()` / `copy_paste_synthesis` 단일 엔진을 경유(아래 호출 그래프)하므로 엔진 한 곳 수정으로 동일하게 적용된다. CLI 토글을 세 조건 동일하게 켜려면 wrapper에도 파라미터 pass-through 필요.

```
aroma  : generate_defects.py main      → generate_defects.run()  → copy_paste_synthesis
random : generate_random.py  run()/CLI → generate_defects.run()  → copy_paste_synthesis
casda  : generate_casda.py   run()/CLI → generate_defects.run()  → copy_paste_synthesis
```

### 이식 대상 참조 (CASDA, ground truth — 수정 금지)

`D:\project\CASDA\CASDA\scripts\extract_clean_backgrounds.py` → `compute_quality_score(patch)`:
- `blur_score` = `cv2.Laplacian(gray, CV_64F).var() >= 100` ? 1.0 : 0.3 (평탄/검은 패치 → 저 Laplacian → 0.3)
- `contrast_score` = `min(np.std(gray)/128, 1.0)` (검은 패치 std≈0 → ≈0)
- `brightness_score` = `0.3 <= np.mean(gray)/255 <= 0.7` ? 1.0 : 0.7
- `noise_score` = `1.0 - min(np.mean(local_var)/100, 1.0)`
- `quality = 0.30*blur + 0.30*contrast + 0.20*brightness + 0.20*noise`
- 순수 검은 패치 ≈0.43 < `min_quality`(기본 0.7) → 배경 풀에서 **거부**. Stage A `--min_suitability 0.5`가 2차 방어.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- normal 배경 풀 구성 (`load_normal_images` 반환 — 검은/평탄 이미지 제외 가능).
- `copy_paste_synthesis`의 paste position 산출 (폴백 경로에서 검은 위치 reject).
- 합성 성공 수(`n_ok`)·선택된 `normal_image` 분포 (annotations).
- 신규 헬퍼 + `run`/CLI 파라미터.

### 그 상태를 전제로 동작하는 기존 로직 (안전성 검토)
- `_foreground_mask`/`_foreground_paste_position` ([[aroma_exp4v2_foreground-placement]]): 위치를 객체 전경으로 제약 → 그 경로에선 배경=객체라 clean-bg 게이트 **불필요(중복)**. 위치 게이트는 foreground 폴백(`_random_paste_position`) 경로에만 끼워 **순서 충돌·이중 거부 방지**.
- `_alpha_composite`: position만 받음 — 무관, 변경 없음.
- B안 mask 저장 / annotations bbox = `[x_paste,y_paste,crop_w,crop_h]`: position이 clean 영역으로 바뀌어도 공식 동일 — 정합 유지.
- `generate_random.py`: `generate_defects.run()` 위임 → 파라미터 thread-through 시 자동 적용.

### "없음(0개)" 상태 생성 가능성 (delete/filter 계열 확인)
- **pool 게이트가 풀을 0장으로 만들 수 있음** (전부 검은 normal dir). → 폴백 필수(아래 엣지 §2). silent 0-output 금지.

---

## 이론적 근거

- 결함 검출 학습은 결함이 **객체 표면 위**에 있을 때만 유효. void 배경의 결함은 "배경에 결함이 있다"는 잘못된 사전분포 주입.
- CASDA가 동일 Severstal에서 이미 채택·검증한 방어. AROMA는 동등 대응이 누락된 상태 → 공정 비교를 위해서도 정합 필요.
- foreground 게이트(배치)와 clean-bg 게이트(배경 선택)는 직교: 전자는 "어디 붙일까", 후자는 "어떤 배경에 붙일까".

---

## 수정 내용

수정 대상: `scripts/aroma/generate_defects.py` (유일한 코드 변경 파일)

### 1. 신규 헬퍼 `_background_quality_score` + `_is_clean_background`
- CASDA `compute_quality_score` 이식. 입력 grayscale ndarray → blur(Laplacian var) + contrast(std) + brightness(mean) + noise(local var) 가중합 → `float`.
- `_is_clean_background(img_or_gray, min_quality=0.7, blur_threshold=100.0) -> bool` 로 임계 비교 분리.
- **`HAS_CV2` 가드 필수**: cv2 없으면 게이트 비활성(`True` 반환, legacy 동작). `_foreground_mask`의 `if not HAS_CV2: return None` 선례 동일.
- RNG 미사용 → **결정론적**.

### 2. 주입 지점 1 (Pool-level, 이미지 단위 거부) — `load_normal_images` (~702행)
- `rglob` 수집 → **정렬**(결정론 강화, 현재 미정렬) → 각 이미지 `_is_clean_background` 평가 → 통과분만 반환.
- 시그니처: `load_normal_images(normal_dir, *, reject_clean_bg=False, min_bg_quality=0.7, blur_threshold=100.0) -> List[str]`.
- **per-image Laplacian은 적재 시 1회만** — `run` 루프 내 반복 평가 금지.
- 거부 수/비율 `logger.info` 로깅(데이터셋별 진단).
- **전량 거부 폴백**: 0장 반환 시 `logger.warning` 후 게이트 무시하고 원본 풀 사용(엣지 §2).

### 3. 주입 지점 2 (Position-level, 폴백 경로 한정) — `copy_paste_synthesis` (~410-424행)
- 현재: `fg_mask=_foreground_mask(...)` → `_foreground_paste_position` → `None`이면 `_random_paste_position`.
- 변경: **`_random_paste_position` 폴백 경로에서만** 후보 `(x,y)` 주변 local 배경 패치(crop 크기)를 잘라 `_is_clean_background` 검사 → 검은/평탄이면 재시도. `max_bg_tries`(기본 20) 루프 + 동일 `rng`. 모두 실패 시 마지막 후보로 진행(합성 자체는 보존).
- foreground 게이트 성공 경로는 **건드리지 않음**(중복 방지).
- 시그니처: `copy_paste_synthesis(..., reject_clean_bg=False, min_bg_quality=0.7, max_bg_tries=20)`.

### 4. 파라미터 노출 — `run(...)` + CLI argparse
- `run`에서 위 파라미터 thread-through.
- CLI: `--reject-clean-bg` (토글), `--bg-injection {pool,position,both}`, `--min-bg-quality`(기본 0.7), `--bg-blur-threshold`(기본 100.0).
- 기본 ON/OFF·injection 기본값은 §미확정 결정 후 반영.

### 5. wrapper 파라미터 pass-through (random/casda 토글 동일화)
세 조건 동일 적용 + CLI 토글 동작을 위해 wrapper에 신규 파라미터 forwarding:
- `generate_random.py`: `run(...)` 시그니처에 `reject_clean_bg`/`min_bg_quality`/`bg_blur_threshold`/`bg_injection` 추가 → `generate_defects.run(...)`로 전달. CLI argparse에도 동일 플래그 추가.
- `generate_casda.py`: 동일 — `run(...)` + CLI에 forwarding.
- 미전달 시 `generate_defects.run()` 기본값으로 동작(셋 다 동일 기본값 보장).

### 6. 수정 불필요 (자동 적용)
- `roi_selection.py`: 선택만 담당, 합성 미경유 → 무관.
- `exp4_v2`: mask PNG에서 bbox 재유도 → position 변경 자동 정합.

---

## 수정 대상 파일

- `scripts/aroma/generate_defects.py` (엔진 — 헬퍼·게이트·CLI)
- `scripts/aroma/generate_random.py` (파라미터 forwarding + CLI)
- `scripts/aroma/generate_casda.py` (파라미터 forwarding + CLI)
- (참조 전용, 수정 금지) `D:\project\CASDA\CASDA\scripts\extract_clean_backgrounds.py`

---

## 암묵적 요구사항 / 엣지 케이스

1. **object-centric vs full-frame**: visa_pcb 등은 객체 주변 어두운 배경이 **정상**. pool 게이트를 전 프레임에 적용하면 정상 object-centric 과도 거부 위험 → pool 게이트 임계 보수적 + **위치 게이트는 foreground 폴백 경로 한정**(foreground 게이트가 이미 배치를 객체로 제약).
2. **전부 검은 normal dir (전량 reject)**: pool 0장 → 합성 불가 → `logger.warning` 후 게이트 무시·원본 풀 사용. 빈 풀 silent 0-output 금지.
3. **단 1장만 통과**: `rng.choice` 동작하나 다양성 붕괴(전 합성 동일 배경) → 통과 수 < 임계(예 N) 시 경고 로깅.
4. **폴라리티(어두운 객체/밝은 배경)**: brightness_score는 단독 거부 불가(가중 0.2). blur+contrast(0.6)가 주력 → 정상적으로 밝거나 어두운(검지만 아닌) 배경 부당 거부 안 됨 확인. `_foreground_mask` corner-vote 폴라리티와 정합.
5. **Reproducibility**: 거부/재시도 루프 동일 `rng`. 품질 평가 결정론(Laplacian/Otsu). 고정 seed → 동일 풀·동일 선택. pool 필터는 `rglob` 정렬로 순서 안정화.
6. **성능**: pool은 1회/이미지. position은 합성마다 local 평가 → `max_bg_tries` 상한. 부하 실측은 자동 금지(load-test-policy) — 비용 추정만.
7. **foreground-placement 상호작용**: 위치 게이트를 foreground 폴백 경로에만 → 순서 충돌·이중 거부 회피.

---

## 테스트 (Colab, pytest 금지)

프로젝트 정책: 신규 테스트 코드 작성·pytest 금지. Colab 직접 검증 + `.md` 가이드.

1. **검은배경 비율 before/after**: severstal / mvtec_* / visa_pcb 각각 게이트 OFF vs ON 합성 → 결함이 void/평탄에 붙은 비율(육안 + 합성 배경 패치 mean/std 분포) 비교 → ON에서 감소 확인.
2. **회귀(visa 과도 거부)**: visa_pcb ON 시 통과 normal 수·합성 성공 `n_ok`가 OFF 대비 유의 감소 없는지. 정상 어두운-배경 부당 거부 없음 검증.
3. **전량 거부 폴백**: 인위적 전부-검은 normal dir → 빈 출력 아닌 경고+폴백 정상 합성.
4. **결정론**: 동일 seed 2회 → annotations `normal_image` 선택·bbox 동일.
5. CLI 형식 `.claude/rules/colab-execution.md` 준수(`!python $SCRIPTS/...`, `$VAR`).

---

## 확정 사항 (사용자 결정 완료)

| 항목 | 결정 | 비고 |
|------|------|------|
| **injection point** | **both (pool + position)** | pool=검은 normal 이미지 풀 제외, position=폴백 위치 검은데 회피. visa 과도거부는 임계 보수 + position을 foreground 폴백 경로 한정으로 방어 |
| **default ON/OFF** | **기본 OFF** | `--reject-clean-bg` 명시 플래그로 켬. 기존 exp4v2 결과와 ablation 비교(검은배경 유무 영향) 가능 |
| **임계값** | **CASDA 기본 그대로** | `min_bg_quality=0.7`, `bg_blur_threshold=100.0`, brightness 0.3~0.7. 동일 Severstal서 검증된 값 |
| **position 게이트 범위** | **foreground 폴백 경로 한정** | foreground 게이트 성공 경로는 미변경(중복 방지) |
| **pool 정렬** | **`rglob` 결과 sort 추가** | 결정론 강화 |
| **전량 거부 처리** | **폴백(게이트 무시·원본 풀)** | `logger.warning` 후 진행, silent 0-output 금지 |

## 남은 TODO (구현 후속, 차단 아님)
- 데이터셋별 최적 임계 튜닝 (1차는 CASDA 기본).
- position 게이트 local 패치 크기(crop 크기 vs 고정 윈도우) 구현 시 결정.

## 구현 결과 (workflow 완료)
- 수정: `generate_defects.py`(헬퍼·게이트·CLI), `generate_random.py`/`generate_casda.py`(forwarding+CLI). py_compile 통과.
- CLI: `--reject-clean-bg`(store_true) / `--min-bg-quality`(0.7) / `--bg-blur-threshold`(100.0) 3개. 세 조건 동일.
- `--bg-injection` 셀렉터는 **미구현** — injection=both 확정이라 게이트 ON 시 pool+position 무조건 동시 적용(별도 토글 불필요).
- 리뷰 반영: (major) 하드 Laplacian 게이트 제거 → CASDA식 `quality>=min_quality` 단일 기준. (minor) `bg_blur_threshold`를 position 게이트까지 thread-through → pool/position 임계 일치.
- 검증: Colab 가이드 `AROMA연구분석/colab_execute/clean_background_gate_verify.md`. commit은 사용자.
