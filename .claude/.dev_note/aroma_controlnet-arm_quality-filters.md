# AROMA controlnet arm — AR 스킵 게이트 + pair-level 텍스처 배경 재추첨 필터 추가

---

## (사용할 skills: feature-dev)

> **진단 근거**: `.claude/.etc/pilot_out/severstal` 파일럿 8장(ROI 4 × normal 2) 분석 —
> 파이프라인 정합성(bbox==mask 8/8, clean-bg gate 작동 실증, fingerprint 캐시)은 전부 정상.
> 육안 품질 결함 3/8장, 전부 bbox AR≥3.1. checkerplate 패턴 단절 1장(역시 AR 3.1과 중첩).

## 개요

ControlNet arm 파일럿에서 확인된 품질 결함 2종에 대한 생성 필터를 추가한다.

1. **필터 2-a (AR gate, controlnet 전용)**: `controlnet_synthesis`는 bbox 전체를 `resolution²`(512²) 정사각으로 squash → 생성 → bbox 크기로 un-squash 한다(generate_defects.py:1262-1271 주석에 알려진 한계로 명시 — "Pilot-gate elongated defects visually"). 파일럿에서 bbox 종횡비(AR)와 품질이 완벽 상관: AR 4.7(1212×256)·3.1(799×256) 샘플 전부 수평 스미어/평탄 패치, AR 1.0·0.5 양호. AR > 임계인 ROI를 생성 전에 스킵한다. copy_paste는 squash가 없어 미적용.
2. **필터 1-b (pair-level 텍스처 거리 재추첨, copy_paste + controlnet 공통)**: normal 배경은 `run()` 루프의 `rng.choice(normal_images)`(line 1834)로 균등 무작위 선택 — ctx 호환성 미적용. checkerplate(무늬강판) normal에 paste 시 seamlessClone이 색만 동화하고 구조적 반복 패턴 위상 단절은 못 감춘다. paste 위치 배경 patch와 ROI 원본 배경 간 텍스처 거리가 임계 초과 시 위치 → normal 순으로 재추첨한다. **arm 간 배경 분포 공정성을 위해 copy_paste arm에도 동일 적용** (controlnet에만 걸면 confound).

---

## 영향도 분석

### 이 기능이 변경하는 상태
- controlnet arm 출력 구성: AR>임계 ROI 스킵 → `n_rois × n_per_roi`보다 적은 출력 (기존 skip_oom/skip_blank와 동일 semantics).
- 합성 이미지의 normal 배경 분포: 텍스처 게이트 ON 시 checkerplate류 normal 회피.

### 그 상태를 전제로 동작하는 기존 로직
- `composed_to_exp4v2.py` 등 다운스트림은 annotations.json 기준으로 동작 — 출력 수 감소는 이미 skip 계열이 만들던 상태라 신규 전제 위반 없음.
- **결정론(RNG stream-shift)**: normal 재추첨은 rng 소비량을 바꿔 이후 모든 샘플의 normal 배정이 이동한다(deterministic-but-stream-shifting — 전례 [[aroma_exp4v2_foreground-void-rejection]]과 동일 이슈). 동일 seed·동일 플래그 재실행은 결정론 유지, 단 **필터 OFF 대비 byte-identical 아님**. 기존 exp 재현성 보존을 위해 두 필터 모두 opt-in 기본값(OFF/무해 기본값)으로 노출.
- **fingerprint 캐시 상호작용**: AR gate는 생성 전 return None → 캐시 write 자체가 없어 무충돌. 텍스처 재추첨으로 normal이 바뀌어도 sidecar가 `normal_image`를 기록(line 1320)하고 `run()`이 캐시 값을 우선(line 1855)하므로 annotations 정합 유지. 단 캐시 히트 샘플은 재추첨을 재수행하지 않음(캐시된 normal 그대로) — 캐시 무효화 없이 필터를 새로 켜려면 `--cn_no_cache` 또는 출력 디렉토리 분리 필요. 가이드에 명시.

### delete / remove / bulk 계열 확인
- 해당 없음 (생성 파이프라인 필터). "0개 출력" 상태는 AR 스킵률 100% + 전 ROI 스킵 시 이론상 가능 — stats 로그로 관찰, 폴백은 하지 않음(스킵이 목적).

---

## 수정 내용

대상: `scripts/aroma/generate_defects.py` (단일 파일 중심)

### 1. 필터 2-a — AR gate

- **삽입 위치**: `controlnet_synthesis` 내 bbox 경계 검증 블록(line 1192-1197, `bw_box>0 and bh_box>0` 보장) **직후**, `box = (...)` 직전. 경계 검증 뒤에 두어 div-by-zero 원천 차단.
- **로직**:
  ```python
  ar = max(bw_box, bh_box) / min(bw_box, bh_box)
  if ar_threshold > 0 and ar > ar_threshold:      # '>' — 임계 정확히 일치는 통과
      stats["skip_ar"] += 1
      logger.warning("controlnet AR %.2f > %.2f — skip %s", ar, ar_threshold, image_id)
      return None
  ```
- stats dict(line 941-945)에 `"skip_ar": 0` 추가, `_log_cn_stats`(line 1342-1347)에 `skip_ar=%d` 추가.
- `ar_threshold`는 `_configure_controlnet_context`가 ctx에 저장 (기존 cn 파라미터 관례).

### 2. 필터 1-b — 텍스처 거리 재추첨

- **신규 헬퍼 2개** (`_background_quality_score` 인근, numpy-only — HAS_CV2 False 안전):
  - `_texture_descriptor(gray_patch) -> Optional[np.ndarray]` — FFT radial power spectrum profile(주기성 peak 감지, checkerplate 대응) + (std, 고주파 에너지 비율) 결합 벡터. 패치가 너무 작거나 빈 경우 None.
  - `_texture_distance(desc_a, desc_b) -> float` — 정규화 profile 간 거리 (구현 시 cosine 기반 권장).
- **source-bg descriptor 계산 (각 arm)**: source defect 이미지의 **mask 밖 dilation ring**(mask 수 px 팽창 테두리 밴드)에서 계산.
  - `copy_paste_synthesis`: bbox 크롭 시점(line 799-803 인근) — `defect_img`·`mask_full` 보유.
  - `controlnet_synthesis`: `roi_image_arr` reload 시점(line 1244 인근).
  - mask가 프레임 전체(ring 공집합)·ellipse fallback(real mask 없음, line 804-814) → descriptor None → **필터 통과(스킵)**.
- **`_paste_and_finalize` 확장** (기존 reject_clean_bg 재샘플 루프 line 636-651에 통합):
  - 신규 파라미터: `source_bg_desc=None`, `normal_pool: Optional[List[str]] = None`, `pool_rng=None`, `texture_dist_threshold: float = 0.0`(0=OFF).
  - **2단 재샘플**: (1단) 같은 normal 내 위치 재샘플 — 기존 clean-bg 게이트에 텍스처 거리 검사 AND 추가, 소수회(3회). (2단) 여전히 초과면 `normal_pool`에서 다른 normal 재추첨(상한 예: 5장), normal 로드는 `_paste_and_finalize`가 수행. checkerplate normal은 위치 무관 초과이므로 normal 단위 탈출 경로 필수.
  - **상한 도달 폴백**: 기존 clean-bg 철학(line 638-640) 준수 — **마지막 후보 그대로 paste + 경고 로그 1줄** (silent drop 금지).
- **호출부 배관**: `run()` 루프(line 1835-1846)에서 `normal_images` pool·rng를 synthesis kwargs로 전달 → `copy_paste_synthesis`/`controlnet_synthesis`가 `_paste_and_finalize`로 하향.

### 3. CLI 플래그 (`_parse_args` line 1933 인근 + `run()`/`main()` 시그니처)

- `--cn-ar-threshold` (dest `cn_ar_threshold`, type float, **default 2.5**, `0`=게이트 OFF) — controlnet 전용.
- `--texture-dist-threshold` (dest `texture_dist_threshold`, type float, **default 0.0=OFF**, opt-in) — 임계값은 Colab 튜닝 후 가이드에 권장값 기록.

### 4. Colab 가이드 갱신

- `AROMA연구분석/colab_execute/controlnet_aroma_arm_execute.md`:
  - 실행 예시에 신규 플래그 추가 (colab-execution.md 형식 — `$VAR`, `!python`).
  - STEP 6 stats 로그 안내에 `skip_ar` 추가.
  - STEP 5 "elongated defects 육안 게이트" 문구 → AR 자동 게이트로 갱신.
  - 캐시 주의: 기존 pilot 캐시 위에서 텍스처 필터를 켜면 캐시 히트 샘플은 미적용 — `--cn_no_cache` 또는 출력 디렉토리 분리 안내.

---

## 수정 대상 파일

- `scripts/aroma/generate_defects.py` — 본 구현 전체
- `AROMA연구분석/colab_execute/controlnet_aroma_arm_execute.md` — 플래그·stats·게이트 문구 갱신
- (범위 제외 — TODO 참조) `generate_random.py` / `generate_casda.py`의 텍스처 필터 미러

---

## 테스트 (Colab 검증 — 테스트 코드 작성·pytest 금지)

1. **AR gate**: severstal pilot ROI(AR 4.7/3.1 포함) 재실행 → stats 로그 `skip_ar=2`(ROI 4 중 2), 파일럿 아티팩트 3장이 출력에서 제거되는지 육안 확인.
2. **AR 임계 탐색**: `--cn-ar-threshold 2.5` vs `3.0` 각 1회 — full run 스킵률 추정(class3 elongated 커버리지 손실 관찰).
3. **텍스처 재추첨**: checkerplate normal 포함 pool로 copy_paste + controlnet 실행(`--texture-dist-threshold` 후보값 2-3개) → annotations.json `normal_image` 분포가 checkerplate 회피하는지 + 재추첨 상한 폴백 경고 로그 확인.
4. **결정론**: 동일 seed·동일 플래그 2회 → 산출물 동일. 필터 전부 OFF(`--cn-ar-threshold 0 --texture-dist-threshold 0`) → 기존 파이프라인과 동일 출력(재현성 보존).
5. **폴백 안전성**: 전 normal이 임계 초과인 극단 pool → 0-output이 아니라 마지막 후보 paste + 경고 확인.

---

## 구현 확정 사항 (2026-07-07 구현 완료 — 계획 대비 변경점)

1. **텍스처 게이트 스코프 반전**: 계획은 random-fallback 한정이었으나, 실측(pilot 이미지에 `_foreground_mask` 적용) 결과 **checkerplate normal이 24~34% 전경을 반환**해 foreground 경로로 배치됨 → fallback 한정 게이트는 주 타깃을 통과시킴. **텍스처 검사는 양 경로 적용**으로 확정 (clean-bg 게이트는 기존대로 fallback 한정).
2. **descriptor는 dilation ring → bbox 4방향 사각 strip 평균**: ring 방식은 자기상관(주기성) 항이 결함 픽셀에 오염됨(노이즈 배경이 c_per=0.67로 측정) → 사각 strip(두께 24px, bbox 밖 = 결함 없음 보장)으로 재설계. 4-D descriptor [std, Laplacian var(tanh), 자기상관 peak, 방향 이방성], 가중 L1 거리 ∈ [0,1].
3. **구조**: `_place_on` nested closure(설계안 A) + `meta["gate_stats"]` pop 집계(설계안 B). 위치 재샘플 예산 = 기존 `max_bg_tries`(20) 재활용, normal 재추첨 상한 `_TEXTURE_MAX_NORMAL_REPICK=5`.
4. **cv2.Laplacian은 CV_32F**: float32 src → CV_64F 조합이 일부 cv2 빌드(4.13)에서 미지원.
5. **로컬 검증 결과** (스크래치 fixture, HEAD 대비): OFF 경로 byte-identical ✓ / ON에서 checkerplate normal 완전 회피(repick 2회) ✓ / ON 결정론 ✓ / annotations 누수 없음 ✓ / descriptor 분리도: 동질 텍스처 거리 0.01~0.06, checkerplate 교차 0.46.
6. **플래그 확정**: `--cn_ar_threshold`(언더스코어 — cn_ 계열 관례), `--texture-dist-threshold`(하이픈 — clean-bg 계열 관례). 가이드 반영: `controlnet_aroma_arm_execute.md`(STEP 5/6, 임계 0.25 시작값), `step4_execute.md`(재현성 경고 포함).

## TODO / 미확정

- **텍스처 거리 임계 기본 권장값**: Colab 시나리오 3에서 튜닝 후 가이드에 기록 (코드 default는 0=OFF 유지).
- **AR 임계 full-run 스킵률**: 시나리오 2로 측정. 과다 시(>20%) 임계 상향 or elongated 전용 타일 생성 트랙 별도 검토(본 노트 범위 밖).
- **random/casda arm 텍스처 필터 미러**(`generate_random.py` line 238-250, `generate_casda.py` line 162-174): arm 간 공정성 원칙상 노출 권장이나 실험 설계 확인 후 별도 노트로 진행.
- **descriptor 구현 세부**(radial bin 수, ring 폭 px, 1단/2단 재시도 횟수 3/5): 구현 시 확정, 상수로 노출.
