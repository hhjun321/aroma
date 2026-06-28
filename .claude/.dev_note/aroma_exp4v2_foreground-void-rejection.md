# AROMA exp4v2 — `_foreground_mask` 전경 void 거부 가드 (검은배경 결함 Mode A 수정)

---

## (사용할 skills: feature-dev)

> **설계 출처**: `design-foreground-mask-fix` 워크플로우(10 에이전트: grounding 3 → 독립 설계 3안 → 적대적 검증 3 → 종합). 점수 minimal 4.5 / robust 5.5 / **principled 7.5** → principled 기반 + 적대적 검증이 잡은 두 구현 결함 보강(minimal의 1-D flatten이 `_background_quality_score`의 Laplacian/filter2D를 깨뜨림, robust의 masked-fill이 소수면적 객체 희석).
> **진단 근거**: [[aroma_exp4v2_aroma-underperformance-diagnosis]] §3.1/§3.2 (Colab 실측 — 검은배경 35장 중 94% foreground 경로 우회).

## 개요

severstal 합성에서 검은/평탄(void) 배경 위에 결함이 붙는 샘플이 ~94% 발생하는 문제(Mode A)를 수정한다. 근본 원인은 `_foreground_mask`의 corner-vote 극성 판정 자체가 아니라, **선택된 전경이 실제로 객체인지 검증하는 단계가 누락**된 것이다. severstal full-frame strip은 마진 void가 커서 4개 코너가 대부분 밝고(`bright_corners >= 2`), 그 결과 어두운 클래스가 전경으로 선택되며 그 어두운 클래스의 largest connected component가 곧 void다. 결함 centroid가 이 void에 배치된다(`_foreground_paste_position`).

본 수정은 `_foreground_mask`에 **전경 void 거부 가드**를 additive로 추가한다. corner-vote 극성(L214-217), Otsu(L202-204), largest-CC(L220-228), ratio gate(L230-232)는 전부 그대로 두고, 선택된 전경이 "평탄 + 검음" void이면 극성을 뒤집지 않고 `None`을 반환해 기존 random-placement 폴백으로 보낸다.

## 영향도 분석

- **대상 함수**: `_foreground_mask` 단일 함수 본문. 시그니처/호출부(`copy_paste_synthesis`) 미변경.
- **dark-object 안전(visa_pcb 등)**: 3중 가드로 보호. (1) corner-vote 극성 미변경, (2) void 판정은 AND 조건(평탄 std AND 검은 mean) — 어둡지만 텍스처 있는 정상 객체는 std가 임계를 넘어 통과, (3) 거부 시 `None` 폴백만, **극성 inversion 절대 금지**(grounding이 경고한 silent semantic inversion 회피). quality는 검증(reject)에만 쓰고 재선택(re-rank)에 쓰지 않는다.
- **mvtec_cable**: 대부분 ratio gate(<2%/>90%)에서 이미 `None`; 새 블록 도달 시에도 케이블 텍스처는 std가 높아 no-op.
- **결정론**: 내부는 RNG-free(Otsu, std/mean, Laplacian/filter2D)라 동일 seed→동일 출력은 보존. 단 **레거시 byte-identical은 보장 못 함** — 호출부가 단일 seed rng(run() 오케스트레이터)를 배치 전체에서 순차 소비하므로 한 장이라도 mask→None으로 바뀌면 그 iteration의 RNG 소비량이 달라져 이후 모든 이미지의 스트림이 이동한다("deterministic-but-stream-shifting"). 데이터셋 분리 실행이라 교차 오염은 제한적. ⚠️ 구현 시 정확한 rng/호출부 라인(run() 내 `random.Random(seed)`, `rng.choice(normal_images)`)을 재확인할 것.
- **cv2 의존성**: 신규 추가 없음(`_foreground_mask`가 이미 cv2 사용, `_background_quality_score` 재사용).
- **성능(Node hot-path 무관 — 오프라인 Python 생성 파이프라인)**: 전경마다 std/mean 2회(저비용) + void 후보에만 bbox crop quality 1회. per-image O(1), Python 픽셀 루프 없음.

## 수정 내용

### 1) 모듈 레벨 상수 신설 (`_foreground_mask` 정의 위)

```python
# 선택된 전경이 평탄+검은 void인지 판정하는 임계 (Colab 튜닝 대상).
# AND 조건: 평탄(std) AND 검음(mean) 둘 다여야 void 후보 → 어둡지만
# 텍스처 있는 정상 객체(visa_pcb PCB, dark wood/metal)는 통과.
FG_VOID_STD = 5.0        # 선택 전경 픽셀 std가 이 미만 = 평탄 후보
FG_VOID_MEAN = 25.0      # 선택 전경 픽셀 mean이 이 미만 = 검음 후보
FG_VOID_QUALITY = 0.5    # 전경 bbox crop quality가 이 미만 = void 확정
                         # (검은 void ~0.43, 배경게이트 0.7보다 충분히 낮춤)
```

### 2) `_foreground_mask` 본문 — L228 직후 / L230(ratio gate) 직전 삽입 (try/except 내부)

```python
fg_mask = (labels == largest).astype(np.uint8) * 255

# ── 전경 void 거부 가드 (Mode A) ─────────────────────────────
# corner-vote(위)는 극성을 결정한다. 이 단계는 그 결과를 '검증'한다:
# severstal 마진 void가 전경으로 선택되면(밝은 void 코너 → 어두운 클래스
# 선택 → 그 largest CC가 곧 void) 결함이 void에 배치된다. 평탄 AND 검음
# 1차 → 2-D bbox-crop quality 2차로 void를 확정하고, 극성을 뒤집는 대신
# None으로 degrade해 random 폴백(+ reject_clean_bg position 게이트)으로
# 보낸다. 정상 dark-object는 텍스처(std)로 1차에서 통과 → 레거시 동일.
sel = fg_mask >= 128
if np.count_nonzero(sel) > 0:
    fg_pixels = gray[sel].astype(np.float32)        # 1-D, orientation-free 통계만 사용
    fg_std = float(np.std(fg_pixels))
    fg_mean = float(np.mean(fg_pixels))
    if fg_std < FG_VOID_STD and fg_mean < FG_VOID_MEAN:
        # 평탄 AND 검음 → 2-D bbox crop을 quality 공식으로 확정 검사
        # (1-D flatten은 Laplacian/filter2D를 깨고, masked-fill은 희석 →
        #  largest-CC의 axis-aligned bbox crop을 그대로 점수화)
        left = int(stats[largest, cv2.CC_STAT_LEFT])
        top = int(stats[largest, cv2.CC_STAT_TOP])
        cw = int(stats[largest, cv2.CC_STAT_WIDTH])
        ch = int(stats[largest, cv2.CC_STAT_HEIGHT])
        crop = gray[top:top + ch, left:left + cw]
        if crop.size > 0 and _background_quality_score(
            crop.astype(np.float32)
        ) < FG_VOID_QUALITY:
            return None     # void 확정 → 폴백

ratio = float(np.count_nonzero(fg_mask)) / float(h * w)   # (기존 L230, 미변경)
if ratio < 0.02 or ratio > 0.90:
    return None
return fg_mask
```

> `stats`(L220), `largest`(L227), `gray`(L195~197)는 모두 함수 내 기존 변수. 추가 cv2 호출 없음. `crop`은 `gray`와 동일 좌표계 (둘 다 `(h, w)`).

### 3) docstring 보강 (코드 동작 변경 없음)

`_foreground_mask` docstring(L168-183)에 void-rejection 단계 추가: "corner-vote가 극성을 결정한 뒤, 선택 전경이 평탄+검은 void(severstal 마진)이면 `None`으로 degrade해 random 폴백으로 보낸다. 극성을 뒤집지 않으므로(inversion 금지) 정상 dark-object 데이터셋은 영향 없음. 효과 극대화는 호출부의 `--reject-clean-bg`(기본 OFF) 동반 필요."

### 4) 실행 권고 (코드 아님 — 가이드 문서에 명시)

severstal 합성 시 `--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0` **동반**. 본 가드는 deterministic void-centroid 타게팅(94% 근본원인)을 제거하지만, 폴백 random이 void에 재착지하지 않게 막는 position 게이트는 `reject_clean_bg=True`(기본 False)일 때만 동작하므로 동반이 효과의 전제다.

## 엣지 케이스

- **HAS_CV2 False**: `_foreground_mask`가 L185-186에서 이미 `None` 반환 → 새 블록 미도달. cv2 의존성 신규 추가 없음, 회귀 없음.
- **전경 픽셀 0개**: `np.count_nonzero(sel) > 0` 가드로 건너뜀.
- **정상 어두운 텍스처 객체(visa_pcb PCB, dark wood/metal)**: 표면 텍스처/에지로 `fg_std`가 `FG_VOID_STD`(5.0)를 넘어 1차 AND 불성립 → 2차 미진입 → `fg_mask` 그대로. **dark-object 핵심 가드.**
- **severstal 마진 void(평탄 ~0 밝기)**: `fg_std`~0-2 < 5.0 AND `fg_mean`<15 < 25.0 → 1차 통과 → bbox crop quality ~0.43 < 0.5 → `None` → 폴백. (Mode A 경로)
- **전경 일부만 void(강판+인접 void 혼합 CC)**: mostly-void면 1차 걸리고 2차 crop quality 낮아 `None`(정답), mostly-steel면 std 높아 1차 통과(정답). 경계 ~50/50은 임계 근방 소수 케이스.
- **새 블록 내 예외**: 기존 try/except(L188/L235)가 감싸 `None` 반환 → 안전 폴백.
- **밝은 void(드문 케이스)**: `fg_mean`<25 불성립 → 새 블록 미발동. 관측된 실패는 '어두운 void 전경'이라 AND(std,mean)가 그 시그니처를 정확히 타겟. 밝은 평탄 이상치는 호출부 position 게이트 영역.

## 수정 대상 파일

- `scripts/aroma/generate_defects.py`
  - 모듈 상수 `FG_VOID_STD` / `FG_VOID_MEAN` / `FG_VOID_QUALITY` 신설 (`_foreground_mask` 위)
  - `_foreground_mask` 본문 L228~L230 사이 void 거부 블록 삽입 + docstring 보강

## 범위 밖 (별도 작업 — 2순위)

- **Mode B**(거대 void가 random-fallback 경로로 유입되는 ~6%): pool 게이트(`load_normal_images`)가 whole-image 평가라 무효(R=0, noise항이 평탄 void 보상)인 문제. 본 수정 범위 밖. pool 게이트를 패치/컴포넌트 단위로 바꾸거나 `--reject-clean-bg` 기본값 정책을 재검토하는 별도 작업. 본 수정으로 94% 경로 제거 후 잔여 검은배경을 STEP 3에서 측정한 뒤 착수 판단.

## 테스트 (Colab)

> 프로젝트 규칙: 새 테스트 코드/pytest 금지. `clean_background_gate_verify.md`의 **Cell 4B**(실제 `_foreground_mask` import probe)를 재사용해 수정 전/후를 대조한다.

1. **수정 전 baseline**: Cell 4B Part A(severstal)로 non-None 비율(~92%)·'어두운 전경' 비율(~27%) 기록, Part B로 검은배경 샘플 중 `via_fg`(전경경로 우회, ~94%) 기록.
2. **dark-object 무회귀 우선**: 수정 후 visa_pcb / mvtec_cable에 Cell 4B Part A 실행 → `_foreground_mask` None-rate가 수정 전 대비 사실상 불변(신규 None ≈ 0)인지. Cell 3 `n_generated` ON/OFF ratio ≥ 0.8 유지.
3. **severstal Mode A 효과**: 수정 후 Cell 4B Part A로 '어두운 전경' 비율이 baseline 대비 크게 감소했는지, Part B로 `via_fg` 건수가 줄고 그만큼 `via_fallback`으로 이동했는지(void 전경이 `None`으로 degrade된 증거).
4. **end-to-end 검은배경 감소**: `--reject-clean-bg` 동반 severstal 재합성 후 Cell 4(배경 mean<30 & std<12 비율)가 수정 전 대비 감소했는지. 동반 플래그 없는 경우(기본 OFF)도 측정해 'centroid 타게팅 제거'만으로의 부분 개선폭 별도 기록.
5. **결정론**: Cell 6로 동일 seed 2회 → `normal_image` 선택 시퀀스 동일(same==True). (동일 seed 재현성이며 레거시 byte-identical이 아님을 보고서에 명시.)
6. **임계 튜닝**: 2/3에서 정상 전경이 새거나 void가 통과하면 Cell 4B Part A 측정 분포에 맞춰 `FG_VOID_*` 상수만 조정 후 2~4 재실행(셀 추가 없음).

> 부하/속도 측정은 load-test 정책상 자동 실행하지 않는다. "void 전경이 줄었는가" 기능 검증만 수행.

## 검증 결과 (Colab 실측, 2026-06-28)

수정본 + `--reject-clean-bg` 로 severstal 재합성 후:

- **Cell 4B Part A** (normal에 `_foreground_mask` probe): None 7.8%→**18.7%**, '어두운 전경' 26.8%→**17.0%**, 전경밝기 median 118→123. 가드가 void 전경 65개 거부 확인(결정론 유지 — 재실행 byte-identical).
- **Cell 4B Part B** (GATE 디렉토리, 실제 합성물): 검은배경 via_fg **33→9**(aroma)/**33→9**(random) — Mode A(전경경로 void 타게팅) 붕괴. via_fallback **~2→14/19** — 빠진 절반이 폴백으로 새서 다시 void 착지.
- **Cell 4** (검은배경 비율 OFF vs ON): aroma **9.1%→6.0%(−34%)**, random **8.8%→7.0%(−20%)**.

**판정**: fix는 **검증된 부분 성공** — Mode A 제거 + 검은배경 실측 감소. 잔여 6~7%는 **Mode B(폴백이 거대-void normal에 재착지, 잔여의 ⅔)** + 약한 void(std 5~12, via_fg 9개)가 지배.

**전략적 함의**: fix는 3조건 대칭(Part B aroma via_fg==random==9)이라 "증강 vs baseline"(절대)에 기여할 뿐 "aroma vs casda"(상대)는 거의 안 바꿈. Mode B로 더 줄여도 대칭이라 상대순위 불변 → **Mode B는 exp4v2가 black-bg를 binding constraint로 지목할 때만 착수**(2순위 유지, speculative 추진 금지). 다음 작업 = production 재합성(random top_k 1690) → exp4v2 재평가.

### Downstream 검증 (exp4v2, severstal, n=2 seed, ratio 0.4)

production 재합성(3조건 top_k 1690/cap 525, parity 확인 — 셋 다 cap 1013 trim) 후 map50 avg:
- baseline 0.3965 ≈ **casda 0.3903 ≈ aroma 0.3872** ≫ random 0.3397.
- old 대비 aroma **+0.066**(두 seed 모두 ↑), casda +0.021, random −0.022, baseline 불변(sanity ✓).
- ★ **연구 crux 충족**: aroma ≈ casda (Δ−0.003, per-seed 주고받음) — 범용이 특화와 동률. **c2 붕괴 해소**: aroma c2 0.082→0.194(≈baseline). aroma > random(ROI 선택 가치 입증).
- ⚠️ 미충족: 증강이 baseline 추월 못함(aroma·casda 둘 다 ≈ baseline; ratio 0.4 synth=real의 28%라 효과 작음). → **ratio 1.0 + n=3**가 다음 확인(재합성 불필요, 학습만, fresh output_dir).
- 📝 위 "전략적 함의"의 *"fix 대칭 → 상대순위 불변"* 예측은 **틀림**: fix가 aroma를 상대적으로도 끌어올림(aroma>random, 특히 c2). 검은배경이 aroma ROI에 더 해로웠던 듯(deficit-aware 결함이 void에 묻혀 무력화 → 유효 배치로 회복).

## 미확정 사항 (TODO / 구현 시 결정)

- `TODO`: `FG_VOID_STD=5.0 / FG_VOID_MEAN=25.0 / FG_VOID_QUALITY=0.5`는 grounding 추론값 — Cell 4B Part A 실측 분포(전경 std/mean/quality)로 calibration 필요. visa_pcb 어두운 PCB가 우연히 세 임계를 모두 만족하면 폴백으로 샘(파괴 아님, object-centric 배치만 상실) → 1순위 확인.
- `TODO`: 전경 largest-CC bbox crop이 강판+void 혼합일 때 2차 quality 변별력. 우선 bbox 그대로(단순·2-D 보존), Colab 분포 보고 fg_mask 재마스킹 여부 결정.
- `<결정 필요>`: `--reject-clean-bg` 기본값을 severstal 한해 True로 바꿀지(코드 변경, 다른 파이프라인 영향) vs 실행 가이드 문서로만 강제할지 — 본 수정 범위 밖, 정책 결정.
- `<확인 필요>`: 구현 시 determinism 분석이 참조한 rng/호출부 라인(run() 내 `random.Random(seed)`, `rng.choice`)을 실제 코드로 재확인.
