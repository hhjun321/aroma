# clean-background 게이트 적용 exp3 성능비교 재실행 Colab 가이드 업데이트

## (사용할 skills: micro-fix)

> 경계선이나 micro-fix로 진행. 코드 변경 없음(`.md` 가이드만), 단일 관심사(`--reject-clean-bg` 플래그 주입 + 재생성 안내), 새 파일·추상화·의존성 없음, 변경 < 50줄. 삭제 절차를 대형 신규 실행 셀로 확장하게 되면 feature-dev 승격 검토.

## 개요

clean-background(검은/평탄 배경 거부) 게이트가 엔진(`scripts/aroma/generate_defects.py`)과 래퍼(`generate_random.py`, `generate_casda.py`)에 구현 완료됨 (참조: [[aroma_exp4v2_clean-background-gate]]). 게이트 ON에서 AROMA가 random 역전·c2 collapse 해소 실측 → 프로젝트 정책 "게이트 항상 ON" 확정.

그러나 CLI 기본값은 `--reject-clean-bg` **OFF(store_true)**. 현재 exp3/step4 Colab 가이드의 합성 명령에 플래그가 없어 **게이트 미적용으로 합성**된다. `AROMA연구분석/exp3-sharpened-spec.md`의 성능비교(Random vs AROMA, 4 데이터셋, FID + PaDiM AUROC)를 게이트 ON 정책 하에서 재실행하려면, 가이드 합성 단계에 플래그를 주입하고 **기존 합성본 삭제 후 재생성**해야 한다(이어붙이기 금지 — 게이트로 출력이 바뀜).

CASDA 비교는 exp3 범위 밖(Severstal 전용 exp4v2, 종결 — [[project_casda_aroma_experiment_concluded]]) → 본 작업 미포함.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `AROMA연구분석/colab_execute/step4_execute.md` 합성 명령 (단일 셀 + 배치 병렬 셀).
- `AROMA연구분석/colab_execute/exp3_execute.md` 0단계 Random 합성 명령.
- 두 가이드에 "재생성 전 기존 합성 디렉토리 삭제" 안내 블록 추가.

### 그 상태를 전제로 동작하는 기존 로직
- exp3 FID/AD 평가(`exp3_generation_quality.py`)는 `synthetic/`·`synthetic_random/`를 **읽기만** 함 → 게이트 인자 불필요, 변경 없음. 단 입력이 재생성본이어야 함.

### delete / 재생성 계열 확인
- 삭제 범위가 **AROMA + Random 양쪽**을 모두 덮어야 함. 한쪽만 게이트 적용 시 비교 비대칭 → 무효.
- seed=42 고정이라도 게이트는 거부 로직 → 출력 개수·구성 달라짐. append 시 옛 게이트-미적용 이미지 잔존 → 반드시 삭제 후 재생성.

---

## 수정 내용

### 1. `AROMA연구분석/colab_execute/step4_execute.md` — AROMA 합성에 게이트 플래그 + 삭제 안내

**1-1. 단일 데이터셋 실행 셀** (`!python ... generate_defects.py`, ~33–42행)
- `--local_staging` 뒤에 추가:
  ```
      --reject-clean-bg \
      --min-bg-quality   0.7 \
      --bg-blur-threshold 100.0
  ```
  (`!python $AROMA_SCRIPTS/...` + `$VAR` 형식 유지 — colab-execution.md)

**1-2. 배치 병렬 실행 셀** (셀 2 `run_one()` 내 `cmd` 리스트, ~144–152행)
- subprocess 리스트라 키/값 분리:
  ```
         '--local_staging',
         '--reject-clean-bg',
         '--min-bg-quality',   '0.7',
         '--bg-blur-threshold', '100.0']
  ```
  (`store_true` 플래그는 값 없이 단독 요소)

**1-3. 기존 합성 삭제 안내**
- "## 실행" 섹션 앞에 주의 블록: 게이트 정책으로 출력 변경 → 재생성 전 단일 실행 시 `$SYNTHETIC_DIR`(=`$AROMA_OUT/synthetic/{ds}`), 배치 실행 시 `synthetic/` 하위 대상 데이터셋 삭제. 텍스트 경고 + 선택적 셀(`!rm -rf $SYNTHETIC_DIR`) 형태.

### 2. `AROMA연구분석/colab_execute/exp3_execute.md` — Random 합성에 게이트 플래그 + 삭제 안내

**2-1. 0단계 Random 합성 실행 셀** (`!python ... generate_random.py`, ~64–83행)
- `--n_per_roi 3` 뒤에 추가:
  ```
          --reject-clean-bg \
          --min-bg-quality   0.7 \
          --bg-blur-threshold 100.0
  ```

**2-2. 기존 합성 삭제 안내** (0단계 도입부, ~40–43행)
- 주의 블록: Random(`$RANDOM_SYNTH_DIR/{ds}` = `synthetic_random/{ds}`)과 AROMA(`$AROMA_SYNTH_DIR/{ds}` = `synthetic/{ds}`) **둘 다** 삭제 후 재생성. AROMA는 step4에서 재생성되므로, 본 가이드 전제(5행 "Step 4 완료")가 "게이트 ON으로 재실행된 step4"임을 명시.

**2-3. 평가 명령(1단계 FID / 2단계 AD)** — **변경 없음**. 재생성된 합성 디렉토리를 읽기만 함.

---

## 수정 대상 파일

- `D:\project\aroma\AROMA연구분석\colab_execute\step4_execute.md` (AROMA 합성: 단일 셀 + 배치 셀 + 삭제 안내)
- `D:\project\aroma\AROMA연구분석\colab_execute\exp3_execute.md` (Random 합성 0단계 + 삭제 안내; 평가 셀 불변)

---

## 암묵적 요구사항 / 엣지 케이스

1. **visa object-centric 과다 거부 리스크**: visa_cashew/visa_pcb는 객체 중심 → 배경 평탄도 높아 게이트 과다 거부 가능. 엔진 단 완화 처리됨(injection=both: pool+position, [[aroma_exp4v2_clean-background-gate]]). 가이드 별도 조치 불필요하나 **재생성 후 이미지 수 검증**으로 사후 포착.
2. **삭제 범위 = AROMA + Random 양쪽**: 비대칭 비교 방지.
3. **배치 병렬 셀도 동일 플래그**: 단일 셀만 고치면 4-데이터셋 일괄 재생성 시 게이트 미적용본 생성. 1-1·1-2 동시 수정 필수.
4. **재생성 후 이미지 수 ≥ 50 확인**: exp3_execute.md 기존 "이미지 수 확인" 셀(~85–95행)을 재생성 후 실행, random/aroma 양쪽 카운트 확인. `fid_unstable`(real 패치 <50)과 별개로 합성본 자체도 게이트로 줄 수 있음.
5. **test/defect는 real-only 유지**: 게이트는 합성(train)에만 작용. 평가 test set은 real 결함만 — 가이드 주의사항(212행) 명시. 본 작업은 합성 디렉토리만 손대므로 불변식 유지.
6. **seed=42 유지** + 삭제 후 재생성 (append 금지).
7. **`store_true` vs 값 인자 구분**: subprocess 리스트에서 키/값 별도 요소.
8. **CASDA 미포함 재확인**: exp3에 generate_casda.py 호출 없음 — 추가 안 함.

---

## 테스트 (Colab, pytest 금지)

코드 변경 없음 → 빌드/유닛 불필요. Colab 실행으로 가이드 정합성·게이트 적용 검증:

1. **삭제 확인**: `synthetic/{ds}`, `synthetic_random/{ds}` 삭제 후 디렉토리 부재 확인.
2. **AROMA 재생성(step4)**: 1개 데이터셋 재생성, 콘솔 clean-bg reject 로그 출력 확인 → 게이트 작동 확인.
3. **Random 재생성(exp3 0단계)**: `generate_random.py` 플래그 전달 후 게이트 로그 확인.
4. **이미지 수 검증**: random/aroma 양쪽 카운트, 각 데이터셋 ≥ 50 (특히 visa 2종 과다 거부 점검).
5. **FID/AD 실행**: 1·2단계 평가 → `exp3_results.json` 생성, `fid_unstable`·AUROC 방향(AROMA > Random > Baseline) 확인.
6. **불변식 확인**: test/defect real-only(합성 미혼입) 확인.

부하/성능 측정 항목 없음(load-test-policy 무관). PaDiM 학습은 기능 동작 검증 목적 → 허용.

---

## 미확정 사항 (TODO)

- **삭제: 코드 셀 vs 텍스트 안내**: `!rm -rf`는 위험 → 텍스트 경고 + 선택적 셀 권장. 최종 결정 필요.
- **배치 삭제 범위 문구**: 단일(`DATASET_KEY`만) vs 전체(`synthetic/` 전부) 구분 안내 문구 확정 필요.
- **visa 과다 거부 폴백**: 재생성 후 이미지 수 50 미만 시 대응. 정책 "게이트 항상 ON" 고정 → 본 범위 밖, 발생 시 "보고"만 남길지 결정.
- **`min-bg-quality`/`bg-blur-threshold` 명시 vs 생략**: CLI 기본값과 동일(0.7 / 100.0)이라 생략 가능하나, 정책 가시성 위해 **명시** 채택(현 계획).
