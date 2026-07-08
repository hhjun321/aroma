# AROMA CCI 무차원화 — expected_range 데이터드리븐 전환 (opt-in, 기본 OFF)

---

## (사용할 skills: feature-dev)

> **진단 근거**: `.claude/.etc/roi_approve/roiCheck.md` 진단 1-1(입력 무차원화). CCI/MCI 스칼라 최종 minmax 정규화가 데이터셋 통계 미적응 상수 사용 확인 (2026-07-08).
> **관련**: [[aroma_phase0_data-driven-anchor]](phase0 데이터드리븐 앵커), [[aroma_step1_cci-bootstrap-sampling]], [[aroma_step1_mci-redesign-class-diversity]].

## 개요

`compute_complexity.py`의 CCI/MCI 스칼라 최종 minmax 정규화가 하드코딩 `expected_range` 상수를 쓴다(예: `texture_entropy:[0,8]`, `valley_count:[0,41]`, `freq_complexity:[0,1]`). 무차원화 자체는 되어 있으나 스케일 앵커가 **전역 상수**라 데이터셋 통계에 적응하지 않는다. 이를 **per-dataset percentile 기반 범위**로 전환하되, **opt-in 기본 OFF**로 설계해 기존 실험 산출물(`.claude/.etc/complexity/*/complexity_report.json`)을 byte-identical 보존한다.

**핵심 위험**: MCI 스칼라는 정책 선택에서 **절대 임계 비교**됨(`high_complexity_mci=0.6`, `prune_low_mci=0.3`). MCI 범위를 데이터셋별로 바꾸면 정책 선택이 달라져 하위 stage 전체에 캐스케이드. **CCI는 리포트·로그 전용이라 안전.** → **1차는 CCI-only adaptive** 권장, MCI adaptive는 별도 dev_note로 분리.

## 영향도 분석

### expected_range 상수 전체 목록 (compute_complexity.py 근처 107-130)
- **MCI** (`_DEFAULT_CONFIG["mci"]["expected_range"]`): `entropy:[0,4]`, `valley_count:[0,41]`
  - `class_diversity`는 expected_range 미사용(`log(n_eff)/log(8)` 별도 정규화, 근처 490-492). `inv_silhouette`는 이미 [0,1](`_clamp01`만).
- **CCI** (`_DEFAULT_CONFIG["cci"]["expected_range"]`): `texture_entropy:[0,8]`, `cluster_count_ctx:[1,8]`, `freq_complexity:[0,1]`, `orient_variance:[0,1.6]`

### 이 기능이 변경하는 상태
- adaptive OFF(기본): 변화 없음, byte-identical.
- adaptive ON: `compute_cci` 반환 `cci` 스칼라 값 변화 → `complexity_report.json`의 CCI 수치·provenance 변화. (MCI는 CCI-only 설계상 불변.)

### 그 상태를 전제로 동작하는 기존 로직
- **MCI 절대 임계 비교 — 건드리면 안 됨**: `run_meta_policy_generator` → `_get_candidate_policies`(근처 657): `if mci >= 0.6: hierarchical 추가`(672), `if mci < 0.3: 계층/log_gmm 제거`(675). 이 정책 후보가 클러스터링 정책 → morph_labels → roi_selection cluster 할당 전체를 좌우. **MCI 범위 변경 = 실험 재현성 붕괴.** → 1차 adaptive는 **CCI에만** 적용.
- **CCI 소비처 = 리포트/로그뿐**: grep 결과 `cci`는 `complexity_report.json` 기록(근처 888)·로그 출력에만. 절대 임계 비교 없음. roi_selection.py는 CCI/MCI 직접 미소비. → CCI adaptive는 정책·클러스터링에 무영향.
- **z-score 경로는 별개**: `_normalize_array`(근처 348)는 MCI 내부 morph_X 표준화(silhouette 입력)용. expected_range와 무관 — 전환 대상 아님.

### 이미 존재하는 opt-in 인프라
`expected_range`는 **이미 config-overridable**: `load_config` → `_deep_merge(_DEFAULT_CONFIG, YAML)`(근처 167-174, 재귀 병합). `cfg["cci"]["expected_range"]`를 런타임/YAML 주입 가능. `_normalize_scalar`(근처 330) 본문 불변, **범위만 교체**.

### delete/remove/bulk 계열
- 해당 없음.

## 수정 내용

### 1. `scripts/aroma/compute_complexity.py` — adaptive range helper (신규)

Phase 0 산출물(`context_X` 등)에서 CCI 4개 성분의 percentile 기반 범위 계산 helper 추가:
- 입력: context feature 배열. `np.isfinite` 마스킹(기존 `_col_mean`/`_col_var` 패턴 재사용).
- 출력: `{성분: [lo, hi]}` — 경계는 `[p1, p99]` (TODO: 경계값 확정).
- degenerate 방어: `hi <= lo`거나 `n < 2`면 해당 성분 **상수 fallback**(현행 `_DEFAULT_CONFIG` 값 유지).

### 2. `scripts/aroma/compute_complexity.py` — override 배선

`compute_cci`(근처 877-878) 호출 **직전**, adaptive ON이면 계산된 range로 `cfg["cci"]["expected_range"]` in-place override. `_normalize_scalar`·`_DEFAULT_CONFIG` 불변. **MCI 경로는 손대지 않음.**

### 3. `scripts/aroma/compute_complexity.py` — CLI + provenance

- `_parse_args`(근처 918)에 `--adaptive_range` (`store_true`, 기본 False), `main`(근처 950) 스레딩.
- adaptive ON이면 계산된 range를 `complexity_report.json`의 `cci_components`/`provenance`에 기록(사후 추적).

## 수정 대상 파일

- `scripts/aroma/compute_complexity.py` (단일 파일)
  - 신규 helper (percentile range)
  - `compute_cci` 호출 직전 override
  - `_parse_args`/`main`에 `--adaptive_range`
  - `_normalize_scalar`(330)·`_DEFAULT_CONFIG`(107) **불변**

## 암묵 요구사항 (엣지케이스/하위호환)

- **빈 데이터**: `context_X` 비었을 때(`empty_context`, 근처 599-605/874) percentile 불가 → 상수 fallback. `n_ctx==0` 가드.
- **degenerate range** (`p99==p1`): `_normalize_scalar`가 `span<1e-9 → 0.0` 이미 방어(근처 336). percentile 단계에서도 `hi<=lo`면 상수 fallback 강제.
- **NaN/inf**: `_normalize_scalar` `math.isfinite` 방어(근처 332). percentile 입력 `np.isfinite` 마스킹.
- **byte-identical**: 플래그 기본 OFF → 기존 `complexity_report.json` 완전 동일. opt-in 켤 때만 값 변화. 프로젝트 컨벤션 충족.
- **무코드 대안**: `load_config` YAML 병합으로 데이터셋별 expected_range 수동 지정도 이미 가능 — 자동 계산 없이 최소 대응 시 병기.

## 테스트 (Colab 검증 — 테스트 코드 작성·pytest 금지)

1. **OFF byte-identical**: adaptive 미지정 → 기존 `.claude/.etc/complexity/*/complexity_report.json`과 동일.
2. **ON range 변화**: `--adaptive_range` → CCI 성분 range·최종 cci 값 변화, provenance 기록 확인. **MCI 값·정책 선택 불변** 확인(정책 로그 대조).
3. **degenerate/빈 데이터 fallback**: 소수 결함 데이터셋에서 상수 fallback 발동 로그 확인.

## TODO / 후속

- **adaptive 적용 범위**: CCI-only(권장, 안전) vs CCI+MCI. MCI 포함 시 `high_complexity_mci`/`prune_low_mci`를 절대→상대(percentile) 전환해야 정합 — **별도 dev_note로 분리**.
- **percentile 경계**: `[p1,p99]` vs `[min,p99]` vs `[p5,p95]` (outlier robust vs 정보손실).
- **산출물 표기**: report에 adaptive flag 남겨 상수/적응 리포트 혼동 방지.
- **무코드 YAML override만으로 충분한지** 자동 계산까지 구현할지 결정.
