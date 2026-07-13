# 09 — 호환성 게이트 (AROMA 핵심 방법)

> **Claude 요약:** AROMA는 CASDA의 ControlNet 생성 위에 두 개의 게이트를 얹는다 — (1) **symmetric compatibility 게이트**(`--compat_mode symmetric --compat_threshold τ`): 결함이 놓일 배경 셀이 "clean 배경에도 실재하고 해당 결함 클러스터의 배경 분포에도 실재하는" 셀인지를 SGM 대칭 호환도로 판정하여 부적합 배치를 거부·재배치, (2) **clean-bg 게이트**(`--reject-clean-bg`): 검은/평탄(void) 배경을 항상 거부. symmetric 매트릭스(`matrix_symmetric` 등 4키)는 phase0 `distribution_profiling`이 패치 단위로 계산하고, τ는 데이터셋별 CPU 사전스캔으로 확정한다(τ=0.5 금지, 사후 튜닝 금지).

---

## 개요

CASDA는 ControlNet으로 결함을 생성한다. AROMA는 그 생성 결과를 **어디에 붙일지**를 통제하기 위해 두 게이트를 추가한다:

- **compat 게이트**: 결함 클러스터별로 "clean 배경 분포와 결함-이미지 배경 분포가 모두 지지하는 컨텍스트 셀"에만 배치를 허용한다. 무정보 재조합(copy-paste 천장)·부적합 배경 배치를 차단한다.
- **clean-bg 게이트**: 검은/평탄 배경(void)을 거부한다. 이 게이트는 **항상 ON**이며, 프로젝트 실측에서 AROMA의 random 역전(random-reversal) 및 severstal c2 붕괴(collapse)를 해소한 것으로 기록되어 있다.

두 게이트는 AROMA arm과 random arm에 **동일하게** 적용되어(clean-bg는 양 arm 공통), 두 arm이 배경 선정/품질 조건에서 대칭(symmetric control)이 되도록 한다.

---

## symmetric compatibility matrix

phase0 `distribution_profiling.py`가 `compatibility_matrix.json`에 아래 4개 키를 **패치 단위(64px)**로 추가 계산한다(`_build_symmetric`). 레거시 이미지-평균 `matrix`와 독립.

| 키 | 정의 |
|----|------|
| `clean_dist` | `{cell: prob}` — 전 good-이미지 패치의 컨텍스트 셀 분포(패치 granularity) |
| `P_def_patch` | `{cluster_str: {cell: prob}}` — 클러스터별 결함-이미지 패치의 셀 분포. 각 패치는 이미지의 `cluster_assignments[image_id]`를 상속. 결함 영역(mask>0.5) 패치는 상류 `_context_worker`에서 이미 제외 |
| `matrix_symmetric` | `{cluster_str: {cell: compat_sym}}` — 클러스터별 max-정규화 SGM. 지지집합 `S_k = {c : P_def_patch[k][c] > 0}`에 대해 `compat_sym(k,c) = sqrt((P_def_patch[k][c] + eps)·(clean_dist[c] + eps))`, 그 뒤 각 행을 행별 max로 나눠 peak=1.0 |
| `symmetric_epsilon` | SGM에 사용한 epsilon(기본 `1e-3`) |

- SGM(대칭 기하평균)은 결함 배경 확률과 clean 배경 확률의 기하평균 → **양쪽 모두 실재하는 셀**만 높은 점수. 한쪽만 있으면 eps로 눌려 낮아진다.
- max-정규화로 각 클러스터 행이 `[0,1]`, peak=1.0 스케일 → τ가 이 대칭 스케일 위에서 해석된다.
- **phase0 검증(assert)**: `compatibility_matrix.json`에 `matrix_symmetric`·`P_def_patch`·`clean_dist`·`symmetric_epsilon`이 존재해야 한다. 없으면 `--compat_mode symmetric`이 **hard-fail**(구 profiling으로는 대칭 게이트 불가). drift 대비 기존 profiling 백업 권장.

---

## compat 게이트 (생성 시)

`generate_defects.py` 관련 플래그:

```python
!python $AROMA_SCRIPTS/generate_defects.py \
    ... \
    --compat_mode symmetric --compat_threshold $TAU \
    --compat_matrix_json $PROF/compatibility_matrix.json \
    --config $PROF/recommended_config.yaml
```

- `--compat_mode symmetric`: `compat_matrix = _compat_json["matrix_symmetric"]`를 사용. 키가 없으면 hard-fail(레거시로 조용히 폴백하지 않음 — τ는 대칭 [0,1] 스케일에 맞춰 캘리브레이션되므로 원시 matrix 값에 적용하면 무의미). `--compat_mode defect`(기본)는 레거시 `matrix`를 그대로 사용(byte-identical).
- `--compat_threshold τ`: `τ>0`일 때만 게이트 활성. `--compat_matrix_json`(+`--config`에서 `bin_edges`) 필요.
- `bin_edges`는 `--config`(recommended_config.yaml)에서 확보 — matrix JSON에 `bin_edges`가 있으면 그것도 수용.

배치 수락/거부 로직(symmetric = `compat_tile` ON):

1. **64px 타일링 query**: 매트릭스·`bin_edges`는 64px 패치로 학습됐으므로 결함 footprint를 full 64px 윈도우(`_tile_anchors`, 이미지 내부로 clamp)로 덮고, 각 윈도우의 gray → 컨텍스트 셀 → `compat_row.get(cell, 0.5)`를 조회해 **평균**(`_COMPAT_TILE_AGG='mean'`, 대안 `'min'`)으로 집계. `agg >= τ`이면 수락.
2. **positive placement (scan-rank-place)**: 무작위 샘플 후 reject-resample 대신, 맞는 위치를 모두 스캔해 footprint 64px 타일 평균 compat로 랭킹하고, void 타일을 걸치는 footprint는 버린 뒤 **top-K(`_POS_TOPK=8`)에서 샘플링**(`_POS_STRIDE=32`, 안전상한 `_POS_MAX_CAND=4096`). `best_mean >= τ`이면 gate pass; 미달 시 stage-2 재-pick/강제-paste 폴백으로 위임(선택 위치는 유지).
3. **이미지 단위 배경 선택(symmetric 전용)**: 균일 무작위 배경 추출 대신, 소스 결함 이미지의 배경 셀 히스토그램(`_dv_bg_hist`, 결함 영역 제외)과 각 clean 이미지의 non-void 셀 히스토그램(`_cell_hist`)의 교집합 유사도(`_hist_intersection`)로 clean 풀을 랭킹해 top-K(`_NORMAL_TOPK=16`)에서 샘플(다양성 유지).
- 폴백률(`fb_rate`)이 0.5를 넘으면 로그 경고(τ가 지나치게 높아 대부분 강제-paste로 새는 신호).

레거시(`compat_tile=False`, defect mode/threshold=0)는 crop-size 패치를 그대로 조회해 `compat_row.get(cell, 0.5) >= τ` — 역사적 게이트와 byte-identical.

---

## clean-bg 게이트

```python
--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

- `--reject-clean-bg`: 검은/평탄(void) 배경 거부 게이트 ON. `_is_clean_background`가 이미지·배치 위치를 품질·blur로 평가.
- `--min-bg-quality 0.7`: 배경 품질 하한(0..1, CASDA 기본 0.7). 이 미만 위치는 void로 간주해 거부.
- `--bg-blur-threshold 100.0`: Laplacian-variance blur 게이트(CASDA 기본 100.0). 이보다 흐린(평탄) 배경 거부.
- **항상 ON**: 이 게이트는 AROMA arm·random arm 양쪽에 동일하게 켠다. 프로젝트 실측 기록상 이 게이트가 AROMA의 random 역전과 severstal c2 붕괴를 해소했다(정책: clean-bg gate 항상 ON).
- clean-bg 사전선정(`clean_bg_selection.py`)은 profiling-유래 파일에서 void 하한(관측 분포 P1)·void_frac 컷(P90)을 **데이터-유래**로 산출하고, 전부 거부되면 전체 풀로 폴백(조용한 0-출력 없음). random arm은 동일 ROI 집합에 균일 무작위 배경을 배정(`--emit_random_arm`)해 대칭 통제를 만든다.

---

## τ 사전스캔

τ는 **step4 말미(생성 직전 아님)**에서 데이터셋별 CPU 사전스캔으로 확정한다(`compat_gate_cpu_diagnosis §10`).

- 출력: `S('compat_gate', ds)/compat_tau_prescan_{ds}.json`의 `ds_tau`.
- **`τ=0.5` 금지**: 대칭 스케일에서 0.5는 발동률을 조사하지 않은 임의값(프로젝트 메모리 `feedback_prescan_thresholds` — GPU 의존 임계는 CPU 사전스캔 후 확정).
- **신 profiling 필수**: 사전스캔은 `matrix_symmetric`을 포함한 새 profiling 위에서만 유효(구 profiling으로는 대칭 τ 산출 불가).
- **aitex 추가 게이트**(elongated 결함용):
  - `--cn_ar_threshold AR_T`(기본 2.5): bbox 종횡비가 임계 이상인 ControlNet ROI 스킵.
  - `--texture-dist-threshold TEX_T`(기본 0.0=off): 후보 패치가 소스 배경 텍스처 기술자에서 임계 거리 이내여야 통과(pair-level 텍스처 게이트).
  - `AR_T`/`TEX_T`도 CPU 사전스캔(+pilot)으로 확정(`controlnet_aroma_arm STEP 5-0/5-0b`).

τ·AR·TEX는 step5 생성이 소비하므로 step4에서 모두 확정한다(생성 직전 재발명 금지).

---

## 주의사항

- **사후 튜닝 금지**: τ·seed·synth_ratio·epochs는 결과 보고 후 변경 금지.
- **mtd의 τ를 aitex에 무검증 재사용 금지**: 각 데이터셋의 τ는 자기 사전스캔 확정값. mtd 값 aitex 전용 금지.
- `matrix_symmetric` 부재 시 symmetric 모드 hard-fail — 반드시 신 profiling 재실행 또는 `--compat_mode defect` 명시.
- clean-bg 게이트는 AROMA·random 양 arm 공통(대칭 통제 불변식). 한쪽만 켜면 통제 깨짐.
- 테스트 코드 신규 작성·pytest 금지(CLAUDE.md) — 검증은 Colab 실행으로.

---

## 관련 노트

[[00-INDEX]] | [[02-Stage0-Prepare-Profiling]] | [[05-Stage3-ControlNet-Generation]]
