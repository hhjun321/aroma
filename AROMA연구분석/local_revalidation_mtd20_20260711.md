# AROMA 개선 framework 로컬 재검증 — mtd 20-ROI (2026-07-11)

> 오늘(07-11) 커밋 13건(`9978826`…`b1bb497`)으로 개선된 framework 전체를 **로컬 mtd**에서 end-to-end 재검증. 이전 5/10-ROI 테스트를 **20-ROI**로 확대. 이 결과로 `colab_execute_new`의 execute 문서를 개선한다.

## 실행 파이프라인 (모두 로컬 CPU)

```
distribution_profiling(신 코드) → roi_selection(top_k=20) → clean_bg_selection(--geometry_prior) → generate_defects(copy_paste, n_per_roi=2)
```
- 데이터: `D:/project/AROMA_DATASET/mtd` (good 956, defect 388). profiling·roi·clean_bg 전부 신 코드로 재생성.
- 산출: 20 ROI → 40 synth(20×2 rep).

---

## 1. phase0 dim 컬럼 (작업 B, `b1bb497`) — ✅ gap 완전 폐쇄

`context_features.csv` 헤더에 `image_w,image_h` 추가 확인. `_image_dim`이 실제 dim 사용:

| 경로 | \|dim − 실제\| L1 (mean) | max | 정확(0px) |
|---|---|---|---|
| grid(구, patch격자 추정) | **63.2px** | 126 | **0 / 956** |
| exact(신, image_w/h) | **0.0px** | 0 | **956 / 956** |

→ 과소추정(≤63px)이 **완전히 사라짐**. 하위호환(구 CSV→grid fallback)도 유지.

## 2. phase0 image_id 고유키 (`31ee0aa`)

mtd stem은 이미 전역 고유(`exp*_num_*`)라 **로직 무변경**(문자열만: good=`_{stem}`, defect=`{class}_{stem}`). 내부 조인 정합 OK, generate_defects `lstrip("_")` 관용 처리 확인. (실효 수혜자는 leather — 본 검증 범위 밖.)

## 3. clean_bg_selection Phase 1/2/3 + size (`ce640a9`…`b2ebd76`) — ✅

- **E1 재현 게이트**: `src_fit_ceiling_mean=0.476` (mtd 기준 ~0.50) → **PASS**. `hist∩` mean 0.460 — mtd는 배경 매칭이 ~random(도메인-조건부, 정직성 유지).
- **Phase 2/옵션1 데이터-유도 3-신호 가중**(no-hardcoding): `w_src=0.559 / w_class=0.282 / w_size=0.160` (lift_src 0.429 / lift_class 0.216 / lift_size 0.123). **size 신호가 유의미 기여**(mtd 배경 크기 편차 有).
- **옵션1·2 size-fit (exact-dim)**: `scale_factor` 리스케일 3/20, min 0.758, **mean 0.976**. exact-dim이라 grid 기반보다 과잉 리스케일이 줄어듦.
- **void 필터**: valid pool 940/956.
- random arm(`clean_bg_random_arm.json`) 대칭대조 방출 확인.

## 4. generate_defects 소비 (`7ecdc5c`,`bfe0264`) — ✅ 전 항목 100%

| 검증 | 결과 |
|---|---|
| clean_bg 소비 telemetry | **used=40 fallback=0 mismatch=0 / 40** |
| 배경 정체성 (rep→pool 배정 일치) | **40 / 40** |
| **위치 정확 반영 (exact-dim, clamp-free)** | **40 / 40** |
| geometry per-class edge+span | 아래 |

**위치 정확 반영 40/40**은 작업 B(exact dim)의 직접 성과 — 구(grid dim)에서는 리스케일 후 위치가 clamp되어 불일치했으나, 이제 effective-wh가 generation 리스케일과 일치해 **precomputed 위치가 그대로 소비**됨.

**Phase 3 geometry (E2 레버) — 실제 클래스 기하 복원:**

| class | 배치 edge+span% | 해석 |
|---|---|---|
| blowhole | **0%** (n=10) | 내부 점결함 → surface 배치 (정확) |
| break | 100% (n=6) | 균열/스팬 → edge 배치 |
| crack | 100% (n=8) | 균열 → edge |
| fray | 100% (n=6) | 연장 결함 → edge/span |
| uneven | 100% (n=10) | 광역 불균일 → span |

→ E2가 지적한 geometry-blind 평탄화(전 클래스 ~50%)가 **클래스별 실제 경향으로 복원**.

## 5. 부수 확인 (Q1 — 산출물 크기)

`roi_candidates.json` = **13.3 MB** vs `roi_selected.json` = **15 KB** (candidates가 **889×**). 단, candidates는 다운스트림(generate_random·build_train_jsonl·exp1/2/6)이 **재소비**하므로 생성 중단 불가 — **슬림 스키마화**가 개선 여지(별도 과제).

---

## 종합 판정

오늘 13개 커밋의 개선 framework가 **로컬 mtd 20-ROI에서 end-to-end 무결 동작**:

- 작업 B(dim 컬럼)로 **위치/스케일 정밀도 완전 확보**(gap 63px→0, 위치 clamp 40/40 해소).
- clean_bg 3-신호 데이터-유도 가중·void 필터·E1 재현 정상.
- generate_defects 소비·배경 정체성·geometry 복원 전부 검증.
- **정직성 유지**: mtd는 배경 hist 매칭이 ~random(mAP 이득 무주장) — 가치는 재현성·대칭대조·기하 정합.

## execute 문서(`colab_execute_new`) 개선 반영 포인트

1. **phase0_execute**: `context_features.csv`에 `image_w/image_h` 신규 컬럼 명시 + image_id 고유키(good=`_{stem}`). phase0 재실행이 dim 컬럼·고유키의 전제.
2. **step3_5_execute**: 가중이 **3-신호(w_src/w_class/w_size)**로 확장, exact-dim 위치·`scale_factor` 필드, E1 게이트는 `src_fit_ceiling_mean`.
3. **step4_execute**: `--geometry_prior` opt-in 시 위치 clamp-free 소비, `clean_bg resolve used/fallback/mismatch` telemetry 확인 문구.
4. **공통**: 구 profiling(컬럼 없음)에서도 clean_bg가 grid fallback으로 무오류 — 재실행 전/후 혼용 무해.
