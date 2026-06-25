# Exp4v2 Severstal 4조건 공정비교 — 다양성·공정성 파이프라인 개선

> 한 세션에서 발견·수정한 7개 변경의 통합 노트. 직전 3조건 결과(baseline > aroma > random)가 **증강이 baseline에 패배**한 원인을 추적하다 복수의 공정성·다양성 결함을 발견, 모두 수정.
> 효과는 재생성 + multi-seed 재실행으로 확정(현재 미확정).

---

## (사용할 skills: feature-dev)

## 개요

직전 exp4v2 3조건(baseline/random/aroma, casda 누락) YOLOv8n multi-class 결과:
`baseline 0.3821 > aroma 0.3595 > random 0.3379` (map50 mean, 3-seed). 증강이 baseline을 못 이김.

원인 조사에서 **3개의 독립 confound + 3개 인프라 결함**을 발견:
1. (핵심·치명) AROMA ROI 선택이 3620 distinct 소스를 **88 distinct crop**으로 붕괴 → CASDA 1692 대비 ~19배 다양성 기아.
2. AROMA/random이 **결함 포함 배경**(context_select)으로 합성 → 무라벨 결함 label-noise.
3. synth pool이 600(real의 0.24:1)로 copy-paste 유효 band(1:1~3:1) 미달 + CASDA(46140)와 pool-size 비대칭.
4. Drive FUSE 순차 I/O로 생성이 비현실적으로 느림.
5. `--condition`이 단일값만 받아 일부 조건 실행 불가.
6. exp4v2 LocalCache(Drive→/tmp)가 워커 수를 Colab 2-vCPU에 묶어(4 워커) seed당 ~22분 throttle.

모두 수정. **핵심은 #1(변경 5)** — 이전 모든 AROMA 결과(MVTec multi-seed 중심주장 포함)가 이 다양성 기아로 오염됐을 수 있음.

---

## 영향도 분석

### 변경하는 상태
- `roi_selection.py` deficit_aware 선택 결과(roi_selected.json)의 distinct (image,bbox) 분포 — **single+multi 모든 AROMA run**.
- 합성 배경 풀(NORMAL_DIR), synth pool 크기(top_k/per_class_cap), 합성/staging 경로 I/O.
- exp4v2 `--condition` 인터페이스.

### 그 상태를 전제로 동작하는 기존 로직
- multi-class class_floor + per_pair_cap fix(`aroma_exp4v2_severstal-synth-multiclass`): **보존**. img_diversity_cap은 직교 skip 조건으로 합성.
- single-class deficit_aware: `img_diversity_cap=None`이면 byte-identical, **default=1로 변경 시 결과 바뀜**(의도 — 옛 결과는 버그).
- random/weighted/top_k 전략: cap 경로 미진입 → 무영향.

### 공정성 (load-bearing)
- 4조건은 **선택 전략만** 달라야 하고 pool 규모·n_per_roi·학습 cap budget·배경 풀·distinct-crop 다양성은 동일해야 함. 위 confound들이 각각 이 불변식을 깸.
- AROMA distinct ~1385 < CASDA 1692 잔차는 **c2 진짜 희소성(117)** 때문 — artifact 아님. 논문에 parity 주장 금지, c2 source-scarcity로 보고.

---

## 수정 내용

### 1. CASDA adapter 양방향 병렬 staging (성능)
`scripts/aroma/casda_roi_adapter.py`, `scripts/aroma/generate_defects.py`, `scripts/aroma/generate_casda.py`
- `adapt()`: `local_staging` 시 manifest(suitability+per_class_cap 통과분)만 ThreadPool 병렬 복사. helper `_stage_workers`(env `AROMA_STAGE_WORKERS`, 기본 16, clamp 1~64) / `_copy_one`(copy2 + 3회 backoff retry + same-size skip) / `_stage_manifest`(dest dir 메인스레드 1회 생성, race 제거). 마스크 1회 디코드(`_bbox_from_mask_array`). copytree-all → manifest 선별.
- `generate_defects._push_outputs()`: 순차 copy2 → ThreadPool 병렬(self-contained, casda_roi_adapter import 안 함 = 순환 회피). same-size skip(--resume), 실패 시 WARN+continue. `_stage_inputs`에 `_is_already_local` 가드(이미 `/content`면 재복사 skip).
- 효과(라이브 검증): 입력 읽기 27분+ → ~30초, 출력 push 순차 ~76분 → 병렬 ~5분, 0 failed.

### 2. exp4_v2 `--condition` 다중선택
`scripts/aroma/experiments/exp4_v2_supervised_detection.py`
- `--condition` → `nargs="+"`, `default=["all"]`. 'all' 포함 시 전체, 아니면 `ALL_CONDITION_KEYS` canonical 순서로 정렬+dedup. 예: `--condition baseline random aroma`.

### 3. synth scale-up (공정 budget)
`exp4v2_execute.md`(Cell B/C), `severstal_exp4v2_multiclass_fix_rerun.md`
- AROMA/random `--top_k 200→1690`(pool 600→5070), CASDA `--per_class_cap None→525`(+`--min_suitability 0.5`, pool ~5076), `n_per_roi=3` 3생성기 동일, `--synth_ratio 1.0` primary(cap 2534).
- 모든 pool ≥ cap → 4조건 동일 budget(trim). 한 번 생성(top_k 1690 / cap 525)으로 1.0/1.5/2.0 sweep 커버.

### 4. 배경 오염 제거 (NORMAL_DIR → train/good)
가이드 `severstal_4cond_rerun_guide.md` P1
- 발견: AROMA/random이 `context_select/context`(**결함 포함**) 배경 사용 → 무라벨 결함이 YOLO에 false-negative 학습 → AROMA/random만 타격(baseline=real만, CASDA=train/good clean 면역).
- `generate_defects` L826 배경은 **context-blind 랜덤**(cell_key는 annotation 저장만, 매칭 안 함) → train/good 전환 시 기능 손실 0, 코드 변경 0.
- 3생성기 전부 `--normal_dir=$AROMA_DATA/severstal/train/good`. context_select(CLIP→KMeans 분포매칭 selector, 결함 필터 아님)는 합성에 불필요.
- 논문 "context-matched 배경 합성" 주장 수정 필요(배경은 랜덤; AROMA context 강점은 ROI/prompt 선택).

### 5. ★source-concentration 버그 fix (핵심)
`scripts/aroma/roi_selection.py`
- 발견: deficit_aware가 3620 distinct 소스풀 → **88 distinct crop**으로 붕괴, 한 이미지 최대 87회 반복(top_k 1690 채움). CASDA 1692 distinct 대비 ~19배 격차 = AROMA 핸디캡.
- root cause: `score_roi`가 image-blind(morph_prior=cluster, ctx_prior/deficit=(cluster,cell) 레벨, image별 항 없음) → 같은 (cluster,cell) bin 내 전 이미지 **동일 score → tie → stable-sort head**(=CSV순 같은 이미지)를 모든 pair에서 반복 선택. `per_pair_cap`은 (cluster,cell)만 막고 per-image 반복 못 막음.
- fix: `img_diversity_cap` 신설. 전역 per-(image,bbox) `Counter`를 모든 선택 루프 skip 조건 + blake2b 결정적 jitter(sub-1e-7, roi_score 1e-6 granularity 미만, 재현 가능) tie-break. cap=1 = 각 source 1회만; 클래스가 진짜 distinct 부족할 때만 bounded repetition + INFO 로그. distinct 통계를 run 로그/roi_summary.md에 출력.
- 검증(in-memory, py_compile): single-class cap=None byte-identical(4전략+floor+per_pair_cap), cap=1 → distinct **88→~1385**(c1/c3/c4 max_rep=1, c2만 117→422 반복), class_floor 균형·per_pair_cap(max≤cap) 유지.

### 6. img_diversity_cap 전체 적용
`roi_selection.py`(default None→1: argparse/`select_rois`/`run` 3 layer + docstring/help/comment) + 전 execute 문서
- belt(default=1) + suspenders(모든 deficit_aware 셀에 명시 `--img_diversity_cap 1`).
- 셀 추가: `step3_execute.md`(기존), `exp1_execute.md`, `severstal_pipeline_execute.md`, `rerun_bbox_pipeline_execute.md`, `severstal_exp4v2_multiclass_fix_rerun.md`(main+sweep), `severstal_4cond_rerun_guide.md`.
- random/weighted/top_k는 cap 경로 미사용 → default 변경 무영향(검증됨). random arm 셀은 불변.
- 사용자 결정: 옛 버그 캐시 보존 안 함, **전체 재실행**.

### 7. LocalCache 워커 throttle 수정 (성능)
`scripts/aroma/experiments/exp4_v2_supervised_detection.py`(`_local_cache_for_yolo`)
- 발견: exp4v2 LocalCache(Drive→/tmp 복사)가 19628 이미지에 ~1303s(~22분) 소요. 조사 결과 코드는 **이미 ThreadPool 병렬**이었고(내 초기 진단 "순차"는 오진, workflow 조사가 정정), 진짜 원인은 워커 수 `num_workers=min(32,(os.cpu_count() or 4)*2)` → Colab 2-vCPU에서 **4 워커로 throttle**. 19k+ FUSE 읽기를 4-wide로 직렬화 = ~22분.
- fix:
  1. 워커를 `_stage_workers()`로 교체 — env `AROMA_STAGE_WORKERS` 재사용(기본 16, clamp 1~64, **os.cpu_count 미사용**으로 Colab 2-vCPU 대응). `generate_defects._push_workers`를 ~15줄 복제(casda_roi_adapter가 generate_defects를 import하므로 여기서 역import = 순환 위험 → 의도적 복제, 새 공유 모듈 안 만듦). 4→16 = ~22분→2-3분.
  2. `_copy_if_missing`(존재만 확인) → `_stage_one`(same-size skip + 3회 backoff retry 0.5/1/2s). multi-seed·`--resume` 재복사 0건 + 중단된 seed의 잘린/부분 파일 치유(존재-only 체크는 부분 파일을 영영 못 고침).
  3. 실패 정책 **RAISE**(push의 WARN+continue와 의도적 차이). 누락 캐시 파일은 downstream에서 조용히 제외(`_image_size`→(0,0)→`n_bad_size`++, 마스크 누락→`n_no_bbox`, synth 경로는 없는 /tmp 파일 가리킴)되어 **에러 없이 파일셋이 바뀜 = 무결성 위반** → fail-loud. retry 소진 시 raise → seed 루프가 catch하여 **해당 seed만 중단**.
  4. dst dir 메인스레드 1회 사전 생성(워커 mkdir 없음), `counts`도 메인스레드만 mutate → shared-state race 없음.
  5. breakdown 로그 추가(train_normal/defect/masks/random/casda/aroma별 건수 + elapsed).
  6. 미사용 `as_completed` import 제거.
- 캐시 파일셋·경로 byte-identical(speed only, 의미 변화 0). py_compile OK.
- 효과: 현재 단일 seed run엔 무효(이미 캐시됨). 3-seed 본실행에서 seed당 ~22분→2-3분 + seed2/3 same-size skip = **~60분 절약**.

---

## 수정 대상 파일

- `scripts/aroma/roi_selection.py` (핵심: img_diversity_cap, default=1)
- `scripts/aroma/casda_roi_adapter.py` (manifest 병렬 staging)
- `scripts/aroma/generate_defects.py` (_push_outputs 병렬, _is_already_local 가드)
- `scripts/aroma/generate_casda.py` (local_staging 전달)
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (--condition nargs; LocalCache 워커 throttle 수정 `_local_cache_for_yolo`)
- `AROMA연구분석/colab_execute/` : `exp4v2_execute.md`, `severstal_exp4v2_multiclass_fix_rerun.md`, `severstal_4cond_rerun_guide.md`(신규 가이드), `step3_execute.md`, `exp1_execute.md`, `severstal_pipeline_execute.md`, `rerun_bbox_pipeline_execute.md`

---

## 테스트

- 정책: pytest 미실행, 신규 테스트 미작성. `python -m py_compile`만 + Colab 검증.
- py_compile: roi_selection.py / generate_defects.py / generate_casda.py / casda_roi_adapter.py / exp4_v2_supervised_detection.py 전부 OK.
- Colab 검증 (필수):
  - ROI 게이트(R §2.1): class 1~4 >0, c2≈117(distinct 천장), 최다 pair ≤ ceil(0.05×1690)=85.
  - **distinct 검증 셀**: AROMA distinct (image,bbox) 88→~1385, random과 동일 자릿수, CASDA 1692. (이게 핵심 회귀 검증.)
  - 4조건 학습 후 `n_synth_train` 3조건 동일(=2534), `n_synth_per_class` 투명 출력.

---

## 재실행 영향

- **모든 AROMA 결과 재생성 필요.** 특히 MVTec multi-seed(중심주장 AROMA>Random)는 source-concentration 버그로 오염됐을 수 있어 재검증 필수.
- Severstal 4조건 재실행 절차 = `severstal_4cond_rerun_guide.md` (Phase 0 pull → 1 AROMA/random 재생성 → 2 CASDA → 3 purge → 4 학습 → 5 검증).
- 변경 7(LocalCache 워커 fix)로 3-seed 본실행에서 **~60분 절약**(seed당 ~22분→2-3분 + seed2/3 same-size skip). 단일 seed run·이미 캐시된 run엔 무효.

---

## 미확정 / 추후 (TODO)

- TODO: inverse-frequency per-class 쿼터(희소 c2에 더, 다수에 적게)는 **ablation으로 분리**(isolate-selection). c2 distinct 117 천장 → 이미지 상한 351(3x)~585(5x), 그 이상은 precision collapse. 필요 시 `generate_defects`에 `--n_per_roi_map`(~15-30줄) micro-fix. 이번 headline(isolate-full-pipeline) 범위 밖.
- TODO: AROMA distinct ~1385 < CASDA 1692 잔차 = AROMA floor 422/class vs CASDA cap 525/class 차 + c2 희소. 완전 매칭하려면 top_k/per_class_cap 정렬 가능(선택).
- TODO: 직전 context_select 오염 크기(p) 측정 — 옛 context_prototypes.json 파일명의 Severstal 라벨 결함 비율.
- TODO: seed 통일 확정(가이드는 42 1337 2025, 직전 run 로그는 42 1 2).
