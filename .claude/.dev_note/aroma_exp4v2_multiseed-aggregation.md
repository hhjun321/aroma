# Exp4v2 Multi-Seed 반복 + mean±std 집계

## (사용할 skills: feature-dev)

## 개요

exp4v2 단일 seed 결과는 작은 val(18~42장)에서 노이즈가 커 AROMA vs RANDOM 차이(평균 +1.7pp)의 통계적 유의성을 판정 불가. multi-seed(예: 42,1,2) 반복 후 조건별 **mean±std + 95% CI**를 산출해, delta가 noise floor 밖인지 정량 확인한다. seed는 이미 train/val split·synth subsample·YOLO train 전체를 관통하므로, seed만 바꿔 반복하면 독립 추정치가 나온다.

## 영향도 분석

### 변경 상태
- `--seeds` 인자(nargs+, default `[42]`) — 하위호환: 단일 seed=기존 동작.
- 결과 JSON 스키마 확장: top-level은 **평균값 유지**(map50 등 → 기존 plot/문서 reader 안 깨짐), `per_seed{}`·`std{}`·`ci95{}`·`n_seeds` 추가.
- checkpoint 경로에 `seed{N}` 레벨 추가 → seed 간 덮어쓰기 방지.
- per-seed 중간 결과 파일 + 최종 집계 파일.

### 그 상태를 전제로 동작하는 기존 로직 (하위호환 유지)
- `plot_exp4v2_results.py` `load()`: `results[ds][model][cond]["map50"]` 읽음 → top-level을 평균으로 유지하면 그대로 동작(평균 그림).
- 문서 결과확인 셀: 동일하게 top-level 평균 읽음.
- resume: per-seed JSON 단위로 기존 resume 재사용.

### 회귀 위험
- `--seeds 42` (또는 미지정) → 기존과 동일 단일 실행. per_seed에 seed 1개, std=0.
- checkpoint 경로 변경: 기존 `{ds}/{model}/{cond}/`에 best.pt 있던 구조 → `{ds}/{model}/{cond}/seed{N}/`. 기존 캐시 best.pt는 경로 달라져 1회 재생성(무해, scratch 설계라 입력으로 안 읽음).

## 이론적 근거

- 소표본 mAP 표준오차 ≈ √(p(1−p)/N_val). val 18~42장 → SE 수 pp. 단일 seed delta가 SE 안에 묻힘.
- multi-seed 반복은 train/val split·subsample 무작위성을 평균화 → mean이 기대성능, std가 추정 변동성. delta > 2·std/√k 면 유의(rough).
- seed별 독립 split이므로 paired/unpaired 비교 가능. 보고는 mean±std + 95% CI(t-분포, 소표본).

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py`

**argparse**: `--seed`(기존, 단일) 유지하되 `--seeds`(nargs="+", type=int, default=None) 추가. `seeds = args.seeds or [args.seed]`.

**`run()` 시그니처**: `seeds: Optional[List[int]] = None` 추가. `seed_list = seeds or [seed]`.

**run() seed 루프 (핵심)**:
```python
per_seed_results = {}
for s in seed_list:
    seed_out = Path(output_dir) / "_seeds" / f"seed{s}" / "exp4v2_results.json"
    existing_s = _load_existing(seed_out) if resume else None
    res_s = _run_detection_mode(..., seed=s, existing_results=existing_s,
                                output_path=str(seed_out))
    per_seed_results[s] = res_s
agg = _aggregate_seeds(per_seed_results, seed_list)
save_json(agg, output_path)          # 최종 집계 (top-level 평균 + per_seed/std/ci95)
_write_summary(agg, ...)             # mean±std 표
```

**`_run_detection_mode`**: checkpoint 경로에 seed 반영. `ckpt_base = Path(output_dir)/ds` → seed별 분리 위해 `ckpt_base = Path(output_dir)/"_seeds"/f"seed{seed}"/ds` (또는 condition 경로에 seed{N} 삽입). 나머지 로직 불변. (seed는 이미 파라미터로 받아 split/subsample/train에 사용 중.)

**신규 `_aggregate_seeds(per_seed_results, seed_list)`**:
- 각 (ds, model, cond)에 대해 seed별 metrics 수집.
- map50/map50_95/precision/recall: mean, std(sample, ddof=1), ci95(t-분포 또는 1.96·std/√k).
- 반환 구조:
```python
results[ds][model][cond] = {
    "map50": <mean>, "map50_95": <mean>,
    "precision": <mean>, "recall": <mean>,       # top-level=평균 (하위호환)
    "n_train": ..., "n_real_train": ..., "n_synth_train": ...,  # seed 동일/대표값
    "n_seeds": k,
    "per_seed": {s: {map50,..,recall} for s in seeds},
    "std": {"map50": .., "map50_95": .., "precision": .., "recall": ..},
    "ci95": {"map50": [lo,hi], ...},
    "weights_path": <seed-0 baseline>,  # 참고용
}
```

**신규 `_write_summary` (또는 기존 _build_summary 확장)**: 조건별 `mean ± std` 표 + Delta(AROMA−Random, AROMA−Baseline) mean±std. n_seeds 명시.

### 2. `AROMA연구분석/colab_execute/exp4v2_execute.md`

`--seeds 42 1 2` 사용 예시 + 결과확인 셀에 mean±std/CI 출력 추가. 소요 시간 = 단일 × seed 수 (3seed ≈ 3배) 명시.

## 수정 대상 파일
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
- `AROMA연구분석/colab_execute/exp4v2_execute.md`

## 암묵적 요구사항 (엣지)
- **seed 1개**(`--seeds 42` 또는 미지정): per_seed 1개, std=0, ci95=[mean,mean]. 기존 동작 보존.
- **per-seed resume**: 중단 시 완료된 (seed, ds, cond) skip. 각 seed JSON 독립 resume.
- **일부 seed 실패**: 성공한 seed만으로 집계, n_seeds 실제값 기록, 로그 경고.
- **checkpoint 디스크**: seed×조건×데이터셋 best.pt 누적 → 디스크 증가. seed별 `_seeds/seed{N}/` 격리로 관리.
- **std 표본 수**: k<2면 std=0/None 처리(ddof=1 division by zero 방지).
- **정수 필드**(n_train 등): seed 간 동일해야 정상(같은 val_frac). 다르면 대표값+경고.
- **하위호환**: 기존 단일-seed JSON을 resume 입력으로 받으면? → 신 구조와 다르므로, multi-seed는 `_seeds/` 하위 새 JSON 사용(기존 top-level JSON과 분리). 최종 집계만 top-level 덮어씀.

## 테스트 (Colab, pytest 금지)
1. `--seeds 42`: 기존과 동일 결과(단일), per_seed 1개·std 0 확인.
2. `--seeds 42 1 2`: per_seed 3개, top-level=평균, std/ci95 채워짐. 로그에 seed별 split 다름 확인.
3. checkpoint `_seeds/seed1/.../best.pt` 등 seed별 생성 확인.
4. resume: 중간 중단 후 재실행 → 완료 seed skip.
5. 기존 plot_exp4v2_results.py로 집계 JSON 로드 → 평균값으로 정상 그림(하위호환).
6. summary.md에 mean±std + delta 표 정상.

## 미확정 TODO
- seed 개수 기본 권장: 3(빠름) vs 5(안정). 1차 3개(42,1,2), 분산 크면 5로.
- CI 방식: 1.96·std/√k(정규근사) vs t-분포(소표본 정확). k≤5면 t 권장.
- synth_ratio sweep과 결합 여부 — 별도 실행(인자만 바꿔 반복)으로 분리, 본 작업은 multi-seed만.
- paired 비교(동일 seed의 random vs aroma 차이 평균) 보고 추가할지 — 분산 줄여 검정력↑, 후속 고려.
