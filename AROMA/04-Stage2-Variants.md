# Stage 2 — Defect Seed 변형 생성

## 목적

각 결함 seed에서 다양한 변형 이미지 생성. Stage 3 배치 평가의 입력.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 입력 | `seed_profile["seed_path"]` (원본 seed 이미지) |
| 출력 | `{cat_dir}/stage2_output/{seed_id}/*.png` |
| Sentinel | `{cat_dir}/stage2_output/{seed_id}/` 에 PNG ≥ `NUM_VARIANTS`개 |

## 스크립트

[[stage2_defect_seed_generation]] → `stage2_defect_seed_generation.py`
- `run_seed_generation(seed_defect, num_variants, output_dir, seed_profile, workers)`

## 핵심 파라미터

```python
NUM_VARIANTS = 50   # seed당 생성할 변형 수
SEED_THREADS = 2    # seed 단위 외부 병렬 (ThreadPoolExecutor)
IMG_WORKERS  = -1   # 변형 이미지 단위 내부 병렬 (cpu_count - 1)
```

## 계산 로직 / 임계값

- `seed_profile.json`의 `subtype` 기반 변형 전략 적용
- 각 seed당 `NUM_VARIANTS`개 PNG 생성 → Stage 3에서 배치 평가
- skip 조건: `stage2_output/{seed_id}/` 에 PNG ≥ `NUM_VARIANTS`개 존재
