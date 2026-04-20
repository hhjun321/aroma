# 병렬 설정 가이드

## Stage별 병렬 설정

| Stage | 외부 병렬 변수 | 내부 병렬 변수 | 병렬 종류 | 근거 |
|-------|-------------|-------------|---------|------|
| 0 | `CAT_THREADS=4` | `workers=-1` | Thread+Process | I/O bound 리사이즈 |
| 1 | `CAT_THREADS=2` | `IMG_WORKERS=-1` | Thread+Process | CPU-bound, Drive 동시쓰기 제한 |
| 1b | `NUM_WORKERS=4` | — | Thread | seed 단위 독립, CPU-bound |
| 2 | `SEED_THREADS=2` | `IMG_WORKERS=-1` | Thread+Process | 내부 병렬로 코어 포화 |
| 3 | 순차 | GPU 배치 | GPU | GPU 인스턴스 공유 불가 |
| 4 | 카테고리 순차 | `IMG_THREADS=4` | Thread | 배경 1회 로드, I/O 절감 |
| 5 | `SEED_THREADS=4` | `IMG_WORKERS=-1` | Process | I/O+CPU 혼합, seed 독립 |
| 6 | `CAT_THREADS=2` | `NUM_IO_THREADS=8` | Thread | I/O-bound Drive 복사 |
| 7 | 순차 (카테고리) | — | — | GPU 단일 점유, `resume=True` |

## run_parallel 동작 조건

```python
# utils/parallel.py
use_parallel = num_workers > 1 and len(tasks) >= 2
# False 이면 순차 실행 (ProcessPoolExecutor 미사용)
```

| workers 값 | num_workers 해석 |
|-----------|----------------|
| `0` | 0 → 항상 순차 |
| `-1` | `max(1, cpu_count - 1)` 자동 |
| `N` | N 그대로 사용 |

## Colab GPU별 권장값 (Stage 5 기준)

| GPU | vCPU 수 | parallel_seeds | workers | 비고 |
|-----|---------|---------------|---------|------|
| T4 | 2 | `2` | `0` | CPU 적음 → 중첩 금지 |
| L4 | 8 | `-1` (=7) | `0` | seed 수준만 자동 병렬 |
| A100 | 12 | `-1` (=11) | `0` | seed 수준만 자동 병렬 |

**중첩 병렬화 주의:**

```
총 프로세스 = parallel_seeds × workers
L4 8-CPU:  parallel_seeds=4, workers=2  →  8 프로세스  (적정)
L4 8-CPU:  parallel_seeds=7, workers=7  → 49 프로세스  (CPU 과부하)
```

## CPU 수 확인

```python
import os
print(f"CPU: {os.cpu_count()}")
# resolve_workers(-1) = max(1, cpu_count - 1)
```
