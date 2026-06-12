# AROMA — execute 가이드 전체 일괄 실행 섹션 추가

## 사용할 skills: feature-dev

## 개요

`phase0_execute.md`, `step1_execute.md` ~ `step4_execute.md` 5개 문서 모두
단일 `DATASET_KEY` 하드코딩 + `!python` 단일 실행 예시만 제공한다.
실제 연구에서는 28개 데이터셋을 순회 실행해야 하므로, 각 문서 하단에
`dataset_config.json` 자동 로드 + 선행 단계 출력물 존재 체크(skip) +
ThreadPoolExecutor 병렬 실행 예시 섹션을 추가한다.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- 5개 문서에 각각 신규 섹션 추가 (기존 단일 실행 섹션 유지)

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (문서 전용 변경, 스크립트 수정 없음)

---

## 수정 내용

### 공통 패턴

모든 문서에 동일한 구조로 섹션 추가:

```
## 전체 데이터셋 일괄 실행 (병렬)
```

**셀 1 — config 로드 + skip 목록 미리 출력**
```python
import json, os
from pathlib import Path

DATASET_CONFIG = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
with open(DATASET_CONFIG) as f:
    cfg = json.load(f)

datasets = [k for k in cfg if not k.startswith('_')]
print(f"총 {len(datasets)}개 데이터셋")

# 선행 단계 출력물 존재 체크 (step별로 다름, 아래 참조)
for ds in datasets:
    ready = <step별 체크 경로>.exists()
    print(f"{'✓' if ready else '↷'} {ds}")
```

**셀 2 — ThreadPoolExecutor 병렬 실행**
```python
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 3  # Drive push I/O 경합 고려. Colab Pro vCPU 기준 2~4 권장

def run_one(ds):
    <step별 선행 체크>
    cmd = ['python', '<script_path>', <step별 인자>]
    r = subprocess.run(cmd, capture_output=True, text=True)
    stderr_tail = r.stderr.strip().splitlines()[-3:] if r.stderr else []
    status = 'ok' if r.returncode == 0 else f'FAIL rc={r.returncode}'
    return ds, status, '\n'.join(stderr_tail)

results = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(run_one, ds): ds for ds in datasets}
    for fut in as_completed(futures):
        ds, status, err = fut.result()
        results[ds] = status
        icon = '✓' if status == 'ok' else ('↷' if status == 'skipped' else '✗')
        print(f"{icon} {ds:25s}  {status}")
        if err:
            print(f"    stderr: {err}")

ok   = sum(1 for s in results.values() if s == 'ok')
skip = sum(1 for s in results.values() if s == 'skipped')
fail = sum(1 for s in results.values() if s not in ('ok', 'skipped'))
print(f"\n완료: {ok}개  skip: {skip}개  실패: {fail}개")
```

> `!python` 매직은 IPython 전용 → 스레드 내에서 사용 불가. 의도적으로 `subprocess.run` 사용.

---

### 파일별 상세

#### 1. `phase0_execute.md` — `distribution_profiling.py`

- **선행 체크**: 없음 (pipeline 첫 단계)
- **skip 조건**: 이미 `{AROMA_OUT}/profiling/{ds}/morphology_features.csv` 존재 시 (재실행 방지)
- **스크립트**: `scripts/distribution_profiling.py` (phase0은 `scripts/` 직속)
- **주요 인자**: `--dataset_config`, `--dataset_key`, `--output_dir`, `--num_workers`
- **병렬 주의**: `--num_workers`(내부 이미지 병렬) + ThreadPoolExecutor(데이터셋 병렬) 중첩
  → `--num_workers 1` 또는 낮은 값 권장. 문서에 주의 문구 필요
- **삽입 위치**: "소요 시간 참고" 섹션 뒤 (현재 마지막)

#### 2. `step1_execute.md` — `compute_complexity.py`

- **선행 체크**: `{AROMA_OUT}/profiling/{ds}/morphology_features.csv` 존재 여부 (phase0 완료 확인)
- **skip 조건**: 위 파일 없으면 skip
- **스크립트**: `scripts/aroma/compute_complexity.py`
- **주요 인자**: `--profiling_dir`, `--output_dir`, `--weight_mode`, `--local_staging`
- **삽입 위치**: "결과 확인" 섹션 뒤 (현재 마지막)

#### 3. `step2_execute.md` — `prompt_generation.py`

- **선행 체크**: `{AROMA_OUT}/complexity/{ds}/` 디렉토리 존재 여부 (step1 완료 확인)
- **skip 조건**: 위 디렉토리 없으면 skip
- **스크립트**: `scripts/aroma/prompt_generation.py`
- **주요 인자**: `--profiling_dir`, `--complexity_dir`, `--output_dir`
- **삽입 위치**: "출력 파일" 섹션 뒤 (현재 마지막)

#### 4. `step3_execute.md` — `roi_selection.py`

- **선행 체크**: `{AROMA_OUT}/prompts/{ds}/prompts.json` 존재 여부 (step2 완료 확인)
- **skip 조건**: 위 파일 없으면 skip
- **스크립트**: `scripts/aroma/roi_selection.py`
- **주요 인자**: `--profiling_dir`, `--prompts_dir`, `--sampling_strategy`, `--top_k`, `--output_dir`
- **삽입 위치**: "출력 파일" 섹션 뒤 (현재 마지막)

#### 5. `step4_execute.md` — `generate_defects.py`

- **선행 체크**: `{AROMA_OUT}/roi/{ds}/roi_selected.json` 존재 여부 (step3 완료 확인)
- **skip 조건**: 위 파일 없으면 skip
- **스크립트**: `scripts/aroma/generate_defects.py`
- **주요 인자**: `--roi_dir`, `--normal_dir`, `--output_dir`, `--method`, `--n_per_roi`, `--feather_px`, `--seed`, `--local_staging`
- **추가**: `normal_dir`은 `cfg[ds]['image_dir']`에서 가져옴 (env var 불필요)
- **local_staging 충돌 없음**: 스테이징 경로가 `/content/tmp/aroma_step4_{dataset}/`로 데이터셋명 포함
- **삽입 위치**: "출력 파일" 섹션 뒤 (현재 마지막)

---

## 수정 대상 파일

- `AROMA연구분석/colab_execute/phase0_execute.md`
- `AROMA연구분석/colab_execute/step1_execute.md`
- `AROMA연구분석/colab_execute/step2_execute.md`
- `AROMA연구분석/colab_execute/step3_execute.md`
- `AROMA연구분석/colab_execute/step4_execute.md`

---

## 테스트 시나리오

Colab Pro 환경 전용, 수동 검증:

1. **config 메타키 필터**: `_comment` 제외 후 28개 키만 추출
2. **skip 동작**: 선행 출력물 없는 데이터셋이 `↷ skipped`로 표시되고 subprocess 미실행
3. **병렬 정상 경로**: `isp_LSM_1`, `mvtec_cable`, `visa_cashew` 동시 실행 후 각 출력물 생성 확인
4. **부분 실패 격리**: 1개 실패 시 나머지 계속 진행, 최종 요약에 `✗ FAIL` 표시
5. **phase0 workers 중첩**: `--num_workers 1` 지정 시 Colab 과부하 없이 완료

---

## TODO (미확정)

- `MAX_WORKERS` 기본값 확정 (Colab Pro vCPU 기준 — 2 vs 3)
- phase0의 `--num_workers` 권장값 (내부 병렬 × 데이터셋 병렬 중첩 임계치)
- step1~step3의 정확한 선행 체크 경로 — 실제 출력 파일명 확인 필요
- subprocess stdout 실시간 스트리밍 vs 완료 후 일괄 출력 여부
