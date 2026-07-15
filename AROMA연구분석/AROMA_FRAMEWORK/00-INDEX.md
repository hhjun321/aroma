# AROMA — Claude Context Map

> **Claude에게:** 새 대화에서 이 파일을 먼저 읽으세요. AROMA 프레임워크 전체 컨텍스트가 여기 있습니다. 각 노트는 `colab_execute_new/`의 실행 가이드와 실제 스크립트(argparse)에 근거합니다.

## 프로젝트 한 줄 요약

데이터셋 분포를 프로파일링해 복잡도(MCI/CCI)로부터 ROI·배치 정책을 **자동 선택**하고, training-free elastic-warping + copy-paste compositing으로 결함을 생성한 뒤 **symmetric 호환성 게이트 + clean-bg 게이트**로 배치를 통제하여 데이터를 증강하고, YOLOv8n 다운스트림 검출로 효과를 검증하는 연구 파이프라인. CASDA에서 진화(수작업 호환성 행렬·형태 규칙을 데이터셋별 자동 유도로 대체).

## 실행 환경

| 항목 | 값 |
|------|----|
| 환경 | Google Colab (GPU T4+) |
| 코드 경로 | `/content/AROMA` |
| 데이터 루트 | `/content/drive/MyDrive/data/Aroma` |
| Python | 3.10+ |
| GPU | 생성(step5)·exp* 필수. profiling/complexity/prompt/ROI/τ 사전스캔은 CPU |

## 공통 환경 셀 (`_SPEC §1` — 전 문서 동일, 수정 금지)

```python
import os, json
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['SYM_ROOT']  = f"{os.environ['AROMA_OUT']}/sym_final"      # 단일 버전 루트
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
```

- Colab 실행 규약: `!python $VAR/x.py` 접두사, 환경변수는 `$VAR`(중괄호 `${VAR}` 금지).
- `distribution_profiling.py`만 `scripts/`(루트) → `$AROMA_REF/scripts/`. 나머지는 `$AROMA_SCRIPTS`(=`scripts/aroma`).

## output 루트 규약 (`_SPEC §2`, stage-first — 반드시 준수)

전 산출물은 `sym_final/{stage}/{ds}` 아래. `S(stage, ds)` 헬퍼로 접근.

| stage | 경로 `S(stage, ds)` | 생성 | 소비 |
|-------|--------------------|------|------|
| `profiling` | `sym_final/profiling/{ds}` | phase0 | step1~3, exp* |
| `complexity` | `sym_final/complexity/{ds}` | step1 | step2 |
| `prompts` | `sym_final/prompts/{ds}` | step2 | step3 |
| `roi` | `sym_final/roi/{ds}` | step3 | step5, exp6 |
| `compat_gate` | `sym_final/compat_gate/{ds}` | step3(τ 사전스캔) | step5 |
| `synth_aroma` | `sym_final/synth_aroma/{ds}` | step5 | exp3/4v2/5/6 |
| `synth_random` | `sym_final/synth_random/{ds}` | step5 | exp3/4v2/5/6 |
| `exp3`/`exp4v2`/`exp5`/`exp6` | `sym_final/{exp}` | 각 exp | — |
| `embed_cache` | `sym_final/embed_cache/{ds}` | exp5/exp6 | exp5·exp6 공유 |

## 파이프라인 흐름

```
prepare_datasets(step -1) → phase0(profiling) → step1(complexity) → step2(prompts)
  → step3(roi_selection / deficit_aware + τ 사전스캔) → step5(생성: AROMA arm=copy-paste+symmetric / random arm)
  → exp3 / exp4v2 / exp5 / exp6
```

## 노트 맵

| 노트 | 내용 | Claude 활용 시점 |
|------|------|-----------------|
| [[01-Overview]] | 전체 아키텍처, Abstract, Highlights, CASDA→AROMA 진화 | 프로젝트 첫 파악 시 |
| [[02-Stage0-Prepare-Profiling]] | prepare_datasets(step -1) + distribution_profiling(phase0) | 데이터 준비·프로파일링 작업·디버깅 시 |
| [[03-Stage1-Complexity-Prompts]] | compute_complexity(step1) + prompt_generation(step2) | 복잡도·프롬프트 작업 시 |
| [[04-Stage2-ROI-Selection]] | roi_selection(step3) + clean-bg 선정(step3.5) | ROI 선택 작업·디버깅 시 |
| [[05-Stage3-Copy-Paste-Generation]] | Copy-paste 결함 생성(step5, AROMA/random arm) | 생성 작업·디버깅 시 |
| [[06-Experiments]] | exp3/exp4v2/exp5/exp6 다운스트림 평가 | 실험 기록·계획 시 |
| [[07-Scripts-Reference]] | 전 스크립트 입출력 표 + 핵심 로직 | 코드 수정·파라미터 확인 시 |
| [[10-Python-Reference]] | 코어 py 파일 **내부 구조**(함수·클래스·제어 흐름·상수·gotcha) | py 파일 코드 레벨 분석·디버깅 시 |
| [[08-Datasets]] | v2-1 4종 데이터셋 + 데이터셋별 규약 | 데이터셋 구성 파악 시 |
| [[09-Compatibility-Gate]] | symmetric compat 게이트 + clean-bg 게이트 + τ 사전스캔 (AROMA 핵심 novelty) | 게이트·τ 작업·디버깅 시 |

## 주요 참조 파일

| 파일 | 위치 | 용도 |
|------|------|------|
| `_SPEC.md` | `AROMA연구분석/colab_execute_new/_SPEC.md` | 전 실행 가이드의 정본(공통 env·output 규약·스테이지별 명령) |
| 실행 가이드 | `AROMA연구분석/colab_execute_new/{phase0,step1~5,exp*}_execute.md` | 스테이지별 Colab 실행 절차 |
| `dataset_config.json` | `/content/AROMA/dataset_config.json` | 데이터셋별 image_dir·class_mode·seed_dirs 등록 |

## 정직/무결성 규약 (`_SPEC §5`)

- 사후 튜닝 금지: τ·seed·synth_ratio·epochs는 결과 보고 후 변경 금지.
- prescan 필수: τ·AR·텍스처 임계는 CPU 사전스캔 확정값. mtd 값을 aitex에 무검증 전용 금지.
- aitex는 tile-level·single-class → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- `--local_staging`: CPU 단계(complexity/random/copy_paste)엔 사용 가능.
- 테스트 코드 신규 작성·pytest 금지(CLAUDE.md). 검증은 Colab 실행으로.
