# Stage 0: In-Place Image Resize Design

**Date:** 2026-04-06
**Status:** Approved

## Problem

AROMA 파이프라인은 512×512 이미지를 기준으로 설계되었으나, 원본 데이터셋의 해상도가
제각각이다 (VisA 3032×2016, MVTec 다양, ISP 256×256~512×512). Stage 1~7 전체를
512 기준으로 재실행하려면, 파이프라인 진입 전에 원본 이미지를 512×512로 통일해야 한다.

추가로 Google Drive 저장소가 대용량 원본 이미지(특히 VisA)로 인해 용량 문제가 발생하고
있어, 인플레이스 리사이즈로 저장 용량도 절감한다.

## Design Decisions

| 결정 사항 | 선택 | 근거 |
|-----------|------|------|
| 비정사각형 처리 | 직접 리사이즈 (비율 왜곡 허용) | 사용자 선택. 결함 검출 모델은 aspect ratio에 민감하지 않음 |
| 기존 출력물 | 전부 삭제 후 재실행 | 사용자 선택. 256 기준 출력물과 512 출력물 혼재 방지 |
| Interpolation | 축소 INTER_AREA, 확대 INTER_LINEAR | 표준 관행 |

## Architecture

### 파일: `stage0_resize.py`

단일 스크립트. `dataset_config.json`을 읽어 카테고리별로 3개 디렉토리의 이미지를
인플레이스 리사이즈한다.

### 리사이즈 대상 (카테고리별 3곳)

| 디렉토리 | 출처 |
|----------|------|
| `image_dir` | dataset_config.json `image_dir` 필드 |
| `seed_dir` | dataset_config.json `seed_dir` 필드 |
| `{cat_dir}/test/good/` | cat_dir = `Path(seed_dir).parents[1]` |

### 핵심 함수

```
resize_directory(dir_path, target_size, dry_run) -> ResizeStats
    각 이미지 파일(png/jpg/bmp)을 순회하며:
    - 이미 target_size×target_size → skip
    - 아니면 → cv2.resize + cv2.imwrite (인플레이스 덮어쓰기)
    - interpolation: 축소 INTER_AREA, 확대 INTER_LINEAR

resize_category(entry, target_size, dry_run) -> ResizeResult
    1. cat_dir 계산
    2. sentinel 체크 → 있으면 skip
    3. 3개 디렉토리에 resize_directory 호출
    4. sentinel 생성 (JSON: timestamp, target_size, stats)

clean_category(entry, dry_run=False) -> list[str]
    stage1~6 출력물 + sentinel 삭제. 삭제된(또는 dry_run 시 삭제 예정) 디렉토리 경로 리스트 반환.
```

### Resume (Sentinel)

- 파일: `{cat_dir}/.stage0_resize_{target_size}_done`
- 내용: JSON `{"timestamp": ..., "target_size": 512, "resized": N, "skipped": M}`
- sentinel 존재 시 해당 카테고리 skip

### CLI

```
python stage0_resize.py [--config dataset_config.json] [--size 512] \
                        [--domain-filter isp] [--dry-run] [--clean]
```

| Flag | 설명 |
|------|------|
| `--config` | dataset_config.json 경로 (기본: `dataset_config.json`) |
| `--size` | 목표 해상도 (기본: 512) |
| `--domain-filter` | isp/mvtec/visa 필터 |
| `--dry-run` | 리사이즈 없이 대상 리포트만 출력 |
| `--clean` | stage1~6 출력물 삭제 |

### `--clean` 삭제 대상

카테고리별:
- `{cat_dir}/stage1_output/`
- `{cat_dir}/stage1b_output/`
- `{cat_dir}/stage2_output/`
- `{cat_dir}/stage3_output/`
- `{cat_dir}/stage4_output/` (Stage 5 출력물인 quality_scores.json도 여기 포함)
- `{cat_dir}/augmented_dataset/` (Stage 6 출력)
- `{cat_dir}/.stage0_resize_*_done`

참고: `stage5_output/`는 별도 디렉토리가 없음 — Stage 5는 `stage4_output/{seed_id}/` 안에 저장.
`--dry-run`은 `--clean`에도 적용되어, 삭제 대상만 리포트하고 실제 삭제하지 않음.

## Error Handling

| 상황 | 처리 |
|------|------|
| 디렉토리 미존재 | warning 로그, skip (Colab에서 미다운로드 카테고리 가능) |
| 이미지 읽기 실패 | error 로그, 계속 진행, 최종 리포트에 포함 |
| 전체 카테고리 실패 | sentinel 미생성 → 재실행 시 재시도 |

## Testing

| 테스트 | 검증 내용 |
|--------|----------|
| `test_resize_images_to_target_size` | 실제 리사이즈 결과 검증 |
| `test_skip_already_correct_size` | 512×512 이미지 skip |
| `test_sentinel_created_on_success` | sentinel JSON 생성 |
| `test_sentinel_skip_on_rerun` | sentinel 있으면 카테고리 skip |
| `test_dry_run_no_modification` | dry_run 시 파일 불변 |
| `test_clean_removes_stage_outputs` | --clean 삭제 검증 |
| `test_interpolation_area_for_downscale` | 축소 시 INTER_AREA 사용 |
| `test_missing_directory_warns_not_errors` | 미존재 디렉토리 warning |

## Colab Integration

`stage1_6_execute.md`의 Progress check 블록 뒤, Stage 1 앞에 Stage 0 셀 추가.
기존 패턴(DOMAIN_FILTER, ThreadPoolExecutor, tqdm, failed list) 따름.
