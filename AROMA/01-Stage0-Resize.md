# Stage 0 — 이미지 리사이즈

## 목적

모든 카테고리 이미지를 512×512로 통일. Stage 1~6 전체 처리의 전제 조건.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `entry["image_dir"]` (원본 이미지) |
| 출력 | `{cat_dir}/stage0_output/` |
| Sentinel | `{cat_dir}/.stage0_resize_512_done` |

## 스크립트

[[stage0_resize]] → `stage0_resize.py`
- `resize_category(entry, target_size, workers)` — 카테고리 단위 리사이즈
- `clean_category(entry)` — Stage 1~6 출력물 삭제

## 핵심 파라미터

```python
TARGET_SIZE = 512
CAT_THREADS = 4   # ThreadPoolExecutor, I/O bound
CLEAN_FIRST = True
```

## 계산 로직 / 임계값

- `CLEAN_FIRST=True`: `clean_category()` 로 이전 Stage 출력 삭제 후 리사이즈
- 카테고리 단위 `ThreadPoolExecutor(max_workers=4)` 병렬 처리
- 완료 시 sentinel 파일 생성 → 재실행 자동 skip
