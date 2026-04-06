# 결함 유형 구조 현황 분석

> 날짜: 2026-04-06
> 목적: 26개 카테고리별 결함 유형 구조 파악 및 향후 다중 분류 실험 방향 정리

## 1. 현황 요약

현재 AROMA 파이프라인은 **카테고리당 단일 결함 유형, 이진 분류(good vs anomaly)** 로 동작한다.

- `dataset_config.json`이 카테고리당 1개의 `seed_dir`만 지정
- Stage 7 벤치마크는 `label = 0 (good) or 1 (anomaly)` 이진 평가
- 결함 유형별 합성 수량 비율을 제어하는 메커니즘 없음

## 2. 도메인별 결함 유형 구조

### ISP (3개 카테고리)

| 카테고리 | 사용 중 | 전체 유형 |
|---------|---------|----------|
| ASM | `area` | area (1개) |
| LSM_1 | `area` | area (1개) |
| LSM_2 | `area` | area (1개) |

ISP 도메인은 결함 유형이 1종(`area`)뿐이므로 다중 분류 해당 없음.

### MVTec (15개 카테고리) — 다중 결함 유형 보유

| 카테고리 | 현재 사용 | 미사용 유형 | 전체 수 |
|---------|----------|-----------|--------|
| bottle | `broken_large` | broken_small, contamination | 3 |
| cable | `cut_outer_insulation` | bent_wire, cable_swap, combined, cut_inner_insulation, missing_cable, missing_wire, poke_insulation | 8 |
| capsule | `scratch` | crack, faulty_imprint, poke, squeeze | 5 |
| carpet | `cut` | color, hole, metal_contamination, thread | 5 |
| grid | `broken` | bent, glue, metal_contamination, thread | 5 |
| hazelnut | `crack` | cut, hole, print | 4 |
| leather | `cut` | color, fold, glue, poke | 5 |
| metal_nut | `scratch` | bent, color, flip | 4 |
| pill | `crack` | color, contamination, faulty_imprint, pill_type, scratch | 6 |
| screw | `scratch_head` | manipulated_front, scratch_neck, thread_side, thread_top | 5 |
| tile | `crack` | glue_strip, gray_stroke, oil, rough | 5 |
| toothbrush | `defective` | (없음) | 1 |
| transistor | `bent_lead` | cut_lead, damaged_case, misplaced | 4 |
| wood | `scratch` | color, combined, hole, liquid | 5 |
| zipper | `broken_teeth` | combined, fabric_border, fabric_interior, rough, split_teeth, squeezed_teeth | 7 |

**MVTec 15개 중 14개 카테고리**가 복수 결함 유형을 보유 (toothbrush만 단일).
현재 각 카테고리에서 1개 유형만 사용하므로, 나머지 유형의 시드 이미지가 활용되지 않고 있음.

### VisA (8개 카테고리)

| 카테고리 | 사용 중 | 전체 유형 |
|---------|---------|----------|
| candle | `anomaly` | anomaly (1개) |
| capsules | `anomaly` | anomaly (1개) |
| cashew | `anomaly` | anomaly (1개) |
| chewinggum | `anomaly` | anomaly (1개) |
| fryum | `anomaly` | anomaly (1개) |
| macaroni | `anomaly` | anomaly (1개) |
| pcb | `anomaly` | anomaly (1개) |
| pipe_fryum | `anomaly` | anomaly (1개) |

VisA 도메인은 `prepare_visa.py`에서 모든 결함을 단일 `anomaly` 폴더로 통합하므로 다중 분류 해당 없음.

## 3. 결함 유형 정보 흐름 — 파이프라인 내 추적

```
dataset_config.json
  seed_dir: .../test/{defect_type}     ← 결함 유형 이름 (파일시스템)
       │
  Stage 1b → seed_profile.json
       subtype: "compact_blob"          ← 형태학적 분류 (원래 이름 ≠ subtype)
       │
  Stage 2~4 → stage4_output/{seed_id}/defect/*.png
                                        ← FLAT: 원래 유형 정보 소실
       │
  Stage 6 →  baseline/test/{defect_type}/   ← seed_dir name에서 보존됨
             aroma_*/train/defect/          ← FLAT: 유형 정보 소실
       │
  Stage 7 →  label = 0 (good) or 1 (anomaly)  ← 이진 분류만
```

**정보 손실 지점:**
1. Stage 1b: 원본 결함 유형명(scratch, crack 등)이 `seed_profile.json`에 저장되지 않음. 대신 형태학적 subtype만 기록.
2. Stage 4/6: 합성된 학습 이미지가 flat `defect/` 폴더에 저장되어 유형 구분 소실.
3. Stage 7: `good` 외 모든 폴더를 label=1로 처리.

## 4. 수량 비율 제어 메커니즘: 현재 없음

| 파이프라인 단계 | 수량 결정 방식 | 유형별 비율 제어 |
|---------------|--------------|----------------|
| Stage 2 (시드 생성) | `num_variants` (flat integer, 기본 10) | 없음 — 모든 시드 동일 수량 |
| Stage 4 (합성) | 시드 × 배경 이미지 수 (균등) | 없음 — 모든 시드 균등 |
| Stage 5 (품질 필터) | `pruning_threshold` (품질 점수) | 없음 — 품질 기반, 유형 무관 |
| Stage 6 (데이터셋 구성) | 전체 합성 이미지 수집 | 없음 — flat 수집 |

## 5. 향후 다중 분류 실험을 위한 변경 필요 지점

현재 이진 분류 파이프라인 재실행(512 기준)이 우선. 이후 다중 분류 실험 시 수정이 필요한 코드 위치:

### (A) 데이터 확장 — 복수 결함 유형 지원

| 변경 대상 | 파일:라인 | 변경 내용 |
|----------|----------|----------|
| Config 구조 | `dataset_config.json` | `seed_dir` → `seed_dirs` 리스트로 확장, 또는 부모 `test/` 디렉토리 자동 탐색 |
| Stage 1b 루프 | `stage1b_seed_characterization.py` | 복수 시드 순회, 원본 결함 유형명 `seed_profile.json`에 기록 |
| Stage 2 비율 | `stage2_defect_seed_generation.py:162` | `num_variants`를 유형별 가중치로 확장 |
| Stage 4 출력 | `stage4_mpb_synthesis.py:177` | `defect/` → `defect/{type}/` 하위 폴더 구조 |
| Stage 6 수집 | `utils/dataset_builder.py:237-266` | 유형별 폴더 구조 유지 또는 비율 기반 샘플링 |

### (B) 평가 확장 — 결함 유형별 성능 측정

| 변경 대상 | 파일:라인 | 변경 내용 |
|----------|----------|----------|
| 테스트 라벨링 | `stage7_benchmark.py:347` | `label = cls_dir.name` (문자열)으로 다중 클래스 |
| 메트릭 | `utils/ad_metrics.py:10-47` | per-class AUROC, macro/micro F1, confusion matrix 추가 |
| 평가 config | `configs/benchmark_experiment.yaml:21-27` | `per_class_auroc`, `macro_f1` 등 추가 |

### (C) 수량 비율 제어

결함 유형별 합성 수량 비율을 맞추는 두 가지 접근:
1. **생성 시 제어**: `num_variants`를 유형별로 다르게 지정 (희소 유형에 더 많은 변형)
2. **데이터셋 구성 시 제어**: Stage 6에서 유형별 over/under-sampling

## 6. 결론

- 현재 파이프라인은 **단일 결함 유형 × 이진 분류** 전용 설계
- MVTec 14개 카테고리에서 2~7개 결함 유형이 미활용
- 다중 분류 실험은 Config → Stage 1b → Stage 4 → Stage 6 → Stage 7 전체에 걸쳐 수정 필요
- **우선순위**: 512 리사이즈 파이프라인 재실행 완료 후, 다중 분류 실험 설계 착수
