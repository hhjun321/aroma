# AROMA Pipeline — Index

**Colab 실행 참조:** `docs/작업일지/stage1_6_execute.md`
**설정 파일:** `dataset_config.json`, `configs/benchmark_experiment.yaml`

## 파이프라인 Stage 목록

| # | Stage | 스크립트 | Sentinel |
|---|-------|---------|---------|
| 0 | 이미지 리사이즈 | `stage0_resize.py` | `{cat_dir}/.stage0_resize_512_done` |
| 1 | ROI 추출 | `stage1_roi_extraction.py` | `{cat_dir}/stage1_output/roi_metadata.json` |
| 1b | Seed 특성 분석 | `stage1b_seed_characterization.py` | `{cat_dir}/stage1b_output/{seed_id}/seed_profile.json` |
| 2 | 변형 생성 | `stage2_defect_seed_generation.py` | `{cat_dir}/stage2_output/{seed_id}/` PNG ≥ 50개 |
| 3 | 레이아웃 로직 | `stage3_layout_logic.py` | `{cat_dir}/stage3_output/{seed_id}/placement_map.json` |
| 4 | MPB 합성 | `stage4_mpb_synthesis.py` | `{cat_dir}/stage4_output/{seed_id}/defect/*.png` |
| 5 | 품질 점수 | `stage5_quality_scoring.py` | `{cat_dir}/stage4_output/{seed_id}/quality_scores.json` |
| 6 | 데이터셋 구성 | `stage6_dataset_builder.py` | `{cat_dir}/augmented_dataset/build_report.json` |
| 7a | 벤치마크 실행 | `stage7_benchmark.py` | `outputs/benchmark_results/{cat}/{model}/{group}/experiment_meta.json` |

## 도메인

| 도메인 | 태스크 | 카테고리 | 비고 |
|--------|--------|---------|------|
| isp | classification | ASM, LSM_1, LSM_2 | LSM_2 벤치마크 제외 |
| mvtec | segmentation | 15개 전 카테고리 | |
| visa | segmentation | candle, capsules, cashew, chewinggum, fryum, macaroni, pcb, pipe_fryum | |

## 핵심 경로 (cat_dir 기준)

```
{cat_dir}/stage1_output/roi_metadata.json
{cat_dir}/stage1b_output/{seed_id}/seed_profile.json
{cat_dir}/stage4_output/{seed_id}/quality_scores.json
{cat_dir}/augmented_dataset/{baseline,aroma_full,aroma_pruned}/
outputs/benchmark_results/{cat_name}/{model}/{group}/experiment_meta.json
```

## 설정 파일

| 파일 | 내용 |
|------|------|
| `dataset_config.json` | 카테고리별 `image_dir`, `seed_dirs[]`, `domain` |
| `configs/benchmark_experiment.yaml` | 모델, 증강비율, pruning 임계값, 평가 메트릭 |

## 노트 목록

- [[01-Stage0-Resize]]
- [[02-Stage1-ROI]]
- [[03-Stage1b-Seed]]
- [[04-Stage2-Variants]]
- [[05-Stage3-Layout]]
- [[06-Stage4-MPB]]
- [[07-Stage5-Quality]]
- [[08-Stage6-Dataset]]
- [[09-Stage7a-Benchmark]]
- [[09-Stage7b-Results]]
- [[10-Dataset-Structure]]
- [[11-Parallel-Guide]]
