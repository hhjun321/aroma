# 데이터셋 디렉터리 구조

## cat_dir 기준 전체 트리

```
{cat_dir}/
├── .stage0_resize_512_done              ← Stage 0 sentinel
│
├── stage1_output/
│   ├── roi_metadata.json                ← Stage 1 sentinel / Stage 3 입력
│   └── masks/
│       ├── global/{image_id}.png
│       └── local/{image_id}_zone{n}.png
│
├── stage1b_output/
│   └── {seed_id}/
│       ├── seed_profile.json            ← Stage 1b sentinel / Stage 2·3 입력
│       └── seed_mask.png
│
├── stage2_output/
│   └── {seed_id}/
│       └── *.png                        ← PNG ≥ 50개 시 Stage 2 완료
│
├── stage3_output/
│   └── {seed_id}/
│       └── placement_map.json           ← Stage 3 sentinel / Stage 4 입력
│
├── stage4_output/
│   └── {seed_id}/
│       ├── defect/
│       │   ├── {image_id}.png           ← Stage 4 sentinel (PNG 존재 시)
│       │   └── {image_id}_mask.png      ← ROI 마스크 / Stage 5 입력
│       └── quality_scores.json          ← Stage 5 sentinel / Stage 6 입력
│
└── augmented_dataset/                   ← Stage 6 출력
    ├── build_report.json                ← Stage 6 sentinel
    ├── baseline/
    │   ├── train/good/
    │   └── test/{good,defect}/          ← 고정 test set (split_seed=42)
    ├── aroma_full/
    │   ├── train/{good,defect}/
    │   └── test/ → symlink → baseline/test
    └── aroma_pruned/
        ├── train/{good,defect}/
        └── test/ → symlink → baseline/test
```

## Benchmark 출력

```
outputs/benchmark_results/
└── {cat_name}/
    └── {yolo11|efficientdet_d0}/
        └── {baseline|aroma_full|aroma_pruned}/
            └── experiment_meta.json
```

## seed_id 명명 규칙

| seed_dirs 수 | seed_id 형식 | 예시 |
|-------------|-------------|------|
| 1개 | `{seed.stem}` | `crack_001` |
| 복수 | `{seed_dir.name}_{seed.stem}` | `broken_large_crack_001` |

## 설정 파일

| 파일 | 내용 |
|------|------|
| `dataset_config.json` | 카테고리별 `image_dir`, `seed_dirs[]`, `domain` |
| `configs/benchmark_experiment.yaml` | 모델, 증강비율, pruning 임계값, 평가 메트릭 |
