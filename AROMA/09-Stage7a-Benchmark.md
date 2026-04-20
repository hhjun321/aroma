# Stage 7a — 벤치마크 실행

## 목적

3개 그룹 데이터셋에 대해 2개 모델을 학습·평가해 AROMA 합성 효과를 수치로 검증.

## 입력 / 출력 / Sentinel

| 항목 | 경로 |
|------|------|
| 입력 | `{cat_dir}/augmented_dataset/{group}/` |
| 입력 | `configs/benchmark_experiment.yaml` |
| 출력 | `outputs/benchmark_results/{cat_name}/{model}/{group}/` |
| Sentinel | `outputs/benchmark_results/{cat_name}/{model}/{group}/experiment_meta.json` |

**전제 조건:** `{cat_dir}/augmented_dataset/baseline/test` 존재 (Stage 6 완료)

## 스크립트

[[stage7_benchmark]] → `stage7_benchmark.py`
- `run_benchmark(config_path, cat_dir, resume, output_dir)`
- `_ensure_test_dir(cat_dir, group)` — test set symlink/copytree 준비

## 핵심 파라미터

```python
DOMAIN_FILTER = "isp"   # isp | mvtec | visa
resume        = True    # 중단 재개 가능
```

## 계산 로직 / 임계값

**모델 설정:**

| 모델 | 백본 | epochs | lr | optimizer | 비고 |
|------|-----|--------|-----|-----------|------|
| `yolo11` | YOLO11n-cls.pt | 30 | 0.01 | SGD | val/ 없으면 임시 YAML 자동 생성 후 삭제 |
| `efficientdet_d0` | EfficientDet-D0 (EfficientNet-B0) | 30 | 0.0001 | Adam | |

**평가 메트릭:**

| 메트릭 | 적용 도메인 |
|--------|------------|
| `image_auroc` | isp, mvtec, visa |
| `image_f1` | isp, mvtec, visa |
| `pixel_auroc` | mvtec, visa만 |

**공통 설정 (benchmark_experiment.yaml):**

```yaml
image_size: 512
train_batch_size: 16
eval_batch_size: 32
eval_chunk_size: 64   # YOLO predict OOM 방지
num_workers: 4
seed: 42
```

**실험 규모:** 2 models × 3 groups × 24 categories = **144 runs**

**카테고리 제외:**
- ISP/`LSM_2`: LSM_1과 동일 센서·라인 → 벤치마크 제외
- VisA: `macaroni2`, `pcb2~4` → dataset_config에서 완전 제거됨

**test set 준비:**
- `aroma_full/test`, `aroma_pruned/test` 없으면 `baseline/test` symlink 자동 생성
- symlink 실패 환경(일부 Drive 마운트): `copytree` 폴백 — Drive FUSE O(n_files)로 느림
