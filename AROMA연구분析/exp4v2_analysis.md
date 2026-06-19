## Exp4v2 Results Analysis

YOLOv8 Supervised Defect Detection — Baseline vs Random vs AROMA

Source data: `.claude/.etc/exp4v2/exp4v2_results.json` (v2, 2026-06-19)  
Previous invalid run: all-zeros run discarded (see §1 below)

---

## §1. 이전 실행 (무효) — 2026-06-19 04:32

모든 조건 mAP=0.0. 파이프라인 3개 버그:
1. Baseline real=0 (정상 배경만 224장으로 학습)
2. AROMA n_synth=0 (이미지 diff threshold=20 너무 높음)
3. Random n_synth=0 (동일 원인)

→ 파이프라인 수정 후 재실행.

---

## §2. 현재 실행 결과 — 2026-06-19 05:28

### 수정된 파이프라인
- Baseline: real 결함 이미지 46장 학습 (50:50 train/val split)
- Local cache 동작 (8.3s / 1516 images → /tmp)
- YOLO baseline cache 동작 (92 real labels built)

### 결과 (mvtec_cable / yolov8n)

| 조건 | mAP50 | mAP50-95 | Precision | Recall | n_real | n_synth |
|------|-------|----------|-----------|--------|--------|---------|
| baseline | **0.4751** | 0.2851 | 0.5285 | 0.4308 | 46 | 0 |
| random | 0.2079 | 0.0784 | 0.0898 | 0.0769 | 46 | **600** |
| aroma | 0.3446 | 0.2073 | 0.1893 | 0.6000 | 46 | **0** ❌ |

### AROMA n_synth=0 원인 (확인됨)

`generate_defects.py --local_staging` 옵션 사용 시 `annotations.json`의 `normal_image` 필드가 Drive 경로가 아닌 로컬 임시 경로(`/content/tmp/aroma_step4_{ds}/`) 로 저장됨.

새 Colab 세션에서 로컬 경로 무효 → `_extract_defect_bboxes` 에서 background=None → return [] → 0 bboxes → 0 labels.

**수정 대상:**
1. `generate_defects.py`: `normal_image` 을 Drive 원본 경로로 저장 (run_synthesis 수정)
2. `exp4_v2_supervised_detection.py`: `source_roi` template matching fallback 추가

---

## §3. 현재 결과 해석

### 핵심 발견: Random synth가 성능을 심각하게 저하

- baseline (real-only): mAP50 = **0.4751**
- random (real + 600 random synth): mAP50 = **0.2079** (−26.7pp, −56% 하락)

Random synthesis는 결함의 모양은 맞지만 배치 위치/맥락이 부자연스러움.  
YOLO가 잘못된 패턴을 학습 → 실제 결함 검출 능력 저하.

### AROMA 조건 (synth=0, 사실상 baseline finetune)

AROMA 조건은 baseline best.pt 에서 동일한 46개 real 이미지로 재학습.  
mAP50 = 0.3446 (baseline 0.4751 대비 −13pp).  
같은 데이터로 두 번 학습 → 약간 과적합.

**이 결과는 AROMA synth 없이 finetune만 한 결과 → 논문 비교 불가능.**

### 기대 결과 (AROMA synth 정상화 후)

AROMA는 배경 맥락에 맞는 결함을 Poisson blending으로 합성 → 결함의 맥락적 일관성 유지.  
예상 순서: **baseline ≤ AROMA < or > baseline >> random**

논문 핵심 주장: AROMA synth ≫ Random synth (품질 차이가 downstream 성능에 반영됨)

---

## §4. 다음 단계

1. **즉시**: `generate_defects.py` 수정 후 mvtec_cable 재합성 (annotations.json 재생성)
   - OR `--no_local_staging` 옵션으로 재실행 (Drive 경로 저장)
2. **exp4v2 재실행**: `--imgsz 640` 추가 권장 (1024×1024 이미지에서 더 높은 해상도)
3. **전체 데이터셋**: isp_LSM_1, visa_cashew 포함 4개 데이터셋으로 확대

### 재실행 권장 명령

```python
# 먼저 AROMA synthesis 재실행 (annotations.json 갱신)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir       $ROI_DIR \
    --normal_dir    $NORMAL_DIR \
    --output_dir    $SYNTHETIC_DIR \
    --method        copy_paste \
    --n_per_roi     3 \
    --feather_px    4 \
    --seed          42
    # --local_staging 제거 (Drive 경로 저장 보장)

# OR 코드 fix 후 --local_staging 재사용 가능
```

```python
# exp4v2 재실행 (imgsz 640 권장)
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model     yolov8n \
    --condition all \
    --dataset_keys          mvtec_cable \
    --random_synthetic_dir  $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir   $AROMA_SYNTH_DIR \
    --real_data_dir         $AROMA_DATA \
    --output_dir            $EXP4V2_OUT \
    --imgsz 640 \
    --baseline_epochs 50 \
    --finetune_epochs 30 \
    --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --resume
```
