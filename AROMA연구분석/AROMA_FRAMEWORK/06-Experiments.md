# 06 — 실험 (다운스트림 평가)

> **Claude 요약:** AROMA 평가는 step5가 만든 두 합성 arm(`synth_aroma` = ControlNet + symmetric compat 게이트 + clean-bg 게이트 grounded 배치, `synth_random` = **naive placement baseline** — grounding·게이트 없이 무작위 배치, 46700af)을 소비하여 4개 실험으로 검증한다. 헤드라인은 exp4v2(YOLOv8n supervised detection)로 `baseline`(real만) / `random`(real + random 합성) / `aroma`(real + AROMA-sym 합성) 3조건을 COCO pretrained에서 **fresh 학습**(graft 미사용)하고 mAP50 Δ(aroma−random)를 본다. 보조로 exp3(생성 품질 FID/AD), exp5(PRDC 커버리지), exp6(임베딩 커버리지 knn/rare)가 기제(mechanism) 증거를 제공한다. 실행 순서 체인은 `phase0 → step1~4 → step5(생성) → exp*`이며, 전 실험은 `_SPEC §1` 공통 환경 셀과 `_SPEC §2` stage-first output 규약(`sym_final/{stage}/{ds}`)을 공유한다.

## 실험 개요

| 실험 | 목적 | 스크립트 | 주요 지표 | GPU |
|------|------|----------|-----------|-----|
| **exp3** | 생성 품질 — random vs aroma 2-way | `experiments/exp3_generation_quality.py` | FID(+KID/LPIPS), PaDiM Image/Pixel AUROC | FID=CPU 가능 / AD=필수 |
| **exp4v2** | **헤드라인** — supervised 검출 이득 (baseline/random/aroma) | `experiments/exp4_v2_supervised_detection.py` | YOLOv8n mAP50, Δ(aroma−random), per-seed 부호 | 필수 (A100 권장) |
| **exp5** | PRDC 커버리지 — real 결함 manifold 커버 | `experiments/exp5_prdc.py` | Precision/Recall/Density/Coverage Δ + permutation p | 임베딩만 (CPU 가능) |
| **exp6** | 임베딩 커버리지 — knn 기제 + rare 타겟팅 | `experiments/exp6_embedding_coverage.py` | 1-NN dist Δ(bootstrap p), rare hit-rate | 캐시 재사용 시 CPU |

- **공통 arm 정의**: `aroma`는 aroma-sym(`generate_defects --method controlnet --compat_mode symmetric` + clean-bg 게이트 + seamless 블렌딩) grounded 배치 산출, `random`은 `generate_random.py`의 **naive baseline**(`random_placement` 기본 ON — placement grounding·clean-bg 게이트 없이 무작위 배치). 두 arm 차이 = ROI 선택 **+ smart-placement 프레임워크 전체**(AROMA 기여). 두 arm 모두 `S('synth_aroma')` / `S('synth_random')`에서 **읽기만** 한다(exp에서 재생성 금지 → n_synth parity 유지).
- **stage-first 루트 규약**(`_SPEC §2`): exp 스크립트에는 **루트만** 넘기고 스크립트가 `/{ds}`를 붙인다. 공유 루트 인자:
  - `--aroma_synthetic_dir $(S('synth_aroma'))` → `{root}/{ds}/annotations.json`
  - `--random_synthetic_dir $(S('synth_random'))` → 동일 규약
  - `--roi_dir_root $(S('roi'))` (exp6 rare 전용) → `{root}/{ds}/roi_*.json`
  - `--embed_cache_dir $(S('embed_cache'))` — **exp5·exp6 공유** DINOv2 캐시(재사용). 합성 재생성 시 `rm -rf $EMBED_CACHE_DIR/{ds}` 후 무효화.

---

## exp3 — 생성 품질 (FID + PaDiM AD)

Random ROI vs AROMA ROI의 2-way 생성 품질을 비교한다. FID는 실제 결함 패치 vs 합성 결함 패치의 분포 거리, AD는 합성본을 train에 섞은 PaDiM 3조건(baseline/random/aroma)의 AUROC. random은 naive 배치(grounding·게이트 없음), aroma는 grounded → 남은 차이는 ROI selection + placement grounding + 생성 기여로 해석.

```python
# FID 모드 (CPU 가능)
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode                 fid \
    --random_synthetic_dir $SYNTH_RANDOM \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         severstal mvtec_leather aitex mtd \
    --output_dir           $EXP3_OUT \
    --seed                 42 \
    --device               cpu

# AD 모드 (PaDiM, GPU 필수)
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode                 ad \
    --random_synthetic_dir $SYNTH_RANDOM \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         severstal mvtec_leather aitex mtd \
    --output_dir           $EXP3_OUT \
    --seed                 42 \
    --image_size           256
```

| 파라미터 | 값 | 비고 |
|----------|----|------|
| `--mode` | `fid` / `ad` / `all` | FID=CPU 가능, AD=GPU 필수 |
| `--seed` | 42 | 고정 |
| `--device` | `cpu`(FID) / cuda(AD 기본) | |
| `--image_size` | 256 | AD PaDiM 입력 크기 |
| `--num_workers` | 4 | 병렬 I/O |

| 지표 | 기대 방향 | 산출 |
|------|-----------|------|
| FID (+KID/LPIPS) | **aroma < random** (낮을수록 real 분포 근접) | `exp3_results.json[ds].fid.{random,aroma}` |
| Image / Pixel AUROC | **aroma > random > baseline** | `exp3_results.json[ds].ad.{baseline,random,aroma}` |

> `fid_unstable: true` = real 결함 패치 < 50개 → 참고용, 결론 근거 금지. 출력: `$EXP3_OUT/exp3_results.json`, `exp3_summary.md`, `checkpoints/{ds}/{cond}/`.

---

## exp4v2 — YOLOv8n Supervised Detection (헤드라인)

step5 산출을 지도학습 검출 레이블로 사용해 `baseline` / `random` / `aroma` 3조건 mAP50을 비교한다. **fresh 전조건 학습** — 세 조건 모두 COCO pretrained에서 독립 학습(graft 미사용, weight 재사용 없음), 합성은 train에만, 평가셋은 항상 real 결함 이미지. legacy exp4(PaDiM unsupervised AD)를 폐기하고 이 supervised 버전으로 대체.

**2 그룹 파라미터 상이** — 반드시 그룹별로 분리 실행(하나라도 섞으면 비교 불가). 공통: `--model yolov8n` · `--condition all` · `--val_frac 0.3` · `--synth_ratio 1.0` · `--batch 64` · `--cache ram` · `--patience 50` · `--resume`.

> **A100 속도 knob(선택, 6c55ed6·2a51686)**: `--workers N`(dataloader 워커, 기본 8=Ultralytics 기본 → 미지정 시 동작 불변; A100은 12 권장) · `--compile`(torch.compile inductor, ~14% 속도↑; 미지원 Ultralytics 조합은 자동 fallback+warn). 둘 다 학습 **결과에 영향 없는** 처리량 knob이라 사후 튜닝 금지 대상 아님.

### 그룹 A — 3종 multi (severstal · mvtec_leather · mtd)

`--class_mode multi` · `--imgsz 640` · `--rect` · `--baseline_epochs 100` · `--seeds 42 1 2`.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys severstal mvtec_leather mtd \
    --class_mode multi \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $YOLO_CACHE \
    --imgsz 640 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 100 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --rect \
    --seeds 42 1 2 \
    --resume
```

### 그룹 B — aitex single (tiled 256×256/stride128)

`--imgsz 256` · `--baseline_epochs 300` · `--seeds 1 2 42`. **`--class_mode` 미지정**(=single, nc=1 'defect') · **`--rect` 미사용**(타일 정사각). 동일 `--output_dir` 유지 — `--resume`이 3종을 건드리지 않고 aitex만 추가.

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys aitex \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $YOLO_CACHE \
    --imgsz 256 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 300 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --seeds 1 2 42 \
    --resume
```

| 파라미터 | 그룹 A (3종) | 그룹 B (aitex) |
|----------|-------------|----------------|
| `--class_mode` | `multi` | 미지정 (single) |
| `--imgsz` | 640 | 256 |
| `--rect` | 사용 | **미사용** |
| `--baseline_epochs` | 100 | 300 |
| `--seeds` | 42 1 2 | 1 2 42 |
| `--condition` | `all` (fresh) | `all` (fresh) |
| `--synth_ratio` | 1.0 | 1.0 |
| `--val_frac` / `--batch` / `--cache` | 0.3 / 64 / ram | 0.3 / 64 / ram |

| 지표 | 기대 방향 | 산출 |
|------|-----------|------|
| mAP50 (mean±std) | aroma ≥ random ≥ baseline | `exp4v2_results.json[ds].yolov8n.{baseline,random,aroma}.map50` |
| Δ(aroma−random) | > 0, 전 seed 부호 일치 | `.aroma.per_seed[s].map50 − .random.per_seed[s].map50` |

> **판정 주의**: near-ceiling(예: mtd baseline~0.90)이면 flat이 흔해 placement 이득 측정 불가 — mtd 단독 headline 금지, headroom 있는 데이터셋이 arbiter. aitex는 tile-level·single-class → Δ만 유효(절대값 3종과 직접 비교 금지). aroma < random/baseline이면 회귀 → step4 τ 사전스캔·step5 fallback률 재점검.

---

## exp5 — PRDC 커버리지

외부 임베딩 좌표계(DINOv2)에서 aroma vs random 합성의 real 결함 manifold 커버리지를 반증 가능하게 검증한다(저비용 L2 증거). random은 naive 배치, aroma는 grounded → 차이는 ROI **선택 전략 + placement grounding**(둘 다 AROMA 기여). 비대칭 예측(Recall만 오르고 Precision 동등)이 사후합리화를 차단.

```python
!python $AROMA_SCRIPTS/experiments/exp5_prdc.py \
    --real_data_dir        $AROMA_DATA \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --nearest_k 3 5 10 \
    --permutation_reps 1000 \
    --val_frac 0.3 --split_seed 42 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP5_OUT
```

| 파라미터 | 값 | 비고 |
|----------|----|------|
| `--nearest_k` | 3 5 10 | 주 보고 k=5, 나머지 sensitivity (cherry-pick 금지) |
| `--permutation_reps` | 1000 | null 분포 |
| `--val_frac` / `--split_seed` | 0.3 / 42 | **exp4v2와 동일 split** (reference = held-out val 정렬) |
| `--backbone` | dinov2_vits14 | 실패 시 InceptionV3 폴백 (`meta.backbone` 확인) |
| `--embed_cache_dir` | `S('embed_cache')` | exp6과 공유 |

| 지표 | 예측 | 판정 |
|------|------|------|
| Precision / Density | 두 조건 동등 | \|Δ\| ∈ permutation null 95% CI |
| Recall / Coverage | **aroma > random** | Δ>0 AND one-sided p < 0.05 (4종 방향 일치) |

> 출력: `$EXP5_OUT/exp5_prdc_results.json`(k별 Δ + p + null CI + meta), `exp5_prdc_summary.md`. `meta.unstable`(n_ref<30) 또는 aitex는 단독 결론 금지(aggregate 보조).

---

## exp6 — 임베딩 커버리지 (knn + rare)

exp5와 패키지로 저비용 L2 기제 증거를 완성한다. ① `knn`: held-out val 결함까지 최근접 거리로 커버리지 기제 측정(합성 arm 비교), ② `rare`: 독립 k-means 모드에서 rare-mode 타겟팅 검증(step3 ROI 선택 산출 `roi_*.json` 직접 평가).

```python
# knn 모드
!python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \
    --mode knn \
    --real_data_dir        $AROMA_DATA \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --val_frac 0.3 --split_seed 42 \
    --bootstrap_reps 2000 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP6_OUT

# rare 모드
!python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \
    --mode rare \
    --real_data_dir $AROMA_DATA \
    --roi_dir_root  $ROI_DIR_ROOT \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --kmeans_k 8 10 12 15 --cluster_seeds 0 1 2 3 4 \
    --null_seeds 30 --rare_quantile 0.25 \
    --val_frac 0.3 --split_seed 42 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP6_OUT
```

| 파라미터 | 값 | 비고 |
|----------|----|------|
| `--mode` | `knn` / `rare` | knn=합성 arm, rare=`--roi_dir_root` |
| `--bootstrap_reps` | 2000 | knn clustered-bootstrap |
| `--kmeans_k` | 8 10 12 15 | rare 그리드 (aitex 소형 시 8 10로 하향) |
| `--cluster_seeds` | 0 1 2 3 4 | rare 그리드 |
| `--null_seeds` / `--rare_quantile` | 30 / 0.25 | rare null (p_emp 하한 ≈ 0.032) |
| `--val_frac` / `--split_seed` | 0.3 / 42 | exp4v2·exp5와 동일 split |

| 모드 | 가설 | 판정 |
|------|------|------|
| knn | Δ(1-NN dist) = mean d1(real+random) − mean d1(real+aroma) > 0 | bootstrap p < 0.05, 4종 방향 일치 |
| rare | rare 모드 hit rate aroma > random-null | 전 그리드(k×cseed) 방향 일치 + 다수 셀 p_emp < 0.05 |

> 기제 증거 한계 — knn·rare 단독으로 downstream 개선 주장 금지, exp5·exp4v2와 패키지로 제시. diversity(pairwise-mean↓ = 뭉침)를 함께 보고(정직). 출력: `$EXP6_OUT/exp6_results.json`(knn/rare 병합), `exp6_knn_summary.md`, `exp6_rare_summary.md`.

---

## 데이터셋별 파라미터

`_SPEC §4` 규약. exp4v2 그룹 파라미터의 정본 출처.

| ds | class_mode | normal_dir | exp4v2 파라미터 |
|----|-----------|-----------|-----------------|
| severstal | multi | dataset_config | imgsz 640, rect, 100ep, seeds 42 1 2 |
| mvtec_leather | multi | dataset_config | imgsz 640, rect, 100ep, seeds 42 1 2 (생성 시 `--cn_no_grayscale`) |
| mtd | multi | dataset_config | imgsz 640, rect, 100ep, seeds 42 1 2 |
| aitex | **single (tiled)** | `aitex_tiled/train/good` | imgsz 256, **no rect**, 300ep, seeds 1 2 42 (생성 시 AR/텍스처 게이트) |

---

## 핵심 로직

- **Fresh 전조건 학습 (exp4v2)**: baseline/random/aroma 모두 COCO pretrained에서 처음부터 독립 학습 — 서로의 weight를 재사용하지 않음(graft 미사용). 합성 결함은 지도학습 레이블 데이터로 train에만 투입, test/defect는 항상 real.
- **Per-condition synth-ratio capping**: `--synth_ratio 1.0` 지정 시 `cap = max(1, int(n_real_train * ratio))`를 random/casda/aroma에 **동일 cap·동일 seed**로 적용해 전체 합성 개수를 맞춘다(`exp4_v2_supervised_detection.py:2418-2434`). 동일 seed 결정적 subsample이므로 step5를 한 번 크게 생성(`--n_per_roi` 상향)해두면 `--synth_ratio`만 바꿔 여러 번 실행 가능(ratio마다 `--output_dir` 분리). 클래스별 개수·라벨화-후 실개수는 강제 매칭하지 않음 — AROMA 선택·배치의 post-treatment 결과를 지우는 bad control 회피(계측만).
- **YOLO real-data cache**: `--yolo_cache_dir`로 real (image,label) 데이터셋을 build-once 후 영속 캐시. 동일 (min_area/val_frac/seed) 조합은 재빌드 없이 재사용하며, per-condition loop 밖으로 hoist되어 baseline/random/aroma가 동일 pre-built pair를 stage.
- **embed_cache DINOv2 재사용**: exp5·exp6이 `S('embed_cache')`를 공유. 캐시 키가 경로(sha256 of image_path 목록) 기반이라, step5 재생성이 동일 파일명으로 덮어쓰면 stale 임베딩 재사용 → 재합성 데이터셋은 `rm -rf $EMBED_CACHE_DIR/{ds}` 후 재실행 필수.

---

## 주의사항

- **사후 튜닝 금지**: τ·seed·synth_ratio·epochs·imgsz·image_size·nearest_k·permutation_reps 등은 확정값을 그대로 쓰고 결과 보고 후 변경하지 않는다. 파라미터를 바꾸면 `--resume` skip에 걸려 재학습되지 않으니 fresh `--output_dir`로만 재실행. FID/AUROC/mAP 방향이 기대와 달라도 재생성·재선택으로 되돌리지 않음.
- **합성 재생성 금지 (parity)**: exp는 step5 산출을 읽기만 한다. arm 정의(aroma=grounded, random=naive)는 step5에서 확정됨 — exp에서 게이트를 다시 켜거나 합성을 다시 만들면 n_synth·arm 정의 parity가 깨진다. clean-bg 게이트는 **AROMA arm 전용**(random은 naive라 미적용)이므로 exp에서 대칭 맞추려 하지 말 것.
- **그룹 파라미터 불변 (exp4v2)**: 3종(multi·640·rect·100ep) / aitex(single·256·no-rect·300ep) 두 그룹을 섞으면 비교 불가. aitex에 `--rect`·`--class_mode multi`를 붙이면 비타일 폐기 regime이 되어 무효.
- **aitex tile-level·single-class**: FID/AUROC/mAP/PRDC 모두 **절대값을 타 데이터셋과 직접 비교 금지**. 동일 데이터셋 내 조건 간 Δ(차이)만 유효하며, 단독 결론 금지(aggregate 보조).
- **기제 vs 인과**: exp5/exp6은 mechanism 증거(임베딩 커버리지)이고, downstream mAP 인과는 exp4v2(L3)가 담당 — 패키지로 함께 제시한다.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md), 시간 실측·처리량 벤치 미수행(load-test policy). 검증은 Colab 실행으로.

---

## 관련 노트

[[00-INDEX]] | [[05-Stage3-ControlNet-Generation]] | [[07-Scripts-Reference]] | [[08-Datasets]]
