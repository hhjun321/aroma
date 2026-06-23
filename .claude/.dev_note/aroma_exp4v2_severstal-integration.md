# Exp4v2 — Severstal 데이터셋 통합 (CASDA 비교용)

## (사용할 skills: feature-dev)

## 개요

CASDA 연구의 Severstal Steel Defect Detection(Kaggle) 셋을 AROMA exp4v2(YOLO supervised detection)에 추가한다. CASDA는 Severstal 위에서 ControlNet 합성 + Poisson blending을 했으나, AROMA 비교는 dev_note `aroma_exp1_severstal-casda-comparison.md` 설계대로 **"동일 copy-paste synthesis, ROI selection 전략만 다름"** 으로 ROI modeling 기여를 분리 측정한다(CASDA augmented 재사용 아님).

핵심 challenge: Severstal mask는 train.csv의 **RLE**(PNG 아님), 결함 **4-class**, 이미지 **1600×256**(비정사각), 합성 파이프라인·dataset_config 미등록. 본 작업은 **baseline(real-only) 평가까지** 우선 구현(합성 random/aroma는 후속). class는 single·4-class 양쪽 모드 지원.

## 결정 사항 (질의 확정)
- **RLE → binary PNG 사전생성** (prepare_severstal.py, prepare_visa first 패턴). 코드 mask=PNG 가정 무수정 재사용. merged(masks/{id}.png) + per-class(masks/class{1-4}/{id}.png).
- **defect_type = ClassId(1~4)** — class_diversity(`_class_diversity_neff`, defect_type distinct 카운트)가 4-class 인지. **Class 2 희귀 → Neff<4 (진짜 불균형)** = MVTec(균형, Neff≈n_types) 대비 DEFICIT 우위 발현 최적 후보. per-class 구조(test/class{1-4} 또는 manifest)로 profiling에 ClassId 전달.
- **class: single + multi(4-class) 둘 다** — `--class_mode {single,multi}`, default `single`(기존 데이터셋 동작 불변).
- **imgsz 640~1280** — CLI 무수정(값만). 1600×256 letterbox.
- **평가 범위: baseline 먼저** — synth 불요. random/aroma는 후속(generate_casda.py).
- **경로: AROMA 베이스** `/content/drive/MyDrive/data/Aroma/severstal/...`, dataset_config 등록.
- **normal/split**: train.csv ImageId=defect, train_images 차집합=normal. visa `_split_normal`(seed42, test_split) 패턴.
- **normal 샘플링 = CLIP 클러스터링 1000 프로토타입** (전량/1:1 head-slice 아님): All normal → CLIP Vision Encoder(512d) → L2 norm → PCA 512→64(opt) → MiniBatch KMeans(K=1000) → cluster medoid → **1000 Representative Context Prototypes**. 용도: 합성 배경(주) + baseline 배경 negative 풀.
- **CASDA 정상 세트 정합: (b) 분포만** — CASDA 목록 직접 교집합 X. CLIP-KMeans 1000이 전체 분포 대표 → 분포 근사로 편향 방어.
- **비교축**: AROMA-ROI vs CASDA-ROI vs Random (동일 copy-paste). CASDA ControlNet 수치 직접비교 X (09-Experiments 수치 공란).

## 영향도 분석

### 변경 상태
- 신규 `prepare_severstal.py`: RLE→PNG mask + normal/defect manifest + Aroma/severstal 레이아웃.
- exp4v2 `_get_image_lists`에 severstal 분기 + class_mode 분기.
- shared 라벨 코드(class_id) — multi 모드만 4-class, single은 class0(불변).
- dataset_config.json severstal 항목.

### 그 상태를 전제로 동작하는 기존 로직 (하위호환)
- `_bboxes_to_yolo_lines(class_id=0)`, `_write_yolo_yaml`(nc:1, names:['defect']): class_mode='single' default → 기존 MVTec/VisA 경로 byte-identical.
- `_mask_to_bboxes`(cv2.imread PNG): RLE→PNG라 무수정.
- `_split_defects`(mask_map 키 있는 것만): severstal mask_map 채우면 동작.

### 회귀 위험
- class_mode default='single' → 기존 전 데이터셋 영향 0.
- multi 모드는 severstal 전용 경로로 격리. shared 함수에 class_id/nc 파라미터 추가하되 기본값=현행.

## 수정 내용

### 1. `scripts/aroma/prepare_severstal.py` (신규)
- 입력: `--train_csv`(ImageId,ClassId,EncodedPixels), `--train_images`, `--output_dir`(Aroma/severstal).
- CASDA `src/utils/rle_utils.py` 디코드 로직 이식/재사용 (rle_decode: 1-indexed, order='F', shape=(256,1600)).
- ImageId별:
  - **per-class mask PNG**: `masks/class{c}/{ImageId}.png` (multi 모드용, ClassId별) — defect_type=ClassId 전달원
  - **merged binary PNG**: `masks/{ImageId}.png` (single 모드용, 전 class OR)
- **결함 조직**: `test/class{1..4}/` 로 ClassId별 분리(profiling이 subfolder→defect_type 매핑) → class_diversity가 4-class·Class2 희귀 인지.
- **normal 도출**: train_images 전체 ∖ train.csv ImageId 집합 (또는 EncodedPixels NaN). `good/` 디렉토리.
- 출력 레이아웃(MVTec 유사):
  ```
  Aroma/severstal/
    train/good/                     (normal 전량 — 프로토타입 선택 입력)
    test/class{1..4}/               (결함 이미지, ClassId별 → defect_type)
    masks/{ImageId}.png             (merged, single)
    masks/class{1..4}/{ImageId}.png (per-class, multi)
    severstal_manifest.json
  ```
- Colab 1회 선행 실행 (prepare_visa 패턴).

### 1b. `scripts/aroma/select_context_prototypes.py` (신규)
- 입력: normal 이미지 디렉토리(`train/good/`), `--k 1000`, `--output`(프로토타입 목록 JSON + 선택 이미지 디렉토리).
- 파이프라인: CLIP Vision Encoder(예: open_clip ViT-B/32) 임베딩(N×512) → L2 normalize → PCA 512→64(optional, `--pca 64`) → MiniBatch KMeans(K=1000, seed=42) → cluster별 **medoid**(센트로이드 최근접 실제 이미지) → 1000 representative context prototypes.
- 출력: `context_prototypes.json`(선택 ImageId 목록) + `context/` 디렉토리(심볼릭/복사). exp4v2 baseline negative 풀 + 합성 배경(후속)이 이 집합 참조.
- GPU 권장(임베딩 12000장). sklearn MiniBatchKMeans, medoid는 cluster 내 센트로이드 L2 최근접.
- 결정적: KMeans seed=42, medoid tie-break 안정 정렬.

### 2. `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
**`_get_image_lists` severstal 분기** (L288 else 직전):
```python
elif dataset_key == "severstal":
    ds = base / "severstal"
    if not ds.exists(): return None
    train_normal = _glob_images(str(ds / "train" / "good"))
    all_defect = _glob_images(str(ds / "test" / "defect"))
    train_normal, test_good = _split_normal(train_normal, test_split, seed)  # 또는 별도
    test_defect = all_defect
    mask_map = _resolve_severstal_masks(test_defect, str(ds / "masks"))  # single: merged
    # multi 모드면 mask_map 값에 per-class 구조 또는 별도 class_map
```
**`--class_mode {single,multi}`** argparse (default single). run()/_run_detection_mode 전파.
- single: 기존 class0 경로(무변경).
- multi: nc=4. `_bboxes_to_yolo_lines`에 실제 class_id, `_write_yolo_yaml`에 nc=4/names=['c1','c2','c3','c4'], mask→bbox 시 per-class mask로 class 할당. severstal 전용.
**`_resolve_severstal_masks`** 신규 (visa/isp resolver 패턴, ImageId stem → masks/{stem}.png).

### 3. `dataset_config.json`
severstal 항목 추가: domain="severstal", image_dir=Aroma/severstal/train/good, (seed_dirs는 RLE라 직접 안 맞음 — 합성 단계에서 crop 산출 후 채움, baseline 단계엔 불요).

## 수정 대상 파일
- `scripts/aroma/prepare_severstal.py` (신규 — RLE→PNG, per-class 조직, normal 도출)
- `scripts/aroma/select_context_prototypes.py` (신규 — CLIP→KMeans(1000)→medoid)
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (severstal 분기 + `--class_mode` + baseline negative를 프로토타입 풀에서 샘플)
- `dataset_config.json` (severstal 등록)
- `AROMA연구분석/colab_execute/` — prepare_severstal + select_context_prototypes + exp4v2 severstal 실행 가이드 (신규)

## 후속 (이번 범위 밖)
- random/aroma 합성: distribution_profiling→complexity→prompt→roi_selection→generate_casda.py 전체 Severstal 1회 실행 (RLE→crop ROI 산출 필요). 별도 dev_note.
- CASDA ROI selection(suitability≥0.5, per_class_cap) → generate_casda.py 이식.

## 암묵적 요구사항 (엣지)
- **RLE column-major**: order='F', shape=(256,1600)=(H,W). 디코드 검증 — 마스크 PNG를 원본 위 오버레이 육안.
- **normal 행 부재**: Kaggle Severstal은 정상 이미지가 train.csv에 없을 수 있음 → train_images ∖ ImageId 차집합으로 normal 도출(NaN 행 가정 금지, 실파일 검증).
- **multi-class mask 겹침**: 한 이미지 다중 class → per-class PNG 분리 저장. single은 OR merge(class 손실 무해).
- **1600×256 letterbox**: imgsz 640 시 가로 축소 → 작은 결함 소실 가능. val map50 진단(기존 경고) 확인. 부족 시 imgsz↑ 또는 후속 타일링.
- **이미지 grayscale**: YOLO는 3채널 기대 — cv2 BGR 로드 자동 처리(기존 _image_size/staging 무관).
- **class_mode 격리**: multi가 shared 함수 건드릴 때 single/타 데이터셋 경로 불변 보장 (default 분기).

## 테스트 (Colab, pytest 금지)
1. prepare_severstal.py 실행 → masks/{ImageId}.png + class{1-4}/ 생성, manifest. mask PNG 오버레이로 RLE 디코드 정합 육안.
2. normal/defect 수 확인 (train_images ∖ csv).
3. exp4v2 `--dataset_keys severstal --condition baseline --class_mode single --imgsz 640` → real-only baseline map50 정상(>0, val 경고 없음).
4. `--class_mode multi` → nc=4 YAML, per-class 라벨, map50/per-class.
5. 기존 데이터셋(mvtec_cable) `--class_mode single`(default) 회귀 → 기존과 동일.

## 미확정 TODO
- normal 정의: train.csv 미등장 전부 normal로? 일부만 샘플? (Severstal normal 수천 장 → test_split 비율 결정)
- multi 모드 class names 명명 (c1~c4 vs defect_1~4).
- 후속 합성 단계 dev_note 분리 작성.
- imgsz 최종값(640 vs 1280) — baseline 검증 후 결정.
