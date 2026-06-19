# AROMA Exp4v2 — Supervised YOLOv8 Detection으로 실험 패러다임 전환

## (사용할 skills: feature-dev)

## 개요

Exp4 (one-class AD 평가)의 근본적 설계 결함 발견:
`_prepare_ad_dataset_with_masks()`가 합성 결함 이미지를 `train/good/`에 복사 →
PatchCore/EfficientAD/RD++ 등 one-class 모델이 결함을 "정상"으로 학습 → AUROC 붕괴.

빠른 검증 결과 (mvtec_cable / simplenet):
- baseline: 0.7646  |  random: 0.5015  |  aroma: 0.4996

baseline >> random ≈ aroma 패턴: AROMA 합성이 정상 분포를 더 심하게 오염시킨 결과.

올바른 패러다임: 합성 결함 이미지 + bbox 레이블 → **supervised YOLOv8 detection 학습**.
합성 데이터가 많을수록 mAP가 높아야 정상 — AROMA가 random보다 우수하면 가설 검증.

---

## 근본 원인 분석

```python
# exp4_downstream_ad.py line 342-343 (버그 위치)
train_dir = tmpdir / "train" / "good"
for i, p in enumerate(synth_image_paths):
    bulk_tasks.append((p, str(train_dir / f"syn_{i:05d}{Path(p).suffix}")))
    #                        ^^^^^^^^^^^ defects → good 폴더에 복사 → AUROC 붕괴
```

One-class AD (PatchCore, SimpleNet, EfficientAD, RD++)는 정상 분포만 학습.
결함 이미지가 train/good에 있으면 결함이 "정상"으로 모델링됨.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- Exp4 파이프라인: one-class AD 평가 → supervised YOLOv8 detection 평가
- 출력 메트릭: Image AUROC → mAP@0.5, mAP@0.5:0.95, Precision, Recall
- exp4_downstream_ad.py: 변경 없음 (기존 유지)

### 새 파일 구조
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (신규)
- `AROMA연구분析/colab_execute/exp4v2_execute.md` (신규)

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — 신규 구현

#### 1.1 bbox 추출: 이미지 차분 방식

합성 이미지에는 별도 마스크 없음. `annotations.json`의 `normal_image` 필드로
배경 이미지를 찾아 composite−background 차분으로 bbox 추출.

```python
def _extract_defect_bboxes(synth_path, normal_path, min_area=200):
    composite = cv2.imread(synth_path, cv2.IMREAD_COLOR)
    background = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(composite, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((3,3)), iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(x, y, bw, bh) for c in contours
            for x, y, bw, bh in [cv2.boundingRect(c)] if bw * bh >= min_area]
```

#### 1.2 annotations.json 로드

```python
def _load_synth_annotations(synth_root, dataset_key):
    # {synth_root}/{ds}/annotations.json 읽기
    # 'dry_run': True 항목 제외
    # 반환: list of {image_path, normal_image, ...}
```

#### 1.3 YOLO 레이블 생성

```python
def _write_yolo_labels(annotations, synth_dir, label_dir):
    # 각 합성 이미지마다 0 cx cy w h (normalized) 형식으로 저장
    # bbox 추출 실패 시 해당 이미지 스킵
```

#### 1.4 조건별 학습

```python
def _run_yolo_condition(images, labels, model_key, epochs, device, output_dir, condition):
    # baseline: zero-shot pretrained YOLOv8 (no training)
    # random/aroma: 합성 이미지 + bbox 레이블로 fine-tune
    # 평가: model.val() → mAP@0.5, mAP@0.5:0.95, precision, recall
```

#### 1.5 heartbeat + 진행 로그 (exp4와 동일 방식)

```python
def _fit_with_heartbeat(model_fn, label, interval_s=60):
    # threading.Event + Thread로 60초마다 경과 시간 출력
```

#### 1.6 CLI 인터페이스

```
--model         {yolov8n, yolov8s, yolov8m}  (기본: yolov8n)
--condition     {baseline, random, aroma, all}
--dataset_keys  {isp_LSM_1, mvtec_cable, visa_cashew, visa_pcb, all}
--epochs        int (기본: 50)
--batch         int (기본: 16)
--imgsz         int (기본: 640)
--seed          int (기본: 42)
--random_synthetic_dir
--aroma_synthetic_dir
--real_data_dir
--output_dir
```

#### 1.7 출력

```
{output_dir}/
  exp4v2_results.json   # {ds: {condition: {model: {map50, map50_95, precision, recall}}}}
  exp4v2_summary.md     # 텍스트 요약 테이블
```

#### 1.8 YOLO 로그 억제

```python
os.environ.setdefault("YOLO_VERBOSE", "False")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
```

---

### 2. `AROMA연구분析/colab_execute/exp4v2_execute.md` — 신규 Colab 가이드

- exp4 실패 원인 설명 (one-class AD + train/good 오염 문제)
- exp4v2 패러다임 설명 (supervised detection)
- 패키지 설치: `!pip install ultralytics -q`
- 빠른 검증 명령: mvtec_cable 단독 / epochs 50
- 전체 실행 명령: 4개 dataset / 기본값

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (신규)
- `AROMA연구분析/colab_execute/exp4v2_execute.md` (신규)

---

## 후속 작업 (이 devnote 범위 외)

- AROMA.txt §4.3 및 Table 11: AUROC → mAP@0.5 메트릭으로 갱신 (실험 결과 확인 후)
- §3.3.2 Benchmark Architectures: YOLOv8 설명으로 교체
- 실험 결과 확인 후 AROMA.txt 논문 본문 업데이트

---

## 테스트

CLAUDE.md: pytest 금지. Colab 직접 검증.

빠른 검증:
```python
os.environ['EXP4V2_OUT'] = f"{os.environ['AROMA_OUT']}/exp4v2"

!pip install ultralytics -q

!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_cable \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seed 42 \
    --epochs 50
```

확인 포인트:
- `annotations.json` 로드 성공 로그
- bbox 추출 성공률 (매칭된 이미지 수)
- YOLO 학습 진행 로그 (heartbeat)
- `exp4v2_results.json` 생성 확인
- aroma mAP > random mAP > baseline mAP 순서 기대
