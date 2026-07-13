# 08 — 데이터셋

> **Claude 요약:** v2-1 4종(`severstal` · `mvtec_leather` · `mtd` · `aitex`)이 sym_final 파이프라인이 소비하는 정본 데이터셋이다. 3종은 `class_mode=multi`, aitex만 `single`(256×256/stride128 tiled). 각 데이터셋은 `dataset_config.json`에 `image_dir`(=`train/good`) + `seed_dirs`(=`test/{class}`)로 등록되어 있고, phase0가 이를 읽어 프로파일링한다. exp4v2 학습 파라미터(imgsz·rect·epochs·seeds)는 데이터셋별로 다르다.

---

## v2-1 데이터셋 4종

| ds | class_mode | normal_dir (`image_dir`) | 비고 |
|----|-----------|--------------------------|------|
| `severstal` | multi | `.../severstal/train/good` | Kaggle RLE→PNG. 4 결함 클래스(class1..4). class2 희소(Neff<4) → multi 불균형. mask는 `masks/`에서 해소(`ground_truth/` 아님). `prepare_severstal.py` + context prototypes(CLIP) 선행 |
| `mvtec_leather` | multi | `.../mvtec/leather/train/good` | MVTec-AD 가죽, **Drive에 표준 레이아웃으로 이미 존재 → 준비 불요(경로 확인만)**. 컬러 가죽이라 생성 시 `--cn_no_grayscale`(grayscale target OFF), CN 학습도 `--no_force_grayscale_target`. 5 클래스(color/cut/fold/glue/poke) |
| `mtd` | multi | `.../mtd/train/good` | 자성타일(Dataset Ninja Supervisely). 1344 imgs(good≈956 / defect≈388), 5 클래스(blowhole/break/crack/fray/uneven), 전부 bitmap. `prepare_mtd.py`가 MVTec-style로 정규화. 이미지 `.jpg` |
| `aitex` | **single** | `.../aitex_tiled/train/good` | 직물. 원본 4096×256(16:1) → **256×256/stride128(50% overlap) tiled**. 전 결함코드를 `test/defect`로 병합(single-class). `prepare_aitex.py --tile 256 --stride 128` |

---

## exp4v2 데이터셋별 파라미터

(`_SPEC.md §4` 정본 — `exp4_v2_supervised_detection.py` fresh 전조건 학습)

| ds | imgsz | rect | epochs | seeds | 그룹 |
|----|-------|------|--------|-------|------|
| `severstal` | 640 | ✓ (`--rect`) | 100 | 42 1 2 | multi 3종 (`--class_mode multi`) |
| `mvtec_leather` | 640 | ✓ | 100 | 42 1 2 | multi 3종. 생성 시 `--cn_no_grayscale` |
| `mtd` | 640 | ✓ | 100 | 42 1 2 | multi 3종 |
| `aitex` | 256 | ✗ (no rect) | 300 | 1 2 42 | single (`--class_mode` 미지정). 생성 시 AR/텍스처 게이트 |

- multi 3종은 한 셀에서 `--dataset_keys severstal mvtec_leather mtd --class_mode multi --imgsz 640 --rect --baseline_epochs 100 --seeds 42 1 2`.
- aitex는 별도 셀: `--dataset_keys aitex --imgsz 256 --baseline_epochs 300 --seeds 1 2 42`(`--rect`·`--class_mode` 없음).
- 공통: `--model yolov8n --condition all`(baseline/random/aroma 모두 fresh, graft 미사용), `--val_frac 0.3 --synth_ratio 1.0 --patience 50 --batch 64 --cache ram --resume`.

---

## dataset_config.json 규약

저장소 루트 `dataset_config.json`에 4종 엔트리가 커밋되어 있다. 각 데이터셋 엔트리 키:

| 키 | 의미 |
|----|------|
| `domain` | 도메인 태그(`severstal` / `mvtec` / `aitex` / `mtd`). mask 해소 분기·이미지 glob 규칙을 결정 |
| `class_mode` | `multi`(severstal/mvtec_leather/mtd) 또는 `single`(aitex). phase0·roi_selection·exp4v2의 클래스 게이트 발동 여부 결정 |
| `image_dir` | `train/good` 절대경로. `normal_dir(ds)`가 반환하는 값이고 생성 `--normal_dir`의 소스 |
| `seed_dirs` | `test/{class}` 절대경로 배열(= 결함 클래스 seed). 일부 단일결함 데이터셋은 `seed_dir`(단수) 사용 |
| `notes` | 준비 방식·해상도·클래스 분포·mask 해소 경로 등 특이사항 서술 |

표준 스키마 예:

```json
"severstal":     { "domain": "severstal", "class_mode": "multi",  "image_dir": ".../severstal/train/good",   "seed_dirs": [".../test/class1", "...class2", "...class3", "...class4"] },
"mvtec_leather": { "domain": "mvtec",     "class_mode": "multi",  "image_dir": ".../mvtec/leather/train/good","seed_dirs": [".../test/color", ".../cut", ".../fold", ".../glue", ".../poke"] },
"aitex":         { "domain": "aitex",     "class_mode": "single", "image_dir": ".../aitex_tiled/train/good",  "seed_dirs": [".../aitex_tiled/test/defect"] },
"mtd":           { "domain": "mtd",       "class_mode": "multi",  "image_dir": ".../mtd/train/good",          "seed_dirs": [".../test/blowhole", ".../break", ".../crack", ".../fray", ".../uneven"] }
```

- 공통 헬퍼: `normal_dir(ds) = CFG[ds]["image_dir"]`, `is_multi(ds) = CFG[ds].get("class_mode") == "multi"`(aitex는 자동 single).
- **경로 정합 원칙**: 전 경로는 `DRIVE=/content/drive/MyDrive/data/Aroma` 기준. config의 절대경로가 실제 Drive와 어긋나면 **config의 해당 경로만 수정**(코드·스크립트 수정 아님).

---

## 데이터셋별 준비 특이사항

- **aitex 타일링 근거**: 원본은 4096×256(16:1)이라 exp4v2 `imgsz=256` letterbox 시 종횡비가 256×16으로 붕괴 → baseline mAP50 0.066(학습 불가). 따라서 **반드시** 256×256/stride128(50% overlap)로 타일링한 `aitex_tiled`만 사용한다. 구 비타일 `aitex` 레이아웃과 **혼용 금지**. 타일 stem에 `__tile_r{r}_c{c}`가 붙어 exp4v2가 동일 원본의 타일을 같은 train/val 측으로 묶는다(50% overlap leak 가드). 결함 없는 타일은 `train/good`(bg 풀)로 이동, sub-`min_tile_area` 파편 타일은 폐기.
- **mvtec_leather 기존/컬러**: MVTec-AD leather는 이미 Google Drive에 표준 AROMA 레이아웃(`train/good`, `test/{defect}`, `ground_truth/{defect}`)으로 존재 → 다운로드·변환·추출 모두 불필요, 경로 확인만. 컬러 가죽이므로 grayscale 강제를 끈다(CN 학습 `--no_force_grayscale_target`, 생성 `--cn_no_grayscale`).
- **severstal**: 원본은 Kaggle 포맷(`train.csv` RLE + `train_images/`)이라 `prepare_severstal.py`로 AROMA 레이아웃(RLE→PNG mask + `train/good` + `test/class1..4`) 변환 필요. mask는 `masks/`(+`masks/class{c}/`)에서 해소된다(`ground_truth/` 아님). class2 희소는 데이터 특성.
- **mtd**: Supervisely bitmap 주석(base64-zlib-PNG)을 full-frame mask로 rasterize. 다중결함 이미지는 각 클래스 아래에 class-union mask로 중복 등장.
- **NEU-DET 탈락**: (준비 문서·SPEC 범위에서 v2-1 4종은 mask+good+breadth 기준으로 선정된 집합이며, bbox-only 데이터셋은 이 레이아웃 요건을 충족하지 못한다. 관련 세부는 프로젝트 메모리 `project_v2-1_datasets` 참조 — 본 문서 소스 파일에는 명시되지 않음.)

---

## 주의사항

- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지. **Δ(개선폭)만 유효**하다.
- aitex는 tiled 전용: `aitex_tiled` 신규 dir만 사용, 구 비타일 `aitex`와 혼용 금지.
- prepare_* 스크립트·추출은 멱등(재실행 안전). severstal/aitex는 원본 dir을 실제 Drive 위치에 맞게 수정.
- 4종 모두 config 검증 셀에서 ✓(`image_dir` + 전 `seed_dirs` 존재)여야 phase0 진입.

---

## 관련 노트

[[00-INDEX]] | [[02-Stage0-Prepare-Profiling]] | [[06-Experiments]]
