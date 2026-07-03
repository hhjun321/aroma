# AROMA 다중도메인 확장 — AITEX(텍스타일) + MTD(자성타일) 데이터셋 통합

## (사용할 skills: feature-dev)

> 신규 도메인 2종 통합: dataset_config 엔트리 + `_find_mask_path` 분기 + prepare 스크립트(신규). 여러 파일·신규 추상화 → feature-dev. 관련: [[aroma_research-core_self-contained-multidomain-design]](breadth 목표), [[aroma_dataset_phase0_analysis]].

## 개요

AROMA breadth(다중도메인 일반성) 실증용 데이터셋을 **4셋**으로 확정: Severstal(강판)·MVTec leather(가죽)·**AITEX(텍스타일)**·**MTD(자성타일)**. 앞 2셋은 이미 네이티브(dataset_config 엔트리 + `_find_mask_path` 지원) → **작업 0(검증만)**. 뒤 2셋(AITEX/MTD)은 신규 통합 대상.

도메인 축 분산: 경성표면(Severstal 강판, MTD 자성타일) + 연성/유기(AITEX 텍스타일, MVTec leather 가죽) → "강철 편중" 탈피.

전제: severstal/mvtec_leather는 **기존 보유**(무수정). 미보유 2셋(AITEX/MTD)은 **Colab 스크립트로 공개소스에서 Drive에 다운로드 후 수행**:
- **AITEX** = Kaggle `nexuswho/aitex-fabric-image-database` (kaggle API). 공개 구조: `Defect_images/`(`nnnn_ddd_ff.png`) + `Mask_images/`(`nnnn_ddd_ff_mask.png`) + `NODefect_images/`(`nnnn_000_ff.png`) 3폴더. ← flat 아님, 이 3폴더 기준.
- **MTD** = Dataset Ninja **Supervisely 배포판** (`meta.json` + `ds/img/*.jpg` + `ds/ann/*.jpg.json` bitmap 주석). `prepare_mtd.py`로 mvtec식 정규화. (abin24 git clone 아님 — 실 Drive 포맷이 Supervisely.)

코드는 이 **공개 구조 기준 작성**, 다운로드·정규화·검증 전부 Colab 가이드 셀로. 실제 확인은 다운로드 후 Colab(경로 assert + Stage1 profiling smoke + mask_source=ground_truth 확인).

### 코드 요건 (기존 파이프라인, 확인됨)
- morphology 6지표는 **pixel mask** 필수 (`distribution_profiling.py` `_morph_worker` :215; GT 없으면 SAM/Otsu fallback, mask_source=`fallback_*`).
- `_find_mask_path`(:119) 도메인 분기: 현재 mvtec/visa/severstal. isp/기타 → None → fallback.
- `good image_dir` 필수(context/deficit, :551-565), `seed_dirs`=결함 이미지 디렉토리.
- dataset_config 엔트리: `{domain, image_dir(good), seed_dirs[], notes}`.

---

## 영향도 분석

### 이 기능이 변경/신설하는 상태
- `dataset_config.json`: **신규 키 `aitex`, `mtd`** 추가.
- `scripts/distribution_profiling.py` `_find_mask_path`: **`elif domain=='aitex'` / `elif domain=='mtd'` 분기 추가**.
- **신규**: `scripts/aroma/prepare_aitex.py`, `scripts/aroma/prepare_mtd.py` (원본 레이아웃 → AROMA 기대 레이아웃 정규화 or mask 경로 해소).
- **신규**: Colab 통합 검증 가이드 `AROMA연구분석/colab_execute/multidomain_integration_verify_execute.md`.

### 그 상태를 전제로 동작하는 기존 로직 (깨지면 안 됨)
- **기존 도메인 분기 무수정**: mvtec/visa/severstal/isp `_find_mask_path` 분기 및 fallback 경로 그대로. 신규는 **`elif`로만 추가** → 회귀 0.
- **mvtec_leather/severstal 엔트리 무수정**: 이미 동작 → 검증만.
- `_morph_worker`/context/GMM 파이프라인 무수정 — 신규 도메인도 동일 경로 통과.

### 신규성 (delete/bulk 아님)
- 순수 추가(엔트리·elif·신규 파일) → 기존 27개 dataset 엔트리·기존 도메인 회귀 없음.

---

## 수정 내용

### 1. AITEX (텍스타일) — domain='aitex'

**원본 레이아웃**(Kaggle nexuswho): **3폴더** — `Defect_images/`(`nnnn_ddd_ff.png`, 105장), `Mask_images/`(`nnnn_ddd_ff_mask.png` binary white=defect), `NODefect_images/`(`nnnn_000_ff.png`, 140장). 4096×256 PNG, 12 defect code(ddd), 7 fabric(ff).

**통합 방식 (권장 (a))**:
- **(a) `prepare_aitex.py`**: 3폴더 → mvtec식 정규화. `NODefect_images/`→`aitex/train/good/`, `Defect_images/{stem}`→`aitex/test/{ddd}/{stem}.png`, `Mask_images/{stem}_mask.png`→`aitex/ground_truth/{ddd}/{stem}_mask.png`. → **`_find_mask_path` mvtec 분기 재사용**(분기 추가 불요). defect_type=ddd code.
- (b) `_find_mask_path` aitex 분기(Defect_images↔Mask_images 폴더 크로스): mask=`.../Mask_images/{stem}_mask.png`, config seed_dirs=Defect_images. prepare 불요이나 good/defect 폴더 이미 분리돼 있어 (a) 정규화가 mvtec 재사용상 더 깔끔.
- → **(a) 권장**.
- ⚠️ Mask_images는 결함 이미지 중 **일부만 mask 제공**(AITEX 알려진 특성 — mask 없는 결함 존재) → mask 없는 결함은 skip+count(정직).

**dataset_config 엔트리**(정규화 후 경로):
```
"aitex": {domain:"mvtec"(정규화로 mvtec 레이아웃) 또는 "aitex", image_dir:".../aitex/train/good", seed_dirs:[".../aitex/test/{ddd}"...], notes:"4096x256, 12 defect codes, 105 defect/140 good. prepare_aitex 정규화."}
```
⚠️ **4096×256 극단 aspect** — 결함 작고 좁음. profiling은 full image 처리하나 morphology는 mask 영역서 산출 → 문제 없음. context patch 그리드가 가로로 길어짐 주의(§TODO).

### 2. MTD (자성타일) — domain='mtd'  ⚠️ 포맷 정정 (abin24 아님)

**실제 원본 = Dataset Ninja Supervisely 배포판** (Drive 실측): `meta.json`(5클래스 전부 bitmap) + `ds/img/{name}.jpg`(1344) + `ds/ann/{name}.jpg.json`(`{size, objects:[{classTitle, bitmap:{data=base64(zlib(PNG RGBA, alpha=mask)), origin:[x,y]}}], tags}`). `objects==[]`=good(956), `objects!=[]`=결함(388, 459 obj: blowhole116/break128/uneven108/crack70/fray37). (앞 가정 abin24 `MT_*/Imgs` co-located PNG는 **틀림** — 폐기.)

**통합 방식 = prepare_mtd 정규화 (AITEX와 동일 패턴)**:
- **`scripts/aroma/prepare_mtd.py`(신규)**: Supervisely bitmap 디코드(`zlib.decompress(base64)→cv2.imdecode→RGBA alpha를 origin에 배치`한 full-frame mask, 실검증 완료) → mvtec식 레이아웃: `train/good/{name}.jpg`, `test/{class}/{name}.jpg`, `ground_truth/{class}/{stem}_mask.png`(class별 union). 빈 mask skip+count(Otsu 날조 방지). 다중결함 이미지는 존재 class마다 엔트리.
- **`_find_mask_path` mtd 분기 = mvtec식**: `image_path.parent.parent.parent/ground_truth/{defect_type}/{stem}_mask.png` (aitex 분기 미러). (구 co-located `with_suffix('.png')` 폐기.)
- seed 이미지 glob: domain=='mtd' → `.jpg`만 (정규화 후 test/에 .jpg만, mask는 ground_truth/ 분리).

**dataset_config 엔트리** (정규화 경로):
```
"mtd": {domain:"mtd", class_mode:"multi",
  image_dir:".../mtd/train/good",
  seed_dirs:[".../mtd/test/blowhole", ".../test/break", ".../test/crack", ".../test/fray", ".../test/uneven"]}
```

### 3. Severstal / MVTec leather — 검증만 (무수정)
- 둘 다 dataset_config 엔트리 존재 + `_find_mask_path` 네이티브(severstal / mvtec). Colab에서 경로 assert + profiling smoke로 정상 동작 재확인만.

### 4. Colab 검증 가이드 (신규)
`multidomain_integration_verify_execute.md`: 4셋 각각 (a) 경로 존재 assert, (b) Stage1 `distribution_profiling` 소규모 실행, (c) 산출 `morphology_features.csv`에서 **mask_source가 `ground_truth`인지**(fallback_* 이면 mask 경로 해소 실패) + morphology 비-degenerate(solidity<1, extent<1 — bbox-as-mask 붕괴 아님) 확인. $VAR 형식, !python, AROMA self-contained env.

---

## 수정 대상 파일

- `dataset_config.json` — `aitex`, `mtd` 엔트리 추가
- `scripts/distribution_profiling.py` — `_find_mask_path`에 aitex(정규화 시 mvtec 재사용이면 불요)/mtd elif 추가
- **신규** `scripts/aroma/prepare_aitex.py`
- **신규** `scripts/aroma/prepare_mtd.py` (또는 분기만으로 대체 시 생략)
- **신규** `AROMA연구분석/colab_execute/multidomain_integration_verify_execute.md`

---

## 테스트 (Colab 전용, 새 테스트코드·pytest 금지)

- 로컬 정적: `python -m py_compile` (신규 prepare 스크립트 + 수정 distribution_profiling).
- Colab(업로드 후): 4셋 경로 assert → Stage1 profiling smoke → morphology_features.csv에서 mask_source=ground_truth + 비-degenerate morphology 확인. AITEX/MTD가 fallback_*로 나오면 분기/경로 오류.

---

## 미확정 사항 (TODO — 업로드 후 확정)

1. **실제 Drive 하위 폴더명/구조**: AITEX(Kaggle 3폴더), MTD(Supervisely ds/ann+ds/img → prepare_mtd 정규화 후 mtd/train/good·test/{class}). 실경로로 config 확정 (완료).
2. **AITEX 통합 (a)prepare 정규화 vs (b)분기** — 권장 (a). ddd defect code→명칭 매핑 테이블 필요 여부.
3. **AITEX 4096×256 aspect** — context patch 그리드/ MCI-CCI 계산에 극단 aspect 영향? 필요 시 리사이즈/패치 정책.
4. **MTD 이미지 glob**: profiling의 seed 이미지 수집이 `.jpg` 확장자 포함하는지 확인(대부분 png 가정 코드면 보강).
5. **AITEX domain 라벨**: 정규화로 mvtec 레이아웃이면 domain='mvtec'로 둘지 'aitex' 별도로 둘지(후자면 분기 추가). breadth 보고엔 'aitex' 명시가 유리.
6. **defect_type 다양성**: AITEX 12종(일부 희소), MTD 5종 — GMM n_clusters 정책은 recommended_config가 자동 결정(무수정).
