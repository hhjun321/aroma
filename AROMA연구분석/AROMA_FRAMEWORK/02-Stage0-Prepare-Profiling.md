# 02 — Stage 0: 데이터 준비 & 분포 프로파일링

> **Claude 요약:** v2-1 4종(`severstal · mvtec_leather · mtd · aitex`)을 AROMA 레이아웃으로 준비(step -1)한 뒤, `distribution_profiling.py`가 결함 형태(morphology)·배경 맥락(context) 분포를 프로파일링하여 클러스터·호환성 행렬·추천 config를 생성한다(phase0). 이 산출물은 이후 전 스테이지(complexity/prompt/ROI/생성)와 symmetric 호환성 게이트의 최상류 입력이다.

## 목적

sym_final 파이프라인 체인 `prepare_datasets(step -1) → phase0 → step1 → … → exp*`의 최상류 두 단계를 담당한다.

- **step -1 (prepare_datasets)**: 4종 원본을 MVTec 스타일 레이아웃(`train/good`, `test/{class}`, ground-truth 마스크)으로 정규화하고 `dataset_config.json`에 등록한다. aitex는 256×256/stride128 타일링 + 단일클래스로 특수 처리.
- **phase0 (distribution_profiling)**: 정규화된 데이터셋에서 결함 형태 특징·64px 패치 맥락 특징을 추출하고, GMM 클러스터링·호환성 행렬(legacy `matrix` + clean-grounded `matrix_symmetric`)·deficit 분석·추천 config를 생성한다. `--compat_mode symmetric` 게이트를 쓰려면 여기서 신규 symmetric 키가 방출되어야 하므로 그 존재를 hard assert 한다.

## 입력 / 출력

경로는 `_SPEC.md`의 stage-first 규약 `S(stage, ds) = $SYM_ROOT/{stage}/{ds}`를 따른다. phase0 산출은 `S('profiling', ds)` = `sym_final/profiling/{ds}`.

| 경로변수 | 값 | 설명 |
|---------|----|----|
| `$AROMA_REF` | `/content/AROMA` | 저장소 루트. `distribution_profiling.py`는 `$AROMA_REF/scripts/`(루트)에 있음 |
| `$AROMA_SCRIPTS` | `/content/AROMA/scripts/aroma` | `prepare_*.py`·`select_context_prototypes.py` 위치 |
| `$DATASET_CONFIG` | `/content/AROMA/dataset_config.json` | 4종 `image_dir`/`seed_dirs`/`domain`/`class_mode` 등록처 |
| `$DRIVE` | `/content/drive/MyDrive/data/Aroma` | 데이터·출력 루트 |
| 입력(prepare) | 원본 raw (`SEV_SRC`, `AITEX_RAW`, `MTD_RAW`) | Kaggle/Supervisely 원시 배포판 |
| 출력(prepare) | `$DRIVE/{severstal,mvtec/leather,aitex_tiled,mtd}` | AROMA 레이아웃 + `*_manifest.json` |
| 출력(phase0) | `S('profiling', ds)` = `$SYM_ROOT/profiling/{ds}` | 프로파일링 산출 10종(하단 참조) |

phase0 산출 파일(각 데이터셋 `S('profiling', ds)` 하위):

- `morphology_features.csv` · `context_features.csv`
- `distribution_analysis.json` · `morphology_clusters.json`
- `compatibility_matrix.json` · `deficit_analysis.json`
- `threshold_policies.json` · `recommended_config.yaml` · `analysis_report.md`
- `figures/*.png` (histograms, compatibility heatmap, deficit bars)

---

## step -1: prepare_datasets

phase0 앞. 4종 모두 준비·검증 ✓여야 phase0 진입. 스크립트는 모두 멱등(재실행 시 기존 산출 위에 재생성).

| 데이터셋 | 원본 | 스크립트 | class_mode | image_dir |
|---------|------|---------|-----------|-----------|
| severstal | Kaggle RLE CSV | `prepare_severstal.py` (+ `select_context_prototypes.py`) | multi | `severstal/train/good` |
| mvtec_leather | **Drive 기존** | 준비 불요(경로 확인만) | multi | `mvtec/leather/train/good` |
| aitex | AITEX.zip (Kaggle) | `prepare_aitex.py --tile 256 --stride 128` | **single** | `aitex_tiled/train/good` |
| mtd | Dataset Ninja Supervisely | `prepare_mtd.py` | multi | `mtd/train/good` |

### severstal — RLE→PNG + 레이아웃 + context prototypes

```python
# RLE→PNG mask + MVTec-style 레이아웃 (train/good, test/class1..4, masks/)
!python $AROMA_SCRIPTS/prepare_severstal.py \
    --train_csv     $SEV_TRAIN_CSV \
    --train_images  $SEV_TRAIN_IMG \
    --output_dir    $SEV_OUT
```

```python
# context prototypes (CLIP 1000, PCA 64) — profiling/생성이 참조
!python $AROMA_SCRIPTS/select_context_prototypes.py \
    --image_dir  $SEV_OUT/train/good \
    --k 1000 --pca 64 --seed 42 --model ViT-B-32 \
    --output     $SEV_OUT/context_select
```

| 파라미터 | 설명 |
|---------|------|
| `--train_csv` | `train.csv` (ImageId, ClassId, EncodedPixels; RLE 1-indexed, column-major order='F', shape 256×1600) |
| `--train_images` | Severstal `train_images/` 디렉터리 |
| `--output_dir` | 출력 루트 (예: `$DRIVE/severstal`) |

- normal = `train.csv`에 없는 원본 파일(집합 차). defect는 primary class(최소 ClassId)로 `test/class{1..4}`에 배치.
- mask는 `masks/{stem}.png`(전 클래스 OR merged) + `masks/class{c}/{stem}.png`(per-class). profiling `severstal` 도메인 분기가 `masks/`에서 해소(ground_truth 아님).
- class2가 희소(Neff<4) → multi-class 불균형은 데이터 특성.

### mvtec_leather — Drive 기존 (준비 제외)

MVTec-AD leather는 이미 표준 AROMA 레이아웃(`train/good`, `test/{color,cut,fold,glue,poke}`, `ground_truth/{defect}`)으로 Drive에 존재한다. **다운로드·변환 없이 경로만 확인**하고, dataset_config의 `mvtec_leather.image_dir`와 일치하지 않으면 config 경로만 수정. mask = `ground_truth/{defect}/{stem}_mask.png`(domain=mvtec).

### aitex — tiled 정규화 (256×256 / stride 128, single-class)

원본 4096×256(16:1)은 exp4v2 imgsz=256 letterbox에서 bbox가 붕괴(baseline mAP50 0.066)하므로 **반드시 타일링**한다.

```python
!unzip -o $DRIVE/AITEX.zip -d $AITEX_RAW
```

```python
# tiled 정규화 (256/stride128, 50% overlap, 단일클래스)
!python $AROMA_SCRIPTS/prepare_aitex.py \
    --defect_images   $AITEX_RAW/Defect_images \
    --mask_images     $AITEX_RAW/Mask_images \
    --nodefect_images $AITEX_RAW/NODefect_images \
    --output_dir      $AITEX_TILED \
    --tile 256 --stride 128 --min_tile_area 50
```

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--defect_images` | — | `Defect_images/` (`nnnn_ddd_ff.png`) |
| `--mask_images` | — | `Mask_images/` (`nnnn_ddd_ff_mask.png`, defect보다 파일 적음 — stem 조인) |
| `--nodefect_images` | — | `NODefect_images/` (fabric별 서브폴더 → recursive 스캔) |
| `--tile` | 256 | 타일 크기. `0` = legacy 비타일 |
| `--stride` | 128 | 타일 stride(50% overlap). `(0, tile]` 범위 |
| `--min_tile_area` | 50 | 타일이 defect 타일로 인정되는 최소 bbox 면적(px). exp4v2 `_mask_to_bboxes` min_area(50)와 일치 유지 |
| `--multi_class` | (off) | per-code `test/{ddd}/` 유지. 기본은 단일 `test/defect/` 병합 |

- 타일 stem에 `__tile_r{r}_c{c}` delimiter → exp4v2 group-aware split이 동일 원본 타일을 같은 train/val로 묶음(50% overlap leak 가드).
- defect 타일(면적≥min_area) → `test/defect/` + `ground_truth/defect/{...}_mask.png`(border fragment 지운 cleaned mask). all-zero 창 → `train/good/`(defect fabric 배경 풀). border fragment-only 창 → 폐기.
- all-zero/unreadable mask는 skip+count (profiling Otsu fallback 날조 방지). 재실행 시 `__tile_` 파일만 purge.
- **`aitex_tiled` 신규 dir 전용, 구 비타일 `aitex`와 혼용 금지.**

### mtd — Supervisely bitmap → full-frame mask

```python
# Supervisely → MVTec-style (train/good, test/{class}, ground_truth/{class}/{stem}_mask.png)
!python $AROMA_SCRIPTS/prepare_mtd.py \
    --supervisely_root $MTD_RAW \
    --output_dir       $MTD_OUT
```

| 파라미터 | 설명 |
|---------|------|
| `--supervisely_root` | Dataset Ninja Supervisely 루트 (`meta.json`, `ds/img/*.jpg`, `ds/ann/*.jpg.json`) |
| `--output_dir` | 정규화 출력 루트 (`$DRIVE/mtd`) |

- 5 클래스(blowhole/break/crack/fray/uneven), 전부 bitmap. `objects==[]` → good, 아니면 결함.
- bitmap = base64(zlib(PNG RGBA)) → alpha를 full-frame 0/255 마스크로 래스터화, class별 union. 다중결함 이미지는 존재하는 각 classTitle마다 (이미지, class-union mask) 엔트리 생성(mvtec per-image-per-class 관례).
- 빈/디코드 실패 mask → skip+count(Otsu 날조 방지, prepare_aitex와 동일 계약). cv2+numpy 필수.

### dataset_config.json 등록 검증

4종 엔트리는 저장소에 이미 커밋됨. 경로가 실제 `DRIVE`와 일치하는지 확인하고 불일치 시 **config의 해당 경로만 수정**(코드/스크립트 수정 아님).

---

## phase0: distribution_profiling.py

⚠️ `distribution_profiling.py`는 `scripts/`(루트)에 있다 — `$AROMA_REF/scripts/`로 호출한다(`$AROMA_SCRIPTS`(scripts/aroma) 아님).

```python
for DS in DATASETS:                 # ["severstal", "mvtec_leather", "mtd", "aitex"]
    os.environ['DS'] = DS
    os.environ['PROF'] = S('profiling', DS)
    !python $AROMA_REF/scripts/distribution_profiling.py \
        --dataset_config $DATASET_CONFIG \
        --dataset_key    $DS \
        --output_dir     $PROF \
        --num_workers    -1
```

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--dataset_config` | `/content/AROMA/dataset_config.json` | dataset_config.json 경로 |
| `--dataset_key` | (필수) | 프로파일링할 데이터셋 키 |
| `--output_dir` | (필수) | 출력 = `S('profiling', ds)` (stage-first 필수) |
| `--num_workers` | -1 | `0`=순차, `-1`=auto(cpu-1), `N`=N. 단일 데이터셋 순차 실행이라 중첩 없음 |
| `--max_images` | None | 디버그용 split당 이미지 상한 |
| `--n_gmm_components` | 5 | GMM 최대 컴포넌트 수(BIC가 `[1,N]`에서 자동 선택) |
| `--sam_checkpoint` | None | fallback 마스크용 SAM(GT 마스크 있는 4종은 불요, 없으면 Otsu fallback 자동) |

### compatibility_matrix.json 구성

legacy 키(`matrix`, `bin_edges`, `n_clusters`, `n_context_bins`, `context_features`)에 **additive**로 4개 symmetric 키가 추가된다(기존 키 보존):

| 키 | 내용 |
|----|------|
| `matrix` | (legacy) `P(context_cell | cluster)` — 이미지-평균 맥락 기반 |
| `clean_dist` | `{cell: prob}` — **good 이미지 전 패치**에 대한 셀 분포(patch-granularity, 이미지-평균 아님). 합=1 |
| `P_def_patch` | `{cluster: {cell: prob}}` — 클러스터별 defect 이미지 전 패치의 셀 분포(각 패치는 image_id로 클러스터 상속). 클러스터별 합=1 |
| `matrix_symmetric` | `{cluster: {cell: compat_sym}}` — clean-grounded SGM. support `S_k={cell: P_def_patch[k][cell]>0}` 위에서 `compat_sym(k,c)=sqrt((P_def_patch[k][c]+ε)·(clean_dist[c]+ε))`, 각 row를 per-cluster max로 나눠 peak=1.0 정규화 |
| `symmetric_epsilon` | SGM에 쓴 ε (`1e-3`) |

### symmetric 키 hard assert 검증

`--compat_mode symmetric`를 쓰려면 위 4키가 반드시 있어야 한다. 없으면 코드 버전 미반영(>=`6c8658f` 필요) 또는 재실행 실패 → symmetric 모드 hard-fail.

```python
c = json.load(open(f"{S('profiling', DS)}/compatibility_matrix.json"))
assert 'matrix_symmetric' in c and 'P_def_patch' in c \
    and 'clean_dist' in c and 'symmetric_epsilon' in c, \
    f"[{DS}] symmetric 키 누락 — --compat_mode symmetric 사용 불가. 코드 버전(>=6c8658f) 확인 후 재실행."
```

추가 검증: 비어있지 않은 cluster row max=1.0(max-norm), `clean_dist` 합=1, `P_def_patch` 클러스터별 합=1, `context_features.csv`의 `image_w`/`image_h` 컬럼 존재(step3.5 정밀 배치 전제, >=`b1bb497`), `image_id` 클래스-고유키(`{defect_type}_{stem}`, good=`_{stem}`, >=`31ee0aa`). 재실행 전 기존 profiling 백업(`*_bak_pre_symmetric`) 후 drift 체크(legacy `matrix`/`bin_edges` 동일 여부).

---

## 핵심 계산 로직

`distribution_profiling.py`는 9-step 파이프라인(`DistributionProfiler.run()`):

- **step1** — dataset_config에서 `image_dir`(good) / `seed_dirs`(defect) 경로 발견, 도메인별 `_find_mask_path`로 GT 마스크 매칭(없으면 SAM/Otsu fallback 카운트).
- **step2 → `morphology_features.csv`** — defect 인스턴스별 6개 형태 특징(`linearity, solidity, extent, aspect_ratio, eccentricity, circularity`) + `area`·`defect_bbox`·full-frame `defect_mask_path`. 마스크는 원본 해상도로 INTER_NEAREST 리사이즈해 bbox 좌표계 정합.
- **step3 → `context_features.csv`** — 64px 패치별 5개 맥락 특징(`local_variance, edge_density(Sobel), texture_entropy(LBP P8R1), frequency_energy(FFT HF비), orientation_consistency(gradient dir entropy)`) + 실제 `image_w`/`image_h`. defect 마스크 과반 패치는 제외.
- **step4 → `distribution_analysis.json`** — feature별 valley 검출(inverted-histogram, Sturges bins, prominence=max(2σ noise, ratio))로 정책 선택: 2+ valley→gmm(multimodal), 1 valley→otsu(bimodal), 0→percentile(unimodal).
- **step5 → `morphology_clusters.json`** — 6-feature 정규화 행렬에 GMM/BIC 클러스터링(sklearn, `random_state=42`; 부족 시 aspect_ratio percentile fallback). cluster centroid·`_auto_label`(linear_scratch/elongated/compact_blob/irregular/general)·`image_id→cluster` 할당.
- **step6 → `compatibility_matrix.json`** — P33/P66 bin edges로 맥락 셀 이산화 → legacy `matrix`(`P(cell|cluster)`) + `_build_symmetric`가 clean_dist·P_def_patch·matrix_symmetric·symmetric_epsilon 4키 추가.
- **step7 → `deficit_analysis.json`** — 클러스터별 `deficit(cell)=max(0, P(cell|good) - P(cell|cluster))` + 정규화 `target_synthetic` + `prior`.
- **step8 → `threshold_policies.json`** — feature별 정책·경계·percentile 집약.
- **step9 → `recommended_config.yaml` + `analysis_report.md` + `figures/*.png`** — Stage1b/3/6가 소비하는 config(morphology 클러스터, context bin_edges, compatibility/deficit 경로), 사람용 리포트, 히스토그램·호환성 heatmap·deficit bar 그림.

## 주의사항

- **경로 규약**: phase0 출력은 반드시 stage-first `S('profiling', ds)`(=`sym_final/profiling/{ds}`). ds-first 금지(exp3/5/6 루트 규약 `root/{ds}`와 깨짐). `distribution_profiling.py`는 `$AROMA_REF/scripts/`(루트), `prepare_*.py`는 `$AROMA_SCRIPTS`(scripts/aroma).
- **drift**: 재실행은 GMM 클러스터·bin_edges를 **재계산**한다. 구 profiling(구 코드 산물)과 cluster 수/라벨이 다를 수 있으므로 step1 백업 후 drift 체크 필수. drift 시 defect-mode·symmetric 모두 신 profiling으로 통일(구/신 혼용 금지, image_id 문자열도 조인 키라 혼용 시 조인 붕괴).
- **버전 게이트**: symmetric 키(`>=6c8658f`), `image_w/image_h`(`>=b1bb497`), image_id 고유키(`>=31ee0aa`).
- **aitex**: tile-level·single-class → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효. leather는 cluster washout(TV~0.13) 경향 → symmetric이 물려도 clean-plausibility 필터로 degrade, 헤드라인 금지.
- **환경변수 형식**: Colab bash 셀에서 `$VAR`(중괄호 `${VAR}` 금지), 스크립트는 `!python` 접두사.
- **정직**: 사후 튜닝 금지(결과 보고 후 파라미터 변경 금지). 테스트 코드 신규 작성·pytest 금지(CLAUDE.md) — 검증은 Colab 셀 실행으로.

## 관련 노트

[[00-INDEX]] | [[03-Stage1-Complexity-Prompts]] | [[07-Scripts-Reference]] | [[08-Datasets]] | [[09-Compatibility-Gate]]
