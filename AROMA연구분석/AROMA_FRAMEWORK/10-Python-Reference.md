# 10 — Python 스크립트 코드 레퍼런스 (파일 내부 구조)

> **Claude 요약:** 07 노트가 스크립트의 **입출력·핵심 로직 요약**을 매핑한다면, 이 노트는 각 py 파일의 **내부 구조**(진입점·argparse·핵심 함수/클래스·제어 흐름·입출력 스키마 키·실제 상수/임계·gotcha)를 코드 근거로 기술한다. 각 step 스크립트가 `import`/join으로 참조하는 파일과 소비하는 JSON 키를 코드 레벨에서 추적할 때 이 문서를 기준으로 삼는다. 모든 항목은 실제 소스(argparse·함수 시그니처)에 근거하며, 커버 범위는 **코어 파이프라인 15개 파일**(prepare_*, profiling, complexity, prompts, roi_selection, clean_bg, build_train_jsonl, train_controlnet, generate_*, exp3/4v2/5/6)이다. plot_*·CASDA 어댑터·validate_pipeline·구버전 exp(1/2/4)는 범위 밖.

## 사용법

- **코드 수정 전**: 대상 파일의 섹션에서 핵심 함수·상수·gotcha를 먼저 확인한다.
- **step 스크립트 디버깅 시**: 소비/생성 JSON의 실제 키가 "입력 구조"/"출력 구조"에 있다. 스키마 불일치는 여기서 대조.
- **경로 규약**: `_SPEC.md` stage-first(`S(stage,ds)=$SYM_ROOT/{stage}/{ds}`). `distribution_profiling.py`·`train_controlnet.py`만 `scripts/`(루트), 나머지는 `scripts/aroma/`.

## 파일 → 섹션 인덱스

| Stage | 파일 | 위치 | CPU/GPU |
|-------|------|------|---------|
| step -1 | `prepare_severstal.py` | `scripts/aroma/` | CPU |
| step -1 | `prepare_aitex.py` | `scripts/aroma/` | CPU |
| step -1 | `prepare_mtd.py` | `scripts/aroma/` | CPU |
| phase0 | `distribution_profiling.py` | `scripts/` (루트) | CPU/SAM 폴백 GPU |
| step1 | `compute_complexity.py` | `scripts/aroma/` | CPU |
| step2 | `prompt_generation.py` | `scripts/aroma/` | CPU |
| step3 | `roi_selection.py` | `scripts/aroma/` | CPU |
| step3.5 | `clean_bg_selection.py` | `scripts/aroma/` | CPU |
| step4a | `build_train_jsonl.py` | `scripts/aroma/` | CPU |
| step4b | `train_controlnet.py` | `scripts/` (루트) | GPU |
| step5 | `generate_defects.py` | `scripts/aroma/` | GPU/CPU |
| step5 | `generate_random.py` | `scripts/aroma/` | CPU |
| exp3 | `exp3_generation_quality.py` | `scripts/aroma/experiments/` | CPU |
| exp4v2 | `exp4_v2_supervised_detection.py` | `scripts/aroma/experiments/` | GPU |
| exp5 | `exp5_prdc.py` | `scripts/aroma/experiments/` | GPU |
| exp6 | `exp6_embedding_coverage.py` | `scripts/aroma/experiments/` | GPU |

---

## step -1 — 데이터셋 준비 (prepare_*)

### prepare_severstal.py (`scripts/aroma/prepare_severstal.py`) — step -1 prepare [CPU]
**역할**: Severstal Steel Defect(Kaggle, RLE 마스크 in train.csv)를 MVTec식 AROMA 레이아웃(train/good, test/class{1..4}, masks)으로 물질화.
**진입점 / argparse**: `main()` → `_parse_args()` → `prepare()` 단일 호출 후 "Done." 출력. 필수 인자: `--train_csv`, `--train_images`, `--output_dir` (3개 모두 required, 옵션 없음).
**핵심 함수**:
| 함수 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `prepare` | `(train_csv, train_images, output_dir) -> Dict` | 전체 오케스트레이션(파싱→스테이징→마스크→manifest) |
| `_inline_rle_decode` | `(rle, shape=(256,1600)) -> np.uint8 (H,W)` | Kaggle RLE 인라인 디코더(1-indexed, order='F') |
| `_get_rle_decoder` | `() -> (callable, name)` | CASDA `src.utils.rle_utils.rle_decode` 시도, 실패 시 인라인 폴백 |
| `_parse_train_csv` | `(csv_path) -> {ImageId: [(ClassId, RLE)]}` | CSV를 ImageId별로 그룹핑, 빈/NaN 행 skip |
| `_list_image_files` | `(dir) -> {filename: abs_path}` | 디렉터리 직속 이미지 파일 목록 |
| `_link_or_copy` | `(src, dst) -> None` | symlink 우선, 실패 시 copy2, 존재 시 skip |
| `_write_mask_png` | `(mask, out_path) -> None` | {0,1}→0/255 그레이 PNG(cv2 우선, PIL 폴백) |

**제어 흐름**: `prepare()`가 → (1) RLE 디코더 선택 → (2) `_parse_train_csv`로 결함 그룹 + `_list_image_files`로 전체 파일 로드 → (3) 출력 골격 생성(train/good, masks, test/class1~4, masks/class1~4) → (4) CSV에 없는 실제 파일 = 정상 → train/good 링크 → (5) 결함 ImageId별로 primary_class(최소 ClassId) 서브폴더에 링크, merged 마스크(OR) + per-class 마스크 작성 → (6) `severstal_manifest.json` 기록.
**입력 구조**: `train.csv`(`ImageId`,`ClassId`,`EncodedPixels`, 헤더 소문자 정규화 후 매칭). RLE는 1-indexed, column-major(order='F'), (256,1600). 정상 이미지 = CSV 미등장 파일(집합 차집합).
**출력 구조**: `train/good/`, `test/class{1..4}/`, `masks/{stem}.png`(merged), `masks/class{c}/{stem}.png`(per-class), `severstal_manifest.json`. manifest 키: `dataset`, `image_shape{height,width}`, `rle_decoder`, `n_classes`, `counts{normal, defect_images, per_primary_class, missing_files}`, `layout`, `defects[]`(`image_id`,`primary_class`,`classes`,`merged_mask`,`per_class_masks`).
**상수·임계**: `SEVERSTAL_H=256`, `SEVERSTAL_W=1600`, `N_CLASSES=4`. ClassId 범위 [1,4] 밖 skip. `_IMG_EXTS`=jpg/jpeg/png/bmp/tiff/tif.
**주의점**: primary_class=존재 클래스 중 최소 ClassId(결정론적 test/ 배치)이나 per-class 마스크는 전 클래스 보존. CSV엔 있으나 파일 없는 ImageId는 `missing_files`로 카운트 후 skip. 정상 판정은 "CSV 미등장 실제 파일". RLE 디코더는 CASDA import 여부에 따라 달라지며 manifest `rle_decoder`에 기록.

### prepare_aitex.py (`scripts/aroma/prepare_aitex.py`) — step -1 prepare [CPU]
**역할**: AITEX Fabric DB(4096×256 PNG, 마스크 별도 폴더)를 타일링(기본 256/stride128)해 MVTec식 단일클래스 레이아웃으로 변환, 다운스트림이 정사각 이미지를 무변경 소비.
**진입점 / argparse**: `main()` → `_parse_args()` → `prepare()`. 필수: `--defect_images`,`--mask_images`,`--nodefect_images`,`--output_dir`. 옵션: `--tile`(256, 0=레거시 비타일), `--stride`(128), `--min_tile_area`(50), `--multi_class`(플래그, 미지정 시 single_class=True).
**핵심 함수**:
| 함수 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `prepare` | `(defect_images, mask_images, nodefect_images, output_dir, tile=256, stride=128, min_tile_area=50, single_class=True) -> Dict` | 전체 오케스트레이션 |
| `_mask_nonzero` | `(mask_path) -> Optional[bool]` | 마스크 nonzero 여부(라이브러리 없으면 None=keep) |
| `_tile_coords` | `(length, tile, stride) -> List[int]` | 타일 top-left 오프셋, 우/하단 잔여 클램프 커버 |
| `_label_components` | `(binary) -> (labels, n)` | 8-연결 컴포넌트 라벨링(cv2/scipy) |
| `_mask_tile_bboxes` | `(mask_tile, min_area, open_borders) -> (bboxes, cleaned_mask, used_union)` | 타일 마스크 분류(대형 유지/경계 파편 소거/소형 내부 union 폴백) |
| `_parse_defect_code` | `(stem) -> Optional[str]` | 파일명 `nnnn_ddd_ff`에서 2번째 토큰(ddd) |
| `_list_pngs` | `(dir, recursive=False) -> List[Path]` | 정렬 PNG(NODefect는 recursive) |
| `_read_image`/`_read_mask_arr`/`_write_image` | I/O | cv2 우선, PIL 폴백 |

**제어 흐름**: `prepare()`가 → (1) defect/nodefect(recursive)/mask PNG 목록 로드 → (2) 이전 manifest 읽어 tiled↔non-tiled 혼합/파라미터 변경 경고 → (3) tiled면 `_require_imaging()`+stride 검증 후 `__tile_` 파일 purge → (4) 마스크 인덱스(stem sans `_mask`) 구축 → (5) 정상 이미지 타일링→train/good → (6) 결함별 코드 파싱→마스크 매칭(없음/빈마스크/판독불가 skip)→타일 분류(빈 window=배경 train/good, bbox 있음=test/defect+ground_truth, 파편만=discard) → (7) `aitex_manifest.json`.
**입력 구조**: `Defect_images/nnnn_ddd_ff.png`(flat), `Mask_images/nnnn_ddd_ff_mask.png`(flat, `_mask` 접미사 키), `NODefect_images/`(fabric별 서브폴더 recursive). 원본 4096×256.
**출력 구조**: `train/good/{stem}__tile_r{r}_c{c}.png`, `test/defect/{...}.png`, `ground_truth/defect/{...}_mask.png`(0/255, cleaned_mask), `aitex_manifest.json`. manifest 키: `dataset`,`tiled`,`tile{size,stride,min_tile_area}`,`single_class`,`classes`,`image_shape_original`,`image_shape`,`source_layout`,`counts{normal_tiles, bg_tiles_from_defect_images, defect_tiles, union_fallback_tiles, discarded_border_fragment_tiles, defect_source_images_matched, ...}`,`defect_codes`,`layout`,`defects[]`(`image_id`,`origin`,`defect_code`,`x0`,`y0`,`n_bbox`,`union_fallback`,`image`,`mask`).
**상수·임계**: `tile=256`,`stride=128`(50% overlap),`min_tile_area=50`(exp4v2 `_mask_to_bboxes` min_area와 일치 요구). `_TILE_DELIM="__tile_"`,`_MASK_SUFFIX="_mask"`. stride는 (0,tile] 강제.
**주의점**: 매칭 마스크 없음/all-zero/판독불가는 각각 카운트 후 skip(Otsu 날조 방지). min_area 미만 컴포넌트는 open border면 소거, 내부 소형 결함이면 union bbox 폴백(소스 소실 방지). `__tile_` delimiter는 exp4v2 group-aware split이 같은 원본 타일을 동일 split에 묶는 계약. 타일 파일은 실제 crop(symlink 아님). 경고 print는 cp949 콘솔 대응 ASCII 전용.

### prepare_mtd.py (`scripts/aroma/prepare_mtd.py`) — step -1 prepare [CPU]
**역할**: MTD(Magnetic-Tile-Defect, Supervisely, base64+zlib bitmap 주석)를 MVTec식 레이아웃(train/good, test/{class}, ground_truth full-frame 마스크)으로 정규화.
**진입점 / argparse**: `main()` → `_parse_args()` → `prepare()`. 필수: `--supervisely_root`,`--output_dir`.
**핵심 함수**:
| 함수 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `prepare` | `(supervisely_root, output_dir) -> Dict` | 전체 오케스트레이션 |
| `_decode_bitmap` | `(data_b64, origin, size) -> Optional[np.ndarray]` | base64→zlib→PNG 디코드, origin 오프셋으로 full-frame 0/255 래스터화 |
| `_link_or_copy` | `(src, dst) -> None` | symlink 우선, copy2 폴백 |

**제어 흐름**: `prepare()`가 → (1) cv2+numpy 확인(없으면 RuntimeError) → (2) `ds/ann`,`ds/img` 검증 → (3) `ds/ann/*.json` 정렬 순회 → (4) `objects==[]`면 good→train/good → (5) 결함이면 classTitle별 union mask(`_decode_bitmap`+`np.maximum`) → (6) 클래스별 test/{cls} 링크 + full-frame 마스크 → (7) `mtd_manifest.json`.
**입력 구조**: `<root>/meta.json`(5클래스, 미직접 파싱), `ds/img/<name>.jpg`, `ds/ann/<name>.jpg.json`(`{size, objects:[{classTitle, bitmap:{data,origin}}], tags}`). bitmap alpha>0=결함.
**출력 구조**: `train/good/`, `test/<class>/`, `ground_truth/<class>/<stem>_mask.png`(full-frame 0/255 class union), `mtd_manifest.json`. 키: `dataset`,`source_format`,`counts{good, defect_entries, per_class, skipped_no_img, skipped_empty_mask}`,`classes`,`layout`,`defects[]`(`image_id`,`defect_type`,`image`,`mask`).
**상수·임계**: 매직 타일값 없음. 경계 클램프 `y0/x0=max(0,origin)`, `y1/x1=min(H/W, origin+patch)`, 무효 영역→None.
**주의점**: 다중결함 이미지는 존재 classTitle마다 (이미지, class-union mask) 엔트리(mvtec per-image-per-class 관례). 전 object 0px인 빈 mask는 skip+count. ann 파일명 `<name>.jpg.json`에서 `[:-5]`로 실제 이미지명 복원. cv2+numpy 없으면 즉시 RuntimeError(폴백 없음). 파일명 정렬 결정론적.

---

## phase0 — 분포 프로파일링

### distribution_profiling.py (`scripts/distribution_profiling.py`) — phase0 profiling [CPU / SAM 폴백 시 GPU]
**역할**: `dataset_config.json`의 한 데이터셋 엔트리에서 결함 morphology와 배경 context 분포를 프로파일링하여, 데이터 기반 임계·클러스터·compatibility 점수·합성 우선순위(deficit)를 유도하는 AROMA Stage 0 오케스트레이터.
**진입점 / argparse**: `main(argv)` → `_parse_args` → `DistributionProfiler(...).run()`(step1~9 순차). 인자: `--dataset_config`(기본 `/content/AROMA/dataset_config.json`), `--dataset_key`(required), `--output_dir`(required), `--num_workers`(-1=auto cpu_count-1, 0=순차), `--max_images`(디버그), `--n_gmm_components`(5, BIC k∈[1,N] 자동), `--sam_checkpoint`(폴백 마스크용, 옵션). `main`은 key 부재 시 available keys 출력 후 exit 1.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `DistributionProfiler` | `(entry, dataset_key, output_dir, num_workers, max_images, n_gmm_max, checkpoint)` | 9-step 오케스트레이터 |
| `.step1_load_and_discover` | `()` | seed/good 이미지·마스크 탐색, morph/context task 구성 |
| `.step2_morphology_features` | `()` | `_morph_worker` 병렬 → morphology_features.csv |
| `.step3_context_features` | `()` | `_context_worker` 병렬 → context_features.csv(스트리밍) |
| `.step4_distribution_analysis` | `()` | feature별 valley 검출 → gmm/otsu/percentile policy |
| `.step5_morphology_clustering` | `()` | 6D morph 벡터 GMM/BIC 클러스터링 |
| `.step6_compatibility_learning` | `()` | legacy `matrix` + `_build_symmetric` SGM |
| `.step7_deficit_analysis` | `()` | good 분포 대비 cluster deficit + target_synthetic |
| `.step8_threshold_policies` | `()` | step4를 threshold_policies.json으로 재패키징 |
| `.step9_reporting` | `()` | yaml/report/figures |
| `_morph_worker` | `(task) -> Optional[dict]` | 이미지 1장 morphology(GT or SAM/Otsu 폴백, full-frame 마스크 저장) |
| `_context_worker` | `(task) -> List[dict]` | non-defect 64px 패치별 context feature |
| `_extract_context_features` | `(patch) -> Dict` | 5 descriptor(variance, Sobel edge, LBP entropy, FFT HF, gradient-orient entropy) |
| `_build_symmetric` | `(context_rows, cluster_assignments, bin_edges, n_clusters, epsilon=1e-3) -> dict` | **compat 핵심**: patch-granularity SGM 대칭 compat(순수 함수) |
| `_detect_valleys` | `(values, feature_name)` | 반전 히스토그램 peak로 multimodality |
| `_fit_gmm_bic` | `(X, max_k)` | BIC 최소 k(random_state=42, n_init=3) |
| `_context_cell_key` | `(feat_dict, bin_edges) -> str` | 5 feature를 `'0_1_2_0_1'` 셀 키로 이산화 |
| `_compute_bin_edges` | `(rows) -> Dict` | context feature별 P33.33/P66.67 경계 |
| `_find_mask_path` | `(domain, image_path, defect_type)` | 도메인별 GT 마스크 경로 해석 |
| `_resolve_seed_entries`/`_seed_defect_type` | seed_dir(str)/seed_dirs(list) 정규화, MTD 클래스 충돌 회피 |
| `_auto_label` | `(centroid) -> str` | centroid 휴리스틱 라벨 |

**제어 흐름 (9-step)**: (1) config 로드 + good/seed 경로 탐색, morph_tasks·context_tasks 구성 → (2) morphology feature 병렬 → CSV → (3) 64px context patch 병렬 → CSV(스트리밍) → (4) morph feature별 분포 분석(valley→policy) → (5) 6D morph GMM/BIC 클러스터링 → (6) compat 학습(legacy matrix + 대칭 SGM) → (7) deficit(target_synthetic) → (8) threshold policy 직렬화 → (9) recommended_config.yaml + analysis_report.md + figures.
**mask 탐색**: `_find_mask_path` 도메인 분기 — `mvtec`/`aitex`/`mtd`=`parent×3/ground_truth/{defect_type}/{stem}_mask.png`; `visa`=`{category}/ground_truth/anomaly/{stem}.{png,jpg}`; `severstal`=per-class `masks/{defect_type}/{stem}.png` 우선, 없으면 merged `masks/{stem}.png`; `isp`/기타=None. None이면 `_morph_worker`에서 `extract_seed_mask(img, checkpoint)`(stage1b) 폴백(SAM 또는 Otsu), `mask_source="fallback_{method}"` 기록. GT라도 `raw.max()==0`이면 폴백. 마스크 해상도 다르면 `INTER_NEAREST` 정렬. context_worker는 `defect_mask[patch].mean()>0.5`(majority defect) 패치 제외.
**compatibility_matrix.json 계산 (step6, 핵심)**: `_compute_bin_edges`로 5 feature의 P33/P66 경계 → 3-bin 이산화 taxonomy. **legacy `matrix`**: defect 이미지의 patch 평균 context(`defect_mean_ctx`, 결측 시 `global_means`)를 1 cell로 이산화 → `counts[cluster][cell]` 공출현 → cluster별 정규화 P(cell|cluster)(이미지-평균 granularity). **`_build_symmetric` (SGM, patch granularity)**: 개별 64px 패치 단위 집계 —
- `clean_dist`: 모든 good 패치 셀 분포 {cell: prob}.
- `P_def_patch`: `{cluster_str:{cell:prob}}` — defect 패치를 그 이미지 cluster(`cluster_assignments[image_id]`)에 귀속.
- `matrix_symmetric`: cluster k의 support S_k={P_def_patch[k][c]>0}에서 기하평균 `compat_sym(k,c)=sqrt((P_def_patch[k][c]+ε)·(clean_dist[c]+ε))` → row별 max로 나눠 peak=1.0 정규화. (defect가 실제 존재한 컨텍스트 ∩ clean 배경이 흔한 컨텍스트를 동시 만족하는 셀을 높게 평가하는 대칭 점수.)
- `symmetric_epsilon`: ε(기본 1e-3).
legacy `matrix`는 유지, SGM 4키는 additive 병합(다운스트림 opt-in).
**출력 구조** (`--output_dir` 하위):
- `morphology_features.csv` — image_id, image_path, defect_type, domain, mask_source, linearity, solidity, extent, aspect_ratio, eccentricity, circularity, area, defect_bbox, defect_mask_path.
- `context_features.csv` — image_id, image_type, patch_xy, local_variance, edge_density, texture_entropy, frequency_energy, orientation_consistency, image_w, image_h.
- `distribution_analysis.json` — feature별 policy/distribution/n_valleys/valley_positions/boundaries/percentiles.
- `morphology_clusters.json` — n_clusters, method(gmm_bic/percentile_fallback), clusters[{cluster_id,n_samples,centroid,label}], cluster_assignments.
- `compatibility_matrix.json` — n_clusters, n_context_bins, context_features, bin_edges, `matrix`(legacy), `P_def_patch`, `clean_dist`, `matrix_symmetric`, `symmetric_epsilon`.
- `deficit_analysis.json` — cluster_str별 prior, deficit{cell: max(0, P(cell|good)−P(cell|cluster))}, target_synthetic.
- `threshold_policies.json`, `recommended_config.yaml`, `analysis_report.md`, `figures/`(morphology_histograms/compatibility_heatmap/deficit_bars), `defect_masks/{defect_type}_{stem}.png`.
**상수·임계**: `GRID_SIZE=64`, `N_CONTEXT_BINS=3`(P33/P66), `MAX_HISTOGRAM_BINS=50`, `VALLEY_PROMINENCE_RATIO=0.1`, `BOUNDED_VALLEY_PROMINENCE_RATIO=0.3`, `FLAT_CV_THRESHOLD=0.5`, `BOUNDED_FEATURES={circularity,eccentricity,extent}`. valley bin Sturges; bin_edges 33.33/66.67. GMM n_init=3, random_state=42, `n_gmm_max=5`. 클러스터링은 `HAS_SKLEARN and len>=4`일 때만 GMM, 아니면 aspect_ratio percentile fallback. LBP P=8,R=1 uniform 10-bin, FFT LF radius `min(h,w)//4`, gradient 18-bin. `_auto_label` 임계 lin>0.7&ar>5.0 등.
**주의점**: MTD는 이미지 확장자를 `.jpg/.jpeg`로 제한(마스크 오인 방지); `_seed_defect_type`도 leaf가 `Imgs`면 부모명 사용(충돌 회피). `_morph_worker`/`_context_worker`는 pickle-safe module-level → `mask_out_path`를 task로 주입. `image_w/h`는 context feature 아님(compat/complexity 벡터 order-index 불변). ISP는 GT 전무 → 전량 SAM/Otsu 폴백+경고. `defect_bbox`는 전체 foreground span이지만 eccentricity/circularity는 max-area region만. SGM `matrix_symmetric`는 legacy와 별개 키(heatmap은 legacy만 참조).

---

## step1~3 — 복잡도 · 프롬프트 · ROI

### compute_complexity.py (`scripts/aroma/compute_complexity.py`) — step1 [CPU]
**역할**: Phase 0 프로파일링 출력을 읽어 데이터셋 단위 MCI/CCI 스칼라 지표를 산출하고, 경험적 정책 평가로 형태·컨텍스트 모델링 정책을 선택.
**진입점 / argparse**: `main()` → `load_config()` → `run()`. 인자: `--profiling_dir`(필수), `--output_dir`(필수), `--config`(기본 `config/aroma_step1.yaml`), `--weight_mode`(equal/entropy_heavy/diversity_heavy), `--local_staging`, `--staging_root`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `load_phase0_outputs` | `(profiling_dir) -> dict` | Phase0 JSON/CSV 로드, morph_X·labels·context_X·status |
| `compute_mci` | `(phase0, cfg, weight_mode) -> (float, dict)` | MCI |
| `compute_cci` | `(phase0, cfg, weight_mode) -> (float, dict)` | CCI |
| `_fit_gmm_bic` | `(X, max_k, seed)` | 2단계 GMM+BIC(탐색 n_init=1 → best_k n_init=3 재적합) |
| `_cluster_context` | `(context_X, cfg, seed)` | 컨텍스트 GMM 클러스터링 |
| `_get_candidate_policies` | `(dist_analysis, mci, cfg)` | valley·MCI 기반 후보 정책 |
| `_apply_policy` | `(policy, X, max_k, seed)` | percentile/otsu/gmm/log_gmm/hierarchical 라벨링 |
| `select_best_policy` | `(candidates, X, cfg)` | silhouette 최고 + 안정성 tie-break |
| `run_meta_policy_generator` | `(phase0, mci, cfg)` | morphology·context 정책 통합 |

**제어 흐름**: `run()` — (1) local_staging 시 `_stage_to_local` → (2) `load_phase0_outputs`(status가 ok/empty_context 아니면 None report 조기 반환) → (3) compute_mci → (4) compute_cci → (5) run_meta_policy_generator(morphology 후보 pruning→평가, context 고정 `["gmm","percentile"]`) → (6) `complexity_report.json`.
**핵심 로직/공식**:
- **MCI** = 4 성분 가중평균: `norm(Entropy)`, `norm(ValleyCount)`, `norm(ClassDiversity)`, `inv_silhouette=clamp01(1−silhouette)`. Entropy=클러스터 라벨 Shannon 엔트로피. ClassDiversity=Hill `Neff=exp(H)`, 정규화 `clamp01(log(max(Neff,1))/log(n_max))`, `n_max=8`.
- **CCI** = `norm(texture_entropy 평균)`, `norm(cluster_count_ctx)`, `norm(freq_complexity=var(frequency_energy))`, `norm(orient_variance=var(orientation_consistency))` 가중평균.
- **정책 후보**: valley=0→`["percentile"]`; ≤1→`["otsu","percentile"]`; >1→`["gmm","otsu"]`. MCI≥0.6→`hierarchical` 추가, <0.3→hierarchical/log_gmm 제거, `max_candidates=3` 절단.
- **tie-break**: 1·2위 silhouette 차 `|Δ|<stability_margin(0.05)`면 `_stability_score`(n_bootstrap=5 seed 평균 silhouette).
**입력 구조**: `distribution_analysis.json`(n_valleys), `morphology_clusters.json`(n_clusters, cluster_assignments), `morphology_features.csv`(6 MORPH_FEATURES + image_id/defect_type), `context_features.csv`(5 CONTEXT_FEATURES, image_type good/normal 우선).
**출력 구조**: `complexity_report.json` — mci, cci, morphology_policy, context_policy, stability_margin, weight_mode, mci_components(raw/normalized/weights/silhouette_score), cci_components, candidate_policies{morphology,context}, evaluation_results, profiling_dir, status, provenance.
**상수·임계**: 가중 프리셋 equal=(0.25×4), entropy_heavy=(0.4,0.2,0.2,0.2), diversity_heavy=(0.2,0.2,0.4,0.2). expected_range: entropy[0,4], valley_count[0,41], texture_entropy[0,8], cluster_count_ctx[1,8], freq_complexity[0,1], orient_variance[0,1.6]. class_diversity_n_max=8, context_gmm max_k=5/max_patches=20000, stability_margin=0.05, high_complexity_mci=0.6, prune_low_mci=0.3, max_candidates=3, n_bootstrap=5, max_k=8. silhouette 샘플 상한 5000.
**주의점**: sklearn 없으면 silhouette=0.0·GMM→단일클러스터 폴백. `_cluster_context` 반환 labels는 서브샘플만 커버(n_clusters만 쓰는 호출자만 안전). CCI context_X는 별도 `default_rng(0)`로 리샘플. `_normalize_array`=zscore, `_normalize_scalar`=minmax(다른 정규화). AROMA_REF 미발견 시 inline I/O 폴백.

### prompt_generation.py (`scripts/aroma/prompt_generation.py`) — step2 [CPU]
**역할**: Phase 0 + Step 1 출력을 읽어 형태 클러스터 × 컨텍스트 bin 조합마다 고정 템플릿 기반 결함 프롬프트 생성(LLM 불필요).
**진입점 / argparse**: `main()` → `run()`. 인자: `--profiling_dir`,`--complexity_dir`,`--output_dir`(모두 필수). status가 ok/None 아니면 `sys.exit(1)`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `generate_morphology_descriptor` | `(centroid) -> str` | aspect_ratio·linearity·solidity 라벨 결합 |
| `generate_context_descriptor` | `(cell_key) -> str` | cell_key 파싱→상위 2 feature 라벨 |
| `generate_prior_modifier` | `(cluster_id, cell_key, matrix) -> str` | 조합 빈도 서술어 |
| `assemble_prompt` | `(morph, ctx, prior) -> str` | `"{morph}, {ctx}, {prior}"` |
| `load_inputs` | `(profiling_dir, complexity_dir)` | 입력 로드 + status |
| `generate_prompts` | `(data) -> dict` | 조합별 프롬프트 dict |
| `_build_summary` | `(prompts, complexity_report)` | 마크다운 테이블 |

**제어 흐름**: `run()` — (1) `load_inputs`(morphology_clusters·compatibility_matrix 없으면 missing_inputs) → (2) `generate_prompts`: cluster마다 centroid로 morph_desc, `matrix[cid]`의 cell_key마다 ctx_desc·prior_mod·prompt, cluster_row 없으면 `{cid}_none` 단일(중립 컨텍스트) → (3) `prompts.json`+`prompts_summary.md`.
**핵심 로직/공식**:
- **morphology 라벨**: ar>3.0 "highly elongated"/>1.5 "elongated"/else "compact"; lin>0.65 "linear"/>0.30 "irregular"/else "scattered"; sol>0.80 "solid"/>0.50 "semi-solid"/else "fragmented".
- **context**: cell_key(`'0_1_2_0_1'`) 파싱, bin 내림차순 상위 2 feature, `_CTX_LABELS[feat][bin]` + "surface background".
- **prior_modifier**: prob≥0.40 "predominantly"/≥0.20 "commonly"/else "rarely observed".
**입력 구조**: `morphology_clusters.json`(clusters, cluster_assignments), `compatibility_matrix.json`(`matrix`), `deficit_analysis.json`(선택), `complexity_report.json`(선택, 요약).
**출력 구조**: `prompts.json` — `"{cluster_id}_{cell_key}"` → {morphology_descriptor, context_descriptor, prior_modifier, prompt, cluster_id, phase0_label, cell_key, prior_prob, deficit, n_cluster_samples}. `prompts_summary.md`.
**상수·임계**: `_AR_HIGH=3.0`/`_AR_MED=1.5`, `_LIN_HIGH=0.65`/`_LIN_LOW=0.30`, `_SOL_HIGH=0.80`/`_SOL_MED=0.50`, `_PRIOR_HIGH=0.40`/`_PRIOR_MED=0.20`, `N_CONTEXT_BINS=3`, top_n=2. centroid 기본 ar=1.0/lin=0.5/sol=0.7.
**주의점**: cluster_row 없을 때만 `_none` 브랜치(deficit=0.0). cell_key 파싱 실패 시 `[0]*5`, feature 수 불일치 시 1 패딩. compat matrix 비어도 경고만. save_json은 `ensure_ascii=False`.

### roi_selection.py (`scripts/aroma/roi_selection.py`) — step3 [CPU]
**역할**: (결함 이미지 × 컨텍스트 bin) 후보를 스코어링하고, 품질 게이트 사전필터 후 4가지 샘플링 전략 중 하나로 ROI 선택.
**진입점 / argparse**: `main()` → `run()` → `build_candidates` → `apply_quality_gate` → `select_rois`. 인자: `--profiling_dir`,`--prompts_dir`,`--output_dir`(필수), `--sampling_strategy`(deficit_aware/compatibility/top_k/weighted/random), `--top_k`(200), `--seed`(42), `--class_mode`(informational), `--nc`(informational), `--class_floor`(flag), `--per_pair_cap_frac`(None), `--rarity_temp`(1.0), `--img_diversity_cap`(**기본 1**), `--min_quality`(0.0=OFF), `--background_type`(directional), `--score_mode`(legacy/realism).
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `quality_proxy` | `(lin,sol,ar,bg)` | subtype 분류 + matching_score(외부 utils, 없으면 (general,1.0)) |
| `score_roi` | `(mp,cp,def,qs,score_mode)` | legacy/realism ROI 스코어 |
| `build_candidates` | `(data,bg,score_mode)` | morph_row × 점유 컨텍스트 bin 교차 스코어링 |
| `apply_quality_gate` | `(cands,min_quality)` | quality_score<min_quality 드롭 |
| `moderated_score` | `(c,rarity_temp)` | 정렬 키(rarity_temp≠1이면 deficit항만 거듭제곱) |
| `_source_key`/`_img_jitter`/`_order_key` | (image_path,bbox) 신원 / 결정적 jitter / 결합 정렬 |
| `_pair_aware_allocation` | 2-phase (cluster,cell) pair 할당 |
| `_stratified_pair_aware` | 클래스 층화 wrapper |
| `_stratified_compat` | compatibility 전략 클래스 층화 |
| `select_rois` | 전략 디스패치 |

**제어 흐름**: `run()` — (1) `load_inputs`(morphology_features.csv·clusters·compat·deficit·prompts.json) → (2) score_mode=realism+rarity_temp≠1 경고 → (3) `build_candidates`(image_id→cluster, cluster_row 없으면 `{"none":0.0}`, quality_proxy 이미지당 1회) → (4) `apply_quality_gate` → (5) `select_rois` → (6) `_diversity_stats` 로깅 → (7) roi_candidates.json·roi_selected.json·roi_summary.md.
**핵심 로직/공식 — score_mode**:
- **legacy**(기본): `ROI_score = 0.4·P(M) + 0.4·P(C) + 0.2·Deficit`, quality 무시. [0,1] 클램프.
- **realism**: `ROI_score = 0.5·P(C) + 0.3·P(M) + 0.2·quality`, deficit 제거(가중 0, JSON 보존), quality를 hard gate→graded 항으로 승격.
**핵심 로직/공식 — 4개 샘플링 전략** (top_k는 `max(1,min(top_k,len))` 클램프):
- **top_k**: roi_score 내림차순 상위 K.
- **random**: `default_rng(seed)` 비복원 균등(Exp2 baseline). 멀티클래스 옵션 무시.
- **weighted**: roi_score 정규화 확률 비복원(합<1e-12면 앞 K).
- **compatibility**: `compat_score=0.6·ctx_prior+0.4·morph_prior`(deficit 없음). class_floor&K>1이면 `_stratified_compat`.
- **deficit_aware**(기본): class_floor&K>1 → `_stratified_pair_aware`, 아니면 `_pair_aware_allocation`.
**`_pair_aware_allocation` (2-phase)**: Phase1 — (cluster_id,cell_key) 그룹화, PairDeficit=mean(deficit), base quota=1(coverage), 잔여를 PairDeficit 비례 Hamilton 분배. Phase1b(cap 시) — eff_cap 초과분 회수→미포화 pair round-robin. pair 내부 `_order_key` 내림차순. Phase2 — 전역 moderated_score backfill. `top_k≤n_pairs`면 coverage-first(pair당 1). img_cap 미충족 시 relaxed_cap +1 bounded-repetition 폴백.
**멀티클래스(`_stratified_pair_aware`)**: class_floor=`top_k//K` 대칭 floor, 잔여 `top_k%K`만 클래스 deficit-mass 최대잉여 분배(floor 배수면 순수 균등, starvation-first). per_pair_cap_frac은 전역 `pair_cap=max(1,ceil(frac·top_k))` POST-CONCAT eviction→class-aware refill→Phase2. img_diversity_cap=`(image_path,defect_bbox)` 소스 crop 최대 선택 횟수(기본 1); cap 활성 시 `_img_jitter`(blake2b [0,1e-7)) tie-break 자동. rarity_temp=deficit 항만 `d**temp`(1.0이면 byte-identical).
**입력 구조**: `morphology_features.csv`(image_id, image_path, defect_bbox, defect_mask_path, linearity, solidity, aspect_ratio, defect_type), `morphology_clusters.json`, `compatibility_matrix.json`(matrix), `deficit_analysis.json`, `prompts.json`.
**출력 구조**: `roi_candidates.json`(전체) / `roi_selected.json`(선택) — 키: image_id, image_path, cluster_id, cell_key, class_key, defect_subtype, quality_score, roi_score, morph_prior, ctx_prior, deficit, prompt, morph_label, ctx_label, defect_bbox, defect_mask_path. `roi_summary.md`.
**상수·임계**: legacy 0.4/0.4/0.2(합=1 assert), realism 0.5/0.3/0.2, compat 0.6/0.4, jitter 1e-7, top_k=200/seed=42/img_diversity_cap=1/min_quality=0.0/score_mode=legacy. blake2b digest_size=8.
**주의점**: 후보 풀은 소스 이미지당 (cluster,cell) bin마다 1개 → 같은 (image_path,bbox) 공유·roi_score image-blind → 동점 붕괴로 img_diversity_cap+jitter 도입(기본 ON). random/top_k/weighted는 멀티클래스 옵션 무시. quality deps 없으면 no-op(score 1.0), morphology 결측 시 quality-unknown→(general,1.0) 통과. `_bootstrap_aroma_ref`가 `parents[2]`를 sys.path에. score_mode=realism+rarity_temp≠1이면 경고. min_quality>0인데 전량 탈락 시 roi_selected.json 빈 채 에러 로그.

---

## step3.5~4 — clean-bg · train.jsonl · ControlNet 학습

### clean_bg_selection.py (`scripts/aroma/clean_bg_selection.py`) — step3.5 [CPU]
**역할**: 선택된 각 ROI에 대해, 프로파일링 산출물(context_features/compatibility_matrix)만으로 "결함 배경과 histogram-intersection이 가장 가까운 clean(good) 배경"을 오프라인 랭킹·할당(픽셀 재스캔 없음).
**진입점 / argparse**: `main()` → `_parse_args()` → `run()`. 인자: `--profiling_dir`,`--roi_dir`,`--output_dir`(필수), `--emit_random_arm`, `--no_reject_clean_bg`, `--void_frac_max`, `--void_floor_pct`(기본 15.0), `--var_floor`/`--edge_floor`(절대 오버라이드), `--pool_k`, `--geometry_prior`, `--seed`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `load_inputs` | `(profiling_dir, roi_dir) -> dict` | 입력 로드, status(never raises) |
| `_patch_void` | `(row, var_floor, edge_floor) -> bool` | void 판정(lv≤floor AND ed≤floor) |
| `_derive_void_floors` | `(good_by_img, floor_pct=15.0)` | 관측 분포 p15 백분위 floor |
| `valid_bg_pool` | `(...)` | void_frac≤cut(기본 0.5 majority) good 필터, all-reject 시 full pool 폴백 |
| `_hist_intersection` | `(p, q) -> float` | 공유 셀 min 합 [0,1] |
| `_image_hist`/`_class_bg_hist` | 이미지/클래스별 non-void 셀 히스토그램 |
| `_size_ok`/`_scale_to_fit`/`_effective_wh` | bbox 사이즈-핏 하드게이트 + fit-rescale(0.95) |
| `build_and_rank` | `(...)` | 2-pass 스코어링/데이터-도출 가중치/랭킹/할당 |
| `random_arm` | `(selected, valid_ids, seed)` | 동일 ROI 균등랜덤 bg(대칭 컨트롤) |

**제어 흐름**: `run()` — (1) `load_inputs`(status≠ok 조기 종료) → (2) `_derive_void_floors` p15 후 CLI 오버라이드(`floor_source` 기록) → (3) `valid_bg_pool` → (4) `build_and_rank` → (5) 각 assignment에 void provenance(`valid_pool_reason`) 부착 → (6) `clean_bg_selected.json` 저장, `--emit_random_arm`이면 `clean_bg_random_arm.json` → (7) `clean_bg_summary.md`.
**핵심 로직**: void 검출은 오프라인이라 픽셀이 아닌 `local_variance`/`edge_density` 컬럼 기반(`_patch_void`: 둘 다 floor 이하). floor는 관측 분포 p15. `build_and_rank`는 `src_fit`(ROI 소스이미지 배경 hist∩)+`class_fit`(클래스 집계 배경 hist∩)+size-fit를, 각 신호의 측정 판별 lift(best−median 평균) 가중 결합→평탄 신호(severstal/mtd)는 ~0, 판별력(aitex) 지배. tot≤0이면 per-source 폴백. blake2b 결정론적 tie-break. pool은 `--pool_k` 또는 p95 컷.
**입력 구조**: `context_features.csv`(image_type good/defect, local_variance, edge_density, patch_xy, image_w/h, 5 context), `compatibility_matrix.json`(context_features, bin_edges), `morphology_features.csv`(image_id, defect_type, defect_bbox), `{roi_dir}/roi_selected.json`(class_key, image_id, defect_bbox, cluster_id).
**출력 구조**: `clean_bg_selected.json`(ROI별: `assigned_normal_id`, `topk_pool`, `topk_positions`/`position`, `score`, `hist_intersection`, `class_fit`, `size_ok`, `size_fit`/`scale_factor`, `n_valid_bg`, `valid_pool_reason`), `clean_bg_random_arm.json`, `clean_bg_summary.md`. 전체 (roi×good) O(수백만)이라 미저장.
**상수·임계**: void floor = 관측 lv/ed의 p15(기본). 도크스트링의 "밝기<25 AND std<10"은 온라인 generate_defects의 픽셀 기준 void 정의이며, 이 오프라인 모듈은 그 아날로그로 lv/ed floor 사용. `void_frac_max` 기본 0.5(majority-void, 과거 상대 p90에서 수정). `_N_CONTEXT_BINS=3`, `_FIT_MARGIN=0.95`(generate_defects lockstep), `_EDGE_MARGIN_FRAC=0.08`, `_SPAN_FRAC=0.80`(geometry prior). pool 컷 p95.
**주의점**: void floor는 최근 p1→p15 상향 수정 — severstal 다크-void 클러스터(lv~0.21/ed~0.98)가 0 위에 몰려 p1 floor는 검은 plate를 놓쳤음(→48% black 합성). `--var_floor`/`--edge_floor`는 도출값 대체 prescan 안전밸브. `src_match_frac<0.5`면 stale roi vs 재프로파일 image_id 불일치 LOUD 경고. `random_arm`은 bg identity만 다른 대칭 컨트롤. `--geometry_prior` 기본 OFF. 정직성: hist 매칭은 도메인-조건부(aitex만 강함).

### build_train_jsonl.py (`scripts/aroma/build_train_jsonl.py`) — step4a [CPU]
**역할**: AROMA 내부에서 실 결함마다 target crop·3채널 hint·prompt를 만들어 `train_controlnet.py`가 그대로 소비하는 `train.jsonl`+`targets/`+`hints/`를 생성(CASDA 경유 대체 경로).
**진입점 / argparse**: `main()` → `_parse_args()` → `build()`. 필수: `--morphology_csv`,`--roi_candidates`,`--context_features`,`--config`(dest=`config_yaml`, recommended_config.yaml),`--output_dir`. 선택: `--image_root`, `--style`(기본 technical).
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `_load_bin_edges` | `(config_yaml)` | config context.features/bin_edges |
| `_bin_value` | `(value, edges) -> int` | 값→오디널 bin(0/1/2) |
| `_load_context_means` | `(context_csv, features)` | image_id 버킷 defect 패치 평균 |
| `_warn_context_collision` | `(pool, context_means)` | image_id당 >1 physical image 충돌 WARN |
| `_derive_background_type` | `(image_id, means, features, bin_edges)` | bin 기반 background_type 정책 |
| `_load_candidate_join` | `(roi_candidates)` | ctx_prior MAX·morph_label·cluster_id 집계 |
| `_load_morphology_pool` | `(morphology_csv)` | 결함 풀(1행=1결함) 파일순 |
| `_metrics_from_row` | `(row)` | hint/prompt용 형태 메트릭 |
| `build` | `(...) -> int` | per-결함 루프, 작성 줄 수 |

**제어 흐름**: `build()` — (1) numpy+cv2/PIL 확인 → (2) `_load_bin_edges`/`_load_candidate_join`/`_load_context_means`/`_load_morphology_pool` → (3) `_warn_context_collision` → (4) targets/hints 생성, `HintImageGenerator`/`PromptGenerator` 준비, negative_prompt 1회 → (5) pool 각 행 루프 → (6) background_type 히스토그램 로깅, 단일 카테고리 붕괴 WARN, n_written==0이면 RuntimeError.
**핵심 로직(per-defect)**: mask_path/bbox 검증 → `(image_id, defect_mask_path)`로 roi_candidates join(morph_label→defect_subtype, ctx_prior MAX→stability_score, 없으면 0.5) → `_derive_background_type`(이미지 defect 패치 context bin) → `_make_crop_pair`로 target crop → 동일 bbox crop 배열로 `HintImageGenerator.generate_hint_image` → `PromptGenerator.generate_prompt`(technical=결정론)+negative → 절대경로로 `train.jsonl` append.
**입력 구조**: `morphology_features.csv`(image_id, image_path, defect_mask_path, defect_bbox, 형태 6열), `roi_candidates.json`(image_id, defect_mask_path, ctx_prior, morph_label, cluster_id), `context_features.csv`(image_type=defect), `recommended_config.yaml`(context.features, context.bin_edges).
**출력 구조**: `targets/`(crop PNG+mask), `hints/`(`{base}_hint.png`), `train.jsonl`. 줄: `{"target","hint","prompt","negative_prompt"(레거시),"source"(==target,레거시)}`. train은 target/hint/prompt만 읽음.
**상수·임계**: `_METRIC_KEYS`=(linearity, solidity, extent, aspect_ratio, eccentricity, circularity, area). background_type: orient bin==2→directional; elif texture==2 or local_var==2→complex_pattern; elif edge==0 AND local_var==0→smooth; else→complex_pattern. bin edge는 오직 config에서. stability 기본 0.5.
**주의점**: `context_features.csv`는 bare image_id 키 — 여러 defect_type이 공유하면(mvtec_cable) background_type이 image_id-버킷 평균, `_warn_context_collision`이 정량 WARN. 단일 카테고리 붕괴 시 WARN. crop/hint/prompt 알고리즘은 `aroma_to_casda_roi`·`utils/hint_generator`·`utils/prompt_generator` 재사용. 스킵 사유별 카운터, 전부 스킵 시 RuntimeError.

### train_controlnet.py (`scripts/train_controlnet.py`) — step4b [GPU]
**역할**: diffusers로 SD1.5 위 ControlNet fine-tune — [hint→결함 이미지] 매핑 학습(UNet/VAE/TextEncoder frozen, ControlNet만 학습).
**진입점 / argparse**: `main()` → `parse_args()` → `train(args)`. 인자: `--data_dir`(필수), `--pretrained_model_name_or_path`(기본 runwayml/stable-diffusion-v1-5), `--controlnet_model_name_or_path`(기본 lllyasviel/sd-controlnet-canny; 빈값이면 UNet init), `--resolution`(512), `--mixed_precision`(no/fp16/bf16), `--gradient_checkpointing`, `--resume_from_checkpoint`(경로/`latest`), `--force_grayscale_target`(기본 True), `--augment`, `--save_fp16`/`--skip_save_pipeline`/`--save_optimizer_state`/`--skip_save_final`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `DefectControlNetDataset` | `Dataset(jsonl_path, image_root, resolution, tokenizer, force_grayscale_target, augment)` | jsonl→target/hint/prompt 텐서 |
| `collate_fn` | `(examples)` | pixel/conditioning/input_ids 스택 |
| `run_sanity_check` | `(dataset, tokenizer, device, num_samples=2)` | shape/NaN/Inf/range 검증 |
| `run_training_sanity_check` | `(controlnet, vae,...) -> bool` | 1-step forward/backward NaN |
| `log_validation` | `(...)` | 검증 이미지 생성(UniPC, 20 step, cond 0.8) |
| `train` | `(args)` | 메인 학습 루프 |
| `save_checkpoint`/`save_full_pipeline`/`_save_controlnet_fp16` | 체크포인트/파이프라인/fp16 |
| `_compute_snr` | `(noise_scheduler, timesteps)` | Min-SNR 가중치 |

**제어 흐름**: `train()` — (1) weight_dtype → (2) 모델 로드(DDPM noise_scheduler, text_encoder/vae/unet frozen; ControlNet은 항상 fp32 로드/`.train()`) → (3) dataset + `run_sanity_check` → (4) DataLoader → (5) AdamW+`get_scheduler`(warmup auto-clamp)+GradScaler → (6) resume 시 controlnet/optimizer/lr_scheduler/global_step 복원 → (7) (비-resume) `run_training_sanity_check` → (8) epoch×batch 루프 → (9) final_model/training_log.json/pipeline 저장.
**핵심 로직**: VAE encode→scaling_factor, add_noise, text encode는 autocast 내; **ControlNet forward는 autocast 밖 fp32**(fp16 NaN gradient 방지); UNet forward는 residual에 `controlnet_conditioning_scale`(학습 1.0) 곱해 주입. Loss=per-sample MSE, prediction_type(epsilon/v_prediction) 분기, `snr_gamma>0`이면 Min-SNR-gamma, 선택 gray_loss(기본 0). grad_accum 경계 unscale→clip→step. NaN/Inf loss·gradient 스킵+카운트, 3회마다 LR 절반, `max_nan_tolerance=10` 초과 abort. 메모리: mixed_precision, gradient_checkpointing, `--save_fp16`(가중치 50%↓), `--skip_save_pipeline`(~4.2GB↓), `--save_optimizer_state`(~1.4GB), `--skip_save_final`. Early stopping: `early_stopping_patience`(20). Best model=avg_epoch_loss<best_loss일 때 `best_model/`. Resume: `latest`→최대 checkpoint-N, resume_step만큼 batch skip.
**입력 구조**: `{data_dir}/train.jsonl`(target/hint/prompt; negative_prompt/source 레거시 미사용). 경로 해석 절대→image_root→data_dir→project_root 폴백.
**출력 구조**: `best_model/`, `final_model/`, `checkpoint-{step}/`(controlnet/ + 옵션 optimizer.pt/lr_scheduler.pt + global_step.txt + training_log.json), `pipeline/` 또는 `pipeline_reference.json`, `validation/step_{step}/`, `training_log.json`, `training_config.json`.
**상수·임계 (parse_args 기본값)**: resolution 512, train_batch_size 1, num_train_epochs 100, gradient_accumulation_steps 4, mixed_precision "no", controlnet_conditioning_scale 1.0(v5), snr_gamma 5.0, early_stopping_patience 20, gray_loss_lambda 0.0(v5 비활성), learning_rate 1e-5, adam 0.9/0.999, weight_decay 1e-2, eps 1e-8, max_grad_norm 1.0, lr_scheduler "cosine", lr_warmup_steps 50(auto max(10, steps//20) clamp), logging_steps 10, checkpointing_steps 500, checkpoints_total_limit 3, validation_steps 200, dataloader_num_workers 0, seed 42. NaN: max_nan_tolerance 10, 3회마다 LR×0.5. 검증: UniPC 20 steps, cond 0.8, seed 42.
**주의점**: hint 채널 R=defect shape/mask, G=structure, B=texture(실제 생성은 `HintImageGenerator`, 이 파일은 3채널 RGB 로드만). `force_grayscale_target` 기본 True — target L 변환 후 RGB 3채널 복제로 VAE latent 채널 분기(컬러 아티팩트) 방지, `--no_force_grayscale_target`(컬러 leather)로 해제. gray_loss는 latent 4채널이 RGB 대응 안 하는 잘못된 가정이라 v5부터 비활성. ControlNet 로드·학습 모두 fp32 강제. random control arm 개념 없음(여기 "random"=seed·noise·flip). augment flip은 공통, brightness/contrast는 target에만. 도크스트링이 "Steel Defect"로 남아 있으나 데이터 도메인 무관.

---

## step5 — 결함 생성

### generate_defects.py (`scripts/aroma/generate_defects.py`) — step5 [GPU controlnet / CPU copy_paste]
**역할**: Step 3에서 선정된 ROI를 정상(good) 배경 위에 합성해 결함 이미지 + `annotations.json`을 생성하는 합성 엔진. copy_paste·controlnet·inpainting(스텁) 3방법을 공유 배치/블렌드 파이프라인으로 실행.
**진입점 / argparse**: `main()` → `_parse_args()` → `run()`. 핵심 CLI —
- 기본: `--roi_dir`,`--normal_dir`,`--output_dir`,`--method{copy_paste,controlnet,inpainting}`,`--n_per_roi(3)`,`--seed(42)`,`--local_staging`.
- 블렌드: `--blend_mode{alpha,seamless}`,`--feather_px(4)`.
- clean-bg 게이트: `--reject-clean-bg`,`--random-placement`(naive),`--min-bg-quality(0.7)`,`--bg-blur-threshold(100.0)`.
- 텍스처/호환: `--texture-dist-threshold(0.0=OFF)`,`--compat_threshold(0.0=OFF)`,`--compat_matrix_json`,`--compat_mode{symmetric,defect}`(기본 defect),`--clean_bg_json`.
- ControlNet: `--controlnet_path`,`--sd_base`,`--cn_steps(30)`,`--cn_cond_scale(0.7)`,`--cn_resolution(512)`,`--cn_no_grayscale`,`--cn_default_bg(complex_pattern)`,`--cn_no_cache`,`--cn_ar_threshold(2.5)`,`--cn_no_ar_fallback`,`--morphology_csv`,`--context_features`,`--config`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `run(...)` | `(roi_dir,normal_dir,output_dir,method,...)->dict` | 전체 오케스트레이션 |
| `copy_paste_synthesis` | `(roi_entry,normal_path,out,...)->meta` | 실 GT bbox+mask 크롭(없으면 타원 fallback), `_paste_and_finalize` 위임 |
| `controlnet_synthesis` | `(roi_entry,normal_path,out,**kw)->meta` | hint+prompt→diffusion→AROMA ROI paste(GT mask=실 seed) |
| `inpainting_synthesis` | `(...)` | 스텁(NotImplementedError) |
| `_paste_and_finalize` | `(normal,crop,mask,out,...)->meta` | fit-rescale→배치→블렌드→풀프레임 GT mask 저장→bbox |
| `_alpha_composite` | `(bg,crop,mask,pos,feather)` | feather(scipy gaussian)+paste |
| `_context_aware_composite` | `(bg,crop,mask,pos,feather)` | Reinhard 전이 + `cv2.seamlessClone(NORMAL_CLONE)` |
| `_reinhard_transfer` | `(src,ref)` | Lab 채널별 mean/std 정합(회색 ref는 L만) |
| `_positive_place` | `(...)->(pos,best_mean,n_nonvoid)` | symmetric 게이트 scan-rank-place |
| `_foreground_mask` | `(normal)` | Otsu+코너투표 전경 추정 |
| `_foreground_paste_position` | `(...)` | 결함 centroid가 전경에 떨어지는 위치 |
| `_is_clean_background` | `(...)->bool` | quality_score≥min_quality면 유효(void 아님) |
| `_background_quality_score` | `(gray,blur_thr)->float` | 0.30 blur+0.30 contrast+0.20 brightness+0.20 noise |
| `_texture_descriptor` | `(patch,mask)` | [contrast,lap-var,periodicity,orient-aniso] |
| `_dv_bg_hist/_cell_hist/_hist_intersection` | 이미지-레벨 배경 분포 히스토그램 유사도 |
| `_cn_generate` | `(ctx,prompt,hint,seed0,...)` | diffusion 호출, blank/OOM 시 seed-shift 재시도 |
| `_configure_controlnet_context` | `(...)` | join 테이블+generator 1회, fingerprint |

**제어 흐름 (`run`)**: (1) `roi_selected.json`(없으면 missing_roi, 빈 경우 empty_roi) → (2) `load_normal_images`(reject_clean_bg 시 이미지별 filter, 전부 거부면 원본 유지) → (3) controlnet이면 인자 검증 후 `_configure_controlnet_context` 1회 → (4) local_staging 복사 → (5) `compat_threshold>0`이면 matrix 로드(symmetric=`matrix_symmetric` 필수·부재시 hard-fail, defect=`matrix`)+bin_edges → (6) `clean_bg_json`(기본 `<roi_dir>/clean_bg_selected.json`) → roi_idx→pool/positions → (7) ROI×rep 이중 루프: 배경 선택(clean_bg 사전풀 → symmetric top-K → 균일 `rng.choice`)→`synthesis_fn`→gate_stats 집계, annotation append → (8) staging push, `annotations.json`, gate/CN 로그.
**호환성 게이트**: `compat_matrix[cluster_id]` 행을 `compat_row`로(`.get(cell,0.5)` — 미관측 중립 soft-match). symmetric(`compat_tile=True`)에서 `_positive_place`가 (scan) 맞는 top-left stride 열거→(rank) footprint를 64px 타일(`_COMPAT_TILE=64`)로 덮어 `_context_cell_key(_extract_context_features(win))`로 cell→`compat_row.get(cell,0.5)` mean, void 타일 걸친 footprint 제외→(place) top-K(=8) `rng.choice`. (reject) `best_nonvoid_mean≥compat_threshold` 미달 시 다른 normal 재추첨(`_TEXTURE_MAX_NORMAL_REPICK=5`), 소진 시 마지막 후보 강제 paste(무단 drop 금지). defect 모드는 crop 크기 패치 1개만 질의(legacy). cell_key는 `distribution_profiling._extract_context_features/_context_cell_key` lazy import(매트릭스 구축과 동일 스케일).
**clean-bg 게이트**: `_is_clean_background`가 `_background_quality_score`≥`min_quality`(0.7, blur_threshold 100.0)면 통과. (a) 정상 풀 로드(이미지 1회 평가), (b) random-placement fallback 위치(void면 `max_bg_tries=20` 재샘플). 전경 경로엔 미적용(legacy). `_foreground_mask` 내부 Mode A void 거부: 전경 std<`_FG_VOID_STD(5.0)` AND bbox-crop quality<`_FG_VOID_QUALITY(0.5)`면 None(밝기 무관). `_FG_VOID_MEAN(25.0)` deprecated.
**블렌드 로직**: `alpha` — mask scipy gaussian(sigma=feather_px) feather 후 `bg.paste`. `seamless` — 국소 배경 Reinhard 전이(회색 ref는 L만) 후 `cv2.seamlessClone(NORMAL_CLONE)`; cv2 없음·mask<16px·경계 초과·예외 시 alpha fallback. poisson은 별도 모드 아니라 seamlessClone(gradient-domain)이 담당.
**입력 구조**: `<roi_dir>/roi_selected.json`(image_path, defect_bbox, defect_mask_path, image_id, cluster_id, cell_key, class_key, morph_label/defect_subtype, ctx_prior, prompt, roi_score, deficit), 선택 `roi_candidates.json`(CN subtype/stability join), `compatibility_matrix.json`, `clean_bg_selected.json`(roi_idx, image_id, topk_pool, topk_positions), `--config`, CN용 morphology/context CSV.
**출력 구조**: `images/syn_{roi:05d}_{rep:02d}.jpg`, `masks/syn_{...}.png`(풀프레임 GT), `annotations.json`. annotation: `{image_path, source_roi, image_id, normal_image, cluster_id, class_key, cell_key, prompt, method, blend_mode, roi_score, deficit, mask_path, bbox:[x,y,w,h], dry_run}`. `run()` 반환: `{status, n_generated, n_skipped, annotations, method, (cn_stats), (gate_stats)}`. bbox는 합성 좌표계 결함 영역.
**상수·임계**: `_COMPAT_TILE=64`, `_POS_STRIDE=32`, `_POS_TOPK=8`, `_POS_MAX_CAND=4096`, `_NORMAL_TOPK=16`, `_NORMAL_SAMPLE_CAP=64`, `_NORMAL_STRIDE=64`. 텍스처: `_TEXTURE_STRIP_PX=24`, `_TEXTURE_MIN_PIX=64`, `_TEXTURE_MAX_NORMAL_REPICK=5`, weights `[0.20,0.20,0.35,0.25]`. CN blank: `_CN_BLANK_STD=6.0`, `_CN_BLANK_RANGE=12.0`, `_CN_MEAN_LO=3.0`, `_CN_MEAN_HI=252.0`, `_CN_MAX_ATTEMPTS=3`. 기본값: cn_steps=30, cn_cond_scale=0.7, cn_resolution=512, cn_ar_threshold=2.5, min_bg_quality=0.7, bg_blur_threshold=100.0, feather_px=4.
**주의점**: GT mask는 항상 **실 seed mask**(CN도 텍스처 픽셀만 생성, paste/GT mask는 seed 모양 유지 → empty-mask parity 붕괴 회피). CN은 실 bbox+mask 필수(타원 fallback 없음, `skip_no_mask`). CN **AR 폴백**: `max(w,h)/min(w,h)>cn_ar_threshold` elongated bbox는 squash-smear 때문에 기본 copy_paste fallback(`method="copy_paste_arfallback"`, parity 유지); `--cn_no_ar_fallback` 시 skip(`skip_ar`). `--random-placement`(naive): forced_xy/compat/전경/void 게이트 전부 우회 균일-랜덤(smart-placement 기여 분리용, copy_paste 전용). CN seed는 content-addressed(`sha1(image_id|bbox|cell_key|rep)`); sidecar `.meta.json` fingerprint 캐시(`--local_staging`은 캐시 무력화 → CN은 Drive output_dir 권장). compat fallback rate>50% 시 "placement uplift 주장 금지" 경고. rng 규율: 게이트 OFF면 새 draw 없이 legacy 스트림 byte-identical.

### generate_random.py (`scripts/aroma/generate_random.py`) — step5 [CPU copy_paste]
**역할**: 랜덤 컨트롤 arm. `roi_candidates.json`에서 top_k ROI 균일 샘플→`roi_selected.json` 작성→`generate_defects.run(method="copy_paste")` 위임. 전 arm 동일 합성 코드, ROI 선정만 다른 baseline.
**진입점 / argparse**: `main()` → `_parse_args()` → `run()`. `_bootstrap`로 `AROMA_REF` sys.path→`generate_defects` import. CLI: `--candidates_json`,`--normal_dir`,`--output_dir`,`--random_roi_dir`(기본 `{output_dir}/_random_roi`),`--top_k(200)`,`--n_per_roi(3)`,`--seed(42)`,`--local_staging`,`--reject-clean-bg`,`--no-random-placement`(기본 naive ON),`--min-bg-quality(0.7)`,`--bg-blur-threshold(100.0)`,`--blend-mode{alpha,seamless}`,`--min_quality(0.0=OFF)`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `run(...)` | `(candidates_json,normal_dir,output_dir,top_k,...)->dict` | 로드→(품질 parity 필터)→랜덤 선정→roi 파일→`generate_defects.run` 위임 |
| `select_random` | `(candidates,top_k,seed)->list` | `default_rng(seed)` 비복원 균일, 인덱스 정렬 |
| `_q(c)` | `->float` | 저장된 `quality_score`(결측→1.0, NaN→0.0) |

**제어 흐름 (`run`)**: (1) `candidates_json` 존재 확인(없으면 exit 1)→로드 → (2) `min_quality>0`이면 `_q(c)≥min_quality` 필터(aroma와 동일 게이트; 풀<top_k면 fallback 없이 전체 사용, 경고) → (3) `select_random` top_k 균일 → (4) `roi_dir/roi_selected.json`+`roi_candidates.json`(수동검사용) → (5) `generate_defects.run(method="copy_paste", random_placement=..., reject_clean_bg=..., blend_mode=..., ...)` → (6) `n_rois_selected` 추가 반환.
**호환성 게이트**: 자체 없음. compat 인자를 넘기지 않음(OFF). 유일 parity 장치는 `min_quality` 사전 필터.
**clean-bg 게이트**: 자체 없음. `reject_clean_bg`/`min_bg_quality`/`bg_blur_threshold`를 forward(기본 reject_clean_bg=False). 판정은 `generate_defects._is_clean_background`.
**블렌드**: 자체 없음. `blend_mode` forward(aroma와 동일 blender).
**입력 구조**: `roi_candidates.json`(공유 후보 풀; `run`이 읽는 키는 `quality_score`뿐). `--normal_dir`.
**출력 구조**: `{output_dir}/_random_roi/roi_selected.json`(선정분), 같은 폴더 `roi_candidates.json`(필터 후 풀, 기록용). 합성 이미지·annotations.json은 `generate_defects.run`이 `--output_dir`에. 반환=`generate_defects.run` 결과 + `n_rois_selected`.
**상수·임계**: top_k=200, n_per_roi=3, seed=42, min_bg_quality=0.7, bg_blur_threshold=100.0, min_quality=0.0(OFF), `random_placement=True`(naive 기본 ON — 설계상 naive 표준 baseline).
**주의점**: 랜덤 arm은 기본 naive placement ON(`--no-random-placement`으로만 grounded 복원) — smart-placement 기여 분리용 의도적 비대칭. ROI 선정 랜덤성(`select_random`)과 별개. `min_quality>0`은 원본 candidates가 quality deps 가용 상태에서 생성됐어야 유의미(deps 불가면 전부 1.0 저장, 이 스크립트는 감지 불가) → step3 "deps unavailable" 경고 확인 필요. 두 rng: ROI 선정 numpy `default_rng(seed)`, 합성 placement `random.Random(seed)`. copy_paste GT mask/AR 폴백은 `generate_defects` 소유(thin wrapper).

---

## exp — 다운스트림 평가

### exp3_generation_quality.py (`scripts/aroma/experiments/exp3_generation_quality.py`) — exp3 [CPU]
**역할**: copy-paste 합성 이미지에 대해 Random ROI vs AROMA ROI의 생성 품질 측정. 합성 모델 자체가 아니라 "adaptive ROI 모델링이 합성 학습데이터 품질을 높이는가"를 평가. `fid` 모드 CPU 기본, `ad`(PaDiM)만 GPU 강제.
**진입점 / argparse**: `main()` → `_parse_args()` → `run()`. 인자: `--mode {fid,ad,all}`(기본 all), `--random_synthetic_dir`, `--aroma_synthetic_dir`, `--real_data_dir`, `--dataset_keys`(nargs+), `--output_dir`, `--seed`(42), `--device`(cpu), `--image_size`(256), `--num_workers`(4).
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `_get_image_lists` | `(dataset_key, real_data_dir, seed, test_split)` | dataset_config 기반 train_normal/test_good/test_defect/mask_map 해석 |
| `_resolve_masks_generic` | `(defect_paths, root)` | mvtec/severstal/visa GT 마스크 규약 first-hit |
| `_mask_to_bbox` | `(mask)` | 논-제로 영역 bbox |
| `_load_real_defect_patches` | `(paths, num_workers)` | full-image 실결함 로드(패치 아님) |
| `_load_source_roi_crops` | `(annotations_path, num_workers)` | annotations.json `image_path` full 합성 이미지 |
| `_compute_fid_score`/`_compute_kid_score`/`_compute_lpips_score` | torchmetrics FID(2048)/KID/lpips(alex) |
| `_run_padim_condition` | anomalib Folder+Padim(resnet18) 1-epoch |
| `_build_summary` | FID/KID/LPIPS/AD 표 + Δ(AROMA−Random) |

**제어 흐름**: `run()` → (1) mode∈{fid,all}→`_run_fid_mode`, (2) mode∈{ad,all}→`_run_ad_mode`, (3) 병합 후 `exp3_results.json`, (4) `_build_summary`→`exp3_summary.md`.
**핵심 로직/통계**: FID/KID/LPIPS는 **real 결함 이미지 vs 합성 이미지 전체(full-image)** 비교(합성 마스크 미저장이라 패치 정렬 불가, ROI 배치 컨텍스트가 측정 대상). FID 299 리사이즈 후 InceptionV3 2048-dim; KID subset_size=min(50, n_synth); LPIPS real 상한 20·synth 상한 50 페어. AD 모드는 train=real_normal+synth, test/good=real held-out, test/defect=real 결함(**합성 미포함**), 마스크 있으면 SEGMENTATION(pixel AUROC) 없으면 CLASSIFICATION. 순열/부트스트랩 없음(단순 Δ).
**입력 구조**: dataset_config(`image_dir`, `seed_dirs`); real `<ds>/train/good`·`test/good`·`test/<type>`+GT 마스크; 합성 `{synth_dir}/{ds}/annotations.json`(`image_path`), `{synth_dir}/{ds}/images/`.
**출력 구조**: `exp3_results.json`(`{ds:{fid:{n_real_patches, random:{fid,fid_unstable}, aroma:{}}, kid, lpips, ad:{baseline/random/aroma:{image_auroc,pixel_auroc}}}}`), `exp3_summary.md`.
**상수·임계**: seed=42, test_split=0.2, FID feature=2048/resize 299, image_size=256, `fid_unstable`=n_real<50, `kid_unstable`=n_synth<50, `lpips_unstable`=n_real<10, num_workers=4.
**주의점**: FID metric thread-unsafe라 리사이즈만 병렬·update 순차; 데이터셋 수준만 병렬. real_data_dir은 시그니처 호환용(경로는 config 절대경로). `ad` 모드는 `_check_gpu()` CUDA 없으면 exit 1. AROMA/Random 동일 copy-paste 엔진 → FID(fidelity) 동등 기대, 차이는 ROI 선택 신호.

### exp4_v2_supervised_detection.py (`scripts/aroma/experiments/exp4_v2_supervised_detection.py`) — exp4v2 [GPU]
**역할**: HEADLINE. YOLOv8 supervised 검출에서 세 조건(baseline/random/aroma; casda 옵션)이 **공통 real labeled defect(train split)** 위에 학습하되 random/aroma만 합성 additive 추가. 모두 COCO 사전학습 시작. 가설: 합성 추가가 baseline 대비 향상, AROMA≥Random.
**진입점 / argparse**: `main()` → `_parse_args()` → `run()`(멀티시드) → `_run_detection_mode()`(단일 seed) → `_run_yolo_condition()`(단일 model×cond). 인자: `--model {yolov8n/s/m/all}`, `--condition`(nargs+, all 또는 baseline/random/casda/aroma), `--dataset_keys`, `--random_synthetic_dir`(필수), `--aroma_synthetic_dir`(필수), `--casda_synthetic_dir`(옵션), `--real_data_dir`, `--output_dir`, `--baseline_epochs`(50), `--val_frac`(**CLI 0.3**, 함수 기본 0.5), `--real_frac`(1.0), `--max_synth_per_ds`, `--synth_ratio`, `--patience`(0), `--imgsz`(256), `--batch`(16), `--cache`(""), `--rect`, `--class_mode {single,multi}`, `--seed`(42), `--seeds`(nargs+), `--resume`, `--yolo_cache_dir`, `--no_local_cache`.
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `_split_defects` | `(defect_paths, mask_map, val_frac, seed)` | 마스크 보유 결함만 disjoint train/val, 타일셋 group-aware |
| `_defect_group_key` | `(path)` | `__tile_` 앞 stem = 소스 그룹 키(타일 누수 방지) |
| `_extract_defect_bboxes` | `(synth, normal, min_area)` | composite−background diff→thresh(8)→contour, Otsu 폴백 |
| `_mask_to_bboxes` | `(mask_path, min_area)` | 마스크 contour bbox, 전멸 시 union 1-bbox 폴백 |
| `_write_yolo_labels` | `(synth_annotations,...)` | 합성 라벨 유도: mask→diff→template match 폴백 |
| `_get_real_test_images_and_labels` | real 결함+GT마스크 → YOLO 라벨셋 |
| `_build_or_load_real_yolo_dataset` | real (image,label)셋 데이터셋당 1회 빌드/Drive 캐시 |
| `_run_yolo_condition` | 1개 (model,cond) 학습+검증 |
| `_run_detection_mode` | datasets×models×conditions 순회(단일 seed) |
| `_aggregate_seeds` | `(per_seed_results, seed_list)` | seed별 mean±std+CI95 |
| `_ci95` | `(samples, mean, std)` | t-분포 95% CI(k<2→[mean,mean]) |
| `_build_summary`/`_build_summary_multiseed` | 단일/멀티시드 마크다운 표 |

**제어 흐름**: `run()` → (1) `seed_list=seeds or [seed]` → (2) seed마다 `_seeds/seed{N}/exp4v2_results.json` resume 로드 → `_run_detection_mode` → per-seed 저장 → (3) `_aggregate_seeds` 통합 → (4) top-level `exp4v2_results.json`(top 키=mean) + summary(seeds>1이면 multiseed). `_run_detection_mode` 데이터셋 루프: (a) `_get_image_lists`, (b) 조건별 합성 annotations, (c) synth cap(`synth_ratio` 우선→`max_synth_per_ds`), (d) 리눅스면 Drive→/tmp 로컬 캐시, (e) `_split_defects`, (f) 타일 bg-leak 가드, (g) `real_frac` 축소, (h) `_build_or_load_real_yolo_dataset`, (i) real셋 데이터셋당 1회 하드링크 스테이징, (j) model×cond 루프(baseline 우선)로 `_run_yolo_condition`, incremental save.
**핵심 로직/통계**: **split** — 마스크 보유 결함만 eligible, non-tiled은 `random.Random(seed).shuffle` 후 val=`max(1,round(n·val_frac))`(train 최소 1 보장); tiled(aitex `__tile_`)은 소스 그룹 단위 greedy 배분(50% overlap 누수 차단), 그룹 1개면 tile-level 폴백. **합성 라벨 유도**: `mask_path` 있으면 authoritative(마스크→bbox, 스케일), 없으면 composite−normal diff, 그래도 없으면 source_roi 템플릿 매칭. min_area=50(real GT 통일). **가중치**: 세 조건 모두 `{model}.pt`(COCO)에서 시작, `save=True`(save=False면 val이 학습전 가중치 측정→map50 붕괴). **집계**: cell별 valid map50 seed만 표본, std ddof=1(k<2→0), CI95 scipy t-ppf(없으면 t-table/1.96). 순열검정 없음. 전 seed 실패 셀 drop.
**입력 구조**: real은 dataset_config/디렉토리 규약(`test/<type>`+GT마스크, severstal `masks/class{c}/`); 합성 `{dir}/{ds}/annotations.json`(`image_path`, `normal_image`, `mask_path`, `source_roi`, `cluster_id`, `defect_bbox`).
**출력 구조**: `exp4v2_results.json` `{ds:{model:{cond:{map50, map50_95, precision, recall, n_train, n_real_train, n_synth_train, n_seeds, per_seed, std, ci95, weights_path?(baseline), per_class?/n_synth_per_class?(multi)}}}}`; per-seed `_seeds/seed{N}/exp4v2_results.json`; 체크포인트 `_seeds/seed{N}/{ds}/{model}/{cond}/weights/best.pt`; `exp4v2_summary.md`.
**상수·임계**: `ALL_MODEL_KEYS=[yolov8n,s,m]`, `ALL_CONDITION_KEYS=[baseline,random,casda,aroma]`, val_frac CLI 0.3, min_area=50, imgsz=256, batch=16, baseline_epochs=50, `_AGG_METRICS=(map50,map50_95,precision,recall)`, `_INT_FIELDS=(n_train,n_real_train,n_synth_train)`, `_TILE_DELIM="__tile_"`.
**주의점**: **resume**는 유효 map50 있는 셀만 skip(전 셀 완료면 스테이징도 skip). **조건 간 n 동일화**는 real셋을 데이터셋당 1회 build해 byte-identical 하드링크 공유; synth만 additive. **leakage 차단**: val=real only, 합성 소스는 train split ROI, 타일 group-aware + bg-leak 가드. `synth_ratio` 지정 시 `max_synth_per_ds` 무시(cap=`max(1,int(n_real_train·ratio))`). 중복 seed dedup(안 하면 std 축소). `_check_gpu()` CUDA 없으면 exit. 빈 synth로 random/aroma 실행 거부(`no_synth_annotations`).
> exp4v2 selective rerun 절차는 `colab_execute_new/exp4v2_rerun_aroma_random_execute.md` 참조 — per-seed JSON에서 대상 condition 키만 제거해 baseline 보존하며 aroma/random만 재학습.

### exp5_prdc.py (`scripts/aroma/experiments/exp5_prdc.py`) — exp5 [GPU]
**역할**: DINOv2 외부 임베딩 좌표계에서 PRDC 커버리지로 ROI 선택 가치 검증. **사전등록 가설**: 동일 copy-paste 엔진이므로 Precision/Density는 aroma≈random(fidelity), Recall/Coverage는 aroma>random — Recall계만 오르고 Precision계 동등해야 입증.
**진입점 / argparse**: `main()`(CUDA 미가용 cpu 강등) → `run(args)` → `_run_one_dataset()`. 인자: `--real_data_dir`, `--aroma_synthetic_dir`, `--random_synthetic_dir`, `--dataset_keys`, `--nearest_k`(nargs+, 기본 `3 5 10`, 주보고 k=5), `--permutation_reps`(1000), `--val_frac`(0.3), `--split_seed`(42), `--seed`(42), `--backbone`(dinov2_vits14), `--embed_cache_dir`, `--output_dir`, `--device`(cuda).
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `_split_defects` | exp4v2 verbatim 복제(stdlib random), byte-identical val split |
| `_load_real_defect_crops` | `(paths, mask_map)` | GT 마스크 bbox crop(full-image 아님) |
| `_load_synth_crops` | `(ann_path)` | 합성 crop, mask→bbox→skip 폴백(full-image 폴백 금지) |
| `_build_backbone` | `(name,device)` | DINOv2(224), 실패 시 InceptionV3(299) 폴백 |
| `_cached_embed` | `.npy Drive 캐시(sha256 manifest 무효화)` |
| `_observed_prdc` | `(real,fake,k)` | prdc 패키지 직접 호출 |
| `_permutation_test` | `(real,aroma,random,k,reps,seed)` | 두 synth 풀 합쳐 재분할 순열검정 |
| `_prdc_from_distances` | 사전계산 거리 위 PRDC 재구현 |
| `_equalize_n` | `(a,r,seed)` | min-n seeded subsample |

**제어 흐름**: `_run_one_dataset()` → (1) `_get_image_lists`+`_split_defects`로 val(reference) → (2) real crop → (3) aroma/random synth crop → (4) 없으면 skip → (5) backbone+캐시 임베딩(real_val/synth_aroma/synth_random) → (6) `_equalize_n` → (7) k별 `_observed_prdc`×2 + `_permutation_test`, parity check → (8) incremental JSON. run()이 데이터셋 순회 후 summary.
**핵심 로직/통계**: **reference=held-out real 결함 val split**(val_frac=0.3, split_seed=42로 exp4v2와 byte-identical → synth는 train split ROI에서만 → leakage 차단). 거리는 **유클리드**(prdc 규약). kNN 반경=`np.partition(d_self,k)[:,k]`. PRDC: precision=`(d_rf<radii_r).any(0).mean()`, recall=`(d_rf<radii_f).any(1).mean()`, density=`(d_rf<radii_r).sum(0).mean()/k`, coverage=`(d_rf.min(1)<radii_r).mean()`. **순열검정**: aroma+random 합쳐(2n) 무작위 재분할, real반경·real×combined·combined×combined 거리 1회 사전계산 후 순열마다 인덱스+fake반경만 재계산; one-sided p=`(#(null≥obs)+1)/(reps+1)`, null CI95. prdc 패키지 값 vs 벡터화 numerical parity assert(atol=1e-6).
**입력 구조**: dataset_config; real GT 마스크; `{aroma,random}_dir/{ds}/annotations.json`(`image_path`, `mask_path`, `bbox`=[x,y,w,h], `dry_run`).
**출력 구조**: `exp5_prdc_results.json` `{ds:{meta:{n,n_ref,backbone,split_seed,val_frac,seed,unstable,n_skipped_synth}, k{K}:{aroma:{P,R,D,C}, random, delta, p_one_sided, null_ci95}}}`; `exp5_prdc_summary.md`(판정: recall·coverage p<0.05 & Δ>0).
**상수·임계**: nearest_k=[3,5,10](주 k=5), permutation_reps=1000, val_frac=0.3, split_seed=42, seed=42, backbone=dinov2_vits14, batch_size=64, unstable=n_ref<30, k별 n≤k or n_ref≤k면 skip.
**주의점**: 조건 간 n 엄격 동일화(`_equalize_n`) 후 순열. `embed_cache`는 sha256(crop_ids) manifest 무효화, exp6와 real_val tag 공유. `dry_run` annotation은 플래그로 명시 제외. full-image crop 폴백 금지(배경 지배 방지). exp3에서 `_get_image_lists`/`_mask_to_bbox` import(exp3는 `__main__` 가드로 부작용 없음).

### exp6_embedding_coverage.py (`scripts/aroma/experiments/exp6_embedding_coverage.py`) — exp6 [GPU]
**역할**: DINOv2 임베딩 커버리지의 두 저비용 모드. `knn`: held-out val 결함 crop이 학습 풀(real / real+random / real+aroma)에 얼마나 가깝게 커버되는지 1-NN cosine 거리 비교(이미지 단위 clustered bootstrap). `rare`: 후보 소스 crop의 k-means 모드 중 rare 모드(빈도 p25 이하 AND val 등장)에 대한 aroma 선택 hit rate를 random 재선택 null 대비 검정. 정보 분리: AROMA 선택은 val/test 미열람.
**진입점 / argparse**: `main()`(CUDA 강등) → `run(args)` → `_run_knn_dataset` 또는 `_run_rare_dataset`. 공통: `--mode {knn,rare}`(필수), `--real_data_dir`, `--dataset_keys`, `--val_frac`(0.3), `--split_seed`(42), `--seed`(42), `--backbone`(dinov2_vits14), `--embed_cache_dir`(exp5 공유), `--output_dir`, `--device`(cuda). knn: `--aroma_synthetic_dir`, `--random_synthetic_dir`, `--bootstrap_reps`(2000). rare: `--roi_dir_root`, `--kmeans_k`(`8 10 12 15`), `--cluster_seeds`(`0 1 2 3 4`), `--null_seeds`(30), `--rare_quantile`(0.25).
**핵심 함수/클래스**:
| 이름 | 시그니처(축약) | 역할 |
|------|----------------|------|
| `_cosine_dist` | `(a,b)` | L2정규화 후 1−cosine(exp5 유클리드와 의도적 상이) |
| `_min_dist_to_pool` | `(val,pool,chunk=512)` | val crop별 풀 최근접(청크 메모리 방어) |
| `_clustered_bootstrap` | `(d1_rand,d1_aroma,img_ids,reps,seed)` | 이미지 단위 재표집 Δ=mean d1(real+random)−mean d1(real+aroma) |
| `_coverage_auc` | `(d1,r_grid)` | C(r)=P(d1≤r) grid 평균 |
| `_diversity_stats` | `(feat,n_sources)` | 풀 내 pairwise cosine(중복 합성 부풀림 감시) |
| `_load_candidate_crops` | `(candidates)` | unique (image,defect_bbox) 후보 crop |
| `_rare_cell_keyed` | `(...)` | (k,cseed) 셀: rare 모드 정의 + observed vs random-null hit rate |

**제어 흐름 (knn)**: `_split_defects`로 train/val → 조건별 synth crop → real val/train + synth aroma/random 임베딩(캐시) → `_equalize_n` → 풀별 d1(real, real+random=min, real+aroma=min) → leakage sanity(d1<1e-6) → `_clustered_bootstrap` → coverage_auc + diversity. **(rare)**: `roi_candidates.json`+`roi_selected.json` → unique 후보 crop 임베딩 → val crop 임베딩(test-presence 필터) → key 인덱스 매핑 → k×cluster_seed 그리드마다 KMeans fit(cand)+predict(val) → `_rare_cell_keyed` → verdict 집계.
**핵심 로직/통계**: **knn 검정**=이미지 단위 clustered bootstrap(같은 이미지 crop 비독립 → 인스턴스 단위 금지), bootstrap p=`(#(Δ_boot≤0)+1)/(reps+1)`, Δ>0이 aroma 우위, CI95. r_grid=d1_real의 10~90 percentile. **rare 검정**: rare 모드=`0<counts≤percentile(nonzero,25) AND val 등장`; observed=선택셋 rare 비율; null은 **candidate 엔트리 단위**(중복 포함) 무중복 균일 추출(generate_random과 budget·표본공간 일치) 30-seed; p_emp=`(#(null≥obs)+1)/(null_seeds+1)`(하한≈0.032). JSD-to-real 미사용(rare 과대표집이 목적).
**입력 구조**: dataset_config; real GT 마스크; knn은 `{dir}/{ds}/annotations.json`(`source_roi`로 distinct sources); rare는 `{roi_dir_root}/{ds}/roi_candidates.json`·`roi_selected.json`(`image_path`, `defect_bbox`, `defect_mask_path`).
**출력 구조**: `exp6_results.json`(top 키=mode: `{knn:{ds:{d1_mean,d1_median,delta_AR,coverage_auc,diversity,meta}}}` / `{rare:{ds:{grid:{k{K}_s{S}:{observed,null_mean,null_ci95,p_emp,n_rare_modes}}, verdict:{n_valid_cells,direction_consistent,n_sig_p05}, meta}}}`); `exp6_{mode}_summary.md`.
**상수·임계**: val_frac=0.3, split_seed=42, seed=42, bootstrap_reps=2000, kmeans_k=[8,10,12,15], cluster_seeds=[0..4], null_seeds=30(KMeans n_init=10), rare_quantile=0.25, knn unstable=고유 val 이미지<10, rare skip=crops<min(kmeans_k), 셀 skip=k≥crops.
**주의점**: **embed_cache exp5 공유** — real_val tag `real_val{val_frac}s{split_seed}`가 exp5와 동일해야 재사용(재합성 후 무효화 필요), synth_aroma/synth_random도 공유. exp5(`_build_backbone`, `_cached_embed`, `_embed_crops`, `_equalize_n`, `_load_real_defect_crops`, `_load_synth_crops`, `_split_defects`)와 exp3 import. rare는 hit 판정 key 단위지만 null/observed는 엔트리 단위. knn cosine vs exp5 유클리드 차이 meta 명시. `run()`은 기존 `exp6_results.json` 로드 후 mode 노드만 갱신(모드 간 결과 공존), incremental save.

---

## 관련 노트

[[00-INDEX]] | [[07-Scripts-Reference]] (I/O·로직 요약) | [[02-Stage0-Prepare-Profiling]] | [[03-Stage1-Complexity-Prompts]] | [[04-Stage2-ROI-Selection]] | [[05-Stage3-ControlNet-Generation]] | [[06-Experiments]] | [[09-Compatibility-Gate]]

> 이 노트는 **파일 내부 구조**(함수·상수·제어 흐름)를, 07은 **스크립트 I/O 요약**을 담당한다. 스텝별 개념·실행 절차는 02~06 노트, 게이트 novelty는 09를 참조.
