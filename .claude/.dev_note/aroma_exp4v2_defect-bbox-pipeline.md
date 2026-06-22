# 결함 mask·bbox 영속화 파이프라인 구축 — copy_paste 타이트 크롭/실제 형태 blend (B안)

## (사용할 skills: feature-dev)

## 개요

AROMA copy_paste 합성이 결함 영역을 크롭하지 않고 full test 이미지를 통째로 paste하여 `bbox=[0,0,1024,1024]`, 타원 mask가 전체를 덮는 버그가 있다. 근본 원인은 결함 mask가 `distribution_profiling._morph_worker`에서 계산(GT 또는 SAM/Otsu fallback)되면서도 **bbox/mask 경로가 어디에도 영속화되지 않고 버려지는 것**이다.

데이터 흐름: `profiling → morphology_features.csv(image_path만) → roi_selection(image_path 전파) → generate_defects(full paste)`.

**B안(정공법)**: profiling 시점에 결함 mask PNG + bbox를 영속화 → CSV 컬럼 additive 추가 → roi_selection이 roi_candidates/roi_selected로 전파 → generate_defects가 bbox로 타이트 크롭 + 실제 mask 형태로 blend + GT mask/bbox를 annotations에 기록. H1(라벨 정확도)와 H2(실제 결함 형태 realism)를 동시 해결한다.

## 영향도 분석

### 변경 상태
- 신규 CSV 컬럼 2개(`defect_bbox`, `defect_mask_path`) additive 추가.
- 3개 파일 본체 수정: profiling(mask PNG 디스크 영속화 + 컬럼), roi_selection(전파만), generate_defects(합성 로직 교체 — P0-A 골격 재사용).
- generate_random은 동일 roi_candidates.json 풀을 dict로 통과 → **코드 수정 불필요, 자동 적용**.

### 역방향 의존 (모두 안전)
- `compute_complexity` MORPH_FEATURES 화이트리스트(L97-99)·`_rows_to_float_array`(L242)는 지정 컬럼만 float 변환 → 신규 컬럼을 **절대 화이트리스트에 추가하지 않으면** 무시되어 안전. (추가 시 float 변환 실패로 전 row drop → 클러스터링 붕괴, **1순위 위험**.)
- distribution_profiling step4~7(deficit/clustering/report/histogram)은 `_load_csv` DictReader + MORPH_FEATURES만 순회 → KeyError·통계 영향 없음.
- `_context_worker`/context_features.csv 스키마 무관.
- exp4_v2는 annotations.json의 `bbox` 키를 직접 읽지 않고 `mask_path`만 소비 → `_mask_to_bboxes`(contour boundingRect)로 재유도. 실제 형태 mask + **전체 캔버스 좌표계**면 자동 정합(L653-662 스케일 경로).

### 재실행 범위 (전체 파이프라인)
컬럼·전파·소비 3계층 모두 변경되므로:
`profiling(step2 재생성) → roi_selection(또는 generate_random) → generate_defects(aroma + random) → exp4_v2 supervised` 전부 재실행 필요.
기존 morphology_features.csv·roi_*.json은 새 컬럼이 없어 빈 bbox를 받으므로 **반드시 재생성**(MEMORY의 mvtec_bottle 재실행 전제와 동일).

## 이론적 근거 (왜 B안)

- A안(generate_defects 단독 수정)으로는 불가능: roi_selected.json에 bbox/mask가 애초에 없어 소비측에서 받을 데이터가 없다.
- `_morph_worker`는 이미 binary mask(L234)와 regionprops max-area region(L238)을 메모리에 보유 → `region.bbox`와 mask crop을 **거의 추가 비용 없이** 산출 가능.
- profiling 시점 영속화가 좌표계 정합의 진실 원천: GT mask 경로(`_find_mask_path`)가 원본과 동일 좌표계이므로 여기서 산출한 bbox가 가장 신뢰도 높다.
- mask는 **full-frame(전체 캔버스)으로 저장하고 bbox는 별도 컬럼**으로 둔다 → 소비측이 `mask[y:y+h, x:x+w]`로 크롭하도록 설계(저장 단계에서 크롭하지 않는 편이 좌표 추적에 안전).

## 수정 내용 (파일별 함수·라인)

### scripts/distribution_profiling.py
1. **`_MORPH_CSV_FIELDS`(L92-96)**: 화이트리스트 끝에 `'defect_bbox'`, `'defect_mask_path'` 2개 추가. `_save_csv`가 `extrasaction='ignore'`(L431)이므로 선언하지 않으면 worker dict의 새 키가 **silent drop** — 반드시 동시 수정.
2. **`DistributionProfiler.__init__`(L484-485)**: `(output_dir / 'defect_masks').mkdir(parents=True, exist_ok=True)` 1줄. 병렬 worker mkdir race 회피를 위해 메인에서 1회 생성.
3. **step1 task 구성(L536-542)**: `task_base`에 `'mask_out_path': str(output_dir/'defect_masks'/f'{defect_type}_{p.stem}.png')` 주입. `_morph_worker`는 module-level pickle-safe 함수라 `self.output_dir` 접근 불가 → 경로를 task로 전달. `defect_type` prefix로 seed_dirs 배열(multi-defect-type) 간 stem 충돌 방지.
4. **`_morph_worker`(L233-259)**: regionprops max-area region(L238) 재사용. `minr,minc,maxr,maxc = region.bbox` → `defect_bbox = [int(minc), int(minr), int(maxc-minc), int(maxr-minr)]` (skimage (row,col)순 → x=col,y=row **명시 변환**). bbox 산출 직전 `mask.shape[:2] != img_color.shape[:2]`이면 `cv2.resize(mask,(W,H),INTER_NEAREST)`로 정합 보정. full binary mask(모든 컴포넌트 0/255)를 `task['mask_out_path']`에 `cv2.imwrite`. 반환 dict에 `'defect_bbox': f'{x},{y},{w},{h}'`, `'defect_mask_path': str(경로)` 추가. props 비었을 경우(L242-244 else) `cv2.boundingRect(binary)` fallback.

### scripts/aroma/roi_selection.py
`build_candidates`(L209-247) **한 곳만** 수정하면 roi_candidates.json/roi_selected.json + generate_random 풀 전부 커버(candidate dict는 모든 전략을 변형없이 통과·JSON 직렬화).
1. L210-211 직후: `bbox_str = row.get('defect_bbox',''); mask_path = row.get('defect_mask_path','')`.
2. bbox 파싱 헬퍼: `'x,y,w,h'` split 후 int 4개, 빈 값/형식 오류면 None.
3. candidate dict(L235-247)에 `'defect_bbox': [x,y,w,h] | None`, `'defect_mask_path': mask_path` 추가(image_path 동일 패턴).

### scripts/aroma/generate_defects.py
`copy_paste_synthesis` 본체 교체 (P0-A의 mask/bbox 영속화 골격 위에 **타원→실제 mask**로 교체). 시그니처/반환(Optional[Dict])은 P0-A에서 준비됨 — 본체만 수정.
1. L198 이후: `roi_entry.get('defect_bbox')`, `roi_entry.get('defect_mask_path')` 읽기. 둘 다 유효 → 실제-mask 경로, 없으면(빈 문자열/None) 기존 타원+full fallback 또는 skip(하위호환).
2. **L211-219 타원 생성 제거** → defect_mask_path PNG 로드 후 defect_bbox `[x,y,w,h]`로 동일하게 crop: `defect_crop = defect_img.crop((x,y,x+w,y+h))`, `mask_crop = mask_png.crop((x,y,x+w,y+h))`. 동일 bbox로 잘라 size 일치 → `_alpha_composite` L136-137 resize 왜곡 경로 회피.
3. L222: `_random_paste_position(normal_img.size, defect_crop.size, rng)` — crop 타이트화로 (0,0) 고정에서 정상 동작.
4. **L232-233**: `bbox = [x_paste, y_paste, crop_w, crop_h]` (composite 좌표계, crop 크기 기준). **`bbox=[0,0,1024,1024]` 버그의 핵심 수정**.
5. L237-241: `full_mask = PILImage.new('L',(bw,bh),0); full_mask.paste(mask_crop, (x_paste,y_paste))` — 실제 형태를 paste 위치에 배치. 전체 캔버스 좌표계 PNG 저장(exp4_v2 계약 정합).
6. crop_w/h > normal size면 `_random_paste_position`의 max(0,..)로 잘림 → resize 또는 skip 분기.

## 수정 대상 파일
- `scripts/distribution_profiling.py`
- `scripts/aroma/roi_selection.py`
- `scripts/aroma/generate_defects.py`
- (수정 불필요·자동 적용: `scripts/aroma/generate_random.py`, `scripts/aroma/experiments/exp4_v2_supervised_detection.py`)
- (변경 금지: `scripts/aroma/compute_complexity.py` — MORPH_FEATURES 화이트리스트 오염 금지)

## 암묵적 요구사항 (엣지)
- **좌표계 정합**: GT mask 해상도 ≠ 원본이면 bbox 직전 `cv2.resize` INTER_NEAREST.
- **skimage 좌표 변환**: region.bbox (y0,x0,y1,x1) → PIL (x,y,w,h) 명시 변환(혼동 시 뒤바뀜).
- **ISP fallback**: GT 없음 → 전부 SAM/Otsu. Otsu가 배경 전체를 잡으면 bbox가 full에 근접 → mask_source를 annotations에 흘려 필터링. area/extent sanity 필터 검토.
- **CSV 직렬화**: `'x,y,w,h'` 콤마 → 4컬럼 쪼개짐 위험. csv.writer quoting + roi_selection 역파싱 형식 일치 필수.
- **빈 bbox/mask_path row**: None vs '' 구분해 타원 fallback/skip(하위호환).
- **다중 결함**: max-area 1개만 crop(기존 동작, 회귀 아님). full mask PNG는 모든 컴포넌트 → exp4_v2가 contour별 다중 box 산출. annotations bbox는 단일 참고용, **mask가 source of truth**.
- **crop > normal**: resize 또는 skip.
- **병렬 I/O**: defect_masks/는 메인에서 1회 mkdir, 파일명 `{defect_type}_{stem}.png` 유니크.
- **절대경로 mask_path**: Colab/Drive 재마운트 깨짐 가능 → exp4_v2 `_resolve_path`로 절대+상대 시도.
- **min_area=50 필터**: 50px 미만 결함은 box 없음 → background negative. 작은 결함 silent drop 주의.

## 테스트 (Colab, pytest 금지)
1. profiling step2 재실행 → `defect_masks/{defect_type}_{stem}.png` 생성 + CSV에 `defect_bbox`/`defect_mask_path` 채워짐 확인(fallback row 포함).
2. CSV의 defect_bbox가 [0,0,W,H] full이 아닌 타이트 박스인지 샘플 확인. fallback_* row bbox 신뢰도 점검.
3. compute_complexity / step5 clustering 재실행 → 컬럼 추가 후에도 float 변환 에러 없이 동작(화이트리스트 미오염).
4. roi_selection(또는 generate_random) 재실행 → roi_candidates/roi_selected.json에 키 전파 확인.
5. generate_defects 재실행 → 결함이 타이트 영역에만 paste되는지 육안 확인. annotations bbox ≠ [0,0,1024,1024], mask PNG가 실제 형태(타원 아님) 확인.
6. exp4_v2 재실행 → `_mask_to_bboxes`가 타이트 YOLO 라벨 산출, min_area=50 의미있게 동작.

> **테스트 정책**: 신규 테스트 코드 작성 금지, pytest 실행 금지. 전 검증 Colab에서 직접 수행.

## 미확정 TODO
- bbox 직렬화 형식(쉼표구분 vs JSON 문자열) 최종 확정 + csv quoting Colab 실측.
- ISP Otsu fallback bbox 품질 — area/extent sanity 필터 추가 여부.
- GT mask 해상도 ≠ 원본 케이스 실제 존재 여부(prepare_visa/prepare_mvtec) 확인.
- crop > normal 시 resize vs skip 정책 확정.
- 기존 생성물 일괄 재생성 순서대로 재실행(mvtec_bottle 등 MEMORY 명시 데이터셋 포함).