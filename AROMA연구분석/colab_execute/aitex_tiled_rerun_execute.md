# AITEX Tiled 재실행 — 타일링(256×256/stride128) + 단일 클래스 전환 후 전체 파이프라인 Colab 실행

> **실행 환경**: prepare/profiling/roi/합성 = CPU 가능 (GPU 불필요) | exp4v2 = GPU (A100 권장)
> **목적**: AITEX 4096×256 원본의 종횡비 붕괴(imgsz letterbox → 콘텐츠 256×16, baseline mAP50 0.066)를 타일링으로 해소하고, 11클래스/71장 표본 문제를 단일 클래스로 병합해 exp4v2 3-seed 재비교의 기반을 복구한다.
> **선행**: 저장소 최신 pull (prepare_aitex.py 타일링, exp4_v2 group-aware split + bg-leak guard, dataset_config.json aitex entry 갱신 커밋 포함).
> **로컬 실측(사전 검증 완료)**: 원본 106결함/141정상 → defect 타일 352 (union_fallback 70), train/good 7,169 타일 (NODefect 4,371 + defect-유래 bg 2,798), 12개 결함 코드 전부 생존, 재실행 byte-identical 결정성 확인.

---

## 0. 개요 / 변경 배경

```
[1] prepare_aitex.py (--tile 256 --stride 128)   4096×256 → 256×256 타일
        ↓ Aroma/aitex_tiled/{train/good, test/defect, ground_truth/defect, aitex_manifest.json}
[2] distribution_profiling.py                     타일 레이아웃 프로파일링
[3] compute_complexity.py → prompt_generation.py  MCI/CCI + 프롬프트
[4] roi_selection.py (compatibility)              ROI 선택
[5] generate_defects.py (aroma) + generate_random.py  합성 재생성 (타일 기준)
[6] yolo_cache/aitex 삭제 + 기존 aitex 결과 제거
[7] exp4v2 3-seed (aitex 단독, --imgsz 256)       baseline 정상화 확인 → 3조건 비교
```

핵심 계약:
- 타일 파일명 `{원본stem}__tile_r{r}_c{c}.png` — exp4v2 `_split_defects`가 이 마커로 **원본 이미지 단위 train/val split** (50% overlap leak 차단).
- 단일 클래스: `test/defect` 하나 → exp4v2 single 모드 자연 축퇴 (nc=1).
- 결함 타일 마스크는 0/255 재인코딩 + 경계 파편 제거(cleaned). 내부 점결함(<50px)은 union bbox로 보존.
- **기존 비타일 `aitex` 레이아웃과 절대 혼용 금지** — 출력은 새 디렉토리 `aitex_tiled`.

---

## 1. 환경변수 설정

```python
import os

os.environ['DRIVE']              = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_ROOT']         = '/content/AROMA'
os.environ['AROMA_SCRIPTS']      = '/content/AROMA/scripts/aroma'
os.environ['AROMA_ROOT_SCRIPTS'] = '/content/AROMA/scripts'
os.environ['DATASET_CONFIG']     = '/content/AROMA/dataset_config.json'

os.environ['AROMA_OUT']    = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AITEX_RAW']    = f"{os.environ['DRIVE']}/aitex_raw"          # Kaggle 3폴더 위치
os.environ['AITEX_TILED']  = f"{os.environ['DRIVE']}/aitex_tiled"        # 신규 출력 (구 aitex와 별도!)

os.environ['PROFILING_AITEX']  = f"{os.environ['AROMA_OUT']}/profiling/aitex"
os.environ['COMPLEXITY_AITEX'] = f"{os.environ['AROMA_OUT']}/complexity/aitex"
os.environ['PROMPTS_AITEX']    = f"{os.environ['AROMA_OUT']}/prompts/aitex"
os.environ['ROI_AITEX']        = f"{os.environ['AROMA_OUT']}/roi/aitex"
os.environ['SYN_AROMA_AITEX']  = f"{os.environ['AROMA_OUT']}/synthetic/aitex"
os.environ['SYN_RAND_AITEX']   = f"{os.environ['AROMA_OUT']}/synthetic_random/aitex"
os.environ['EXP4V2_OUT']       = f"{os.environ['AROMA_OUT']}/exp4v2"

for k in ('AITEX_RAW','AITEX_TILED','PROFILING_AITEX','ROI_AITEX','EXP4V2_OUT'):
    print(f"{k:18s}: {os.environ[k]}")
```

> 환경변수 참조는 `$VAR` (중괄호형 금지), 실행은 `!python` 접두사 (colab-execution.md).
> **주의**: 기존 profiling/roi/synthetic의 aitex 산출물은 비타일 레이아웃 기준이라 전부 재생성 대상. 위 경로가 기존과 같다면 아래 각 단계가 덮어쓴다 — 보존이 필요하면 실행 전 `..._old`로 rename.

---

## 2. prepare — 타일링 실행

```python
!python $AROMA_SCRIPTS/prepare_aitex.py \
    --defect_images   $AITEX_RAW/Defect_images \
    --mask_images     $AITEX_RAW/Mask_images \
    --nodefect_images $AITEX_RAW/NODefect_images \
    --output_dir      $AITEX_TILED \
    --tile 256 --stride 128 --min_tile_area 50
```

> `--tile 256 --stride 128 --min_tile_area 50`은 기본값이라 생략 가능(명시 권장). `--min_tile_area`는 exp4v2 `_mask_to_bboxes` min_area=50과 반드시 동일값.
> 재실행 시 이전 `*__tile_*` 산출물은 자동 purge 후 재생성(결정적). 파라미터 변경 시 NOTE 로그 출력.

**산출 확인 (로컬 실측 기대값 포함):**

```python
import json, os
mani = f"{os.environ['AITEX_TILED']}/aitex_manifest.json"
m = json.load(open(mani)); c = m['counts']
print("tiled              :", m['tiled'], m['tile'])          # True {size:256, stride:128, min_tile_area:50}
print("defect_tiles       :", c['defect_tiles'])              # 기대 352
print("union_fallback     :", c['union_fallback_tiles'])      # 기대 70 (점결함 union bbox 보존)
print("normal_tiles       :", c['normal_tiles'])              # 기대 4371 (NODefect 141×31)
print("bg_from_defect     :", c['bg_tiles_from_defect_images'])  # 기대 2798
print("border_frag_discard:", c['discarded_border_fragment_tiles'])
print("sources matched    :", c['defect_source_images_matched'])  # 기대 102
print("sources zero-tile  :", c['defect_sources_zero_defect_tiles'])  # 기대 0 ← 0이 아니면 점결함 소실
print("회계 폐합          :", c['defect_source_images_matched'] + c['defect_sources_zero_defect_tiles']
      + c['skipped_no_mask'] + c['skipped_empty_mask'] + c['unreadable'] + c['unparsed_code'],
      "==", c['defect_images_total'])                          # 106 == 106
print("defect_codes(12)   :", m['defect_codes'])               # 025 포함 12개 전부
```

---

## 3. dataset_config 확인

저장소 커밋에 이미 반영됨 — 값만 확인:

```python
import json, os
cfg = json.load(open(os.environ['DATASET_CONFIG']))['aitex']
print(cfg['class_mode'])   # single
print(cfg['image_dir'])    # .../aitex_tiled/train/good
print(cfg['seed_dirs'])    # ['.../aitex_tiled/test/defect'] 단일 항목
for d in [cfg['image_dir']] + cfg['seed_dirs']:
    print("exists:", os.path.isdir(d), d)
```

> Drive 경로가 위 §1의 `$AITEX_TILED`와 다르면 dataset_config.json의 aitex entry 경로를 실제 위치로 수정.

---

## 4. profiling → complexity → prompts → roi

```python
!python $AROMA_ROOT_SCRIPTS/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG \
    --dataset_key    aitex \
    --output_dir     $PROFILING_AITEX \
    --num_workers    2
```

> 확인: `morphology_features.csv` rows ≈ defect 타일 수(352), 로그 `fallback masks: 0` (전부 ground_truth 해소).

```python
!python $AROMA_SCRIPTS/compute_complexity.py \
    --profiling_dir $PROFILING_AITEX \
    --output_dir    $COMPLEXITY_AITEX \
    --local_staging

!python $AROMA_SCRIPTS/prompt_generation.py \
    --profiling_dir  $PROFILING_AITEX \
    --complexity_dir $COMPLEXITY_AITEX \
    --output_dir     $PROMPTS_AITEX
```

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROFILING_AITEX \
    --prompts_dir       $PROMPTS_AITEX \
    --output_dir        $ROI_AITEX \
    --sampling_strategy compatibility \
    --top_k             200 \
    --seed              42
```

> 단일 클래스라 `--class_mode multi`/`--class_floor` 불필요 (single 기본값).

---

## 5. 합성 재생성 (aroma + random, 타일 기준)

```python
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $ROI_AITEX \
    --normal_dir  $AITEX_TILED/train/good \
    --output_dir  $SYN_AROMA_AITEX \
    --method      copy_paste \
    --blend_mode  seamless \
    --n_per_roi   3 \
    --seed        42 \
    --local_staging \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0

!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI_AITEX/roi_candidates.json \
    --normal_dir      $AITEX_TILED/train/good \
    --output_dir      $SYN_RAND_AITEX \
    --top_k           200 \
    --seed            42 \
    --n_per_roi       3 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

**확인:** 두 출력 모두 `annotations.json` 항목 수 > 0, `images/` 파일이 256×256인지 스팟 체크.

---

## 6. 캐시/기존 결과 정리 (필수)

```python
# (a) yolo_cache의 aitex real 캐시 삭제 — 구(비타일·leaky split) 캐시가 카운트 일치로
#     우연히 valid 판정될 가능성을 원천 차단 (group-aware split은 총수를 안 바꾸므로).
!rm -rf $AROMA_OUT/yolo_cache/aitex

# (b) 기존 exp4v2 결과에서 aitex 항목 제거 — --resume skip 방지.
import json, os, glob
for f in [f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json"] + \
         glob.glob(f"{os.environ['EXP4V2_OUT']}/_seeds/seed*/exp4v2_results.json"):
    if os.path.exists(f):
        d = json.load(open(f))
        if 'aitex' in d:
            d.pop('aitex'); json.dump(d, open(f, 'w'), indent=2)
            print("aitex 항목 제거:", f)
```

> Colab `/tmp`는 세션마다 초기화되므로 구 staging 파일은 신경 쓸 필요 없음. 세션을 이어 쓰는 경우에만 `/tmp` 하위 exp4v2 캐시 디렉토리 삭제.

---

## 7. exp4v2 재실행 (aitex 단독, 3-seed)

**7-A. baseline 스모크 (1 seed) — 타일링 효과 판정 게이트:**

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition baseline \
    --dataset_keys aitex \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $AROMA_OUT/synthetic \
    --real_data_dir        $DRIVE \
    --output_dir           $EXP4V2_OUT/_aitex_smoke \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 256 \
    --val_frac 0.3 \
    --baseline_epochs 100 \
    --patience 30 \
    --batch 64 \
    --cache ram \
    --seed 1
```

> **imgsz 256** — 타일 원 해상도 1:1 (기존 640 rect는 타일에 불필요한 업스케일).
> **통과 기준**: baseline mAP50이 기존 0.066 대비 유의미하게 상승(문헌상 0.3~0.7 구간 기대). 로그에서 `group-aware split — N source groups` 와 `tiled bg-leak guard` 라인 확인 — 이 두 줄이 없으면 타일 마커가 유실된 것(§2 재확인).
> 여전히 ~0.1 이하면 aitex는 벤치마크 제외 결정 → 여기서 중단.

**7-B. 3-seed 전체 (3조건):**

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys aitex \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $AROMA_OUT/synthetic \
    --real_data_dir        $DRIVE \
    --output_dir           $EXP4V2_OUT \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
    --imgsz 256 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --baseline_epochs 300 \
    --patience 50 \
    --batch 64 \
    --cache ram \
    --seeds 1 2 43 \
    --resume
```

> exp4v2 seed 규약 `1 2 43` 유지. `--resume` 필수 (세션 한도 대비).
> 다른 데이터셋(severstal/leather/mtd)은 이 실행에 포함하지 않는다 — imgsz가 다르고(640 rect) 기존 결과 유지.

---

## 8. 결과 해석 시 유의점

1. **tile-level mAP**: 50% overlap로 동일 결함이 인접 타일 2~3개에 중복 계수됨. baseline/random/aroma 3조건 동일 적용이라 상대 비교는 공정하나, 절대값을 타 데이터셋과 직접 비교하지 말 것. 논문 표기 시 "tile-level" 명시.
2. **union_fallback 타일(70/352, ~20%)**: 점결함이 union bbox 1개로 라벨됨. exp4v2의 union-bbox fallback과 동일 의미론.
3. **합성 배경의 val-그룹 타일 잔존 leak(알려진 한계)**: baseline의 배경 네거티브는 val 그룹 bg 타일을 seed별로 필터하지만, random/aroma 합성 이미지는 seed-독립적으로 1회 생성되어 배경에 val 그룹 bg 타일이 포함될 수 있다(배경 텍스처 수준, 결함 픽셀 아님). 3조건 중 synth 두 조건에 동일 적용.
4. **기존(비타일) aitex 결과와 비교 불가** — 평가셋 자체가 다름. exp4v2 요약표에서 aitex 행은 이번 재실행 이후 값만 사용.
