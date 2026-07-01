# Compounding 실험 — AROMA ROI 선택 × CASDA 학습된 ControlNet 엔진 (Severstal)

**목적 (핵심 연구):** AROMA의 ROI 선택을 CASDA의 **실제 학습된 ControlNet 생성 엔진** 위에 얹어, *좋은 생성* 위에서도 ROI 선택이 효과가 있는지(직교 + compound) 검증한다.
**단일 인자 swap:** 3개 arm은 **입력 `roi_metadata.csv`만 다르고**, packager + ControlNet + Poisson compose는 완전히 동일하다.
- **casda arm** — CASDA 자신의 `roi_metadata.csv` (CASDA-suitability 선택)
- **aroma arm** — AROMA `roi_selected.json`(compatibility) → 호환 어댑터로 변환한 csv
- **random arm** — AROMA `roi_selected.json`(random) → 동일 어댑터로 변환한 csv

> **load-bearing 비교** = `aroma vs casda` (생성 엔진 동일, ROI 선택만 다름) 및 `aroma vs random`.
> dev_note: `.claude/.dev_note/aroma_research-core_thesis-and-compounding.md`.

> ⚠️ **이 가이드의 엔진 정의 (TASK 결정, 재논의 금지):** 엔진 = CASDA의 *실제 inference* = `StableDiffusionControlNetPipeline`(sd-controlnet-canny) + Poisson compose. 레포 in-repo `stage4_diffusion_synthesis.py`(ControlNet-**Inpaint**)가 **아니다.** CASDA 스크립트를 **수정하지 않고** 그대로 호출한다. dev_note ⑤의 "옵션 A(stage4 스텁 구현)"는 *대안* 설계였고, 본 실험은 그와 별개로 **학습된 CASDA ControlNet 충실 재현** 경로를 택한다.

---

## 스테이지별 런타임 배지

| § | 스테이지 | 런타임 | 비고 |
|---|---------|--------|------|
| 1 | AROMA ROI 선택 (compatibility / random) | 🟢 **CPU** | DL 없음, JSON 집계 |
| 2 | 어댑터 `aroma_to_casda_roi.py` (roi_selected → roi_metadata.csv + crops) | 🟢 **CPU** | crop/mask PNG 저장, DL 없음 |
| 3a | CASDA `prepare_controlnet_data.py` (hint + train.jsonl) | 🟢 **CPU** | Canny/hint 생성, DL 없음 |
| 3b | CASDA `test_controlnet.py` (ControlNet 생성) | 🔴 **GPU** | SD-v1.5 + ControlNet inference |
| 3c | CASDA Stage C Poisson compose | 🟢 **CPU** | `cv2.seamlessClone`(MIXED/NORMAL), DL 없음 |
| 4 | exp4v2 4-way detection (YOLOv8) | 🔴 **GPU** | Colab Pro A100 권장 |
| 5 | 결과 읽기 + 해석 | 🟢 **CPU** | JSON 집계 |

> §1·§2·§3a·§3c는 CPU 런타임에서 수행 가능. **§3b·§4만 GPU 런타임**. GPU 비용 절감을 위해 §3b 생성물(per-arm `generated/`, Poisson `synthetic`)은 **도메인·arm당 1회 생성·캐시**하고 §4 YOLO seed마다 재생성하지 않는다.

---

## 선행조건 / 주의 (반드시 먼저 확인)

1. **best_model 실재 (확인됨):** `controlnet_training_v5.5/best_model/`에 `config.json` + `diffusion_pytorch_model.safetensors`. `ControlNetModel.from_pretrained(BEST_MODEL)`로 로드. 없으면 전 단계 중단.
2. **도메인 = Severstal 한정:** 이 ControlNet은 **Severstal 강판에 튜닝**됐다. carpet/VisA 등 타 도메인은 zero-shot fidelity 미검증 → **본 실험은 severstal만**. 타 도메인 확장은 도메인별 **fidelity pilot(시각 검증)로 gate-in** 후 별도 진행(범위 밖).
3. **CASDA Stage C = `compose_casda_images.py` ✅ 확인됨** (Poisson, 출력 `images/`+`masks/`full-frame+`metadata.json`). **composed→exp4v2 변환은 §3d 셀로 해소.** + CASDA Severstal 데이터 루트(`DRIVE_SEVERSTAL=/content/drive/MyDrive/data/Severstal`, train_images/train.csv)는 AROMA 루트(`/content/drive/MyDrive/data/Aroma`)와 별개 — §0 env에 둘 다 설정.
4. **GPU 필수:** §3b(ControlNet 생성)·§4(YOLO 학습)는 CUDA 없이 실행 불가. §1·§2·§3a·§3c는 CPU 가능.
5. **두 Drive 루트 동시 마운트:** CASDA = `/content/drive/MyDrive/data/Severstal`, AROMA = `.../data/Aroma`. 둘 다 필요.
6. **CASDA 코어 무수정 원칙:** `prepare_controlnet_data.py` / `test_controlnet.py` / Stage C는 **그대로** 호출. 변환은 전적으로 어댑터(§2)에서 처리.
7. **CASDA 스크립트 CLI 미확정 플래그**는 본문에서 **⚠️ CLI 확인 필요**로 표기했다. 클론 후 `--help`로 확인하고 교체할 것 (플래그를 임의 발명하지 말 것).
8. **신규 .ipynb 금지** — 본 .md 셀을 Colab에 복사 실행. pytest 금지(Colab 직접 검증).
9. **부하/성능 실측 자동 금지(load-test 정책)** — mAP는 기능 검증 목적이라 허용, 생성 throughput 실측은 안 함.

---

## 0. 환경변수 (🟢 CPU)

```python
import os, json

# --- 저장소 클론 (CASDA / AROMA) ---
# !git clone --single-branch -b approve https://github.com/hhjun321/CASDA.git /content/CASDA
# !git clone <AROMA repo> /content/AROMA
os.environ['CASDA']         = "/content/CASDA"
os.environ['CASDA_SCRIPTS'] = "/content/CASDA/scripts"
os.environ['AROMA']         = "/content/AROMA"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"

# --- Drive 루트 (두 개가 다름에 주의) ---
os.environ['DRIVE']           = "/content/drive/MyDrive/data/Severstal"   # CASDA 측
os.environ['DRIVE_SEVERSTAL'] = os.environ['DRIVE']                       # 별칭(§3c compose에서 사용)
os.environ['AROMA_DATA']      = "/content/drive/MyDrive/data/Aroma"       # AROMA 측

# --- 학습된 ControlNet 체크포인트 (확인됨) ---
os.environ['BEST_MODEL'] = f"{os.environ['DRIVE']}/controlnet_training_v5.5/best_model"

# --- AROMA Stage 0/2 산출 (profiling + prompts, severstal) ---
os.environ['AROMA_OUT'] = f"{os.environ['AROMA_DATA']}/aroma_output"
os.environ['PROF']      = f"{os.environ['AROMA_OUT']}/profiling/severstal"
os.environ['PROM']      = f"{os.environ['AROMA_OUT']}/prompts/severstal"
os.environ['MORPH_CSV'] = f"{os.environ['PROF']}/morphology_features.csv"   # ⚠️ 실제 위치 확인

# --- AROMA ROI 선택 출력 (compatibility=aroma, random) ---
os.environ['ROI_AROMA']  = f"{os.environ['AROMA_OUT']}/roi_compound/severstal_aroma"
os.environ['ROI_RANDOM'] = f"{os.environ['AROMA_OUT']}/roi_compound/severstal_random"

# --- 어댑터 출력: CASDA 호환 roi_metadata.csv + crops ---
os.environ['ARM_ROOT']      = f"{os.environ['AROMA_OUT']}/compound_arms/severstal"
# CASDA 원본 Stage A roi_metadata.csv (읽기 전용 — capping 입력). 버전 폴더명 확인:
#   !ls {DRIVE} | grep roi_patches   (예: roi_patches_v5.1)
os.environ['CSV_CASDA_FULL'] = f"{os.environ['DRIVE']}/roi_patches_v5.1/roi_metadata.csv"
# capped 출력 (3 arm 모두 ARM_ROOT 아래 — 원본 절대 덮어쓰지 않음)
os.environ['CSV_CASDA']     = f"{os.environ['ARM_ROOT']}/casda/casda_roi_metadata_capped.csv"
os.environ['CSV_AROMA']     = f"{os.environ['ARM_ROOT']}/aroma/aroma_roi_metadata.csv"
os.environ['CSV_RANDOM']    = f"{os.environ['ARM_ROOT']}/random/random_roi_metadata.csv"
os.environ['CROPS_AROMA']   = f"{os.environ['ARM_ROOT']}/aroma/crops"
os.environ['CROPS_RANDOM']  = f"{os.environ['ARM_ROOT']}/random/crops"

# --- per-arm ControlNet dataset / generated / Poisson 합성 출력 ---
# CN_ROOT(hints + train.jsonl + generated)는 Drive에 — CPU(§3a)→GPU(§3b) 세션 분리 시
# 산출물 영속 필요(세션 재기동하면 /content 로컬은 소실). prepare가 느리면 --workers -1로 완화.
os.environ['CN_ROOT']     = f"{os.environ['AROMA_OUT']}/compound_cn/severstal"
os.environ['SYN_CASDA']   = f"{os.environ['AROMA_OUT']}/compound_synth/severstal_casda"
os.environ['SYN_AROMA']   = f"{os.environ['AROMA_OUT']}/compound_synth/severstal_aroma"
os.environ['SYN_RANDOM']  = f"{os.environ['AROMA_OUT']}/compound_synth/severstal_random"

# --- exp4v2 출력 ---
os.environ['EXP4_OUT']    = f"{os.environ['AROMA_OUT']}/exp4v2_compound"

SEED = "42"
os.environ['SEED'] = SEED
for k in ['BEST_MODEL','MORPH_CSV','CSV_CASDA_FULL','CSV_CASDA','CSV_AROMA','CSV_RANDOM','EXP4_OUT']:
    print(f"{k:12s} = {os.environ[k]}")

# best_model 실재 확인 (없으면 즉시 중단)
from pathlib import Path
bm = Path(os.environ['BEST_MODEL'])
ok = (bm/'config.json').exists() and any(bm.glob('*.safetensors'))
print("\nbest_model OK:", ok, "—", "진행 가능" if ok else "❌ 체크포인트 없음, 중단")
```

> 전제: AROMA Stage 0(profiling) + Stage 2(prompts)가 severstal에 존재(`$PROF`, `$PROM`). `morphology_features.csv`는 profiling 산출 — 실제 경로 확인 후 `$MORPH_CSV` 교체.

---

## 1. AROMA ROI 선택 — compatibility(aroma) + random  · 🟢 CPU

severstal은 multi-class(`--class_mode multi --class_floor`). method_pivot 가이드(`method_pivot_rerun_execute.md` §1)와 동일 패턴, 여기선 severstal 단일.

```python
import sys, subprocess
ROI_SEL = f"{os.environ['AROMA_SCRIPTS']}/roi_selection.py"

def select(strategy, out_dir):
    cmd = [sys.executable, ROI_SEL,
           "--profiling_dir", os.environ['PROF'],
           "--prompts_dir",   os.environ['PROM'],
           "--output_dir",    out_dir,
           "--sampling_strategy", strategy,
           "--top_k", "1690",            # MAX_ROIS=1690 도달용 (최근 severstal roi_selected와 동일; random도 1690 선택)
           "--seed", os.environ['SEED'],
           "--class_mode", "multi", "--class_floor",
           "--background_type", "complex"]   # severstal context
    if strategy == "compatibility":
        cmd += ["--img_diversity_cap", "1", "--min_quality", "0.0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = '\n'.join((r.stderr or '').splitlines()[-4:])
    print(f"{'✓' if r.returncode==0 else '✗'} {strategy:13s} -> {out_dir}")
    if r.returncode != 0: print(tail)
    return r.returncode == 0

select("compatibility", os.environ['ROI_AROMA'])   # aroma arm
select("random",        os.environ['ROI_RANDOM'])  # random arm
# → 각 출력 디렉터리에 roi_selected.json 생성
```

> roi_selection.py 확정 플래그(코드 확인): `--profiling_dir --prompts_dir --output_dir --sampling_strategy{deficit_aware,compatibility,top_k,weighted,random} --top_k --seed --class_mode{single,multi} --class_floor --img_diversity_cap --min_quality --background_type{smooth,directional,periodic,organic,complex} --rarity_temp --per_pair_cap_frac --nc`.

---

## 2. 어댑터 (NEW) `aroma_to_casda_roi.py` — roi_selected.json → CASDA roi_metadata.csv  · 🟢 CPU

> **신규 스크립트.** `scripts/aroma/aroma_to_casda_roi.py`로 작성한다(아래 *어댑터 사양* 충족). 기존 `casda_roi_adapter.py`는 **역방향**(CASDA→AROMA)이므로 재사용 불가 — 별개 파일.

### 2.1 어댑터 사양 (CASDA packager 동작에 정합)

CASDA `prepare_controlnet_data.py` → `ControlNetDatasetPackager.package_single_roi`는 csv의 **필드에서 hint·prompt를 재생성**한다. csv의 `prompt` 텍스트는 무시·재생성된다. 따라서 어댑터는 다음 **필드**를 채우고 **crop-aligned** `roi_image_path`/`roi_mask_path`를 만들면 된다.

**CASDA roi_metadata.csv 헤더 (21열, 순서 정확히 동일하게 출력):**
```
image_id,class_id,region_id,roi_bbox,defect_bbox,centroid,area,linearity,solidity,extent,aspect_ratio,defect_subtype,background_type,suitability_score,matching_score,continuity_score,stability_score,recommendation,prompt,roi_image_path,roi_mask_path
```

**필드 매핑 (AROMA roi_selected.json[+morphology_features.csv] → CASDA 열):**

| CASDA 열 | 출처 / 산출 |
|----------|-------------|
| `image_id` | roi_selected `image_id` |
| `class_id` | roi_selected `class_key`("class1"→1, "class2"→2, …) 또는 `defect_type`. 파싱 실패 시 1 |
| `region_id` | 동일 image_id 내 0-base 시퀀스 |
| `roi_bbox` | crop 영역 = `defect_bbox`에 약간 패딩한 ROI 사각형. **문자열 튜플** `"(x1, y1, x2, y2)"` (절대좌표) |
| `defect_bbox` | roi_selected `defect_bbox`는 **[x,y,w,h]** → **(x, y, x+w, y+h)** 로 변환 후 **문자열 튜플** `"(x1, y1, x2, y2)"` |
| `centroid` | mask 무게중심 또는 bbox 중심 `"(cx, cy)"` |
| `area` | mask 픽셀 수(int). mask 없으면 `w*h` |
| `linearity,solidity,extent,aspect_ratio` | `morphology_features.csv`를 image_id로 join (CASDA hint/prompt가 이 값을 소비) |
| `defect_subtype` | roi_selected `defect_subtype` (CASDA PromptGenerator 입력) |
| `background_type` | **`complex_pattern`** 고정 (CASDA severstal 라벨; hint G/B 채널 로직과 정합) — `--background_type` 인자로 받음 |
| `suitability_score` | roi_selected `roi_score`(0~1 클램프). 없으면 `quality_score` |
| `matching_score` | roi_selected `quality_score`(0~1). 없으면 0.5 |
| `continuity_score,stability_score` | 보유 시 매핑, 없으면 중립 0.5 (hint stability 채널이 소비) |
| `recommendation` | `"acceptable"` 고정(또는 suitability 임계 기반) |
| `prompt` | **빈 문자열 또는 placeholder** — packager가 재생성하므로 무관 |
| `roi_image_path` | **defect crop PNG** 절대경로 (아래 `--make_crops`로 생성) |
| `roi_mask_path` | **crop-aligned mask PNG** 절대경로 (roi_image와 **동일 크기**) |

**crop 생성 규칙 (`--make_crops`):** 원본 `image_path`에서 `roi_bbox`로 잘라 `roi_image_path`에 저장하고, `defect_mask_path`(있으면)를 동일 `roi_bbox`로 잘라 `roi_mask_path`에 저장한다. mask가 없으면 defect_bbox 사각형을 채운 binary mask를 crop 크기로 생성. **roi_image와 roi_mask는 반드시 동일 H×W** (CASDA `package_single_roi`가 crop-aligned 배열을 가정).

### 2.2 어댑터 CLI

> ⚠️ **arm 균등화 + GPU 비용 통제 (필수):** roi_selected는 arm마다 수가 다르다 (severstal AROMA multi-class floor=1690, random=200, CASDA csv=수천). 생성 수가 arm마다 다르면 **불공정**(단일인자 swap 무효) + ControlNet 추론 **폭증**(ROI×compositions). → **세 arm 모두 동일 `MAX_ROIS`로 cap.** 어댑터 `--max_rois N`은 class-stratified round-robin(rare 클래스 균형, 결정론적)으로 cap. CASDA arm csv도 동일 N으로 stratified 샘플.

```python
ADAPTER = f"{os.environ['AROMA_SCRIPTS']}/aroma_to_casda_roi.py"   # NEW
MAX_ROIS = 1383  # 생성 풀 cap (보고 하이퍼파라미터 아님 — 아래 reviewer 노트 참조).
                 # = 가장 작은 조건(CASDA 가용 ROI=1383)에 맞춘 공통 풀 → 엄격 per-arm parity.
                 # aroma(1690)·random(1690) → 1383 trim, casda 전량. compose ×1 → 풀 1383.
                 # 1383 ≥ 1013(학습 budget) → exp4v2 --synth_ratio 0.4 가 1013으로 trim.
                 # (CASDA 가용 ROI 수가 바뀌면 그 값으로 재설정.)

def adapt(roi_selected, output_csv, crops_dir):
    cmd = [sys.executable, ADAPTER,
           "--roi_selected",   roi_selected,
           "--morphology_csv", os.environ['MORPH_CSV'],
           "--output_csv",     output_csv,
           "--crops_dir",      crops_dir,
           "--background_type","complex_pattern",
           "--max_rois",       str(MAX_ROIS),
           "--make_crops"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"{'✓' if r.returncode==0 else '✗'} adapt -> {output_csv}")
    if r.returncode != 0: print('\n'.join((r.stderr or '').splitlines()[-6:]))
    return r.returncode == 0

# aroma arm (compatibility 선택)
adapt(f"{os.environ['ROI_AROMA']}/roi_selected.json",
      os.environ['CSV_AROMA'], os.environ['CROPS_AROMA'])
# random arm
adapt(f"{os.environ['ROI_RANDOM']}/roi_selected.json",
      os.environ['CSV_RANDOM'], os.environ['CROPS_RANDOM'])

# casda arm: CASDA 원본 roi_metadata.csv(CSV_CASDA_FULL, 읽기전용)를 동일 MAX_ROIS로
#            class-stratified 샘플 → CSV_CASDA(capped, 별도 경로). 원본 안 건드림.
import pandas as pd
from pathlib import Path
Path(os.environ['CSV_CASDA']).parent.mkdir(parents=True, exist_ok=True)
_cd = pd.read_csv(os.environ['CSV_CASDA_FULL'])   # CASDA Stage A 원본 (읽기전용)
if len(_cd) > MAX_ROIS:
    _per = max(1, MAX_ROIS // _cd['class_id'].nunique())
    _cd = (_cd.groupby('class_id', group_keys=False)[_cd.columns.tolist()]
              .head(_per)                         # 선택순 상위 per-class (결정론적, 경고 없음)
              .head(MAX_ROIS))
# ⚠️ 패키징 필터 3종(edge / area / quality) 3 arm 균일 무력화 —
#    어댑터가 aroma/random에 적용하는 pass-through(roi_bbox==defect_bbox로 edge skip,
#    area 큼, score 0.7)를 casda에도 동일 적용. 안 하면 casda만 edge+area 필터로
#    1383→465 컷 → synth cap(1013) 미달 → parity 붕괴. (선택은 각 arm 상류서 확정;
#    패키징 필터는 generation 엔진 아님 → 균일 무력화가 단일인자 swap 정합.)
_cd = _cd.copy()
_cd['roi_bbox']        = _cd['defect_bbox']                 # edge filter skip (aroma와 동일)
_cd['area']            = _cd['area'].clip(lower=100)        # area>=100 통과
_cd['stability_score'] = 0.7                                # quality_filter 통과
_cd['matching_score']  = 0.7
_cd['recommendation']  = 'acceptable'
_cd['background_type'] = 'complex_pattern'                  # compose bg-matching 균일 통과
                                                           #   (안 하면 compose가 casda 다양 bg-type drop: 1152→651)
_cd.to_csv(os.environ['CSV_CASDA'], index=False)
print(f"casda arm: {os.environ['CSV_CASDA_FULL']} → {len(_cd)} rows (cap {MAX_ROIS}, "
      f"edge/area/quality 무력화) → {os.environ['CSV_CASDA']}")
```

> **검증(어댑터 산출):** `pd.read_csv(CSV_AROMA)`의 컬럼이 §2.1 헤더 21열과 **순서까지 동일**한지, `roi_image_path`/`roi_mask_path` 파일이 실재하고 **동일 크기**인지 1~2개 샘플 확인. py_compile은 작성 후 `python -m py_compile scripts/aroma/aroma_to_casda_roi.py`로 로컬 점검(테스트 코드 작성 금지, Colab 직접 검증).

---

## 3. arm별 CASDA 파이프라인 (casda / aroma / random)  · 🟢/🔴 혼합

3개 arm은 **입력 csv만 다르고** packager + ControlNet + Poisson은 동일하다(단일 인자 swap).

```python
ARMS = {
    "casda":  os.environ['CSV_CASDA'],
    "aroma":  os.environ['CSV_AROMA'],
    "random": os.environ['CSV_RANDOM'],
}
SYN = {"casda": os.environ['SYN_CASDA'],
       "aroma": os.environ['SYN_AROMA'],
       "random": os.environ['SYN_RANDOM']}
```

### 3a. `prepare_controlnet_data.py` — hint + train.jsonl  · 🟢 CPU

CASDA packager가 csv를 읽어 hint 이미지 + `train.jsonl`을 생성. **CASDA 코어 무수정.**

```python
PREP = f"{os.environ['CASDA_SCRIPTS']}/prepare_controlnet_data.py"

def prepare(arm):
    out = f"{os.environ['CN_ROOT']}/{arm}"
    cmd = [sys.executable, PREP,
           "--roi_metadata", ARMS[arm],
           "--output_dir",   out,
           # ✅ 확인됨: train_images/train_csv default가 상대경로('train_images'/'train.csv')
           #    → 미전달 시 없음 → 실패. Severstal 절대경로 전달.
           "--train_images", f"{os.environ['DRIVE']}/train_images",
           "--train_csv",    f"{os.environ['DRIVE']}/train.csv",
           "--skip_validation",   # ROI 추출 검증은 본 실험 무관(packaging만 필요)
           "--workers", "-1"]     # ⚡ default 0=순차(느림) → -1 자동 병렬. 핵심 속도 레버.
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"{'✓' if ok else '✗'} prepare[{arm}] -> {out}/train.jsonl")
    if not ok:
        print("STDERR tail:\n" + '\n'.join((r.stderr or '').splitlines()[-12:]))
        print("STDOUT tail:\n" + '\n'.join((r.stdout or '').splitlines()[-8:]))
    return out

CN = {arm: prepare(arm) for arm in ARMS}
# → 각 $CN_ROOT/{arm}/ 아래 hints + train.jsonl

### 3a.5 — (비활성, 결정 (i)) AROMA step2 prompt patch 미사용

> ⛔ **이 단계는 실행하지 않는다.** smoke 결과: AROMA step2 자유서술 prompt를 트레인된 ControlNet(CASDA-prompt 학습)에 넣으니 **OOD → 생성 비현실**(A/B 확인: 동일 hint에 CASDA식 prompt는 그럴듯, AROMA step2 prompt는 깨짐).
>
> **결정 (i)**: prompt patch 제거 → **packager가 재생성한 CASDA식 prompt 사용**. 이 prompt는 AROMA의 `defect_subtype`(형태 분류) + background_type로 CASDA 템플릿으로 생성 → **in-distribution + AROMA 형태결정 반영**("AROMA-informed prompt").
>
> **train.jsonl이 이미 patch됐다면 §3a prepare를 재실행**(packager 재생성 prompt로 깨끗한 jsonl 복원)한 뒤 이 셀을 건너뛰고 §3b로 진행. 어댑터 csv의 `prompt` 컬럼(AROMA step2 prompt)은 packager가 무시하므로 무해(prompt-ablation 시 참조용).
>
> 📝 논문: AROMA Stage2 prompt-generation은 **별도 컴포넌트로 기술**하되, ControlNet 생성은 트레인 모델 분포에 맞춘 in-distribution rendering(AROMA subtype → CASDA 템플릿)을 사용함을 명시.
```

> ⚠️ **CLI 확인 필요:** `prepare_controlnet_data.py`의 정확한 플래그(`--roi_metadata` vs `--roi_csv`, `--output_dir` vs `--out`)는 클론 후 `!python $PREP --help`로 확인 후 교체.

### 3b. `test_controlnet.py` — ControlNet 생성  · 🔴 GPU

학습된 best_model로 hint→결함 이미지 생성. `StableDiffusionControlNetPipeline`(canny), 30 steps, conditioning_scale 0.7, res 512.

> ⚠️ **세션 재기동(CPU→GPU) 시 필수**: §0 env 셀을 **먼저 재실행**한 뒤, 아래 재초기화 셀로 `CN/ARMS/SYN/GEN` 변수를 env에서 재구성. (§1~§3a.5는 산출물이 Drive에 영속하므로 재실행 불요 — train.jsonl/hints 존재만 확인.)

```python
# === GPU 세션 재초기화 (§0 env 재실행 후) — §3a 메모리 변수 복원 ===
import os, subprocess, sys, glob, json
arms = ['casda', 'aroma', 'random']
ARMS = {'casda':  os.environ['CSV_CASDA'],
        'aroma':  os.environ['CSV_AROMA'],
        'random': os.environ['CSV_RANDOM']}
CN   = {a: f"{os.environ['CN_ROOT']}/{a}"  for a in arms}      # §3a prepare 산출(Drive)
GEN  = {a: f"{CN[a]}/generated"            for a in arms}
SYN  = {'casda':  os.environ['SYN_CASDA'],
        'aroma':  os.environ['SYN_AROMA'],
        'random': os.environ['SYN_RANDOM']}
# 영속 산출 확인 (재기동 후에도 Drive에 있어야 함)
for a in arms:
    jl = f"{CN[a]}/train.jsonl"
    n  = sum(1 for _ in open(jl, encoding='utf-8')) if os.path.exists(jl) else 'MISSING'
    h  = len(glob.glob(f"{CN[a]}/hints/*.png"))
    print(f"{a:7} jsonl={n}  hints={h}")
# GPU 확인
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

#### 3b-smoke. aroma 4장 먼저 (best_model 로드 + AROMA-prompt 생성품질 확인)

전량 전에 aroma 4개만 생성 → best_model 로드·파이프라인·OOD prompt 영향 확인.

```python
import json, subprocess, sys, os, glob
TEST = f"{os.environ['CASDA_SCRIPTS']}/test_controlnet.py"

src  = f"{CN['aroma']}/train.jsonl"
mini = f"{CN['aroma']}/train_smoke.jsonl"
lines = [l for l in open(src, encoding='utf-8') if l.strip()][:4]
open(mini, 'w', encoding='utf-8').writelines(lines)
print("smoke:", len(lines), "entries")
for l in lines[:2]:
    e = json.loads(l); print("  prompt:", e.get('prompt','')[:90])

out = f"{CN['aroma']}/generated_smoke"
r = subprocess.run([sys.executable, TEST,
    "--model_path", os.environ['BEST_MODEL'],
    "--jsonl_path", mini,
    "--output_dir", out,
    "--num_inference_steps","30",
    "--guidance_scale","7.5",
    "--controlnet_conditioning_scale","0.7",
    "--resolution","512"], capture_output=True, text=True)
print("rc:", r.returncode)
print('STDOUT tail:\n' + '\n'.join((r.stdout or '').splitlines()[-15:]))
if r.returncode != 0:
    print('STDERR tail:\n' + '\n'.join((r.stderr or '').splitlines()[-20:]))
```

```python
# 생성물 육안 (AROMA prompt OOD 영향: 결함이 그럴듯한가?)
from IPython.display import Image as IPImage, display
gens = sorted(glob.glob(f"{out}/**/*.png", recursive=True))
print("generated:", len(gens))
for g in gens[:4]:
    print(g); display(IPImage(g))
```

> smoke OK(rc=0 + 결함 그럴듯)면 아래 전량 셀 실행. rc≠0이면 STDERR 보고 — best_model 경로/diffusers 버전/hint 경로 resolution 점검.

#### 3b-full. 3 arm 전량 생성

```python
TEST = f"{os.environ['CASDA_SCRIPTS']}/test_controlnet.py"

def generate(arm):
    out = f"{CN[arm]}/generated"
    cmd = [sys.executable, TEST,
           "--model_path",                   os.environ['BEST_MODEL'],
           "--jsonl_path",                   f"{CN[arm]}/train.jsonl",
           "--output_dir",                   out,
           "--num_inference_steps",          "30",
           "--guidance_scale",               "7.5",
           "--controlnet_conditioning_scale","0.7",
           "--resolution",                   "512"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"{'✓' if r.returncode==0 else '✗'} generate[{arm}] -> {out}")
    if r.returncode != 0: print('\n'.join((r.stderr or '').splitlines()[-8:]))
    return out

GEN = {arm: generate(arm) for arm in ARMS}
```

> ✅ **CLI 확인됨** (CASDA 04-StageB Step5): `--model_path --jsonl_path --output_dir --num_inference_steps --guidance_scale --controlnet_conditioning_scale --resolution`. (클래스별 생성 수 차등 원하면 `--num_images_per_class '{"1":2,"2":10,"3":1,"4":2}'`.) 모델은 `ControlNetModel.from_pretrained(BEST_MODEL)`(`config.json`+`*.safetensors`).
> **synth budget cap:** 어댑터 `--max_rois 1690` (class-stratified) + compose `--compositions-per-roi 1` → arm당 생성 풀 = **1690**. exp4v2 `--synth_ratio 0.4`가 학습 시 **1013**으로 trim(최근 run 정합). 1690 생성은 1회·캐시 후 §4 재사용. (pool이 1013보다 커야 trim 가능 — 1690 OK.)

### 3c. Stage C — Poisson compose  · 🟢 CPU

생성된 결함을 severstal good 배경에 Poisson 합성, `annotations.json`(exp4v2 호환) 생성.

```python
# ✅ 확인됨: CASDA Stage C = compose_casda_images.py (Poisson Blending, CASDA 04-StageC Step6)
COMPOSE = f"{os.environ['CASDA_SCRIPTS']}/compose_casda_images.py"
GOOD_BG = f"{os.environ['DRIVE_SEVERSTAL']}/train_images"       # CASDA clean-images (good 강판)
TRAIN_CSV = f"{os.environ['DRIVE_SEVERSTAL']}/train.csv"

def compose(arm):
    cmd = [sys.executable, COMPOSE,
           "--generated-dir",    GEN[arm],                          # test_controlnet 출력 /generated
           "--hint-dir",         f"{CN[arm]}/hints",
           "--metadata-csv",     f"{CN[arm]}/packaged_roi_metadata.csv",
           "--summary-json",     f"{GEN[arm]}/generation_summary.json",
           "--clean-images-dir", GOOD_BG,
           "--train-csv",        TRAIN_CSV,
           "--output-dir",       f"{SYN[arm]}/severstal",
           "--workers",          "8",
           "--compositions-per-roi", "3"]   # ROI당 3합성(다른 bg) → 풀 = gen×3 (casda 3456/aroma 4134/random 4140)
                                            #   → 모두 ≥2534 = ratio 1.0 cap 커버 → synth_ratio 0.2~1.0 sweep 가능.
                                            #   compose=CPU Poisson(저렴), generated 재사용(GPU regen 불요).
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"{'✓' if r.returncode==0 else '✗'} compose[{arm}] -> {SYN[arm]}/severstal")
    if r.returncode != 0: print('\n'.join((r.stderr or '').splitlines()[-8:]))

for arm in ARMS: compose(arm)
# → 각 $SYN[arm]/severstal/ (CASDA composed 형식)
```

> ✅ **스크립트 확인됨**: `compose_casda_images.py` (하이픈 플래그: `--generated-dir --hint-dir --metadata-csv --summary-json --clean-images-dir --train-csv --output-dir --workers --bg-cache --compositions-per-roi`).
> ✅ **변환 해소(§3d)**: `compose_casda_images.py` 출력(`images/`+`masks/`full-frame+`metadata.json`)을 §3d 셀이 exp4v2 `annotations.json`으로 변환. mask_path=full-frame mask, source_roi/cluster_id=파일명 `_class{N}_`. §3→§4 연결 완성.

### 3d. composed → exp4v2 annotations 변환  · 🟢 CPU  (§3→§4 연결, 갭 해소)

> 💾 **저장 정책**: SYN(=compose 합성 결과: `images/`+`masks/`+`annotations.json`)은 **반드시 Drive**(`SYN_*`=AROMA_OUT 하위). 이 합성셋은 **향후 성능비교(exp4v2 multi-seed/ratio sweep 등)서 반복 재사용**되므로 영속 필수. annotations.json의 `image_path`/`mask_path`는 Drive 절대경로 → 세션 넘어 유효. (§3b `generated/`만 로컬 중간물.) compose 출력이 로컬이면 안 됨 — §3c `--output-dir`이 `{SYN[arm]}/severstal`(Drive)인지 확인.

`compose_casda_images.py` 출력 = `{images/, masks/(full-frame 1600×256), metadata.json}`, 파일명에 `_class{N}_`. exp4v2 `_load_synth_annotations`는 `{synth_root}/severstal/annotations.json`(list, 각 `image_path`/`mask_path`(full-frame)/`source_roi`(class{N} 포함)/`cluster_id`(1-4)/`dry_run`)를 읽음. 아래로 변환:

```python
import json, re, glob
from pathlib import Path

def composed_to_exp4v2(composed_dir, syn_out):
    """CASDA composed({images,masks}) → exp4v2 {severstal/annotations.json}."""
    imgs = sorted(glob.glob(f"{composed_dir}/images/*.png"))
    ann = []
    for ip in imgs:
        name = Path(ip).name
        m = re.search(r"_class(\d+)", name)          # exp4v2 source_roi/cluster 라벨
        cid = int(m.group(1)) if m else 1
        mp = f"{composed_dir}/masks/{name}"
        ann.append({
            "image_path": ip,
            "mask_path":  mp if Path(mp).exists() else None,  # full-frame → exp4v2 bbox 유도
            "source_roi": name,        # class{N} 포함 → _parse_severstal_class 매칭
            "cluster_id": cid,         # 1-4 fallback
            "dry_run": False,
        })
    out = Path(syn_out) / "severstal"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(ann, open(out / "annotations.json", "w"), indent=2)
    print(f"  {composed_dir} -> {out}/annotations.json  ({len(ann)} entries)")
    return str(out)

for arm in ARMS:
    composed_to_exp4v2(f"{SYN[arm]}/severstal", SYN[arm])
# → 각 $SYN[arm]/severstal/annotations.json (exp4v2 호환)
```

> ✅ mask_path = compose의 full-frame mask → exp4v2가 bbox 유도. source_roi 파일명 `_class{N}_` → class 라벨. cluster_id = 동일값 fallback. **이로써 §3→§4 연결 완성.**

### 3 단계 무결성 (single-factor swap)

- 3개 arm은 **동일** packager(`prepare_controlnet_data.py`) + **동일** best_model + **동일** Stage C Poisson을 통과. 차이는 **입력 roi_metadata.csv 1개뿐**.
- **seed 처리:** 생성·compose seed를 3 arm 동일(`$SEED`)로 고정. 동일 (defect,bg) 조건이 arm마다 다른 latent로 디노이즈되지 않도록 seed 결정성 유지(dev_note ④ 치명결함2 — 본 경로는 CASDA inference를 그대로 쓰므로 CASDA seed 처리에 위임, 결과의 arm간 재현성을 §5 negative control로 점검).
- **생성·캐시 1회:** §3b·§3c 산출은 도메인·arm당 1회 만들어 Drive에 캐시. §4 YOLO seed(42/1/2)마다 재생성 금지.

---

## 4. exp4v2 — Severstal 4-way detection  · 🔴 GPU

exp4v2는 condition-agnostic(synthetic dir만 교체) + **model-agnostic**. 4 arm = baseline / random / casda / aroma. **`--model all` = yolov8n/s/m 3사이즈** (동일 synth 재사용, detector만 재학습 → multi-architecture robustness). seeds 42 1 2.

> **multi-model 근거**: 단일 detector(nano)로는 "증강 효과 일반화 + 아키텍처 의존성" 미검증 → reviewer 취약. synth는 arm당 1회 생성·재사용이므로 **n/s/m 추가는 detector 학습 반복만**(저비용). 계열내 스케일 robustness 확보. (더 강한 주장엔 타계열 detector 1개 추가 권장 — RT-DETR 등, 별도 통합.)

```python
os.environ['EXP4'] = f"{os.environ['AROMA_SCRIPTS']}/experiments/exp4_v2_supervised_detection.py"
print("EXP4 =", os.environ['EXP4'])   # env var 여야 !python 에서 치환됨

# ratio sweep: 합성 풀은 1회 생성(§3b~§3d, ≥2534)·재사용, exp4v2 학습만 ratio별 반복.
# 각 ratio는 별도 --output_dir (덮어쓰기 방지). 비용: ratio당 model(3)×cond(4)×seed(3).
for ratio in ["0.2", "0.4", "0.7", "1.0"]:
    out = f"{os.environ['EXP4_OUT']}_r{ratio}"
    print(f"\n===== synth_ratio {ratio} → {out} =====")
    !python $EXP4 \
        --model all \
        --condition baseline random casda aroma \
        --dataset_keys severstal \
        --class_mode multi \
        --random_synthetic_dir $SYN_RANDOM \
        --casda_synthetic_dir  $SYN_CASDA \
        --aroma_synthetic_dir  $SYN_AROMA \
        --real_data_dir        $AROMA_DATA \
        --output_dir           "$out" \
        --seeds 42 1 2 \
        --synth_ratio {ratio} \
        --val_frac 0.3 \
        --imgsz 640 \
        --baseline_epochs 100 \
        --patience 20 \
        --batch 64 --cache ram --rect \
        --yolo_cache_dir $AROMA_OUT/yolo_cache \
        --resume
```

> **ratio sweep 설계**: 합성 풀(compose ×3 → 각 arm ≥2534)은 **1회 생성·재사용**. exp4v2가 `--synth_ratio`별로 cap=int(2534×ratio)만큼 **각 arm에서 seeded subsample** → 모든 ratio·모든 arm 동일 cap = parity. ratio당 별도 `_r{ratio}` output. baseline(synth 무관)은 ratio 간 동일 → 첫 ratio 후 `--resume`이 skip(중복 학습 방지). **비용 급증** 주의: 4 ratio × 3 model × 4 cond × 3 seed. 처음엔 ratio 1개(예 0.4) + `--model yolov8n`로 smoke.

> ⚠️ **비용 급증 주의**: `--model all`(3) × 4 condition × 3 seed = **36 학습 run** (baseline은 model×seed=9, 나머지는 synth 공유). GPU 시간 3배(단일 model 대비). 처음엔 `--model yolov8n --seeds 42`로 smoke 후 `--model all --seeds 42 1 2` 전량 권장. yolo_cache로 real 데이터 build-once 재사용.

> **확정 플래그(exp4v2, 코드 확인):** `--condition`(nargs+), `--dataset_keys`, `--class_mode{single,multi}`, `--random_synthetic_dir/--casda_synthetic_dir/--aroma_synthetic_dir`, `--real_data_dir`, `--output_dir`, `--seeds`, `--synth_ratio`, `--val_frac`, `--imgsz`, `--baseline_epochs`, `--patience`, `--batch/--cache/--rect`, `--yolo_cache_dir`, `--resume`. (exp4v2_execute.md 참조 — 이 스크립트는 AROMA 레포 소속이라 CLI 확정.)
>
> **synth_ratio / pool parity (load-bearing):** cap = max(1, int(n_real_train×ratio)). 각 arm composed 풀 ≥ cap이어야 3조건 등가 trim. **casda background_type=complex_pattern 균일화** 후 composed 풀 = casda ~1152 / aroma 1378 / random 1379 → **셋 다 ≥ 1013** → **`--synth_ratio 0.4`** (cap 1013, 최근 cleanbg 0.4 run 정합) → 3 arm 1013 균일 trim → `n_synth_train` 동일(§5 parity 게이트 **필수 확인**).
>
> ⚠️ casda 균일화 안 하면 compose가 casda 다양 background_type을 drop(1152→651) → 0.4 불가. cap셀 `background_type='complex_pattern'`(또는 packaged csv 패치+recompose)로 해결. bg 선택만 균일화(hints/generated는 실제 bg-type로 생성됨 — 최종 clean본은 cap셀부터 complex_pattern로 regen 권장).
>
> `--resume`은 기존 severstal 결과 skip → 재실행 시 삭제 또는 fresh `--output_dir`.
>
> 📝 **Reviewer 방어 — "왜 1383?"**: 1383은 **보고 하이퍼파라미터가 아니다.** 보고하는 증강 budget = **synth:real = 0.4 (조건당 1013장)**, 전 조건 동일 + 이전 실험과 동일. 1383은 조건별 **생성 풀** 크기로, **가장 작은 조건(CASDA 가용 ROI)에 맞춰** 세 조건 풀을 동일화한 값(엄격 parity, CASDA ROI 전량 사용). 1383 ≥ 1013이라 학습 budget을 충분히 커버. 논문 본문엔 budget=ratio(0.4/1013)만, 1383은 부록/구현 노트로.
>
> ⚠️ **annotations.json 호환:** `--*_synthetic_dir` 아래 `severstal/annotations.json`이 exp4v2 스키마로 존재해야 함(§3c). 없으면 해당 arm `no_synth_annotations` 거부.

---

## 5. 결과 읽기 + 해석  · 🟢 CPU

```python
import json, os
with open(f"{os.environ['EXP4_OUT']}/exp4v2_results.json") as f:
    results = json.load(f)

CONDS = ["baseline", "random", "casda", "aroma"]
arms = results["severstal"]["yolov8n"]

# (1) map50 mean ± std + 95% CI + n_synth_train
print(f"{'arm':<10}{'map50 (mean±std)':>22}{'95% CI':>22}{'n_synth':>10}")
print("-"*64)
for c in CONDS:
    cell = arms.get(c, {})
    m  = cell.get("map50"); sd = (cell.get("std") or {}).get("map50")
    ci = (cell.get("ci95") or {}).get("map50"); ns = cell.get("n_synth_train")
    ms  = f"{m:.4f} ± {sd:.4f}" if isinstance(m,float) and isinstance(sd,float) else (f"{m:.4f}" if isinstance(m,float) else "N/A")
    cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci,list) and ci[0] is not None else "N/A"
    print(f"{c:<10}{ms:>22}{cis:>22}{str(ns):>10}")

# (2) 공정성 parity: random/casda/aroma의 n_synth_train 동일?
ns = {c: arms.get(c,{}).get("n_synth_train") for c in ("random","casda","aroma")}
uniq = set(v for v in ns.values() if v is not None)
print(f"\nn_synth_train parity {ns} -> {'OK' if len(uniq)<=1 else '⚠️ 불일치(cap 누락 의심)'}")

# (3) load-bearing delta
def md(a,b):
    x,y = arms.get(a,{}).get("map50"), arms.get(b,{}).get("map50")
    return (x-y)*100 if isinstance(x,float) and isinstance(y,float) else None
d_ac, d_ar = md("aroma","casda"), md("aroma","random")
print(f"\nΔ(aroma - casda)  = {d_ac:+.2f}pp   [핵심: 생성 동일, 선택만 다름]" if d_ac is not None else "Δ(aroma-casda)=N/A")
print(f"Δ(aroma - random) = {d_ar:+.2f}pp" if d_ar is not None else "Δ(aroma-random)=N/A")

# (4) per-class AP (희소 클래스 c2/c3/c4에서 선택 효과 노출)
classes = sorted({c for cond in CONDS for c in (arms.get(cond,{}).get("per_class") or {})})
print(f"\n{'class':<6}" + "".join(f"{c:>10}" for c in CONDS) + f"{'Δ(A-C)':>10}")
for cls in classes:
    vals = {cond: ((arms.get(cond,{}).get('per_class') or {}).get(cls) or {}).get('map50') for cond in CONDS}
    row = "".join((f"{vals[c]:.4f}" if isinstance(vals[c],float) else "N/A").rjust(10) for c in CONDS)
    a,cc = vals.get("aroma"), vals.get("casda")
    dac = f"{(a-cc):+.4f}" if isinstance(a,float) and isinstance(cc,float) else "N/A"
    print(f"{cls:<6}{row}{dac:>10}")
```

### 해석 (사전등록 — dev_note ④ 음성결과 규칙 준수)

- **load-bearing = `aroma vs casda`** (생성 엔진·packager·Poisson 동일, **ROI 선택만 다름**). 이것이 compounding의 직접 증거.
- **사전등록 판정:**
  - (i) `aroma ≥ casda` ∧ `aroma > random` → **compounding SUPPORTED** (좋은 생성 위에서도 ROI 선택이 기여).
  - (ii) `aroma ≈ casda`, 둘 다 `> random` → "deficit-aware가 좋은 생성서 worse 아님"(약한 방어).
  - (iii) `aroma ≈ casda ≈ random` → 좋은 생성 위에선 ROI 무관(thesis (a) 일반화 실패, **정직 보고**).
  - (iv) `aroma < baseline` (전 arm aug<baseline) → 생성품질이 원인 아님, severstal 데이터 충분/포화 → 주장 bound(severstal은 falsification arm).
- **null 보고 전 control 필수:**
  - **negative control(seed 결정성):** 동일 arm을 동일 seed로 2회 §4 실행 시 map50 재현 — 안 되면 latent/placement confound, null 무의미.
  - **parity(§5 (2)):** `n_synth_train` 3조건 동일 — 다르면 cap 누락 rig, 재확인.
- **표기 규칙:** 이제 `casda` arm은 **진짜 학습된 ControlNet+Poisson**이므로 'CASDA'로 표기 가능(이전 exp4v2의 copy-paste 'casda-ROI'와 다름 — 본 실험은 diffusion 실행됨). 단 **severstal 단일 도메인** 한정임을 caption에 명시.

---

## 출력 파일

| 경로 | 내용 |
|------|------|
| `$ROI_AROMA/roi_selected.json`, `$ROI_RANDOM/roi_selected.json` | AROMA compatibility / random 선택 ROI |
| `$CSV_AROMA`, `$CSV_RANDOM` | 어댑터 산출 CASDA 호환 roi_metadata.csv (+ crops/) |
| `$CSV_CASDA` | CASDA 자체 roi_metadata.csv (Stage A 산출) |
| `$CN_ROOT/{arm}/train.jsonl` + hints | packager 산출 (arm별) |
| `$CN_ROOT/{arm}/generated/` | ControlNet 생성 결함 (arm별, GPU) |
| `$SYN_{CASDA,AROMA,RANDOM}/severstal/{images,annotations.json}` | Poisson 합성 (arm별) |
| `$EXP4_OUT/exp4v2_results.json` | 4-way mAP (multi-seed mean±std/CI, per_class) |
| `$EXP4_OUT/exp4v2_summary.md` | 비교표 + per-class map50 |

## 체크리스트

- [ ] best_model 실재 확인(§0) — `config.json`+`*.safetensors`
- [ ] 어댑터 산출 csv 헤더 21열 순서 일치 + roi_image/roi_mask 동일 크기 검증(§2.2)
- [ ] CASDA `prepare_controlnet_data.py` / `test_controlnet.py` / Stage C **CLI `--help`로 확인 후 ⚠️ placeholder 교체**(§3)
- [ ] §3b·§3c 산출 1회 생성·캐시(YOLO seed마다 재생성 금지)
- [ ] §4 `n_synth_train` 3조건 동일(parity, §5 (2))
- [ ] negative control(seed 결정성) 통과 후에만 null 해석(§5)
- [ ] 모든 figure/caption에 **severstal 단일 도메인** 한정 명시
