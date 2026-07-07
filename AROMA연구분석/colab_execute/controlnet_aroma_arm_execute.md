# AROMA ControlNet-생성 arm 실행 가이드 (leather / mtd / severstal)

**목적**: AROMA arm의 합성을 copy-paste → **ControlNet 생성(텍스처) + AROMA ROI paste**로 교체한 뒤, exp4v2에서 random(copy-paste, 기존 결과 재사용)과 재비교.
**설계 (단일변수 비교)**: 기존 `roi_selected.json`을 **그대로 재사용** — ROI 선택·배치·블렌딩·GT mask(seed mask)는 동일, **결함 crop 픽셀 생성만** ControlNet으로 교체. copy-paste aroma(20260705) vs ControlNet aroma vs random 삼각 비교 가능.
**대상 데이터셋**: `mvtec_leather` · `mtd` · `severstal` · `aitex`
> **aitex 특수 처리**: aitex는 tiled(256×256/stride128)+단일클래스 전환으로 baseline이 정상화되어 편입한다. tiled 재실행 결과(`exp4v2/20260706_aitex`, seeds 42/1/2, 300ep)가 확보되어(2026-07-07 업로드) **aitex도 이식 대상**이다 — baseline/random을 20260706_aitex에서 이식하고 aroma-CN만 학습(8-C), aroma-CP(0.4847)와의 삼각 비교도 성립. 파라미터는 tiled 규약(**imgsz 256**, rect 미사용, epochs 300, **seed 1 2 42**, grayscale target ON, single class)을 그대로 따라야 이식이 유효하다. 나머지 3종은 20260705 이식(seed 42 1 2, imgsz 640 rect, epochs 100) 그대로.
**런타임**: 학습·생성·벤치마크 GPU 필수 (Colab Pro A100 권장)

> **parity 설계 노트**: GT mask는 항상 실물 seed mask이므로 생성물이 흐릿해도 bbox는 유효 — dev_note ⑧의 empty-mask parity 붕괴(casda 771/1013)를 구조적으로 회피한다. blank 생성물은 seed-shift 재시도(2회) 후 skip되며 `cn_stats`로 관측된다. **AR 게이트 elongated ROI는 copy_paste로 폴백**(개수·클래스·bbox parity 유지, 2026-07-07 Option 3) — blank/oom skip을 제외하면 출력이 `n_rois × n_per_roi`에 근접하게 유지된다.

---

## 워크플로우 개요

| STEP | 내용 | 런타임 | 데이터셋별 반복 |
|------|------|--------|----------------|
| 0 | 환경변수 | CPU | — |
| 1 | 패키지 설치 | CPU | — |
| 2 | 전제 확인 (profiling/roi/기존 결과) | CPU | — |
| 3 | ControlNet 학습 데이터 빌드 (`build_train_jsonl`) | CPU | 4종 |
| 4 | ControlNet 학습 (`train_controlnet`) | **GPU** | 4종 (세션 분리 권장) |
| 5 | 생성 pilot 육안 gate | GPU | 4종 |
| 6 | 전량 생성 + paste (`generate_defects --method controlnet`) | **GPU** | 4종 |
| 7 | parity / cn_stats 게이트 | CPU | — |
| 8 | exp4v2 — 4종 모두 baseline/random 이식(3종=20260705, aitex=20260706_aitex) + aroma-CN만 학습 | **GPU** | — |
| 9 | 결과 비교 | CPU | — |

---

## STEP 0 — 환경변수

```python
import os, json

# AROMA 기본 (기존 세션 셀과 동일)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"
os.environ['AROMA_REF']     = "/content/AROMA"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')

# 본 실험 전용 출력 루트 (기존 산출과 분리)
os.environ['CN_DATA']    = f"{os.environ['AROMA_OUT']}/controlnet_data"      # train.jsonl + targets/hints
os.environ['CN_MODELS']  = f"{os.environ['AROMA_OUT']}/controlnet_models"    # 학습 체크포인트/best_model
os.environ['SYN_CN']     = f"{os.environ['AROMA_OUT']}/synthetic_cn"         # ControlNet 합성 산출
os.environ['EXP4V2_CN']  = f"{os.environ['AROMA_OUT']}/exp4v2/20260705_cn"   # fresh 벤치 출력
os.environ['EXP4V2_REF'] = f"{os.environ['AROMA_OUT']}/exp4v2/20260705"      # 기존 결과 (이식 소스, 3종 전용)
os.environ['EXP4V2_AITEX_REF'] = f"{os.environ['AROMA_OUT']}/exp4v2/20260706_aitex"  # aitex tiled 재실행 (aitex 이식 소스)

# aitex tiled 레이아웃 (구 비타일 aitex와 별도). normal_dir은 dataset_config aitex entry로 자동 해소됨.
os.environ['AITEX_TILED'] = f"{os.environ['DRIVE']}/aitex_tiled"

# 전체 대상. STEP 3~7은 4종 공통 루프. STEP 8은 이식 소스만 분리(3종=20260705, aitex=20260706_aitex).
DATASETS       = ["mvtec_leather", "mtd", "severstal", "aitex"]
GRAFT_DATASETS = ["mvtec_leather", "mtd", "severstal"]   # 20260705 baseline/random 이식 대상
AITEX_GRAFT    = ["aitex"]                                # 20260706_aitex baseline/random 이식 (2026-07-07 전환, 구 FRESH)
for k in ['CN_DATA','CN_MODELS','SYN_CN','EXP4V2_CN','EXP4V2_REF','AITEX_TILED']:
    print(k, '=', os.environ[k])

with open(os.environ['DATASET_CONFIG']) as f:
    CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]
```

---

## STEP 1 — 패키지 설치

```python
!pip install diffusers transformers accelerate safetensors -q
!pip install scikit-image opencv-python-headless -q
# xformers는 선택 (없으면 SDPA로 자동 fallback)
```

```python
!nvidia-smi
```

---

## STEP 2 — 전제 확인

```python
import pathlib

need = []
for ds in DATASETS:
    prof = f"{os.environ['AROMA_OUT']}/profiling/{ds}"
    roi  = f"{os.environ['AROMA_OUT']}/roi/{ds}"
    checks = {
        "morphology_features.csv": f"{prof}/morphology_features.csv",
        "context_features.csv":    f"{prof}/context_features.csv",
        "recommended_config.yaml": f"{prof}/recommended_config.yaml",
        "roi_selected.json":       f"{roi}/roi_selected.json",
        "roi_candidates.json":     f"{roi}/roi_candidates.json",
        "random annotations":      f"{os.environ['AROMA_OUT']}/synthetic_random/{ds}/annotations.json",
        "기존 aroma annotations":   f"{os.environ['AROMA_OUT']}/synthetic/{ds}/annotations.json",
    }
    for name, p in checks.items():
        ok = pathlib.Path(p).exists()
        if not ok: need.append(p)
        print(f"{'✓' if ok else '✗'} {ds:14s} {name}: {p}")

# 20260705 per-seed 결과 (3종 baseline/random 이식 소스, 실측 seed = 42 1 2)
for s in (42, 1, 2):
    p = f"{os.environ['EXP4V2_REF']}/_seeds/seed{s}/exp4v2_results.json"
    print(f"{'✓' if pathlib.Path(p).exists() else '✗'} seed{s}: {p}")

# 20260706_aitex per-seed 결과 (aitex baseline/random 이식 소스, seed = 42 1 2)
for s in (42, 1, 2):
    p = f"{os.environ['EXP4V2_AITEX_REF']}/_seeds/seed{s}/exp4v2_results.json"
    print(f"{'✓' if pathlib.Path(p).exists() else '✗'} aitex seed{s}: {p}")

print("\nMISSING:", len(need))
```

> 모두 ✓여야 진행. `roi_selected.json`은 **20260705 aroma arm이 쓴 것과 동일 파일**이어야 단일변수 비교가 성립한다.
> ⚠️ 3종(GRAFT_DATASETS) 규약: 가이드 표기는 `--seeds 1 2 43`이지만 **20260705 실측 per_seed는 42/1/2** — 본 가이드는 실측에 맞춰 `--seeds 42 1 2`를 사용한다.
> ⚠️ **aitex는 20260705가 아니라 `EXP4V2_AITEX_REF`(20260706_aitex, tiled 재실행)에서 이식**한다 — `EXP4V2_REF`에 aitex가 없어도 정상. aitex 필수 ✓: `profiling/aitex`·`roi/aitex`·`synthetic_random/aitex`·`synthetic/aitex` + 위 20260706_aitex per-seed 3개. seed는 **1 2 42**(tiled 재실행의 1 2 43은 오실행이므로 정정 — 20260706_aitex 실측 per_seed는 42/1/2).

---

## STEP 3 — ControlNet 학습 데이터 빌드 (CPU)

학습·생성이 **같은 hint/prompt 생성기**를 쓰도록 `build_train_jsonl.py`로 빌드한다 (style=technical 고정).

```python
for ds in DATASETS:
    os.environ['DS'] = ds
    os.environ['PROF'] = f"{os.environ['AROMA_OUT']}/profiling/{ds}"
    os.environ['ROI']  = f"{os.environ['AROMA_OUT']}/roi/{ds}"
    print(f"\n=== {ds} ===")
    !python $AROMA_SCRIPTS/build_train_jsonl.py \
        --morphology_csv   $PROF/morphology_features.csv \
        --roi_candidates   $ROI/roi_candidates.json \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --output_dir       $CN_DATA/$DS \
        --style            technical
```

**확인**: 각 `$CN_DATA/{ds}/train.jsonl` 라인 수 출력. `background_type collapsed` WARN이 뜨면 그대로 진행하되 기록해 둔다(학습·생성 동일 분포이므로 parity는 유지됨).

```python
for ds in DATASETS:
    p = f"{os.environ['CN_DATA']}/{ds}/train.jsonl"
    n = sum(1 for _ in open(p)) if pathlib.Path(p).exists() else 0
    print(f"{ds}: train.jsonl lines = {n}")
```

---

## STEP 4 — ControlNet 학습 (GPU, 데이터셋당 1세션 권장)

공통: SD1.5 base + canny ControlNet init, fp16, gradient checkpointing, early stopping, checkpoint 재개.
데이터셋별 차등:

| 데이터셋 | 규모 | epochs | --augment | grayscale target |
|---------|------|--------|-----------|------------------|
| severstal | 수천 crop | 100 | OFF | 기본 ON (강판 grayscale) |
| mtd | 수백 | 150 | **ON** | 기본 ON (자성타일 grayscale) |
| aitex | 수백 타일(~352) | 150 | **ON** | 기본 ON (직물 grayscale) — mtd 준용 |
| mvtec_leather | ~수십·소수 | 300 | **ON** | **OFF** (`--no_force_grayscale_target`, 컬러 가죽) |

```python
# ── severstal (최장 — 단독 세션 권장) ──
os.environ['DS'] = "severstal"
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA/$DS \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 100 \
    --early_stopping_patience 20 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

```python
# ── mtd ──
os.environ['DS'] = "mtd"
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA/$DS \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 150 \
    --augment \
    --early_stopping_patience 20 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

```python
# ── aitex (tiled/single: mtd 준용 — 수백 타일, augment ON, grayscale 기본 ON) ──
os.environ['DS'] = "aitex"
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA/$DS \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 150 \
    --augment \
    --early_stopping_patience 20 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

```python
# ── mvtec_leather (소수 데이터: augment 필수, overfitting 허용) ──
os.environ['DS'] = "mvtec_leather"
!python /content/AROMA/scripts/train_controlnet.py \
    --data_dir $CN_DATA/$DS \
    --output_dir $CN_MODELS/$DS \
    --mixed_precision fp16 \
    --gradient_checkpointing \
    --num_train_epochs 300 \
    --augment \
    --no_force_grayscale_target \
    --early_stopping_patience 30 \
    --checkpointing_steps 500 \
    --save_optimizer_state \
    --save_fp16 \
    --resume_from_checkpoint latest
```

> 세션 끊김 → 동일 셀 재실행 (`--resume_from_checkpoint latest`).
> 학습 중 validation 이미지가 `$CN_MODELS/{ds}/validation/step_*/`에 주기 저장됨 — STEP 5의 1차 육안 소스.
> 산출: `$CN_MODELS/{ds}/best_model/` (ControlNetModel.from_pretrained 로드 가능).

---

## STEP 5-0 — AR 분포 사전 스캔 (CPU, 수 초 — GPU 진입 전 필수)

AR 게이트 발동 여부는 `roi_selected.json`의 `defect_bbox`만으로 완전 결정되므로, **생성 전에 임계별 ar_fallback 비율을 정확히 예측**할 수 있다. 데이터셋별 임계를 여기서 확정한다 (실측 사례: aitex는 임계 2.5에서 fallback 98% — CN arm이 사실상 copy_paste가 되어 측정 불가).

```python
import json, numpy as np

CAND = [2.0, 2.5, 3.0, 4.0, 6.0, 8.0]

for ds in DATASETS:
    sel = json.load(open(f"{os.environ['AROMA_OUT']}/roi/{ds}/roi_selected.json"))
    ars = []
    for e in sel:
        bb = e.get("defect_bbox")
        if not bb or len(bb) != 4 or min(bb[2], bb[3]) <= 0:
            continue
        ars.append(max(bb[2], bb[3]) / min(bb[2], bb[3]))
    ars = np.array(ars)
    pct = {t: float((ars > t).mean()) for t in CAND}
    print(f"{ds:14s} n={len(ars):4d}  AR med={np.median(ars):5.2f} p75={np.percentile(ars,75):5.2f} "
          f"p90={np.percentile(ars,90):5.2f} max={ars.max():6.1f}")
    print("               fallback%: " + "  ".join(f"@{t}={100*p:.0f}%" for t, p in pct.items()))
```

**판정 규칙**:

1. 임계 후보별 fallback 비율을 보고 **CN 생성 비중이 유의미(권장: fallback ≤ 50%, 이상적 ≤ 30%)**해지는 최소 임계를 데이터셋별로 고른다.
2. 고른 임계가 2.5보다 크면 = 그 AR 수준의 squash 왜곡을 감수한다는 뜻 → **STEP 5 pilot에서 반드시 그 임계로 생성해 육안 재검** (pilot ROI를 AR 상위에서 뽑아 왜곡 최악 케이스 확인). 왜곡 불가 판정이면 임계를 되돌리고 해당 데이터셋은 "CN 측정 범위 제한(fallback X%)" 또는 제외로 정직 보고.
3. 어떤 임계로도 fallback을 못 낮추면(aitex처럼 구조적 elongated) → 그 데이터셋은 CN 비교에서 제외/한정 보고가 정답. 임계만 0(OFF)으로 올려 왜곡물을 학습에 넣는 것은 금지 — pilot 육안 통과 없이 OFF 금지.
4. 확정값을 아래 `AR_THRESHOLD`에 기록 — STEP 5/6이 이 dict를 사용한다.

```python
# STEP 5-0 스캔 + pilot 육안으로 확정한 데이터셋별 AR 임계 (0 = 게이트 OFF)
AR_THRESHOLD = {"mvtec_leather": 2.5, "mtd": 2.5, "severstal": 2.5, "aitex": 2.5}  # ← 스캔 결과로 교체
```

### 5-0b — 텍스처 거리 분포 사전 스캔 (CPU, 수 분)

텍스처 게이트도 같은 원리로 **생성 전에 거부율을 예측**할 수 있다 — source descriptor(ROI 주변 배경)와 normal 무작위 patch의 거리 분포는 CPU만으로 계산된다. 임계가 분포보다 과하게 낮으면 전 normal 거부 → repick 소진 → fallback(마지막 후보 paste) 폭주 = 게이트 무력 + 재샘플 낭비.

```python
import sys, random
import numpy as np
from PIL import Image
sys.path.insert(0, os.environ['AROMA_SCRIPTS'])
import generate_defects as gd

TEX_CAND = [0.15, 0.20, 0.25, 0.30, 0.35]
prng = random.Random(0)

for ds in DATASETS:
    sel = json.load(open(f"{os.environ['AROMA_OUT']}/roi/{ds}/roi_selected.json"))
    normals = gd.load_normal_images(normal_dir(ds))
    descs = []
    for e in prng.sample(sel, min(40, len(sel))):
        bb, mp, ip = e.get("defect_bbox"), e.get("defect_mask_path"), e.get("image_path")
        if not (bb and mp and ip and pathlib.Path(mp).exists() and pathlib.Path(ip).exists()):
            continue
        d = gd._source_bg_descriptor(
            np.asarray(Image.open(ip).convert("RGB")),
            np.asarray(Image.open(mp).convert("L")), tuple(int(v) for v in bb))
        if d is not None:
            descs.append((d, bb))
    dists = []
    for d, bb in descs:
        for _ in range(5):  # ROI당 무작위 normal patch 5개
            nim = np.asarray(Image.open(prng.choice(normals)).convert("RGB"))
            h, w = nim.shape[:2]
            bw, bh = min(int(bb[2]), w), min(int(bb[3]), h)
            x, y = prng.randint(0, max(0, w - bw)), prng.randint(0, max(0, h - bh))
            dists.append(gd._texture_distance(d, gd._texture_descriptor(nim[y:y+bh, x:x+bw])))
    a = np.array(dists)
    rej = "  ".join(f"@{t}={100*(a>t).mean():.0f}%" for t in TEX_CAND)
    print(f"{ds:14s} n_desc={len(descs):3d} pairs={len(a):4d}  dist med={np.median(a):.3f} "
          f"p90={np.percentile(a,90):.3f}   거부율: {rej}")
```

**판정 규칙**: 거부율이 게이트의 "선별력"이다 — **10~50% 밴드가 적정** (0%에 가까우면 게이트 무의미, 80%+면 fallback 폭주). 데이터셋별로 적정 밴드에 드는 임계를 골라 아래에 기록. descriptor가 대부분 None(n_desc 급감)인 데이터셋은 게이트 자체가 자동 통과되므로 0(OFF)과 동일 — 그대로 두면 됨.

```python
# 5-0b 스캔으로 확정한 데이터셋별 텍스처 임계 (0 = OFF)
TEX_THRESHOLD = {"mvtec_leather": 0.25, "mtd": 0.25, "severstal": 0.25, "aitex": 0.25}  # ← 스캔 결과로 교체
```

> ⚠️ 임계를 바꿔 재생성할 때는 **새 `--output_dir` 또는 `--cn_no_cache`** (STEP 6 캐시 주의 참조 — stale sidecar 오귀속 방지).
> **원칙 (aitex 98% 폴백 사후 교훈)**: GPU 단계가 의존하는 모든 필터·임계는 발동률을 CPU로 예측 가능한 한 **반드시 사전 스캔 후 확정**한다. 단일 데이터셋 pilot에서 잡은 값을 타 데이터셋에 무검증 일괄 적용 금지.

---

## STEP 5 — 생성 pilot 육안 gate (GPU, 전량 생성 전 필수)

각 데이터셋에서 ROI 4개만 잘라 실제 생성→paste 경로를 태워 확인한다.

```python
import json, shutil, pathlib

def pilot(ds, n=4):
    roi = f"{os.environ['AROMA_OUT']}/roi/{ds}"
    pilot_dir = f"/content/pilot_roi/{ds}"
    pathlib.Path(pilot_dir).mkdir(parents=True, exist_ok=True)
    sel = json.load(open(f"{roi}/roi_selected.json"))
    json.dump(sel[:n], open(f"{pilot_dir}/roi_selected.json", "w"))
    return pilot_dir

for ds in DATASETS:
    os.environ['DS'] = ds
    os.environ['PROF'] = f"{os.environ['AROMA_OUT']}/profiling/{ds}"
    os.environ['PILOT_ROI'] = pilot(ds)
    os.environ['NORMAL'] = normal_dir(ds)
    gray_flag = "--cn_no_grayscale" if ds == "mvtec_leather" else ""
    os.environ['GRAY_FLAG'] = gray_flag
    os.environ['AR_T'] = str(AR_THRESHOLD[ds])    # STEP 5-0에서 확정
    os.environ['TEX_T'] = str(TEX_THRESHOLD[ds])  # STEP 5-0b에서 확정
    print(f"\n=== pilot {ds} (ar={os.environ['AR_T']}, tex={os.environ['TEX_T']}) ===")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir    $PILOT_ROI \
        --normal_dir $NORMAL \
        --output_dir /content/pilot_out/$DS \
        --method controlnet \
        --controlnet_path  $CN_MODELS/$DS/best_model \
        --morphology_csv   $PROF/morphology_features.csv \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --n_per_roi 2 --seed 42 \
        --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --cn_ar_threshold $AR_T \
        --texture-dist-threshold $TEX_T \
        $GRAY_FLAG
```

> 임계를 2.5보다 올린 데이터셋은 pilot ROI를 `sel[:n]` 대신 **AR 상위에서** 뽑아 왜곡 최악 케이스를 확인할 것:
> `sel = sorted(sel, key=lambda e: -(max(e['defect_bbox'][2], e['defect_bbox'][3]) / min(e['defect_bbox'][2], e['defect_bbox'][3])))` 후 상위 n개.

> **신규 필터 2종** (severstal pilot 분석에서 도입 — dev_note `aroma_controlnet-arm_quality-filters.md`):
> - `--cn_ar_threshold 2.5` — bbox 종횡비 max(w,h)/min(w,h) > 2.5 ROI를 AR 게이트. 세장형 bbox는 512² squash-unsquash 과정에서 수평 스미어/평탄 패치가 생김(pilot: 아티팩트 3장 전부 AR≥3.1, AR≤2.1 전부 양호). **기본 동작 = copy_paste 폴백**(2026-07-07, dev_note Option 3): AR 게이트 ROI를 스킵하지 않고 copy_paste로 채워 출력 개수·클래스·bbox를 random/aroma-CP와 **완전 parity** 유지. 폴백 샘플은 annotations `method="copy_paste_arfallback"`로 태깅되고 `controlnet stats`의 `ar_fallback`으로 카운트. → aroma-CN vs aroma-CP는 elongated에서 양쪽 동일 copy_paste, delta는 ControlNet이 실제 생성한 non-elongated에서만 발생(통제된 비교). `--cn_no_ar_fallback`으로 순수-arm 스킵 복원 시 `skip_ar` 카운트. `0`이면 게이트 OFF.
> - `--texture-dist-threshold $TEX_T` (데이터셋별, 5-0b 스캔으로 확정) — 소스 결함 주변 배경과 paste 위치 배경의 텍스처 descriptor 거리(0..1)가 임계 초과 시 위치 재샘플 → normal 재추첨(상한 5장). checkerplate(무늬강판) 같은 구조 반복 패턴 배경으로의 오배치를 차단. 로그의 `texture-gate stats`(active/repick_draws/fallback)로 동작 확인. `0`이면 OFF. **튜닝 밴드 0.15–0.35**: fallback이 자주 찍히면(전 normal 거부) 임계를 올리고, checkerplate 오배치가 남으면 내린다.

```python
# 육안 확인: 결함이 그럴듯한가 (특히 leather)
from PIL import Image
import matplotlib.pyplot as plt
for ds in DATASETS:
    imgs = sorted(pathlib.Path(f"/content/pilot_out/{ds}/images").glob("*.jpg"))[:4]
    fig, axes = plt.subplots(1, max(1, len(imgs)), figsize=(16, 4)); fig.suptitle(ds)
    for ax, p in zip(axes if len(imgs) > 1 else [axes], imgs):
        ax.imshow(Image.open(p)); ax.axis('off')
    plt.show()
```

**gate 기준**: (a) 로그의 `controlnet stats`에서 `blank_rate < 0.2`, (b) 결함 텍스처가 배경과 이질적이지 않은가, (c) `ar_fallback` 비율 — 세장형 결함은 `--cn_ar_threshold`가 **자동으로 copy_paste 폴백** 처리(수동 육안 게이트 불필요, 개수 parity 유지). `ar_fallback / (호출 수)`가 과도하면(예: >30%, severstal class3처럼 elongated 위주) 논문에 "elongated 결함은 양 arm copy_paste 처리, delta는 non-elongated 기여"로 명시하면 됨 — 커버리지 손실이 아니라 비교 통제. 임계 상향(2.5→3.0)으로 ControlNet 생성 비중을 늘릴지는 pilot 품질로 판단. 실패 시 → 해당 데이터셋 학습 하이퍼 조정(augment/epoch) 또는 `--cn_cond_scale`(0.7→0.9) 조정 후 재-pilot. leather가 끝내 비현실적이면 **"controlnet arm 부적합"으로 정직 보고**하고 leather 제외.

> pilot 재실행은 안전: 캐시가 모델 가중치 서명(size+mtime)·steps·cond_scale을 fingerprint로 검증하므로, 재학습·파라미터 변경 후 재실행하면 구산출물은 자동 무효화되고 새로 생성된다.

---

## STEP 6 — 전량 생성 + paste (GPU)

`n_per_roi`는 **기존 copy-paste aroma 합성과 동일 풀 크기**가 되도록 기존 annotations에서 역산한다 (parity: random cap 이상 보장).

> ⚠️ **`--local_staging` 사용 금지** — ControlNet 경로는 이미지별 sidecar 캐시(`*.meta.json`)로 세션 재개 시 GPU를 건너뛴다. Drive 직결이어야 캐시가 살아남는다. 세션이 끊기면 동일 셀 재실행 = 이어서 생성.
> 캐시는 fingerprint(모델 가중치 서명 + steps/cond_scale/resolution/grayscale + 샘플 identity/seed)를 검증한다 — **재학습·파라미터 변경 후 재실행 시 stale 산출물은 자동 재생성**되며, `roi_selected.json`이 바뀌어 roi 인덱스가 다른 결함을 가리키게 돼도 오귀속되지 않는다.

```python
for ds in DATASETS:
    ann_old = json.load(open(f"{os.environ['AROMA_OUT']}/synthetic/{ds}/annotations.json"))
    sel = json.load(open(f"{os.environ['AROMA_OUT']}/roi/{ds}/roi_selected.json"))
    npr = max(1, round(len(ann_old) / max(1, len(sel))))
    print(f"{ds}: 기존 합성 {len(ann_old)}장 / ROI {len(sel)}개 → n_per_roi = {npr}")
```

```python
N_PER_ROI = {"mvtec_leather": 3, "mtd": 3, "severstal": 3, "aitex": 3}   # ← 위 셀 출력으로 교체

for ds in DATASETS:
    os.environ['DS'] = ds
    os.environ['PROF'] = f"{os.environ['AROMA_OUT']}/profiling/{ds}"
    os.environ['ROI']  = f"{os.environ['AROMA_OUT']}/roi/{ds}"
    os.environ['NORMAL'] = normal_dir(ds)
    os.environ['NPR'] = str(N_PER_ROI[ds])
    os.environ['GRAY_FLAG'] = "--cn_no_grayscale" if ds == "mvtec_leather" else ""
    os.environ['AR_T'] = str(AR_THRESHOLD[ds])    # STEP 5-0/5에서 확정
    os.environ['TEX_T'] = str(TEX_THRESHOLD[ds])  # STEP 5-0b에서 확정
    print(f"\n=== generate {ds} (n_per_roi={os.environ['NPR']}, ar={os.environ['AR_T']}, tex={os.environ['TEX_T']}) ===")
    !python $AROMA_SCRIPTS/generate_defects.py \
        --roi_dir    $ROI \
        --normal_dir $NORMAL \
        --output_dir $SYN_CN/$DS \
        --method controlnet \
        --controlnet_path  $CN_MODELS/$DS/best_model \
        --morphology_csv   $PROF/morphology_features.csv \
        --context_features $PROF/context_features.csv \
        --config           $PROF/recommended_config.yaml \
        --n_per_roi $NPR --seed 42 \
        --blend_mode seamless \
        --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
        --cn_ar_threshold $AR_T \
        --texture-dist-threshold $TEX_T \
        $GRAY_FLAG
```

> blend/gate 설정(seamless + clean-bg gate)은 20260705 copy-paste aroma arm과 동일 — 생성 방식 외 전 변수 고정. AR/텍스처 임계는 STEP 5-0/5-0b 스캔 + STEP 5 pilot으로 확정한 `AR_THRESHOLD`/`TEX_THRESHOLD` 데이터셋별 값을 그대로 사용.
> ⚠️ **캐시와 필터 설정**: fingerprint는 AR·텍스처 임계를 포함하지 않는다. 따라서 —
> - `--texture-dist-threshold`만 바꿔 기존 캐시 위에서 재실행하면 **캐시 히트 샘플은 이전 normal 배정이 유지**된다.
> - AR 폴백 샘플은 sidecar를 쓰지 않으므로 세션 재개 시 매번 copy_paste로 재생성된다(CPU, 동일 seed → 결정론적, 무해). 단 **같은 output_dir에서 `--cn_ar_threshold`를 바꿔 재실행하면** 이전 controlnet 산출물 위에 폴백 composite가 덮이면서 옛 sidecar가 남고, 이후 임계를 되돌리면 stale sidecar가 폴백 이미지를 controlnet 생성물로 오귀속시킬 수 있다.
> - **결론: AR/텍스처 필터 설정을 바꿨으면 새 `--output_dir`를 쓰거나 `--cn_no_cache`로 1회 재생성할 것.** 동일 설정의 단순 세션 재개는 안전.
> 소요: A100 기준 이미지당 약 2–4초(30 steps) → severstal 수천 장 = 수 시간. 캐시 재개 가능.

---

## STEP 7 — parity / cn_stats 게이트 (CPU)

```python
# random arm의 n_synth_train 실측 (3종=20260705, aitex=20260706_aitex).
# 이 값 이상이어야 synth_ratio=1.0 cap 충족.
RANDOM_N = {"mvtec_leather": 64, "mtd": 272, "severstal": 2534, "aitex": 246}

def _live_labelable(path):
    ann = json.load(open(path))
    return [a for a in ann if not a.get("dry_run") and a.get("bbox")]

for ds in DATASETS:
    live = _live_labelable(f"{os.environ['SYN_CN']}/{ds}/annotations.json")
    with_mask = sum(1 for a in live if a.get("mask_path"))
    n_arfb = sum(1 for a in live if a.get("method") == "copy_paste_arfallback")
    need = RANDOM_N[ds]
    ok = len(live) >= need
    print(f"{'✓' if ok else '✗ PARITY FAIL'} {ds:14s} labelable={len(live)} "
          f"(mask={with_mask}, ar_fallback={n_arfb})  needed>={need}")
```

> AR 폴백(Option 3) 덕에 elongated ROI가 스킵되지 않으므로 labelable은 대개 첫 실행에서 need를 충족한다. `method="copy_paste_arfallback"` 비중(`n_arfb / len(live)`)을 기록 — 논문에 "aroma-CN 중 X%는 elongated이라 copy_paste 처리, delta는 나머지 ControlNet 생성분 기여"로 명시.
> 그럼에도 ✗이면(blank/oom skip 과다) `N_PER_ROI[ds] += 1` 후 STEP 6 재실행 — 캐시 덕에 기존 생성분은 skip되고 부족분만 추가 생성된다.
> STEP 6 로그의 `controlnet stats` 라인(gen_ok / ar_fallback / skip_blank / blank_rate / join_miss)을 여기 기록해 둔다 — 논문 부록·재현성 증빙.

---

## STEP 8 — exp4v2 (4종 모두 이식 + aroma-CN만 학습, GPU)

### 8-A. baseline/random 결과 이식 (3종=20260705, aitex=20260706_aitex, aroma 항목만 제거)

```python
import copy, pathlib, json

# (소스 ref 디렉토리, 이식 대상 데이터셋 필터) 쌍 — 두 소스를 같은 per-seed JSON에 병합
GRAFT_SOURCES = [
    (os.environ['EXP4V2_REF'],       set(GRAFT_DATASETS)),  # 3종 (20260705)
    (os.environ['EXP4V2_AITEX_REF'], {"aitex"}),            # aitex (20260706_aitex)
]

for s in (42, 1, 2):
    dst_dir = pathlib.Path(f"{os.environ['EXP4V2_CN']}/_seeds/seed{s}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for ref, keep in GRAFT_SOURCES:
        src = f"{ref}/_seeds/seed{s}/exp4v2_results.json"
        res = json.load(open(src))
        for ds, models in res.items():
            if ds not in keep:
                continue
            out[ds] = {}
            for model, conds in models.items():
                # aroma(=copy-paste arm) 결과는 제거 — 이 run의 aroma 키는 aroma-CN이 채운다.
                out[ds][model] = {c: v for c, v in conds.items() if c != "aroma"}
    json.dump(out, open(dst_dir / "exp4v2_results.json", "w"), indent=2)
    kept = {ds: list(out[ds].get('yolov8n', {})) for ds in out}
    print(f"seed{s}: 이식 완료 → {kept}")
```

> aitex도 이식 대상이다(2026-07-07, 20260706_aitex 결과 확보) — baseline/random을 이식하고 8-C에서 **aroma-CN만** 학습한다(aitex 1 ds × 3 seed = 3 run, 기존 9 run 대비 GPU 6 run 절약). 20260706_aitex의 aroma(=aroma-CP, mAP50 0.4847)는 이식하지 않고 STEP 9에서 참조 비교로만 사용. GRAFT 3종과 aitex의 seed 집합이 {1,2,42}로 동일하므로 같은 `_seeds/seed*` 디렉토리를 공유하되, `--dataset_keys`로 대상이 분리되어 서로 덮어쓰지 않는다(exp4v2 --resume은 기존 키 보존 병합).
> ⚠️ **이식 유효 조건**: 8-C 파라미터가 20260706_aitex 실행과 동일해야 한다 — imgsz 256, rect 미사용, epochs 300/patience 50, val_frac 0.3, synth_ratio 1.0, batch 64, seeds 1 2 42. 하나라도 다르면 이식된 baseline/random과 비교 불가.

### 8-B. GRAFT 3종 실행 — 파라미터는 20260705과 완전 동일 (seeds 42 1 2, imgsz 640 rect)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys mvtec_leather mtd severstal \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $SYN_CN \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_CN \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
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

> `--resume` + 이식된 per-seed JSON → baseline/random은 skip, **aroma(ControlNet)만 학습** (3 ds × 3 seed = 9 run).
> ⚠️ 위 파라미터(imgsz/val_frac/synth_ratio/epochs/batch/rect)를 하나라도 바꾸면 이식된 baseline/random과 비교 불가.
> ⚠️ **epochs=100** — 가이드 구버전 표기는 300이었으나 20260705 실측 results.csv(severstal seed42, 3조건 모두)가 **정확히 100 epochs**로 확인됨(2026-07-07 조사). 이식 정합을 위해 100 고정. (aitex 8-C는 이식 소스가 tiled 재실행(20260706_aitex, 300/50)이므로 300 유지 — 별개 규약.)

### 8-C. aitex 실행 — 20260706_aitex 이식 + aroma-CN만 학습 (seeds 1 2 42, imgsz 256, rect 미사용)

8-A에서 baseline/random이 이식됐으므로 `--resume`이 두 조건을 skip하고 **aroma-CN만 학습**한다(1 ds × 3 seed = 3 run). 파라미터는 20260706_aitex 실행과 완전 동일해야 이식 유효(8-A 경고 참조).

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n \
    --condition all \
    --dataset_keys aitex \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --aroma_synthetic_dir  $SYN_CN \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_CN \
    --yolo_cache_dir       $AROMA_OUT/yolo_cache \
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

> `--condition all` + `--resume` + 이식된 per-seed JSON → baseline/random은 skip, **aroma(ControlNet)만 학습** (8-B와 동일 패턴). 8-B와 **출력 디렉토리(`$EXP4V2_CN`)는 동일**하되 seed JSON에 aitex 키만 추가된다.
> ⚠️ aitex 파라미터는 `aitex_tiled_rerun_execute.md` §7-B(=20260706_aitex 실행)와 정합 — **imgsz 256, `--rect` 미사용**(타일이 정사각 256), epochs 300/patience 50. seed는 **1 2 42**. 하나라도 바꾸면 이식된 baseline/random과 비교 불가.

---

## STEP 9 — 결과 비교 (CPU)

```python
import json, os

cn  = json.load(open(f"{os.environ['EXP4V2_CN']}/exp4v2_results.json"))
ref = json.load(open(f"{os.environ['EXP4V2_REF']}/exp4v2_results.json"))
ref_aitex = json.load(open(f"{os.environ['EXP4V2_AITEX_REF']}/exp4v2_results.json"))

# aroma-CP 참조 소스: 3종=20260705, aitex=20260706_aitex — 전 4종 동일 포맷 삼각 비교
CP_REF = {"mvtec_leather": ref, "mtd": ref, "severstal": ref, "aitex": ref_aitex}

print(f"{'dataset':<14} {'baseline':>9} {'random':>9} {'aroma-CP':>9} {'aroma-CN':>9} {'Δ(CN-R)':>9} {'Δ(CN-CP)':>9}")
print("-" * 75)
for ds in DATASETS:
    b  = cn[ds]["yolov8n"]["baseline"]["map50"]
    r  = cn[ds]["yolov8n"]["random"]["map50"]
    a2 = cn[ds]["yolov8n"]["aroma"]["map50"]                 # ControlNet arm
    a1 = CP_REF[ds][ds]["yolov8n"]["aroma"]["map50"]         # copy-paste arm (참조)
    print(f"{ds:<14} {b:>9.4f} {r:>9.4f} {a1:>9.4f} {a2:>9.4f} {a2-r:>+9.4f} {a2-a1:>+9.4f}")

print("\nmAP50-95:")
for ds in DATASETS:
    r  = cn[ds]["yolov8n"]["random"]["map50_95"]
    a2 = cn[ds]["yolov8n"]["aroma"]["map50_95"]
    a1 = CP_REF[ds][ds]["yolov8n"]["aroma"]["map50_95"]
    print(f"{ds:<14} random={r:.4f}  aroma-CP={a1:.4f}  aroma-CN={a2:.4f}  Δ(CN-R)={a2-r:+.4f}")
```

per-seed paired delta(부호 일치)도 확인: `cn[ds]['yolov8n']['aroma']['per_seed']` vs `['random']['per_seed']` (전 4종 seed {42,1,2} 공통).

> **aitex 판독 기준** (20260706_aitex 실측, tile-level mAP50): baseline 0.3719 / random 0.3881 / **aroma-CP 0.4847**(±std 0.054) — 4개 데이터셋 중 유일하게 copy-paste aroma가 random을 +9.7pp 이긴 케이스. 따라서 aitex의 관전점은 Δ(CN-R)만이 아니라 **Δ(CN-CP)**: ControlNet 생성이 이미 강한 copy-paste 재조합(0.4847)을 넘는지가 "외형 신규성" 기여의 직접 증거(H1 개선판). CN-CP가 음수라도 CN-R이 양수면 "생성 arm도 random은 이긴다"로 해석 분리.

---

## 주의사항

- **OOM**: 512px SD1.5+ControlNet fp16 피크 ≈ 5–6GB — T4도 가능하나 A100 권장. 코드가 attention/vae slicing + empty_cache 재시도 → 재-OOM 시 해당 샘플 skip(`skip_oom` 카운트).
- **blank_rate > 0.2 경고**가 전량 생성에서 뜨면: 학습 미흡(특히 leather) 또는 hint/prompt 분포 이탈. STEP 4 하이퍼 조정 후 재학습이 원칙.
- **재현성**: diffusion seed = content-hash(image_id, bbox, cell_key, rep) — 재실행·재개 시 동일 latent, 동일 (image,bbox)가 복수 cell로 선택돼도 latent 분리. `--seed 42`는 배경 선택용(기존과 동일 규약).
- **prompt/hint 학습 분포 일치**: defect_subtype·stability는 `roi_candidates.json` join(학습과 동일 소스: morph_label + MAX ctx_prior)에서 취한다 — `--roi_dir`에 `roi_candidates.json`이 함께 있어야 완전 일치(없으면 entry 필드 fallback + WARN).
- **디스크**: `$SYN_CN` 이미지 + sidecar meta + `$CN_MODELS` 체크포인트 누적. 학습 완료 후 `checkpoint-*` 정리 가능 (`best_model/`만 유지).
- **aitex tile-level mAP**: aitex는 256×256 타일 단위 평가라 50% overlap로 동일 결함이 인접 타일에 중복 계수된다. baseline/random/aroma-CN 3조건 동일 적용이라 상대 비교는 공정하나 절대값을 타 데이터셋과 직접 비교하지 말 것(논문 표기 시 "tile-level" 명시). 상세는 `aitex_tiled_rerun_execute.md` §8.
- **aitex도 이식(GRAFT)**: baseline/random은 20260706_aitex(tiled 재실행)에서 이식, 8-C는 aroma-CN만 학습(3 run). aroma-CP 0.4847은 STEP 9 참조 비교로 사용. 이식 유효 조건 = 파라미터 완전 동일(imgsz 256, 300/50, no rect, seeds 1 2 42) — 8-A 경고 참조.
- 시간 실측·처리량 벤치는 수행하지 않는다 (load-test policy).
