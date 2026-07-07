# AROMA ControlNet-생성 arm 실행 가이드 (leather / mtd / severstal)

**목적**: AROMA arm의 합성을 copy-paste → **ControlNet 생성(텍스처) + AROMA ROI paste**로 교체한 뒤, exp4v2에서 random(copy-paste, 기존 결과 재사용)과 재비교.
**설계 (단일변수 비교)**: 기존 `roi_selected.json`을 **그대로 재사용** — ROI 선택·배치·블렌딩·GT mask(seed mask)는 동일, **결함 crop 픽셀 생성만** ControlNet으로 교체. copy-paste aroma(20260705) vs ControlNet aroma vs random 삼각 비교 가능.
**대상 데이터셋**: `mvtec_leather` · `mtd` · `severstal` · `aitex`
> **aitex 특수 처리**: aitex는 tiled(256×256/stride128)+단일클래스 전환으로 baseline이 정상화되어 편입한다. 단 **20260705 결과에 aitex가 없어 이식 불가**하고, 이번 실험에서 **aroma-CP(copy-paste) 비교를 하지 않으므로** 이식할 이점이 없다. 따라서 aitex만 **이식 스킵 + `--condition all` fresh 학습**(baseline/random/aroma-CN 한 run), **seed 1 2 42**, **imgsz 256**(rect 미사용), grayscale target ON, single class. 나머지 3종은 기존 이식(seed 42 1 2, imgsz 640 rect) 그대로.
**런타임**: 학습·생성·벤치마크 GPU 필수 (Colab Pro A100 권장)

> **parity 설계 노트**: GT mask는 항상 실물 seed mask이므로 생성물이 흐릿해도 bbox는 유효 — dev_note ⑧의 empty-mask parity 붕괴(casda 771/1013)를 구조적으로 회피한다. blank 생성물은 seed-shift 재시도(2회) 후 skip되며 `cn_stats`로 관측된다. **AR 게이트 elongated ROI는 copy_paste로 폴백**(개수·클래스·bbox parity 유지, 2026-07-07 Option 3) — blank/oom skip을 제외하면 출력이 `n_rois × n_per_roi`에 근접하게 유지된다.

---

## 워크플로우 개요

| STEP | 내용 | 런타임 | 데이터셋별 반복 |
|------|------|--------|----------------|
| 0 | 환경변수 | CPU | — |
| 1 | 패키지 설치 | CPU | — |
| 2 | 전제 확인 (profiling/roi/기존 결과) | CPU | — |
| 3 | ControlNet 학습 데이터 빌드 (`build_train_jsonl`) | CPU | 3종 |
| 4 | ControlNet 학습 (`train_controlnet`) | **GPU** | 3종 (세션 분리 권장) |
| 5 | 생성 pilot 육안 gate | GPU | 3종 |
| 6 | 전량 생성 + paste (`generate_defects --method controlnet`) | **GPU** | 3종 |
| 7 | parity / cn_stats 게이트 | CPU | — |
| 8 | exp4v2 — 3종: baseline/random 이식 + aroma arm만 학습 / **aitex: 이식 스킵 + `--condition all` fresh** | **GPU** | — |
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

# aitex tiled 레이아웃 (구 비타일 aitex와 별도). normal_dir은 dataset_config aitex entry로 자동 해소됨.
os.environ['AITEX_TILED'] = f"{os.environ['DRIVE']}/aitex_tiled"

# 전체 대상. STEP 3~7은 4종 공통 루프. STEP 8만 이식 대상(GRAFT)과 fresh(aitex)로 분리.
DATASETS       = ["mvtec_leather", "mtd", "severstal", "aitex"]
GRAFT_DATASETS = ["mvtec_leather", "mtd", "severstal"]   # 20260705 baseline/random 이식 대상
FRESH_DATASETS = ["aitex"]                                # 이식 없이 --condition all fresh 학습
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

# 20260705 per-seed 결과 (baseline/random 이식 소스, 실측 seed = 42 1 2)
for s in (42, 1, 2):
    p = f"{os.environ['EXP4V2_REF']}/_seeds/seed{s}/exp4v2_results.json"
    print(f"{'✓' if pathlib.Path(p).exists() else '✗'} seed{s}: {p}")

print("\nMISSING:", len(need))
```

> 모두 ✓여야 진행. `roi_selected.json`은 **20260705 aroma arm이 쓴 것과 동일 파일**이어야 단일변수 비교가 성립한다.
> ⚠️ 3종(GRAFT_DATASETS) 규약: 가이드 표기는 `--seeds 1 2 43`이지만 **20260705 실측 per_seed는 42/1/2** — 본 가이드는 실측에 맞춰 `--seeds 42 1 2`를 사용한다.
> ⚠️ **aitex(FRESH_DATASETS)는 위 20260705 seed 체크와 무관** — 이식하지 않고 fresh 학습하므로 `EXP4V2_REF`에 aitex가 없어도 정상이다. aitex는 `profiling/aitex`·`roi/aitex`·`synthetic_random/aitex`·`synthetic/aitex`(tiled 재실행 산출)만 ✓이면 된다. seed는 **1 2 42**(tiled 재실행의 1 2 43은 오실행이므로 정정).

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
    print(f"\n=== pilot {ds} ===")
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
        --cn_ar_threshold 2.5 \
        --texture-dist-threshold 0.25 \
        $GRAY_FLAG
```

> **신규 필터 2종** (severstal pilot 분석에서 도입 — dev_note `aroma_controlnet-arm_quality-filters.md`):
> - `--cn_ar_threshold 2.5` — bbox 종횡비 max(w,h)/min(w,h) > 2.5 ROI를 AR 게이트. 세장형 bbox는 512² squash-unsquash 과정에서 수평 스미어/평탄 패치가 생김(pilot: 아티팩트 3장 전부 AR≥3.1, AR≤2.1 전부 양호). **기본 동작 = copy_paste 폴백**(2026-07-07, dev_note Option 3): AR 게이트 ROI를 스킵하지 않고 copy_paste로 채워 출력 개수·클래스·bbox를 random/aroma-CP와 **완전 parity** 유지. 폴백 샘플은 annotations `method="copy_paste_arfallback"`로 태깅되고 `controlnet stats`의 `ar_fallback`으로 카운트. → aroma-CN vs aroma-CP는 elongated에서 양쪽 동일 copy_paste, delta는 ControlNet이 실제 생성한 non-elongated에서만 발생(통제된 비교). `--cn_no_ar_fallback`으로 순수-arm 스킵 복원 시 `skip_ar` 카운트. `0`이면 게이트 OFF.
> - `--texture-dist-threshold 0.25` — 소스 결함 주변 배경과 paste 위치 배경의 텍스처 descriptor 거리(0..1)가 임계 초과 시 위치 재샘플 → normal 재추첨(상한 5장). checkerplate(무늬강판) 같은 구조 반복 패턴 배경으로의 오배치를 차단. 로그의 `texture-gate stats`(active/repick_draws/fallback)로 동작 확인. `0`이면 OFF. **튜닝 밴드 0.15–0.35**: fallback이 자주 찍히면(전 normal 거부) 임계를 올리고, checkerplate 오배치가 남으면 내린다.

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
    print(f"\n=== generate {ds} (n_per_roi={os.environ['NPR']}) ===")
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
        --cn_ar_threshold 2.5 \
        --texture-dist-threshold 0.25 \
        $GRAY_FLAG
```

> blend/gate 설정(seamless + clean-bg gate)은 20260705 copy-paste aroma arm과 동일 — 생성 방식 외 전 변수 고정. AR/텍스처 필터 임계는 STEP 5 pilot에서 확정한 값을 그대로 사용.
> ⚠️ **캐시와 텍스처 필터**: fingerprint는 텍스처 임계를 포함하지 않으므로, **기존 캐시 위에서 `--texture-dist-threshold`만 바꿔 재실행하면 캐시 히트 샘플은 이전 normal 배정이 유지**된다. 필터 설정을 바꿨으면 새 `--output_dir`를 쓰거나 `--cn_no_cache`로 1회 재생성할 것. (AR 게이트는 생성 전 스킵이라 캐시와 무충돌.)
> 소요: A100 기준 이미지당 약 2–4초(30 steps) → severstal 수천 장 = 수 시간. 캐시 재개 가능.

---

## STEP 7 — parity / cn_stats 게이트 (CPU)

```python
# 3종(GRAFT): random arm의 n_synth_train (20260705 실측). 이 값 이상이어야 synth_ratio=1.0 cap 충족.
# aitex(FRESH): 20260705에 없으므로 synthetic_random/aitex 실측 라벨수를 런타임 계산해 요구치로 사용.
RANDOM_N = {"mvtec_leather": 64, "mtd": 272, "severstal": 2534}

def _live_labelable(path):
    ann = json.load(open(path))
    return [a for a in ann if not a.get("dry_run") and a.get("bbox")]

for ds in DATASETS:
    live = _live_labelable(f"{os.environ['SYN_CN']}/{ds}/annotations.json")
    with_mask = sum(1 for a in live if a.get("mask_path"))
    n_arfb = sum(1 for a in live if a.get("method") == "copy_paste_arfallback")
    if ds in RANDOM_N:
        need = RANDOM_N[ds]
    else:  # aitex: random pool 실측에서 동적 산출
        need = len(_live_labelable(f"{os.environ['AROMA_OUT']}/synthetic_random/{ds}/annotations.json"))
    ok = len(live) >= need
    print(f"{'✓' if ok else '✗ PARITY FAIL'} {ds:14s} labelable={len(live)} "
          f"(mask={with_mask}, ar_fallback={n_arfb})  needed>={need}")
```

> AR 폴백(Option 3) 덕에 elongated ROI가 스킵되지 않으므로 labelable은 대개 첫 실행에서 need를 충족한다. `method="copy_paste_arfallback"` 비중(`n_arfb / len(live)`)을 기록 — 논문에 "aroma-CN 중 X%는 elongated이라 copy_paste 처리, delta는 나머지 ControlNet 생성분 기여"로 명시.
> 그럼에도 ✗이면(blank/oom skip 과다) `N_PER_ROI[ds] += 1` 후 STEP 6 재실행 — 캐시 덕에 기존 생성분은 skip되고 부족분만 추가 생성된다.
> STEP 6 로그의 `controlnet stats` 라인(gen_ok / ar_fallback / skip_blank / blank_rate / join_miss)을 여기 기록해 둔다 — 논문 부록·재현성 증빙.

---

## STEP 8 — exp4v2 (GRAFT 3종: 이식 + aroma만 학습 / aitex: fresh `--condition all`, GPU)

### 8-A. 20260705 baseline/random 결과 이식 (GRAFT_DATASETS만, aroma 항목만 제거)

```python
import copy, pathlib, json

for s in (42, 1, 2):
    src = f"{os.environ['EXP4V2_REF']}/_seeds/seed{s}/exp4v2_results.json"
    dst_dir = pathlib.Path(f"{os.environ['EXP4V2_CN']}/_seeds/seed{s}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    res = json.load(open(src))
    out = {}
    for ds, models in res.items():
        if ds not in GRAFT_DATASETS:   # aitex는 이식 대상 아님(20260705에 없음, fresh 학습)
            continue
        out[ds] = {}
        for model, conds in models.items():
            out[ds][model] = {c: v for c, v in conds.items() if c != "aroma"}
    json.dump(out, open(dst_dir / "exp4v2_results.json", "w"), indent=2)
    kept = {ds: list(out[ds].get('yolov8n', {})) for ds in out}
    print(f"seed{s}: 이식 완료 → {kept}")
```

> aitex는 8-A에서 이식하지 않는다 — 8-C에서 baseline/random/aroma를 fresh 학습해 per-seed JSON(seed1/2/42)에 병합한다. GRAFT 3종과 aitex의 seed 집합이 {1,2,42}로 동일하므로 같은 `_seeds/seed*` 디렉토리를 공유하되, `--dataset_keys`로 대상이 분리되어 서로 덮어쓰지 않는다(exp4v2 --resume은 기존 키 보존 병합).

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
> ⚠️ **epochs=100** — 가이드 구버전 표기는 300이었으나 20260705 실측 results.csv(severstal seed42, 3조건 모두)가 **정확히 100 epochs**로 확인됨(2026-07-07 조사). 이식 정합을 위해 100 고정. (aitex 8-C는 이식이 없어 tiled 재실행 규약 300/50을 따름 — 별개.)

### 8-C. aitex 실행 — 이식 없음, `--condition all` fresh (seeds 1 2 42, imgsz 256, rect 미사용)

aitex는 20260705에 없어 이식 소스가 없고 aroma-CP 비교도 하지 않으므로, **baseline/random/aroma-CN 세 조건을 한 run에서 fresh 학습**한다. 한 run 내 세 조건이 동일 seed·split을 공유하여 단일변수 비교가 내부적으로 완결된다.

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

> aitex는 baseline/random을 이식하지 않으므로 `--condition all`이 세 조건 모두 학습 (aitex 1 ds × 3 seed × 3 조건). 8-B와 **출력 디렉토리(`$EXP4V2_CN`)는 동일**하되 seed JSON에 aitex 키만 추가된다.
> ⚠️ aitex 파라미터는 `aitex_tiled_rerun_execute.md` §7-B와 정합 — **imgsz 256, `--rect` 미사용**(타일이 정사각 256). seed는 **1 2 42**(tiled 재실행의 1 2 43 오실행 정정). 이 값을 tiled baseline과 다르게 두면 tiled 재실행 결과와 교차 참조가 불가.
> ⚠️ aitex의 tiled 재실행(`$EXP4V2_OUT/exp4v2`) 결과와는 **별개 출력**(`$EXP4V2_CN`)이다 — CN arm은 여기서 baseline/random까지 새로 학습하므로 tiled 재실행의 aitex 항목을 재사용/이식하지 않는다.

---

## STEP 9 — 결과 비교 (CPU)

```python
import json, os

cn  = json.load(open(f"{os.environ['EXP4V2_CN']}/exp4v2_results.json"))
ref = json.load(open(f"{os.environ['EXP4V2_REF']}/exp4v2_results.json"))

# ── GRAFT 3종: aroma-CP(copy-paste, ref) 대비까지 포함 ──
print(f"{'dataset':<14} {'baseline':>9} {'random':>9} {'aroma-CP':>9} {'aroma-CN':>9} {'Δ(CN-R)':>9} {'Δ(CN-CP)':>9}")
print("-" * 75)
for ds in ["mvtec_leather", "mtd", "severstal"]:
    b  = cn[ds]["yolov8n"]["baseline"]["map50"]
    r  = cn[ds]["yolov8n"]["random"]["map50"]
    a2 = cn[ds]["yolov8n"]["aroma"]["map50"]           # ControlNet arm
    a1 = ref[ds]["yolov8n"]["aroma"]["map50"]          # copy-paste arm (참조)
    print(f"{ds:<14} {b:>9.4f} {r:>9.4f} {a1:>9.4f} {a2:>9.4f} {a2-r:>+9.4f} {a2-a1:>+9.4f}")

print("\nmAP50-95:")
for ds in ["mvtec_leather", "mtd", "severstal"]:
    r  = cn[ds]["yolov8n"]["random"]["map50_95"]
    a2 = cn[ds]["yolov8n"]["aroma"]["map50_95"]
    a1 = ref[ds]["yolov8n"]["aroma"]["map50_95"]
    print(f"{ds:<14} random={r:.4f}  aroma-CP={a1:.4f}  aroma-CN={a2:.4f}  Δ(CN-R)={a2-r:+.4f}")

# ── aitex: aroma-CP 없음 → CP 열 제외 (baseline/random/aroma-CN/Δ(CN-R)만) ──
if "aitex" in cn:
    print(f"\n{'dataset':<14} {'baseline':>9} {'random':>9} {'aroma-CN':>9} {'Δ(CN-R)':>9}   (aroma-CP 없음)")
    print("-" * 60)
    for metric in ("map50", "map50_95"):
        b  = cn["aitex"]["yolov8n"]["baseline"][metric]
        r  = cn["aitex"]["yolov8n"]["random"][metric]
        a2 = cn["aitex"]["yolov8n"]["aroma"][metric]
        print(f"{'aitex('+metric+')':<14} {b:>9.4f} {r:>9.4f} {a2:>9.4f} {a2-r:>+9.4f}")
```

per-seed paired delta(부호 일치)도 확인: `cn[ds]['yolov8n']['aroma']['per_seed']` vs `['random']['per_seed']` (GRAFT 3종=seed 42/1/2, aitex=seed 1/2/42 키).

---

## 주의사항

- **OOM**: 512px SD1.5+ControlNet fp16 피크 ≈ 5–6GB — T4도 가능하나 A100 권장. 코드가 attention/vae slicing + empty_cache 재시도 → 재-OOM 시 해당 샘플 skip(`skip_oom` 카운트).
- **blank_rate > 0.2 경고**가 전량 생성에서 뜨면: 학습 미흡(특히 leather) 또는 hint/prompt 분포 이탈. STEP 4 하이퍼 조정 후 재학습이 원칙.
- **재현성**: diffusion seed = content-hash(image_id, bbox, cell_key, rep) — 재실행·재개 시 동일 latent, 동일 (image,bbox)가 복수 cell로 선택돼도 latent 분리. `--seed 42`는 배경 선택용(기존과 동일 규약).
- **prompt/hint 학습 분포 일치**: defect_subtype·stability는 `roi_candidates.json` join(학습과 동일 소스: morph_label + MAX ctx_prior)에서 취한다 — `--roi_dir`에 `roi_candidates.json`이 함께 있어야 완전 일치(없으면 entry 필드 fallback + WARN).
- **디스크**: `$SYN_CN` 이미지 + sidecar meta + `$CN_MODELS` 체크포인트 누적. 학습 완료 후 `checkpoint-*` 정리 가능 (`best_model/`만 유지).
- **aitex tile-level mAP**: aitex는 256×256 타일 단위 평가라 50% overlap로 동일 결함이 인접 타일에 중복 계수된다. baseline/random/aroma-CN 3조건 동일 적용이라 상대 비교는 공정하나 절대값을 타 데이터셋과 직접 비교하지 말 것(논문 표기 시 "tile-level" 명시). 상세는 `aitex_tiled_rerun_execute.md` §8.
- **aitex는 fresh 학습**: baseline/random을 이식하지 않고 8-C `--condition all`로 함께 학습하므로 GRAFT 3종보다 run 수가 많다(aitex 3조건 × 3 seed = 9 run). seed는 1 2 42.
- 시간 실측·처리량 벤치는 수행하지 않는다 (load-test policy).
