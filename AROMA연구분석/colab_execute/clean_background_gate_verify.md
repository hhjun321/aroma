# Clean-Background Gate (검은/평탄 배경 거부) Colab 검증 가이드

**목적**: CASDA의 검은/평탄(void) 배경 거부 품질 게이트를 AROMA copy-paste 합성에 이식한 결과를 검증한다.
게이트는 **생성 시점 차단**(사후 제거 아님)이며 `--reject-clean-bg` 플래그로 켜고, **aroma / random / casda 세 조건 모두 동일 적용**(전부 `generate_defects.run()` 단일 엔진 경유).

**검증 핵심 4가지**:
1. **배경 분포 before(OFF) vs after(ON)** — 합성셋 배경 패치 밝기(mean)·대비(std) 분포가 ON에서 검은쪽 꼬리가 줄어드는가.
2. **검은배경 위 결함 샘플 비율** — ON에서 감소하는가.
3. **회귀(object-centric 과도 거부)** — visa_pcb 등 객체 중심 데이터셋에서 ON 시 통과 normal 수·합성 성공(`n_generated`)이 OFF 대비 유의 감소 없는가.
4. **결정론** — 동일 seed 2회 합성 → annotations `normal_image` 선택이 byte-identical 한가.

**전제**:
- `00_setup.md` 로 저장소 clone + Drive 마운트 완료.
- `cv2`(opencv) 가용. 게이트는 cv2 없으면 자동 비활성(legacy 동작, `HAS_CV2 False` 시 `_is_clean_background` → True).
- 게이트 기본값 **OFF** — 명시적으로 `--reject-clean-bg` 를 줘야 켜진다.

---

## 게이트 동작 요약 (구현 확인용)

| 항목 | 값 |
|------|-----|
| CLI 플래그 (3 생성기 공통) | `--reject-clean-bg` (토글), `--min-bg-quality` (기본 0.7), `--bg-blur-threshold` (기본 100.0) |
| injection point | **both** — pool(검은 normal 이미지 제외) + position(폴백 위치 검은데 회피). 별도 토글 없이 게이트 ON 시 둘 다 적용 |
| pool 게이트 | `load_normal_images` 에서 각 normal 이미지 1회 평가, 통과분만 배경 풀 사용. rglob 결과 **정렬**(결정론) |
| position 게이트 | `_random_paste_position` **폴백 경로 한정**. foreground 배치 성공 경로는 미변경(중복 방지). `max_bg_tries=20` 재시도 후 마지막 후보로 진행 |
| 품질 공식 (CASDA 이식) | `quality = 0.30*blur + 0.30*contrast + 0.20*brightness + 0.20*noise`, accept iff `quality >= min_bg_quality` |
| 전량 거부 폴백 | pool 0장이면 `logger.warning` 후 게이트 무시·원본 풀 사용 (silent 0-output 금지) |
| 결정론 | 거부/재시도 동일 `rng`. 품질 평가 자체는 RNG 없음 |

> **참고**: dev_note에 거론된 `--bg-injection {pool,position,both}` 플래그는 최종 구현에 없다. 게이트 ON 시 pool+position 양쪽이 항상 적용된다.

---

## Cell 0 — 환경변수 설정

> `.claude/rules/colab-execution.md` 준수: `!python $VAR` 형식, `${VAR}` 금지.
> 기존 `exp4v2_execute.md` 와 동일한 env 체계를 따른다. 본인 세션 값과 맞는지 `print` 로 확인.

```python
import os

# AROMA 기본 (기존 셀에서 설정했으면 생략 가능)
os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"

# 게이트 OFF (기존, baseline 비교군) 합성 출력
os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['CASDA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic_casda"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"

# 게이트 ON (신규, 별도 출력 디렉토리 — OFF 와 직접 비교)
os.environ['RANDOM_SYNTH_DIR_GATE'] = f"{os.environ['AROMA_OUT']}/synthetic_random_cleanbg"
os.environ['CASDA_SYNTH_DIR_GATE']  = f"{os.environ['AROMA_OUT']}/synthetic_casda_cleanbg"
os.environ['AROMA_SYNTH_DIR_GATE']  = f"{os.environ['AROMA_OUT']}/synthetic_cleanbg"

# 공통 합성 파라미터
os.environ['N_PER_ROI'] = "3"

for k in ('AROMA_DATA','AROMA_SYNTH_DIR','AROMA_SYNTH_DIR_GATE',
          'RANDOM_SYNTH_DIR','RANDOM_SYNTH_DIR_GATE',
          'CASDA_SYNTH_DIR','CASDA_SYNTH_DIR_GATE'):
    print(f"{k:24s}: {os.environ[k]}")
```

> 별도 출력 디렉토리(`*_cleanbg`)를 쓰는 이유: OFF/ON 합성셋을 **나란히 보존**해 동일 검증 셀로 before/after 를 직접 대조하기 위함. 기존 exp4v2 OFF 결과를 덮어쓰지 않는다.

---

## Cell 1 — 패키지 / 입력 경로 확인

```python
!pip install opencv-python-headless pillow -q   # cv2 + PIL (게이트 평가에 필요)

import os
# severstal(full-frame strip) / visa_pcb(object-centric 회귀) / mvtec_cable(full-frame) 3종 점검.
# 각 데이터셋의 ROI/normal 입력이 존재해야 한다 (Step3/Step4 산출물).
DATASETS = {
    "severstal":  {  # full-frame strip
        "roi_dir":        f"{os.environ['AROMA_OUT']}/roi/severstal",          # roi_selected.json (aroma)
        "candidates":     f"{os.environ['AROMA_OUT']}/roi/severstal/roi_candidates.json",  # random
        "normal_dir":     f"{os.environ['AROMA_DATA']}/severstal/train/good",
    },
    "visa_pcb":   {  # object-centric — 회귀 점검 대상
        "roi_dir":        f"{os.environ['AROMA_OUT']}/roi/visa_pcb",
        "candidates":     f"{os.environ['AROMA_OUT']}/roi/visa_pcb/roi_candidates.json",
        "normal_dir":     f"{os.environ['AROMA_DATA']}/visa_pcb/train/good",
    },
    "mvtec_cable":{  # full-frame
        "roi_dir":        f"{os.environ['AROMA_OUT']}/roi/mvtec_cable",
        "candidates":     f"{os.environ['AROMA_OUT']}/roi/mvtec_cable/roi_candidates.json",
        "normal_dir":     f"{os.environ['AROMA_DATA']}/mvtec_cable/train/good",
    },
}
for ds, p in DATASETS.items():
    print(f"\n=== {ds} ===")
    for k, v in p.items():
        print(f"  {k:12s}: {v}  exists={os.path.exists(v)}")
```

> `exists=False` 면 본인 세션의 실제 경로로 교체(데이터셋별 roi 산출 위치가 다를 수 있음). 본 가이드는 aroma=`--roi_dir`, random=`--candidates_json`, casda=`--metadata_csv` 입력을 쓴다.

---

## Cell 2 — OFF vs ON 재합성 (severstal, mvtec_cable: 3조건 동일 플래그)

게이트 OFF(기존)와 ON(`--reject-clean-bg`)을 **별도 출력 디렉토리**로 각각 합성한다.
세 조건(aroma/random/casda) 모두 동일 플래그를 준다. casda 는 Severstal 전용이라 severstal 블록에만 포함.

> **속도**: 검증 목적이므로 합성 자체 시간 측정은 하지 않는다(load-test 정책). `--local_staging` 으로 Drive I/O 만 줄인다.

### 2-A. severstal — OFF (기존 비교군)

```python
# aroma OFF
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $AROMA_OUT/roi/severstal \
    --normal_dir  $AROMA_DATA/severstal/train/good \
    --output_dir  $AROMA_SYNTH_DIR/severstal \
    --n_per_roi   $N_PER_ROI \
    --seed 42 \
    --local_staging

# random OFF
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $AROMA_OUT/roi/severstal/roi_candidates.json \
    --normal_dir      $AROMA_DATA/severstal/train/good \
    --output_dir      $RANDOM_SYNTH_DIR/severstal \
    --top_k 845 \
    --n_per_roi $N_PER_ROI \
    --seed 42 \
    --local_staging

# casda OFF (Stage A roi_metadata.csv 가 ROI_DIR 에 있어야 함 — exp4v2_execute.md Cell A 참조)
!python $AROMA_SCRIPTS/generate_casda.py \
    --metadata_csv  $AROMA_OUT/roi/severstal/roi_metadata.csv \
    --normal_dir    $AROMA_DATA/severstal/train/good \
    --output_dir    $CASDA_SYNTH_DIR/severstal \
    --per_class_cap 525 \
    --min_suitability 0.5 \
    --n_per_roi $N_PER_ROI \
    --seed 42 \
    --local_staging
```

### 2-B. severstal — ON (`--reject-clean-bg`, 세 조건 동일 플래그)

```python
# aroma ON
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $AROMA_OUT/roi/severstal \
    --normal_dir  $AROMA_DATA/severstal/train/good \
    --output_dir  $AROMA_SYNTH_DIR_GATE/severstal \
    --n_per_roi   $N_PER_ROI \
    --seed 42 \
    --reject-clean-bg \
    --min-bg-quality 0.7 \
    --bg-blur-threshold 100.0 \
    --local_staging

# random ON
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $AROMA_OUT/roi/severstal/roi_candidates.json \
    --normal_dir      $AROMA_DATA/severstal/train/good \
    --output_dir      $RANDOM_SYNTH_DIR_GATE/severstal \
    --top_k 845 \
    --n_per_roi $N_PER_ROI \
    --seed 42 \
    --reject-clean-bg \
    --min-bg-quality 0.7 \
    --bg-blur-threshold 100.0 \
    --local_staging

# casda ON
!python $AROMA_SCRIPTS/generate_casda.py \
    --metadata_csv  $AROMA_OUT/roi/severstal/roi_metadata.csv \
    --normal_dir    $AROMA_DATA/severstal/train/good \
    --output_dir    $CASDA_SYNTH_DIR_GATE/severstal \
    --per_class_cap 525 \
    --min_suitability 0.5 \
    --n_per_roi $N_PER_ROI \
    --seed 42 \
    --reject-clean-bg \
    --min-bg-quality 0.7 \
    --bg-blur-threshold 100.0 \
    --local_staging
```

### 2-C. mvtec_cable — OFF & ON (aroma/random 2조건; casda 미적용)

```python
# OFF
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $AROMA_OUT/roi/mvtec_cable --normal_dir $AROMA_DATA/mvtec_cable/train/good \
    --output_dir $AROMA_SYNTH_DIR/mvtec_cable --n_per_roi $N_PER_ROI --seed 42 --local_staging
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $AROMA_OUT/roi/mvtec_cable/roi_candidates.json --normal_dir $AROMA_DATA/mvtec_cable/train/good \
    --output_dir $RANDOM_SYNTH_DIR/mvtec_cable --top_k 200 --n_per_roi $N_PER_ROI --seed 42 --local_staging

# ON
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $AROMA_OUT/roi/mvtec_cable --normal_dir $AROMA_DATA/mvtec_cable/train/good \
    --output_dir $AROMA_SYNTH_DIR_GATE/mvtec_cable --n_per_roi $N_PER_ROI --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --local_staging
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $AROMA_OUT/roi/mvtec_cable/roi_candidates.json --normal_dir $AROMA_DATA/mvtec_cable/train/good \
    --output_dir $RANDOM_SYNTH_DIR_GATE/mvtec_cable --top_k 200 --n_per_roi $N_PER_ROI --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --local_staging
```

> **로그 확인**: ON 실행 로그에 `clean-bg pool gate: kept K / N normal images (R rejected, min_quality=0.70 blur=100.0)` 가 출력된다. severstal/mvtec(full-frame)은 검은 normal 이 거의 없으면 `R` 이 작을 수 있고, void 가 많은 데이터셋은 `R` 이 크다. 전량 거부 시 `reject_clean_bg ... gate disabled` 또는 풀 폴백 warning 이 보여야 한다(silent 0-output 금지).

---

## Cell 3 — visa_pcb 회귀 점검 (object-centric 과도 거부 여부)

visa_pcb 는 **객체 주변 어두운 배경이 정상**이라, pool 게이트가 정상 object-centric normal 을 과도 거부하면 안 된다.
ON 시 통과 normal 수·합성 성공(`n_generated`)이 OFF 대비 유의 감소 없는지 확인한다.

```python
# OFF
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $AROMA_OUT/roi/visa_pcb --normal_dir $AROMA_DATA/visa_pcb/train/good \
    --output_dir $AROMA_SYNTH_DIR/visa_pcb --n_per_roi $N_PER_ROI --seed 42 --local_staging
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $AROMA_OUT/roi/visa_pcb/roi_candidates.json --normal_dir $AROMA_DATA/visa_pcb/train/good \
    --output_dir $RANDOM_SYNTH_DIR/visa_pcb --top_k 200 --n_per_roi $N_PER_ROI --seed 42 --local_staging

# ON
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $AROMA_OUT/roi/visa_pcb --normal_dir $AROMA_DATA/visa_pcb/train/good \
    --output_dir $AROMA_SYNTH_DIR_GATE/visa_pcb --n_per_roi $N_PER_ROI --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --local_staging
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $AROMA_OUT/roi/visa_pcb/roi_candidates.json --normal_dir $AROMA_DATA/visa_pcb/train/good \
    --output_dir $RANDOM_SYNTH_DIR_GATE/visa_pcb --top_k 200 --n_per_roi $N_PER_ROI --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --local_staging
```

```python
# n_generated (n_ok) 회귀 비교: annotations.json 항목 수 = 합성 성공 수
import json, os
from pathlib import Path

def n_synth(synth_root, ds):
    ann = Path(synth_root) / ds / "annotations.json"
    if not ann.exists():
        return None
    raw = json.loads(ann.read_text())
    items = (raw.get("annotations") or raw.get("items") or []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    return sum(1 for e in items if isinstance(e, dict) and e.get("dry_run") is not True)

print(f"{'dataset':12s} {'cond':8s} | {'OFF n':>7} {'ON n':>7} {'ratio':>7}")
print("-"*48)
for ds in ("visa_pcb", "severstal", "mvtec_cable"):
    for cond, off_dir, on_dir in (
        ("aroma",  os.environ['AROMA_SYNTH_DIR'],  os.environ['AROMA_SYNTH_DIR_GATE']),
        ("random", os.environ['RANDOM_SYNTH_DIR'], os.environ['RANDOM_SYNTH_DIR_GATE']),
    ):
        off, on = n_synth(off_dir, ds), n_synth(on_dir, ds)
        if off is None or on is None:
            continue
        ratio = on/off if off else float('nan')
        flag = "  ⚠️ 과도거부 의심" if (off and ratio < 0.8) else ""
        print(f"{ds:12s} {cond:8s} | {off:7d} {on:7d} {ratio:7.3f}{flag}")
```

> **합격 기준**: visa_pcb 의 `ratio (ON/OFF)` 가 **0.8 이상**(정상 object-centric normal 이 과도 거부되지 않음). pool 전량 거부 시엔 폴백으로 원본 풀을 쓰므로 `ratio≈1.0` 이 되어야 한다. ratio 가 크게 낮으면 visa 어두운 배경이 부당 거부된 것 → `--min-bg-quality` 하향 또는 데이터셋별 임계 튜닝 검토.

---

## Cell 4 — 배경 분포 before(OFF) vs after(ON) + 검은배경 위 결함 비율

각 합성 이미지에서 **결함 paste 영역 주변 배경 패치**의 밝기(mean)·대비(std)를 측정해 OFF vs ON 분포를 비교한다.
배경이 검은/평탄이면 mean 이 낮고 std 가 작다. ON 에서 그 꼬리가 줄어야 한다.

```python
import json, os, numpy as np, cv2
from pathlib import Path
import matplotlib.pyplot as plt

def resolve_path(base, val):
    if not val: return None
    p = Path(val)
    if p.is_absolute() and p.exists(): return str(p)
    for cand in (base/val, base/"images"/Path(val).name):
        if cand.exists(): return str(cand)
    return str(p) if p.exists() else None

def load_anns(synth_root, ds):
    base = Path(synth_root) / ds
    ann = base / "annotations.json"
    if not ann.exists(): return base, []
    raw = json.loads(ann.read_text())
    items = (raw.get("annotations") or raw.get("items") or []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    return base, [e for e in items if isinstance(e, dict) and e.get("dry_run") is not True]

def bbox_of(e):
    bb = e.get("bbox") or e.get("defect_bbox")
    if not bb: return None
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [int(round(v)) for v in bb]
    return None

def bg_patch_stats(synth_root, ds, max_n=400):
    """결함 bbox 바로 바깥 테두리(배경) 픽셀의 mean/std 수집 + 검은배경 비율."""
    base, items = load_anns(synth_root, ds)
    means, stds = [], []
    n_black = 0; n_tot = 0
    for e in items[:max_n]:
        img_p = resolve_path(base, e.get("image_path"))
        if not img_p: continue
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        H, W = img.shape[:2]
        bb = bbox_of(e)
        if bb is None: continue
        x, y, w, h = bb
        pad = max(8, int(0.3*max(w, h)))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(W, x+w+pad), min(H, y+h+pad)
        region = img[y0:y1, x0:x1].astype(np.float32)
        if region.size == 0: continue
        # 결함 내부를 빼고 테두리(배경)만 — 외곽 ring
        ring = region.copy()
        iy0, ix0 = (y-y0), (x-x0)
        iy1, ix1 = iy0+h, ix0+w
        ring[max(0,iy0):max(0,iy1), max(0,ix0):max(0,ix1)] = np.nan
        vals = ring[~np.isnan(ring)]
        if vals.size == 0: continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        means.append(m); stds.append(s)
        n_tot += 1
        # 검은/평탄 배경 휴리스틱: 배경 mean < 30 (0..255) 이고 std < 12
        if m < 30 and s < 12:
            n_black += 1
    return np.array(means), np.array(stds), (n_black, n_tot)

DS = "severstal"   # severstal / visa_pcb / mvtec_cable 로 바꿔가며 실행
COND_DIRS = {
    "aroma":  (os.environ['AROMA_SYNTH_DIR'],  os.environ['AROMA_SYNTH_DIR_GATE']),
    "random": (os.environ['RANDOM_SYNTH_DIR'], os.environ['RANDOM_SYNTH_DIR_GATE']),
}

print(f"=== {DS}: 검은배경 위 결함 샘플 비율 (배경 mean<30 & std<12) ===")
print(f"{'cond':8s} | {'OFF black/tot':>16} {'ON black/tot':>16}")
print("-"*46)
for cond, (off_dir, on_dir) in COND_DIRS.items():
    _, _, (boff, toff) = bg_patch_stats(off_dir, DS)
    _, _, (bon,  ton)  = bg_patch_stats(on_dir,  DS)
    roff = f"{boff}/{toff} ({100*boff/toff:.1f}%)" if toff else "N/A"
    ron  = f"{bon}/{ton} ({100*bon/ton:.1f}%)"     if ton  else "N/A"
    print(f"{cond:8s} | {roff:>16} {ron:>16}")

# 분포 시각화 (aroma 기준)
moff, soff, _ = bg_patch_stats(COND_DIRS['aroma'][0], DS)
mon,  son,  _ = bg_patch_stats(COND_DIRS['aroma'][1], DS)
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
if len(moff): axs[0].hist(moff, bins=40, alpha=0.5, label=f"OFF (med={np.median(moff):.0f})")
if len(mon):  axs[0].hist(mon,  bins=40, alpha=0.5, label=f"ON  (med={np.median(mon):.0f})")
axs[0].set_title(f"{DS} aroma — 배경 밝기(mean) 분포"); axs[0].set_xlabel("bg mean (0..255)"); axs[0].legend()
if len(soff): axs[1].hist(soff, bins=40, alpha=0.5, label=f"OFF (med={np.median(soff):.1f})")
if len(son):  axs[1].hist(son,  bins=40, alpha=0.5, label=f"ON  (med={np.median(son):.1f})")
axs[1].set_title(f"{DS} aroma — 배경 대비(std) 분포"); axs[1].set_xlabel("bg std"); axs[1].legend()
plt.tight_layout(); plt.show()
```

> **합격 기준**:
> - severstal/mvtec(full-frame): ON 에서 **검은배경 비율이 OFF 대비 감소**(또는 둘 다 ≈0 이면 애초에 void 가 적은 데이터 → 게이트가 해 끼치지 않음 확인). 배경 mean 분포의 저밝기 꼬리(좌측)가 ON 에서 얇아지면 게이트 작동.
> - visa_pcb(object-centric): 검은배경 비율은 정상적으로 높을 수 있으나, **Cell 3 의 n_generated ratio≥0.8** 가 우선 합격 조건. 분포는 참고용.

---

## Cell 4B — foreground-mask probe (게이트가 severstal에서 무효인 이유 직접 규명)

**배경**: Cell 4 에서 severstal OFF==ON 이 **byte-identical**(검은배경 비율·개수 완전 동일)로 나오면, clean-bg 게이트가 severstal 합성을 전혀 바꾸지 못한 것이다. 코드 추적상 그 원인은 — pool 게이트는 검은 normal 이 없어 R=0(no-op), **position 게이트는 폴백(`_random_paste_position`) 경로 한정**인데 severstal 이 `_foreground_mask` 로 전경을 찾아 **foreground 배치 경로로 들어가면 position 게이트를 우회**하기 때문이다. 이 셀은 그 가설을 **실제 `_foreground_mask` 함수를 import 해** 데이터로 확정한다(재구현 아님).

> 전제: Cell 1(`!pip install opencv-python-headless`) 실행 완료, `cv2` 가용. `HAS_CV2 False` 면 `_foreground_mask` 가 항상 None → 결과 해석 무의미(먼저 cv2 설치).

### Part A — `_foreground_mask` None 비율 + 전경 밝기

```python
import os, sys, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# 실제 코드의 함수 그대로 import — 진짜 동작을 검증(재구현 금지)
sys.path.insert(0, os.environ['AROMA_SCRIPTS'])     # /content/AROMA/scripts/aroma
from generate_defects import _foreground_mask, _is_clean_background, HAS_CV2
print("HAS_CV2:", HAS_CV2, "(False면 _foreground_mask 항상 None → 해석 주의)")

NORMAL_DIR = f"{os.environ['AROMA_DATA']}/severstal/train/good"   # 합성에 쓴 바로 그 풀
IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
paths = sorted(str(p) for p in Path(NORMAL_DIR).rglob("*") if p.suffix.lower() in IMG_EXTS)
N = min(600, len(paths))
print(f"normal images: {len(paths)}, probing {N}")

n_none = 0
fg_ratios, fg_means, bg_means = [], [], []
n_dark_fg = 0
for p in paths[:N]:
    rgb  = np.asarray(PILImage.open(p).convert("RGB"))   # 합성과 동일 입력 형식
    gray = rgb.mean(axis=2)
    fg = _foreground_mask(rgb)                            # ← 실제 함수 호출
    if fg is None:
        n_none += 1
        continue
    m = fg >= 128
    fg_mean = float(gray[m].mean())   if m.any()    else float('nan')
    bg_mean = float(gray[~m].mean())  if (~m).any() else float('nan')
    fg_ratios.append(float(m.mean())); fg_means.append(fg_mean); bg_means.append(bg_mean)
    # '어두운 전경' = void 를 전경으로 오검출했을 신호: 전경이 배경보다 어둡거나 절대적으로 어둡다
    if (not np.isnan(fg_mean)) and (fg_mean < bg_mean - 10 or fg_mean < 40):
        n_dark_fg += 1

n_fg = N - n_none
print(f"\n_foreground_mask (N={N}):")
print(f"  None  → 폴백 경로(position 게이트 작동) : {n_none:4d} ({100*n_none/N:.1f}%)")
print(f"  non-None → foreground 배치(게이트 우회) : {n_fg:4d} ({100*n_fg/N:.1f}%)")
if n_fg:
    print(f"     └ '어두운 전경'(void 오검출 의심)   : {n_dark_fg:4d} ({100*n_dark_fg/n_fg:.1f}%)")
    print(f"     └ 전경 밝기 median={np.median(fg_means):.0f} / 배경 밝기 median={np.median(bg_means):.0f} / 전경비율 median={np.median(fg_ratios):.2f}")

# 전경 vs 배경 밝기 분포
if n_fg:
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(fg_means, bins=40, alpha=0.5, label=f"foreground (med={np.median(fg_means):.0f})")
    ax.hist(bg_means, bins=40, alpha=0.5, label=f"background (med={np.median(bg_means):.0f})")
    ax.set_title("severstal — _foreground_mask 가 잡은 전경 vs 배경 밝기"); ax.set_xlabel("mean(0..255)"); ax.legend()
    plt.tight_layout(); plt.show()
```

### Part B — Cell 4 가 잡은 '실제 검은배경 샘플'이 foreground 경로로 배치됐는지 교차 확인

```python
# Cell 4 의 load_anns/resolve_path/bbox_of 가 메모리에 있어야 한다(Cell 4 먼저 실행).
DS = "severstal"
def _norm_path(nrm):
    if not nrm: return None
    c = Path(nrm)
    if c.exists(): return str(c)
    b = Path(NORMAL_DIR) / Path(nrm).name      # staging 절대경로 → basename 으로 복원
    return str(b) if b.exists() else None

def crosscheck(synth_root, ds, max_n=400):
    base, items = load_anns(synth_root, ds)
    n_dark = n_dark_via_fg = n_dark_via_fallback = n_dark_nonorm = 0
    for e in items[:max_n]:
        img_p = resolve_path(base, e.get("image_path"));  bb = bbox_of(e)
        if not img_p or bb is None: continue
        import cv2
        comp = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if comp is None: continue
        H, W = comp.shape[:2]; x,y,w,h = bb
        pad = max(8, int(0.3*max(w,h)))
        x0,y0,x1,y1 = max(0,x-pad),max(0,y-pad),min(W,x+w+pad),min(H,y+h+pad)
        reg = comp[y0:y1, x0:x1].astype(np.float32)
        ring = reg.copy(); iy0,ix0 = y-y0,x-x0
        ring[max(0,iy0):iy0+h, max(0,ix0):ix0+w] = np.nan
        vals = ring[~np.isnan(ring)]
        if vals.size == 0: continue
        if not (float(vals.mean()) < 30 and float(vals.std()) < 12):  # Cell 4 와 동일 '검은' 정의
            continue
        n_dark += 1
        # 이 검은 샘플의 normal 에 _foreground_mask 적용 → bbox 중심이 전경 안인가?
        np_path = _norm_path(e.get("normal_image"))
        if np_path is None: n_dark_nonorm += 1; continue
        fg = _foreground_mask(np.asarray(PILImage.open(np_path).convert("RGB")))
        cx, cy = int(x + w/2), int(y + h/2)
        if fg is not None and 0 <= cy < fg.shape[0] and 0 <= cx < fg.shape[1] and fg[cy, cx] >= 128:
            n_dark_via_fg += 1          # 전경 픽셀 위 = foreground 배치 경로 = 게이트 우회
        else:
            n_dark_via_fallback += 1    # 전경 밖/None = 폴백 경로였어야(게이트가 잡았어야)
    return n_dark, n_dark_via_fg, n_dark_via_fallback, n_dark_nonorm

for cond, root in (("aroma", os.environ['AROMA_SYNTH_DIR']),
                   ("random", os.environ['RANDOM_SYNTH_DIR'])):
    nd, via_fg, via_fb, nonorm = crosscheck(root, DS)
    print(f"[{cond}] 검은배경 샘플 {nd}장 중 — foreground경로(우회)={via_fg}, 폴백경로={via_fb}, normal결측={nonorm}")
```

> **판정**:
> - **Part A 에서 non-None 비율이 높다** → severstal 은 foreground 배치 경로를 탄다 → position 게이트(폴백 한정) **구조적 우회 확정** → Cell 4 byte-identical 설명됨.
> - **non-None 중 '어두운 전경' 비율이 높다** → `_foreground_mask` 가 void 를 전경으로 오검출 → 결함을 어두운 곳에 놓음 → **검은배경 결함의 직접 원인**.
> - **Part B 에서 검은 샘플 대부분이 `via_fg`(전경경로)** → 그 검은배경 결함들은 게이트가 손댈 수 없는 경로로 배치됐음이 데이터로 확정.
> - 반대로 **Part A None 비율이 높거나 Part B `via_fallback` 이 많다면** → 폴백에서 position 게이트가 돌았어야 하는데 byte-identical 이므로 모순 → 게이트 quality 공식이 그 패치를 통과(0.7↑)시킨 것 → 게이트 임계/공식 재검토 대상.
>
> **결론 방향**: severstal 의 검은배경을 실제로 줄이려면 void 체크를 **폴백이 아니라 `_foreground_paste_position`(전경 배치 경로)** 에 넣어야 한다. 현 게이트 설계로는 전경배치 데이터셋에서 무효.

---

## Cell 5 — 검은배경 결함 샘플 육안 비교 (OFF만, 게이트 효과 시각 확인)

OFF 합성에서 배경이 검은 상위 샘플을 띄워, ON 에서 이런 샘플이 사라졌는지 눈으로 확인한다.

```python
import matplotlib.patches as mpatches

def darkest_bg_samples(synth_root, ds, k=8):
    base, items = load_anns(synth_root, ds)
    scored = []
    for e in items:
        img_p = resolve_path(base, e.get("image_path"))
        bb = bbox_of(e)
        if not img_p or bb is None: continue
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        H, W = img.shape[:2]
        x, y, w, h = bb
        pad = max(8, int(0.3*max(w, h)))
        reg = img[max(0,y-pad):min(H,y+h+pad), max(0,x-pad):min(W,x+w+pad)].astype(np.float32)
        if reg.size == 0: continue
        scored.append((float(np.mean(reg)), img_p, bb))
    scored.sort(key=lambda t: t[0])   # 가장 어두운 것부터
    return scored[:k]

DS = "severstal"
for cond, (off_dir, on_dir) in COND_DIRS.items():
    samples = darkest_bg_samples(off_dir, DS, k=8)
    if not samples: continue
    fig, axs = plt.subplots(2, 4, figsize=(16, 8)); axs = np.array(axs).reshape(-1)
    for ax in axs: ax.axis("off")
    for ax, (m, p, bb) in zip(axs, samples):
        ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        x, y, w, h = bb
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", lw=1.5))
        ax.set_title(f"bg-region mean={m:.0f}", fontsize=8)
    fig.suptitle(f"[OFF] {DS} {cond} — 가장 어두운 배경 8장 (ON 에선 줄어야)", fontsize=12)
    plt.tight_layout(); plt.show()
```

> 위 OFF 샘플 중 명백히 검은/void 위에 결함이 붙은 것이 ON 합성셋에는 없거나 크게 줄어야 한다. ON 디렉토리를 같은 함수에 넣어 교차 확인 가능.

---

## Cell 6 — 결정론 확인 (동일 seed 2회 → normal_image 선택 동일)

게이트 ON 상태로 **동일 seed 로 2회** 합성해, annotations 의 `normal_image`(배경 선택) 시퀀스가 byte-identical 한지 확인한다.

```python
import os
# seed 42 로 두 번, 별도 출력 디렉토리
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $AROMA_OUT/roi/severstal --normal_dir $AROMA_DATA/severstal/train/good \
    --output_dir $AROMA_OUT/_det_run1/severstal --n_per_roi $N_PER_ROI --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --local_staging

!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $AROMA_OUT/roi/severstal --normal_dir $AROMA_DATA/severstal/train/good \
    --output_dir $AROMA_OUT/_det_run2/severstal --n_per_roi $N_PER_ROI --seed 42 \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --local_staging
```

```python
import json, os
from pathlib import Path

def normal_seq(root, ds):
    base = Path(root) / ds
    raw = json.loads((base / "annotations.json").read_text())
    items = (raw.get("annotations") or raw.get("items") or []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    items = [e for e in items if isinstance(e, dict) and e.get("dry_run") is not True]
    # 결정론은 합성 순서에 대한 것 — annotations 기록 순서 그대로 normal_image 추출
    # (staging 절대경로 차이를 무시하려 basename 으로 비교)
    return [Path(str(e.get("normal_image") or "")).name for e in items]

s1 = normal_seq(f"{os.environ['AROMA_OUT']}/_det_run1", "severstal")
s2 = normal_seq(f"{os.environ['AROMA_OUT']}/_det_run2", "severstal")
print(f"run1 n={len(s1)}  run2 n={len(s2)}")
same = (s1 == s2)
print("normal_image 선택 시퀀스 동일:", same)
if not same:
    # 첫 불일치 위치 출력
    for i, (a, b) in enumerate(zip(s1, s2)):
        if a != b:
            print(f"  첫 불일치 @ idx {i}: run1={a!r}  run2={b!r}")
            break
    print("  ⚠️ 결정론 위반 — rng 사용 경로 점검 필요")
else:
    print("  ✅ 결정론 OK — 동일 seed → 동일 배경 선택")
```

> **합격 기준**: `same == True`. 게이트의 거부/재시도가 동일 `rng` 를 쓰고 품질 평가가 RNG-free(Laplacian/filter2D) 이므로, 동일 seed → 동일 풀·동일 선택이어야 한다. basename 비교로 staging 절대경로 차이는 무시한다.

---

## 판정 요약

| 검증 | Cell | 합격 기준 |
|------|------|-----------|
| pool 게이트 작동 | 2 로그 | `clean-bg pool gate: kept K/N (R rejected ...)` 출력, void 많은 데이터셋에서 `R>0` |
| 전량 거부 폴백 | 2 로그 | 전부-검은 풀에서 silent 0 아닌 warning + 원본 풀 사용 |
| 검은배경 결함 비율 ↓ | 4 | severstal/mvtec ON 비율 < OFF 비율 (또는 둘 다 ≈0) |
| 배경 분포 개선 | 4 | ON 배경 mean 분포의 저밝기 꼬리 감소 |
| visa 회귀 없음 | 3 | visa_pcb `n_generated` ON/OFF ratio ≥ 0.8 |
| 결정론 | 6 | 동일 seed 2회 → `normal_image` 시퀀스 동일 |

> 모든 항목 통과 시: 게이트가 (1) 검은배경 합성을 생성 시점에 차단하고 (2) object-centric 데이터셋을 과도 거부하지 않으며 (3) 결정론을 보존함이 확인된다.
> visa ratio < 0.8 이면 데이터셋별 임계 튜닝(남은 TODO) 대상이며, 기본값 OFF 이므로 기존 exp4v2 파이프라인에는 영향 없다.
