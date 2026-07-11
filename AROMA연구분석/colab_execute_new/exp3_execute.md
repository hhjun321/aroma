# exp3 — Cross-Domain 생성 품질 평가 실행 가이드 (sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 입력 루트는 `_SPEC §2` output 규약(`S('synth_aroma')` / `S('synth_random')`)만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: Random ROI vs AROMA ROI 2-way 생성 품질 비교 (FID + PaDiM AUROC).
**런타임**: FID 모드 = **CPU 가능** | AD 모드 = **GPU 필수** (Colab Pro A100 권장).
**데이터셋**: v2-1 4종 `severstal · mvtec_leather · aitex · mtd`. aitex = tiled(256×256/stride128, single-class).

**전제 (step5 완료)**: step5에서 4종 모두 AROMA arm(`S('synth_aroma', ds)`)·random arm(`S('synth_random', ds)`) 합성본이 생성되어 있어야 한다.
- **AROMA 입력은 이제 aroma-sym(ControlNet + symmetric compat 게이트)** 산출이다 — 구 copy_paste arm이 아니다. step5의 `generate_defects.py --method controlnet --compat_mode symmetric`가 만든 결과를 그대로 소비한다.
- **clean-bg 게이트 parity는 step5에서 이미 적용**되어 있다 (AROMA·random 양 arm 모두 `--reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0` 동일 조건). 따라서 본 문서는 **합성 재생성을 하지 않는다** — 두 arm 모두 step5 산출을 읽기만 한다.

---

## STEP 0 — 공통 환경 셀 (`_SPEC §1` 그대로 — 수정 금지)

```python
import os, json

# ===== 공통 환경 (sym_final 전 문서 동일 — 수정 금지) =====
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
# ===== 단일 버전 루트 (stage-first: {stage}/{ds}) =====
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step4 산출, step5 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
```

### exp3 입력/출력 루트 (환경변수 export)

exp3 스크립트는 `--aroma_synthetic_dir` / `--random_synthetic_dir` 아래에 자동으로 `/{ds}`를 붙인다 (root-first 규약). 따라서 **루트(=`S('synth_aroma')` / `S('synth_random')`)만** 넘긴다.

```python
os.environ['SYNTH_AROMA']  = S('synth_aroma')    # sym_final/synth_aroma  (스크립트가 /{ds} 부착)
os.environ['SYNTH_RANDOM'] = S('synth_random')   # sym_final/synth_random (스크립트가 /{ds} 부착)
os.environ['EXP3_OUT']     = S('exp3')           # sym_final/exp3

print("SYNTH_AROMA  :", os.environ['SYNTH_AROMA'])
print("SYNTH_RANDOM :", os.environ['SYNTH_RANDOM'])
print("AROMA_DATA   :", os.environ['AROMA_DATA'])
print("EXP3_OUT     :", os.environ['EXP3_OUT'])
```

---

## STEP 1 — 패키지 설치 (최초 1회)

```python
!pip install torchmetrics[image] anomalib lpips -q
```

---

## STEP 2 — 전제 확인 (step5 합성본 존재)

두 arm 모두 4종의 합성 이미지가 있어야 진행한다. 없으면 step5를 먼저 완료할 것.

```python
import pathlib

print(f"{'데이터셋':<16} {'aroma-sym':>12} {'random':>12}")
print("-" * 42)
missing = []
for ds in DATASETS:
    counts = {}
    for label, root in [("aroma", os.environ['SYNTH_AROMA']), ("random", os.environ['SYNTH_RANDOM'])]:
        p = pathlib.Path(f"{root}/{ds}/images")
        n = len(list(p.glob("*.jpg")) + list(p.glob("*.png"))) if p.exists() else 0
        counts[label] = n
        if n == 0: missing.append(f"{label}/{ds}")
    print(f"{ds:<16} {counts['aroma']:>12} {counts['random']:>12}")
print("\nMISSING:", missing if missing else "없음 (진행 가능)")
```

> AROMA 쪽 이미지는 **aroma-sym**(ControlNet + symmetric 게이트) 산출이다. count가 0이면 step5의 AROMA arm(`generate_defects.py --method controlnet --compat_mode symmetric`)이 완료되지 않은 것이다.

---

## STEP 3 — FID 평가 (CPU 가능)

실제 결함 패치 vs 합성 결함 패치의 분포 거리 측정. `--device cpu`로 실행한다.

```python
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode                 fid \
    --random_synthetic_dir $SYNTH_RANDOM \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         severstal mvtec_leather aitex mtd \
    --output_dir           $EXP3_OUT \
    --seed                 42 \
    --device               cpu
```

**FID 결과 확인:**

```python
import json

with open(f"{os.environ['EXP3_OUT']}/exp3_results.json") as f:
    results = json.load(f)

print(f"{'데이터셋':<16} {'FID_Random':>12} {'FID_AROMA':>12} {'n_real':>8} {'unstable':>10}")
print("-" * 62)
for ds, data in sorted(results.items()):
    fd = data.get("fid", {})
    r  = fd.get("random", {})
    a  = fd.get("aroma", {})
    fid_r = f"{r['fid']:.2f}" if isinstance(r.get("fid"), float) else "N/A"
    fid_a = f"{a['fid']:.2f}" if isinstance(a.get("fid"), float) else "N/A"
    n_real = fd.get("n_real_patches", "?")
    unstable = "⚠" if (r.get("fid_unstable") or a.get("fid_unstable")) else ""
    print(f"{ds:<16} {fid_r:>12} {fid_a:>12} {str(n_real):>8} {unstable:>10}")
```

---

## STEP 4 — AD 평가 (GPU 필수)

PaDiM 3조건(baseline / random / aroma-sym) 학습 및 AUROC 측정. **Colab Pro A100 권장.**

```python
!python $AROMA_SCRIPTS/experiments/exp3_generation_quality.py \
    --mode                 ad \
    --random_synthetic_dir $SYNTH_RANDOM \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --real_data_dir        $AROMA_DATA \
    --dataset_keys         severstal mvtec_leather aitex mtd \
    --output_dir           $EXP3_OUT \
    --seed                 42 \
    --image_size           256
```

**AD 결과 확인:**

```python
import json

with open(f"{os.environ['EXP3_OUT']}/exp3_results.json") as f:
    results = json.load(f)

print(f"{'데이터셋':<16} {'조건':<10} {'Image AUROC':>12} {'Pixel AUROC':>12}")
print("-" * 54)
for ds, data in sorted(results.items()):
    ad = data.get("ad", {})
    for cond in ["baseline", "random", "aroma"]:
        m  = ad.get(cond, {})
        ia = f"{m['image_auroc']:.4f}" if isinstance(m.get("image_auroc"), float) else "N/A"
        pa = f"{m['pixel_auroc']:.4f}" if isinstance(m.get("pixel_auroc"), float) else "N/A"
        print(f"{ds:<16} {cond:<10} {ia:>12} {pa:>12}")
    print()
```

---

## STEP 5 — Markdown 요약 확인

```python
with open(f"{os.environ['EXP3_OUT']}/exp3_summary.md") as f:
    print(f.read())
```

---

## 판정

| 지표 | 기대 방향 |
|------|---------|
| FID | **aroma-sym < Random** (낮을수록 실제 결함 분포에 가까움) |
| Image AUROC | **aroma-sym > Random > Baseline** |
| Pixel AUROC | **aroma-sym > Random > Baseline** |

- **FID**: aroma-sym의 FID가 Random보다 낮으면, ControlNet + symmetric 호환 게이트가 만든 결함 패치가 real 결함 패치 분포에 더 근접함을 의미한다. clean-bg 게이트 parity는 step5에서 이미 양 arm 동일하게 적용되었으므로, 남은 차이는 **selection(deficit_aware/realism) + 생성(controlnet/symmetric)** 기여로 해석한다.
- **AD(PaDiM AUROC)**: 합성본을 train에 섞었을 때 이상탐지 성능이 얼마나 오르는가를 본다. aroma-sym 조건이 random·baseline을 상회하면 생성 품질이 다운스트림 유용성으로 이어진다는 근거다. baseline은 합성 미포함(real train only) 기준선이다.
- 방향이 뒤집히거나 데이터셋별로 엇갈리면 **사후 튜닝하지 말고** 그대로 보고한다(무결성 §). 특히 aitex는 tile-level·single-class라 절대값 비교가 아니라 Δ(조건 간 차이)만 유효하다.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP3_OUT/exp3_results.json` | 데이터셋 × FID + AD 수치 |
| `$EXP3_OUT/exp3_summary.md` | Markdown 비교 테이블 + delta 섹션 |
| `$EXP3_OUT/checkpoints/{ds}/{cond}/` | PaDiM 체크포인트 |

---

## 공통 무결성 / 정직 (`_SPEC §5`)

- **사후 튜닝 금지**: seed·image_size 등은 위 커맨드 값을 그대로 쓰고, 결과 보고 후 변경하지 않는다. FID/AUROC 방향이 기대와 달라도 재생성·재선택으로 되돌리지 않는다.
- **합성 재생성 금지 (parity)**: 본 문서는 step5 산출(`synth_aroma`/`synth_random`)을 **읽기만** 한다. clean-bg 게이트는 step5에서 양 arm 동일 조건으로 이미 적용되었다 — exp3에서 게이트를 다시 켜거나 합성을 다시 만들면 parity가 깨진다.
- **aroma-sym 입력 명시**: AROMA arm은 ControlNet + symmetric compat 게이트(구 copy_paste 아님) 산출이다. 결과 해석·논문 기술 시 이 arm 정의를 정확히 표기한다.
- **FID 소표본 경고**: `fid_unstable: true`이면 real 결함 패치가 50개 미만이라는 뜻 — 해당 값은 참고용으로만 쓰고 결론 근거로 삼지 않는다.
- **aitex tile-level 주석**: aitex는 tiled(256×256/stride128)·single-class다. FID/AUROC **절대값을 타 데이터셋과 직접 비교 금지**, 동일 데이터셋 내 조건 간 Δ만 유효하다.
- **Test set 원칙**: 합성 이미지는 train에만 포함. test/defect는 항상 real 이미지만 사용한다(스크립트 내장 규칙).
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행** (load-test policy).
