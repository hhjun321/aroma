# exp5 — PRDC 커버리지 평가 실행 가이드 (sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 입력·출력 루트 규약은 `_SPEC §2·§3`(exp*)만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: aroma vs random 합성의 real 결함 manifold 커버리지 비교 — **외부 임베딩 좌표계(DINOv2)** 에서 ROI 선택 가치를 반증 가능하게 검증 (저비용 L2 증거, `low_compute_validation_plan.md` 1순위).

**AROMA arm 정의**: 여기서 비교되는 aroma 합성은 **aroma-sym**(step4 = `generate_defects --method controlnet` + `--compat_mode symmetric` + clean-bg 게이트 + seamless 블렌딩)의 산출물이다. random arm은 동일 clean-bg 게이트를 통과한 `generate_random.py` 통제군이다. 두 arm 모두 `S('synth_aroma')`·`S('synth_random')`(step4 산출)에서 읽는다.

**데이터셋 (v2-1 확정 4종)**: `severstal · mvtec_leather · aitex · mtd`. **aitex = tiled(256×256/stride128, single-class)**.

**런타임**: 임베딩 추출 T4 데이터셋당 1–3분(캐시 후 0), PRDC·permutation은 CPU. **전체 30분 이내**.

**전제**: step4까지 완료되어 4종의 aroma(`S('synth_aroma',ds)/annotations.json`)·random(`S('synth_random',ds)/annotations.json`) 합성이 존재한다. mvtec_leather aroma 미생성 시 해당 데이터셋 skip 로그 후 제외(step4 생성 후 재실행).

---

## 사전 등록 가설 (판정 기준)

aroma/random은 **동일 copy-paste/블렌딩 엔진 + 동일 clean-bg 게이트** → 차이는 ROI **선택 전략**뿐. 따라서:

| 지표 | 예측 | 판정 |
|------|------|------|
| Precision / Density | **두 조건 동등** | \|Δ\|가 permutation null 95% CI 내 |
| Recall / Coverage | **aroma > random** | Δ>0 AND one-sided p < 0.05 (4종 방향 일치) |

> Recall 계열만 오르고 Precision 계열이 동등해야 "선택의 가치" 입증. 둘 다 오르거나 Precision이 깨지면 가설 기각 — 이 비대칭 예측이 사후합리화 반박을 차단한다.

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
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step5 산출, step4 소비)
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p

DATASETS = ["severstal", "mvtec_leather", "mtd", "aitex"]   # v2-1 4종
with open(os.environ['DATASET_CONFIG']) as f: CFG = json.load(f)
def normal_dir(ds): return CFG[ds]["image_dir"]                 # aitex → aitex_tiled/train/good
def is_multi(ds):   return CFG[ds].get("class_mode") == "multi" # aitex=single (자동)
```

### exp5 입력·출력 루트 (stage-first `S()` → 환경변수)

`--aroma_synthetic_dir`/`--random_synthetic_dir`는 **루트만** 넘기고 스크립트가 `/{ds}`를 붙인다(`{root}/{ds}/annotations.json`). `colab-execution.md` 규약대로 `S()` 값을 환경변수로 고정한 뒤 `!python` 셀에서 `$VAR`로 참조한다.

```python
os.environ['AROMA_SYNTH_DIR']  = S('synth_aroma')    # step4 산출 (aroma-sym)
os.environ['RANDOM_SYNTH_DIR']  = S('synth_random')   # step4 산출 (통제)
os.environ['EMBED_CACHE_DIR']  = S('embed_cache')    # exp5·exp6 공유 캐시
os.environ['EXP5_OUT']         = S('exp5')

for k in ('AROMA_SYNTH_DIR', 'RANDOM_SYNTH_DIR', 'EMBED_CACHE_DIR', 'EXP5_OUT'):
    print(f"{k:18s}: {os.environ[k]}")
```

> `EMBED_CACHE_DIR = S('embed_cache')`는 **exp6(knn·rare)가 재사용**하는 공유 캐시 — 통상 삭제하지 말 것.
> ⚠️ **단, 합성 재생성 후에는 반드시 무효화**: 캐시 키가 경로 기반(sha256 of image_path 목록)이라, step4 재실행이 **동일 파일명**(`syn_00000_00.jpg`)으로 내용을 덮어쓰면 stale 임베딩이 재사용된다. 재합성한 데이터셋은 `!rm -rf $EMBED_CACHE_DIR/{ds}` 후 재실행.

### 전제 확인 (합성 존재)

```python
import pathlib
for ds in DATASETS:
    a = pathlib.Path(f"{S('synth_aroma', ds)}/annotations.json")
    r = pathlib.Path(f"{S('synth_random', ds)}/annotations.json")
    print(f"{ds:14s} aroma={'✓' if a.exists() else '✗'}  random={'✓' if r.exists() else '✗'}")
```

> aroma가 ✗인 데이터셋은 step4 미완 — exp5는 자동 skip 로그를 남기며, step4 생성 후 재실행한다.

---

## STEP 1 — 패키지 설치

```python
!pip install prdc -q
# torch/torchvision 은 Colab 기본 제공. DINOv2 는 torch.hub 에서 자동 다운로드.
```

## STEP 2 — 스모크 (severstal 단독, reps=100)

```python
!python $AROMA_SCRIPTS/experiments/exp5_prdc.py \
    --real_data_dir        $AROMA_DATA \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --dataset_keys severstal \
    --nearest_k 3 5 10 \
    --permutation_reps 100 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP5_OUT/_smoke
```

**통과 기준**: JSON에 `k3/k5/k10` 존재, `meta.n`이 두 조건 동일(equalize), 로그에 `parity mismatch` 경고 **없음**(prdc 패키지 vs 벡터화 permutation 경로 일치 검증).

## STEP 3 — 음성 대조 (permutation 구현 sanity)

동일 synth 풀을 반으로 갈라 aroma/random 자리에 넣으면 **Δ≈0, p가 균등분포 근처**여야 한다:

```python
import os, json, random, pathlib

# random 합성 하나를 반분해 가짜 aroma/random 구성
SRC = f"{os.environ['RANDOM_SYNTH_DIR']}/severstal/annotations.json"
NEG = "/content/tmp/exp5_negctl"
anns = json.load(open(SRC))
random.Random(0).shuffle(anns)
half = len(anns) // 2
for name, part in (("a", anns[:half]), ("b", anns[half:])):
    d = pathlib.Path(f"{NEG}/{name}/severstal"); d.mkdir(parents=True, exist_ok=True)
    json.dump(part, open(d / "annotations.json", "w"))

!python $AROMA_SCRIPTS/experiments/exp5_prdc.py \
    --real_data_dir        $AROMA_DATA \
    --aroma_synthetic_dir  /content/tmp/exp5_negctl/a \
    --random_synthetic_dir /content/tmp/exp5_negctl/b \
    --dataset_keys severstal \
    --nearest_k 5 --permutation_reps 500 \
    --output_dir /content/tmp/exp5_negctl/out

neg = json.load(open("/content/tmp/exp5_negctl/out/exp5_prdc_results.json"))
k5 = neg["severstal"]["k5"]
print("Δ:", k5["delta"], "\np:", k5["p_one_sided"])   # Δ≈0, p ≈ 0.2~0.8 기대
```

## STEP 4 — 전체 실행 (4종)

```python
!python $AROMA_SCRIPTS/experiments/exp5_prdc.py \
    --real_data_dir        $AROMA_DATA \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --nearest_k 3 5 10 \
    --permutation_reps 1000 \
    --val_frac 0.3 --split_seed 42 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP5_OUT
```

> `--val_frac 0.3 --split_seed 42` = **exp4v2 운영 규약과 동일 split** — reference(held-out val)가 downstream 실험과 정렬됨. 바꾸면 leakage 방어 논리가 깨지므로 고정.

## STEP 5 — 결과 확인 / 판정

```python
import json, os

with open(f"{os.environ['EXP5_OUT']}/exp5_prdc_results.json") as f:
    res = json.load(f)

print(f"{'dataset':<14} {'ΔPrec':>8} {'ΔDens':>8} {'ΔRecall(p)':>16} {'ΔCover(p)':>16} {'판정':>12}")
print("-" * 80)
for ds in sorted(res):
    node = res[ds]
    if node.get("skipped"):
        print(f"{ds:<14} skip: {node.get('reason')}"); continue
    kk = node.get("k5", {})
    if kk.get("skipped"):
        print(f"{ds:<14} k5 skip"); continue
    d, p = kk["delta"], kk["p_one_sided"]
    ok = d["recall"] > 0 and d["coverage"] > 0 and p["recall"] < 0.05 and p["coverage"] < 0.05
    flag = "✅" if ok else "❌"
    if node["meta"].get("unstable"): flag += " ⚠n_ref<30"
    print(f"{ds:<14} {d['precision']:>+8.4f} {d['density']:>+8.4f} "
          f"{d['recall']:>+8.4f} ({p['recall']:.3f}) {d['coverage']:>+8.4f} ({p['coverage']:.3f}) {flag:>12}")
```

Markdown 요약: `$EXP5_OUT/exp5_prdc_summary.md` (k=5 주표 + k-sensitivity 부표 + 판정).

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP5_OUT/exp5_prdc_results.json` | 데이터셋 × k별 관측 PRDC(양 조건) + Δ + one-sided p + null 95% CI + meta(n/n_ref/backbone/split/skip 카운트) |
| `$EXP5_OUT/exp5_prdc_summary.md` | 주표(k=5) + k-sensitivity + 가설 판정 |
| `$EMBED_CACHE_DIR/{ds}/*.npy` | DINOv2 임베딩 캐시 (exp6 kNN·rare-mode 실험과 **공유**) |

## 공통 무결성 / 정직 (`_SPEC §5`)

- **사후 튜닝 금지**: `val_frac`·`split_seed`·`nearest_k`·`permutation_reps`는 위 값 고정, 결과 보고 후 변경 금지.
- **k cherry-pick 금지**: 주 보고는 k=5, 나머지 k는 sensitivity로 함께 제시 — 방향이 k에 따라 뒤집히면 결론 보류.
- **aitex는 tile-level·single-class** → val 결함 수가 적어 `unstable: true` 가능. 절대값을 타 데이터셋과 직접 비교 금지, Δ만 유효하며 단독 결론 금지(aggregate 보조).
- **leakage 방어 명문화**: 논문에 "AROMA 선택은 test(val)를 보지 않으며, reference는 synth 소스(train split)와 분리된 held-out"임을 기술.
- **synth crop 폴백**: mask_path → bbox → **skip**(full-image crop 금지 — 배경이 임베딩 지배). `n_skipped_synth`가 크면 합성 산출물 점검.
- **백본 폴백**: DINOv2 로드 실패 시 InceptionV3 — 결과 JSON `meta.backbone`으로 어느 쪽인지 확인(혼용 비교 금지).
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행** (load-test policy). GPU 없이 `--device cpu`도 가능(임베딩만 느려짐).
