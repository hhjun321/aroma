# AROMA Exp 5 — PRDC 커버리지 평가 실행 가이드 (Workflow)

**목적**: aroma vs random 합성의 real 결함 manifold 커버리지 비교 — **외부 임베딩 좌표계(DINOv2)** 에서 ROI 선택 가치를 반증 가능하게 검증 (저비용 L2 증거, `low_compute_validation_plan.md` 1순위)
**데이터셋 (v2-1 확정 4종)**: `severstal`·`mvtec_leather`·`aitex`·`mtd`
**런타임**: 임베딩 추출 T4 데이터셋당 1–3분(캐시 후 0), PRDC·permutation은 CPU. **전체 30분 이내**

---

## 사전 등록 가설 (판정 기준)

aroma/random은 **동일 copy-paste 엔진** → 차이는 ROI **선택 전략**뿐. 따라서:

| 지표 | 예측 | 판정 |
|------|------|------|
| Precision / Density | **두 조건 동등** | \|Δ\|가 permutation null 95% CI 내 |
| Recall / Coverage | **aroma > random** | Δ>0 AND one-sided p < 0.05 (4종 방향 일치) |

> Recall 계열만 오르고 Precision 계열이 동등해야 "선택의 가치" 입증. 둘 다 오르거나 Precision이 깨지면 가설 기각 — 이 비대칭 예측이 사후합리화 반박을 차단한다.

**선행 전제**: 4종의 aroma(`synthetic/`)·random(`synthetic_random/`) 합성 존재 (mvtec_leather aroma 미생성 시 해당 데이터셋 skip 로그 후 제외 — step4로 생성 후 재실행).

---

## STEP 0 — 환경변수

```python
import os

os.environ['AROMA_OUT']     = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_SCRIPTS'] = "/content/AROMA/scripts/aroma"
os.environ['AROMA_DATA']    = f"{os.environ['DRIVE']}/Aroma"

os.environ['RANDOM_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['AROMA_SYNTH_DIR']  = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['EXP5_OUT']         = f"{os.environ['AROMA_OUT']}/exp5_prdc"
os.environ['EMBED_CACHE_DIR']  = f"{os.environ['AROMA_OUT']}/embed_cache"

print("EXP5_OUT        :", os.environ['EXP5_OUT'])
print("EMBED_CACHE_DIR :", os.environ['EMBED_CACHE_DIR'])
```

> `EMBED_CACHE_DIR`는 후속 실험(kNN test-coverage·rare-mode)이 **재사용**하는 공유 캐시 — 통상 삭제하지 말 것.
> ⚠️ **단, 합성 재생성 후에는 반드시 무효화**: 캐시 키가 경로 기반(sha256 of image_path 목록)이라, `generate_defects`/`generate_random` 재실행이 **동일 파일명**(`syn_00000_00.jpg`)으로 내용을 덮어쓰면 stale 임베딩이 재사용된다. 재합성한 데이터셋은 `!rm -rf $EMBED_CACHE_DIR/{ds}` 후 재실행.

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
import os, json, random, pathlib, shutil

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
| `$EMBED_CACHE_DIR/{ds}/*.npy` | DINOv2 임베딩 캐시 (후속 kNN·rare-mode 실험 공유) |

## 주의사항

- **k cherry-pick 금지**: 주 보고는 k=5, 나머지 k는 sensitivity로 함께 제시 — 방향이 k에 따라 뒤집히면 결론 보류.
- **aitex n_ref 소표본**: val 결함 수가 적어 `unstable: true` 가능 — 수치는 보고하되 단독 결론 금지(aggregate 보조).
- **leakage 방어 명문화**: 논문에 "AROMA 선택은 test(val)를 보지 않으며, reference는 synth 소스(train split)와 분리된 held-out"임을 기술.
- **synth crop 폴백**: mask_path → bbox → **skip**(full-image crop 금지 — 배경이 임베딩 지배). `n_skipped_synth`가 크면 합성 산출물 점검.
- **백본 폴백**: DINOv2 로드 실패 시 InceptionV3 — 결과 JSON `meta.backbone`으로 어느 쪽인지 확인(혼용 비교 금지).
- **GPU 없이도 가능**: `--device cpu`로 실행 가능(임베딩만 느려짐, 수십 분).
