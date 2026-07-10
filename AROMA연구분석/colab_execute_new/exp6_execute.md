# exp6 — 임베딩 커버리지 실행 가이드 (knn + rare, sym_final)

> 이 문서는 `colab_execute_new/_SPEC.md`를 **정본**으로 따른다. STEP 0 환경 셀은 `_SPEC §1`을 그대로 복사한 것이며, 입력·출력 루트 규약은 `_SPEC §2·§3`(exp*)만 사용한다. env·output 루트 규약을 재발명하지 않는다.

**목적**: exp5(PRDC)와 함께 저비용 L2 증거 완성 — ① `--mode knn`: held-out val 결함까지의 최근접 거리로 **커버리지 기제** 측정, ② `--mode rare`: 독립 k-means 모드에서 **rare-mode 타겟팅** 검증 (`low_compute_validation_plan.md` 2·3순위).

**AROMA arm 정의**: knn 모드가 비교하는 aroma 합성은 **aroma-sym**(step4 = `generate_defects --method controlnet` + `--compat_mode symmetric` + clean-bg 게이트 + seamless 블렌딩)의 산출물이며, random arm은 동일 clean-bg 게이트를 통과한 `generate_random.py` 통제군이다(둘 다 `S('synth_aroma')`·`S('synth_random')`에서 읽음). rare 모드는 합성 산출이 아니라 **step3 ROI 선택 산출**(`S('roi',ds)/roi_selected.json`+`roi_candidates.json`)을 직접 평가한다.

**데이터셋 (v2-1 확정 4종)**: `severstal · mvtec_leather · aitex · mtd`. **aitex = tiled(256×256/stride128, single-class)**.

**런타임**: exp5 임베딩 캐시 재사용 시 **CPU 수 분**. 신규 임베딩(real_train·cand_src)만 T4 수 분.

---

## 사전 등록 가설

| 모드 | 가설 | 판정 |
|------|------|------|
| knn | Δ(1-NN dist) = mean d1(real+random) − mean d1(real+aroma) **> 0** | clustered-bootstrap p < 0.05, 4종 방향 일치 |
| rare | rare 모드(빈도 p25↓ AND val 등장) hit rate **aroma > random-null** | 전 그리드(k×cseed) 방향 일치 + 다수 셀 p_emp < 0.05 |

> **정보 분리**: AROMA 선택은 val/test를 보지 않는다(선택 입력 = train 결함 프로파일링) — val 기준 평가와 선택은 정보적으로 분리.
> **rare p 하한**: null 30-seed → p_emp 최소 ≈ 0.032. 그보다 낮은 p는 구조적으로 불가(해석 시 유의).
> **한계(정직)**: 두 모드 모두 **기제(mechanism) 증거** — downstream mAP 인과는 exp4v2(L3)가 담당. 단독 주장 금지, exp5와 패키지로 제시.

**전제**: exp5 STEP 4까지 실행돼 `EMBED_CACHE_DIR`에 real_val·synth 임베딩 캐시 존재(없어도 자동 생성 — 시간만 추가). rare 모드는 step3 산출(`S('roi',ds)/roi_selected.json`+`roi_candidates.json`) 필요.

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

### exp6 입력·출력 루트 (stage-first `S()` → 환경변수)

knn은 `--aroma_synthetic_dir`/`--random_synthetic_dir`(루트, 스크립트가 `/{ds}` 부착), rare는 `--roi_dir_root`(스크립트가 `{root}/{ds}/roi_*.json` 조회). `embed_cache_dir`는 **exp5와 공유**(`S('embed_cache')`). `colab-execution.md` 규약대로 `$VAR`로 참조.

```python
os.environ['AROMA_SYNTH_DIR']  = S('synth_aroma')    # step4 산출 (aroma-sym) — knn
os.environ['RANDOM_SYNTH_DIR']  = S('synth_random')   # step4 산출 (통제)      — knn
os.environ['ROI_DIR_ROOT']     = S('roi')            # step3 산출            — rare
os.environ['EMBED_CACHE_DIR']  = S('embed_cache')    # exp5·exp6 공유 캐시
os.environ['EXP6_OUT']         = S('exp6')

for k in ('AROMA_SYNTH_DIR', 'RANDOM_SYNTH_DIR', 'ROI_DIR_ROOT', 'EMBED_CACHE_DIR', 'EXP6_OUT'):
    print(f"{k:18s}: {os.environ[k]}")
```

> ⚠️ **재합성 후 캐시 무효화**(exp5 규약 동일): 합성/ROI 재생성 시 `!rm -rf $EMBED_CACHE_DIR/{ds}` 후 재실행. 캐시 키가 경로 기반이라 동일 파일명 덮어쓰기 시 stale 임베딩 재사용.
> ⚠️ **부분 재실행 주의**: 같은 모드를 `--dataset_keys` 일부만으로 재실행하면 그 모드 노드가 **통째로 교체**되어 다른 데이터셋 결과가 사라진다 — 재실행 시 항상 전체 4종을 지정하거나 fresh `--output_dir` 사용.
> ⚠️ **quality gate 정합**(rare): step3/step4에서 clean-bg·`--min_quality` 게이트를 켠 경우, null의 표본공간(`roi_candidates`)이 실제 random 생성과 같은 gated 풀인지 확인 — aroma/random/null 셋 다 동일 게이트여야 공정([[quality-gate-fairness 규약]]).

---

## STEP 1 — 패키지

추가 설치 불필요 — torch/sklearn Colab 기본, prdc 불요. DINOv2는 torch.hub 자동 다운로드.

## STEP 2 — knn 모드 실행

```python
!python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \
    --mode knn \
    --real_data_dir        $AROMA_DATA \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --val_frac 0.3 --split_seed 42 \
    --bootstrap_reps 2000 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP6_OUT
```

## STEP 3 — knn 결과 해석

```python
import json, os
res = json.load(open(f"{os.environ['EXP6_OUT']}/exp6_results.json"))["knn"]
for ds in sorted(res):
    n = res[ds]
    if n.get("skipped"): print(f"{ds}: skip ({n['reason']})"); continue
    b, m, dv = n["delta_AR"], n["d1_mean"], n["diversity"]
    ok = "✅" if (b["delta"] > 0 and b["p_boot_one_sided"] < 0.05) else "❌"
    print(f"{ds:<14} d1 R/rnd/aroma={m['real']:.4f}/{m['real+random']:.4f}/{m['real+aroma']:.4f}"
          f"  Δ(A-R)={b['delta']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] p={b['p_boot_one_sided']:.4f} {ok}")
    print(f"{'':<14} diversity pairwise-mean A/R = {dv['aroma']['pairwise_mean']}/{dv['random']['pairwise_mean']}"
          f"  distinct-src A/R = {dv['aroma']['n_distinct_sources']}/{dv['random']['n_distinct_sources']}")
```

- **diversity 체크**: aroma의 coverage↑가 pairwise-mean↓(뭉침)와 동반되면 "중복으로 부풀린 커버"일 수 있음 — 표에 함께 보고(정직).
- 로그에 `leakage 신호` 경고가 있으면 소스 분리(train-only) 위반 — 결과 사용 금지, 합성 소스 점검.
- 상세 표: `$EXP6_OUT/exp6_knn_summary.md`.

## STEP 4 — rare 모드 실행

```python
!python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \
    --mode rare \
    --real_data_dir $AROMA_DATA \
    --roi_dir_root  $ROI_DIR_ROOT \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --kmeans_k 8 10 12 15 --cluster_seeds 0 1 2 3 4 \
    --null_seeds 30 --rare_quantile 0.25 \
    --val_frac 0.3 --split_seed 42 \
    --embed_cache_dir $EMBED_CACHE_DIR \
    --output_dir $EXP6_OUT
```

## STEP 5 — rare 결과 해석

```python
import json, os
res = json.load(open(f"{os.environ['EXP6_OUT']}/exp6_results.json"))["rare"]
for ds in sorted(res):
    n = res[ds]
    if n.get("skipped"): print(f"{ds}: skip ({n['reason']})"); continue
    v, meta = n["verdict"], n["meta"]
    print(f"{ds:<14} 방향일치 {v['direction_consistent']}/{v['n_valid_cells']}"
          f"  p<0.05: {v['n_sig_p05']}/{v['n_valid_cells']}"
          f"  (unique={meta['n_unique_crops']} sel={meta['n_selected']} unmatched={meta['n_unmatched']})")
```

- **전 그리드 함께 보고** — 특정 (k,seed)만 뽑는 cherry-pick 금지. 방향이 그리드에 따라 뒤집히면 결론 보류.
- `n_unmatched`가 크면 roi_selected↔candidates 스키마 불일치 신호 — 점검.
- 그리드 상세: `$EXP6_OUT/exp6_rare_summary.md`.

## STEP 6 — 대조 검증 (구현 sanity)

**knn 음성 대조** — random 합성 반분을 aroma/random 자리에 → Δ≈0, p 비유의:

```python
import os, json, random, pathlib

SRC = f"{os.environ['RANDOM_SYNTH_DIR']}/severstal/annotations.json"
NEG = "/content/tmp/exp6_negctl"
anns = json.load(open(SRC)); random.Random(0).shuffle(anns)
half = len(anns) // 2
for name, part in (("a", anns[:half]), ("b", anns[half:])):
    d = pathlib.Path(f"{NEG}/{name}/severstal"); d.mkdir(parents=True, exist_ok=True)
    json.dump(part, open(d / "annotations.json", "w"))

!python $AROMA_SCRIPTS/experiments/exp6_embedding_coverage.py \
    --mode knn --real_data_dir $AROMA_DATA \
    --aroma_synthetic_dir  /content/tmp/exp6_negctl/a \
    --random_synthetic_dir /content/tmp/exp6_negctl/b \
    --dataset_keys severstal --bootstrap_reps 500 \
    --output_dir /content/tmp/exp6_negctl/out

neg = json.load(open("/content/tmp/exp6_negctl/out/exp6_results.json"))["knn"]["severstal"]
print("Δ:", neg["delta_AR"]["delta"], " p:", neg["delta_AR"]["p_boot_one_sided"])  # Δ≈0, p 비유의 기대
```

**rare 양성 대조**(선택) — `roi_selected.json`을 "특정 소수 모드 crop만"으로 인위 구성해 넣으면 p가 하한(≈0.032)에 붙어야 함 — 검정 방향 확인.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `$EXP6_OUT/exp6_results.json` | `knn`/`rare` 노드 병합(incremental) — knn: d1/Δ/CI/p/coverage AUC/diversity, rare: (k×cseed) 그리드 + verdict |
| `$EXP6_OUT/exp6_knn_summary.md` / `exp6_rare_summary.md` | 모드별 판정 표 |
| `$EMBED_CACHE_DIR/{ds}/real_train*·cand_src*` | 신규 임베딩 캐시(exp5와 공유 풀에 추가) |

## 트러블슈팅

- `too_few_crops`: 후보 unique crop < max(k) — 소형 데이터셋은 `--kmeans_k`를 낮춰 재실행(예: aitex `--kmeans_k 8 10`).
- `no_rare_modes`: 모드 빈도가 균등해 p25 컷이 비거나 val 미등장 — 해당 셀 skip 표기(정상). 전 셀 skip이면 rare_quantile 상향 검토(사유 명시).
- `⚠few-imgs`(knn): val 이미지 < 10 — bootstrap 신뢰 낮음, aggregate 보조로만.
- mvtec_leather aroma 합성 부재 시 knn은 skip — step4 합성 후 재실행.

## 공통 무결성 / 정직 (`_SPEC §5`)

- **사후 튜닝 금지**: `kmeans_k`·`cluster_seeds`·`null_seeds`·`rare_quantile`·`val_frac`·`split_seed`는 위 값 고정, 결과 보고 후 변경 금지.
- **prescan/게이트 정합**: aroma/random/null 세 표본공간이 동일 clean-bg 게이트를 통과한 풀이어야 공정(quality-gate-fairness).
- **aitex는 tile-level·single-class** → 절대값을 타 데이터셋과 직접 비교 금지, Δ·방향일치만 유효. rare는 `too_few_crops` 시 `--kmeans_k` 하향(사유 명시).
- **기제 증거 한계**: knn·rare 단독으로 downstream 개선을 주장하지 않는다 — exp5(PRDC)·exp4v2(mAP)와 패키지로 제시.
- **테스트 코드 신규 작성·pytest 금지**(CLAUDE.md). 검증은 Colab 실행으로.
- **시간 실측·처리량 벤치 미수행** (load-test policy). GPU 없이 `--device cpu`도 가능(임베딩만 느려짐).
