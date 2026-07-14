# KolektorSDD (single-class) — 생성 파이프라인 + exp4v2 실행 가이드

> **목적**: KolektorSDD v1(금속 정류자 표면)을 aitex 와 동일하게 **single-class(nc=1 'defect')** 로 exp4v2 3조건(baseline/random/aroma) 평가.
> **⚠️ 사전 경고 (CCI)**: kolektor **CCI=0.224**(로컬 실측) — leather(0.20)/mtd(0.25) **dead zone**. compat placement 레버 약함 → **aroma≈random 가능성 높음**. 그럼에도 실 mAP 로 empirical 확정(CCI 는 proxy). headroom(baseline)은 §5 에서 실측.
> **실행 환경**: 생성 = CPU(copy_paste 무학습) | exp4v2 = GPU(A100).
> **참조**: 생성 명령 규약은 `mvtec_zipper_transistor_baseline_execute.md §9` 및 `colab_execute_new/stepN_execute.md` 정본과 동일.

---

## 0. 전제
- prepare 완료 필요(아래 §2). dataset_config 에 `kolektor` 등록됨(domain=aitex, class_mode=single). exp4v2 `("aitex","mtd","kolektor")` 분기 인식.
- 단일클래스라 roi_selection 은 multi 플래그 없이(전역 top-k), exp4v2 는 `--class_mode` 생략(single 기본).

## 1. 공통 환경 (`_SPEC §1`)

```python
import os, json, pathlib
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']     = f"{os.environ['DRIVE']}"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p
DS = "kolektor"
os.environ['SYNTH_AROMA']=S('synth_aroma'); os.environ['SYNTH_RANDOM']=S('synth_random')
os.environ['EXP4V2_OUT']=S('exp4v2'); os.environ['YOLO_CACHE']=f"{os.environ['AROMA_OUT']}/yolo_cache"
```

## 2. prepare (원본 → AROMA 레이아웃, 1회)

```python
!python $AROMA_SCRIPTS/prepare_kolektor.py \
    --kolektor_root $DRIVE/KolektorSDD-boxes \
    --output_dir    $DRIVE/kolektor
```
> 기대: `good=347 defect=52 skipped=0`. 산출: `kolektor/{train/good, test/defect, ground_truth/defect/{stem}_mask.png}`.

## 3. 생성 파이프라인 (step2→step5, CPU)

```python
os.environ['DS']=DS
os.environ['PROF']=S('profiling',DS); os.environ['CPLX']=S('complexity',DS)
os.environ['PROMPTS']=S('prompts',DS); os.environ['ROI']=S('roi',DS)
os.environ['NORMAL']=json.load(open(os.environ['DATASET_CONFIG']))[DS]['image_dir']
os.environ['COMPAT']=f"{S('profiling',DS)}/compatibility_matrix.json"
```

**3-0. phase0 profiling + step1 complexity** (아직 Drive 에 없으면):
```python
!python $AROMA_REF/scripts/distribution_profiling.py \
    --dataset_config $DATASET_CONFIG --dataset_key $DS --output_dir $PROF --num_workers -1
!python $AROMA_SCRIPTS/compute_complexity.py --profiling_dir $PROF --output_dir $CPLX
```
> 로컬 실측: MCI=0.328, CCI=0.224 (mask 52 매칭, ground_truth). Colab 재현값 사용.

**3-1. step2 prompt**:
```python
!python $AROMA_SCRIPTS/prompt_generation.py --profiling_dir $PROF --complexity_dir $CPLX --output_dir $PROMPTS
```

**3-2. step3 roi_selection (single — multi 플래그 없음)**:
```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir $PROF --prompts_dir $PROMPTS \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 200 --img_diversity_cap 1 \
    --output_dir $ROI
```

**3-3. step3_5 clean_bg_selection (+random arm)**:
```python
!python $AROMA_SCRIPTS/clean_bg_selection.py \
    --profiling_dir $PROF --roi_dir $ROI --output_dir $ROI --emit_random_arm
```
> kolektor 는 균일표면 → clean_bg void 검출 fallback 로그 주시(과대거부 시 `--void_floor_pct`/`--no_reject_clean_bg` 조정).

**3-4. step4c τ 사전스캔** (ControlNet 학습 SKIP, τ만): `colab_execute_new/step4_execute.md §4c [4c-0~4c-5]` 를 `DS='kolektor'` 로 실행 → `compat_tau_prescan_kolektor.json`.
```python
p=f"{S('compat_gate',DS)}/compat_tau_prescan_{DS}.json"
tau=json.load(open(p)).get('ds_tau'); assert tau and 0<tau<0.5, f"τ 미확정 {tau}"
os.environ['TAU']=str(tau); print('ds_tau=',tau)
```

**3-5. step5 AROMA arm (copy_paste, min-bg-quality 0.42)**:
```python
os.environ['OUT']=S('synth_aroma',DS)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir $ROI --normal_dir $NORMAL --output_dir $OUT \
    --method copy_paste --n_per_roi 3 --seed 42 --blend_mode seamless \
    --reject-clean-bg --min-bg-quality 0.42 --bg-blur-threshold 100.0 \
    --compat_mode symmetric --compat_threshold $TAU --compat_matrix_json $COMPAT \
    --local_staging
```
> ⚠️ **min-bg-quality 0.42** (kolektor 사전스캔). **기본 0.7 사용 금지** — full-image max 0.549 라 0.7 이면 배경 풀 전멸(생성 붕괴). 로그 `clean_bg assignment ON` + `compat gate ON` + `placement-gate fallback<50%` 확인.

**3-6. step5 random arm**:
```python
os.environ['OUT_R']=S('synth_random',DS)
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI/roi_candidates.json --normal_dir $NORMAL \
    --output_dir $OUT_R --top_k 200 --n_per_roi 3 --seed 42 --local_staging
```

**3-7. parity 확인**:
```python
for lbl,root in [("aroma",S('synth_aroma',DS)),("random",S('synth_random',DS))]:
    ann=pathlib.Path(root)/"annotations.json"
    n=sum(1 for a in json.load(open(ann)) if not a.get('dry_run') and a.get('bbox')) if ann.exists() else 0
    print(f"{lbl} labelable={n}")   # 둘 다 >0 이어야 exp4v2 유효
```

## 4. exp4v2 3-arm (single, GPU)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --model yolov8n --condition all \
    --dataset_keys kolektor \
    --aroma_synthetic_dir  $SYNTH_AROMA \
    --random_synthetic_dir $SYNTH_RANDOM \
    --real_data_dir $AROMA_DATA \
    --output_dir    $EXP4V2_OUT \
    --yolo_cache_dir $YOLO_CACHE \
    --imgsz 640 --val_frac 0.3 --synth_ratio 1.0 \
    --baseline_epochs 100 --patience 25 \
    --batch 128 --workers 12 --cache ram --rect \
    --seeds 42 1 2 --resume
```
> `--class_mode` 생략 = single(nc=1). `--rect`(tall 2.5:1). `--compile` 미사용. `$EXP4V2_OUT` 기존 셋과 동일 dir — 키 'kolektor' 상이라 충돌 없음(통합 results.json).

## 5. 판정
```python
r=json.load(open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json"))['kolektor']['yolov8n']
for c in ('baseline','random','aroma'):
    cell=r.get(c,{}); print(f"{c:9} map50={cell.get('map50')} ±{(cell.get('std') or {}).get('map50')}")
a=r.get('aroma',{}).get('per_seed',{}); rd=r.get('random',{}).get('per_seed',{})
seeds=sorted(set(a)&set(rd)); d=[a[s]['map50']-rd[s]['map50'] for s in seeds]
print('Δ(A-R) per-seed',[round(x,4) for x in d],'mean=%+.4f'%(sum(d)/len(d) if d else 0))
```
**해석**:
- headroom 먼저: baseline map50 ≳0.9 near-ceiling(부적합) / 0.5~0.85 headroom有.
- CCI 0.224(약레버) → **aroma≈random 예상**. Δ(A−R)≈0 이면 "균일표면→레버없음"(leather 유형) 확정. 예상외 양(+)이면 잔여 레버 존재.
- aitex(CCI 0.44)·severstal(0.31) 가 여전히 주 arbiter. kolektor 는 **low-CCI 대조 사례**(AROMA 가 이질성 필요함을 보이는 증거)로 유효.
