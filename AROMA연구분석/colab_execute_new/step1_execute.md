# step1 — `compute_complexity.py` (sym_final 정본)

> **목적**: phase0 프로파일링 출력을 읽어 MCI / CCI를 계산하고 형태학·컨텍스트 정책을 선택한다. 입력 `S('profiling',ds)` → 출력 `S('complexity',ds)`.
> **실행 환경**: **CPU**. `--local_staging`으로 Drive CSV I/O를 로컬로 스테이징 → 데이터셋당 ~1–2분.
> **전제**: phase0(`distribution_profiling.py`) 완료 — 각 데이터셋 `S('profiling',ds)/morphology_features.csv` 존재.
> ⚠️ **phase0 재실행 시 step1도 반드시 재실행**: phase0 재실행은 GMM 클러스터링을 **재계산**해 `morphology_features.csv`의 `cluster_id`·`image_id`가 바뀐다. step1의 MCI/CCI·정책은 이 클러스터에 종속되므로, 구 complexity를 그대로 두면 신 profiling과 **조용히 불일치**한다(오류 없이 다운스트림 mAP로만 드러남). 구/신 혼용 금지 — phase0를 다시 돌렸으면 step1→step2→step3까지 연쇄 재실행한다.
> **실행 순서 체인**: phase0 → **step1** → step2 → step3 → step4(ControlNet 학습) → step5(생성) → exp3/exp4v2/exp5/exp6.
> **데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex는 tiled(single-class) — phase0에서 자동 해소된 프로파일링을 그대로 소비.

---

## STEP 0 — 공통 환경 셀 (sym_final 전 문서 동일 — 그대로 복사)

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

---

## STEP 1 — 실행 (DATASETS 4종 루프)

`compute_complexity.py`는 `scripts/aroma/`에 있으므로 `$AROMA_SCRIPTS/`로 호출한다.

```python
from pathlib import Path

for DS in DATASETS:
    os.environ['DS']    = DS
    os.environ['PROF']  = S('profiling', DS)
    os.environ['CPLX']  = S('complexity', DS)
    if not Path(f"{os.environ['PROF']}/morphology_features.csv").exists():
        print(f"↷ skip {DS} — phase0 출력 없음"); continue
    print(f"\n===== complexity: {DS} → {os.environ['CPLX']} =====")
    !python $AROMA_SCRIPTS/compute_complexity.py \
        --profiling_dir $PROF \
        --output_dir    $CPLX \
        --weight_mode   equal \
        --local_staging
```

> `--local_staging`은 CPU 단계(complexity)에서 사용 가능 — Drive CSV를 로컬로 복사해 I/O 단축. (ControlNet **생성** 단계에서는 sidecar 캐시 Drive 직결이 필요하므로 미사용.)

---

## STEP 2 — 결과 확인

```python
for DS in DATASETS:
    rp = f"{S('complexity', DS)}/complexity_report.json"
    if not os.path.exists(rp):
        print(f"[{DS}] complexity_report.json 없음"); continue
    r = json.load(open(rp))
    print(f"\n===== [{DS}] =====")
    print(f"MCI={r['mci']:.4f}  CCI={r['cci']:.4f}  "
          f"Morphology Policy={r['morphology_policy']}  Context Policy={r['context_policy']}")

    print('  --- MCI 구성 요소 ---')
    for k, v in r.get('mci_components', {}).get('raw', {}).items():
        nv = r['mci_components']['normalized'].get(k)
        ns = f"{nv:.4f}" if isinstance(nv, (int, float)) else "-.----"
        print(f'    {k:<20} raw={v:.4f}  norm={ns}')
    sil = r.get('mci_components', {}).get('silhouette_score')
    if sil is not None:
        print(f"    {'silhouette_score':<20} {sil:.4f}  (diagnostic)")

    print('  --- CCI 구성 요소 ---')
    for k, v in r.get('cci_components', {}).get('raw', {}).items():
        nv = r['cci_components']['normalized'].get(k)
        ns = f"{nv:.4f}" if isinstance(nv, (int, float)) else "-.----"
        print(f'    {k:<20} raw={v:.4f}  norm={ns}')

    print('  --- 후보 정책 평가 ---')
    for ev in r.get('evaluation_results', []):
        print(f"    [{ev.get('axis','?'):12s}] {ev.get('policy','?'):<14} "
              f"silhouette={ev.get('silhouette',0):.4f}  stability={ev.get('stability','-')}")
```

---

## 판정 / 다음 단계

- [ ] 4종 전부 `S('complexity',ds)/complexity_report.json` 생성
- [ ] MCI/CCI 및 morphology/context policy 값이 산출됨

통과 시 → **step2**(`prompt_generation.py`, 입력 `S('profiling',ds)`+`S('complexity',ds)`).

---

## 무결성 / 정직 (_SPEC §5)

- output 경로는 반드시 stage-first `S('complexity', ds)`(=`sym_final/complexity/{ds}`). ds-first 금지.
- aitex는 tile-level·single-class → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- 사후 튜닝 금지(결과 보고 후 파라미터 변경 금지). 테스트 코드 신규 작성·pytest 금지(CLAUDE.md) — 검증은 본 Colab 셀 실행으로.
