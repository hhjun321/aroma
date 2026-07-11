# step2 — `prompt_generation.py` (sym_final 정본)

> **목적**: phase0 프로파일링 + step1 complexity 출력을 읽어, 형태학 클러스터 × 컨텍스트 빈 조합별 자연어 프롬프트를 생성한다. 입력 `S('profiling',ds)`+`S('complexity',ds)` → 출력 `S('prompts',ds)`.
> **실행 환경**: **CPU**.
> **전제**: step1(`compute_complexity.py`) 완료 — 각 데이터셋 `S('complexity',ds)/complexity_report.json` 존재.
> **실행 순서 체인**: phase0 → step1 → **step2** → step3 → step5(ControlNet 학습) → step4(생성) → exp3/exp4v2/exp5/exp6.
> **데이터셋**: v2-1 4종 `severstal · mvtec_leather · mtd · aitex`. aitex는 tiled(single-class).

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
os.environ['CN_MODELS'] = f"{os.environ['SYM_ROOT']}/controlnet_models"   # ControlNet 학습본(step5 산출, step4 소비)
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

`prompt_generation.py`는 `scripts/aroma/`에 있으므로 `$AROMA_SCRIPTS/`로 호출한다.

```python
from pathlib import Path

for DS in DATASETS:
    os.environ['DS']      = DS
    os.environ['PROF']    = S('profiling', DS)
    os.environ['CPLX']    = S('complexity', DS)
    os.environ['PROMPTS'] = S('prompts', DS)
    if not Path(f"{os.environ['CPLX']}/complexity_report.json").exists():
        print(f"↷ skip {DS} — step1 출력 없음"); continue
    print(f"\n===== prompts: {DS} → {os.environ['PROMPTS']} =====")
    !python $AROMA_SCRIPTS/prompt_generation.py \
        --profiling_dir  $PROF \
        --complexity_dir $CPLX \
        --output_dir     $PROMPTS
```

---

## STEP 2 — 결과 확인

```python
from collections import defaultdict

for DS in DATASETS:
    pp = f"{S('prompts', DS)}/prompts.json"
    if not os.path.exists(pp):
        print(f"[{DS}] prompts.json 없음"); continue
    prompts = json.load(open(pp))
    print(f"\n===== [{DS}] 총 프롬프트 수: {len(prompts)} =====")

    by_cluster = defaultdict(list)
    for key, entry in prompts.items():
        by_cluster[entry['cluster_id']].append(entry)

    for cid in sorted(by_cluster.keys()):
        entries = by_cluster[cid]
        label = entries[0].get('phase0_label', '')
        print(f"  Cluster {cid} [{label}]  조합수={len(entries)}")
        for e in sorted(entries, key=lambda x: x.get('deficit', 0), reverse=True)[:2]:
            print(f"    [{e['cell_key']}] deficit={e['deficit']:.3f}  prob={e['prior_prob']:.3f}")
            print(f"      → {e['prompt']}")
```

출력 파일:

| 파일 | 내용 |
|------|------|
| `prompts.json` | `"{cluster_id}_{cell_key}"` 키 → prompt, descriptor, deficit, prior_prob |
| `prompts_summary.md` | 마크다운 테이블 (전체 조합 목록) |

---

## 판정 / 다음 단계

- [ ] 4종 전부 `S('prompts',ds)/prompts.json` 생성
- [ ] 클러스터별 조합·deficit·prompt가 정상 출력됨

통과 시 → **step3**(`roi_selection.py`, 입력 `S('profiling',ds)`+`S('prompts',ds)` → 출력 `S('roi',ds)`).

---

## 무결성 / 정직 (_SPEC §5)

- output 경로는 반드시 stage-first `S('prompts', ds)`(=`sym_final/prompts/{ds}`). ds-first 금지.
- aitex는 tile-level·single-class → 절대값 타 데이터셋 직접 비교 금지, Δ만 유효.
- 사후 튜닝 금지(결과 보고 후 파라미터 변경 금지). 테스트 코드 신규 작성·pytest 금지(CLAUDE.md) — 검증은 본 Colab 셀 실행으로.
