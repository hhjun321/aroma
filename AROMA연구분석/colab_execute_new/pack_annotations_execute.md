# 배치 검증용 산출물 수집 — Colab → 로컬 zip

> **목적**: `ring_sgm` 배치가 파이프라인 끝까지 의도대로 살아남았는지 **로컬에서 정량 대조**하기 위해, Drive 산출물 중 필요한 JSON만 모아 zip 으로 내린다.
> **성격**: 벤치마크 아님. ROI selection · clean_bg_selection · ring 배치의 **이론적 확인**용.
> **실행 환경**: CPU. 이미지 불요 — `annotations.json` 에 `bbox`·`normal_image`·`cluster_id` 가 있어 정량 전량이 JSON 만으로 계산된다.
> **체인 위치**: step5(생성) 완료 후. exp* 와 무관.

---

## 왜 이미지가 필요 없나

| 필요한 정보 | 출처 |
|---|---|
| 붙은 좌표 | `annotations.json` 의 `bbox` |
| 붙은 배경 | `annotations.json` 의 `normal_image` |
| 결함 형태 군집 | `annotations.json` 의 `cluster_id` |
| 배경의 문맥 셀 | 로컬 `context_features.csv` (이미 있음) |
| 목표 분포 `q_k` | `compatibility_matrix.json` 의 `matrix_symmetric` |
| 결함 인접 문맥 (참조 분포) | 로컬 `defect_tiles.json` (이미 있음) |

`context_features.csv` 는 severstal 121MB 라 **패키지에서 제외**한다. 로컬 미러를 쓰되, `compatibility_matrix.json` 의 `bin_edges` 가 로컬과 같은지 압축 해제 후 먼저 대조한다(다르면 셀 코드가 어긋나 대조가 무의미).

---

## STEP 0 — 공통 환경 셀 (`_SPEC §1` 그대로 — 수정 금지)

```python
import os, json

# ===== 공통 환경 (sym_final 전 문서 동일 — 수정 금지) =====
os.environ['DRIVE']          = '/content/drive/MyDrive/data/Aroma'
os.environ['AROMA_REF']      = '/content/AROMA'
os.environ['AROMA_SCRIPTS']  = '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']      = f"{os.environ['DRIVE']}/aroma_output"
os.environ['DATASET_CONFIG'] = os.environ.get('DATASET_CONFIG', '/content/AROMA/dataset_config.json')
# ===== 단일 버전 루트 (stage-first: {stage}/{ds}) =====
os.environ['SYM_ROOT'] = f"{os.environ['AROMA_OUT']}/sym_final"
def S(stage, ds=None):
    p = f"{os.environ['SYM_ROOT']}/{stage}"
    return f"{p}/{ds}" if ds else p
```

---

## STEP 1 — 수집 + zip

```python
import os, json, shutil, zipfile
from pathlib import Path

ALL  = ["severstal", "mvtec_leather", "mtd", "aitex", "kolektor"]
PACK = Path("/content/aroma_pack")            # 임시 수집 루트
if PACK.exists(): shutil.rmtree(PACK)

manifest, missing = [], []

def grab(src, rel):
    """src → PACK/rel 복사. 없으면 missing 기록(중단하지 않음)."""
    s = Path(src)
    if not s.exists():
        missing.append(rel); return
    d = PACK / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
    manifest.append({"path": rel, "bytes": s.stat().st_size})

for DS in ALL:
    # 실제 배치 결과 (arm 별)
    grab(f"{S('synth_aroma_tobe', DS)}/annotations.json", f"synth_aroma_tobe/{DS}/annotations.json")
    grab(f"{S('synth_random',     DS)}/annotations.json", f"synth_random/{DS}/annotations.json")
    # 배경·자리 선정 산출 (step3.5)
    grab(f"{S('roi', DS)}/clean_bg_selected.json", f"roi/{DS}/clean_bg_selected.json")
    grab(f"{S('roi', DS)}/clean_bg_summary.md",    f"roi/{DS}/clean_bg_summary.md")
    grab(f"{S('roi', DS)}/roi_selected.json",      f"roi/{DS}/roi_selected.json")
    # 정합 확인용 (context_features.csv 는 용량 때문에 제외 — 로컬 미러 사용)
    grab(f"{S('profiling', DS)}/compatibility_matrix.json", f"profiling/{DS}/compatibility_matrix.json")
    grab(f"{S('profiling', DS)}/morphology_clusters.json",  f"profiling/{DS}/morphology_clusters.json")

(PACK / "MANIFEST.json").write_text(json.dumps(
    {"datasets": ALL, "files": manifest, "missing": missing}, indent=1), encoding="utf-8")

ZIP = "/content/aroma_pack.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in sorted(PACK.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(PACK))

print(f"파일 {len(manifest)}개, 누락 {len(missing)}개")
for m in missing: print("  누락:", m)
print(f"\n{ZIP}   {os.path.getsize(ZIP)/1e6:.1f} MB")
for e in sorted(manifest, key=lambda x: -x['bytes']):
    print(f"  {e['bytes']/1e6:8.2f} MB  {e['path']}")
```

**담기는 것 — 5종 × 7파일 = 35개**

| 파일 | 용도 |
|---|---|
| `synth_aroma_tobe/{ds}/annotations.json` | **실제 붙은 좌표** (ring arm) |
| `synth_random/{ds}/annotations.json` | 실제 random arm |
| `roi/{ds}/clean_bg_selected.json` | 배정 배경 · `position` · `topk_pool` |
| `roi/{ds}/clean_bg_summary.md` | `w_k` · ring fallback 통계 (실행 기록) |
| `roi/{ds}/roi_selected.json` | ROI 정렬 확인 |
| `profiling/{ds}/compatibility_matrix.json` | `q_k` — **로컬 정합 확인 필수** |
| `profiling/{ds}/morphology_clusters.json` | cluster 배정 |

**예상 용량**: `clean_bg_selected.json` 이 지배적(severstal 로컬 기준 9MB). 압축 후 **3~8MB**.

> `누락` 이 비어 있어야 한다. `synth_random/{ds}/annotations.json` 이 없으면 random arm 을 아직 안 돌린 것이며, 그 경우 실제 random arm 대조만 빠지고 나머지는 진행된다(스크립트가 같은 배경에서 균등 무작위를 시뮬레이션해 대체 arm 을 만든다).

---

## STEP 2 — 다운로드

```python
from google.colab import files
files.download("/content/aroma_pack.zip")
```

> 파일이 커서 브라우저 다운로드가 끊기면 Drive 에 두고 Drive 클라이언트로 받는다:
> ```python
> import shutil; shutil.copy2("/content/aroma_pack.zip", f"{os.environ['AROMA_OUT']}/aroma_pack.zip")
> print("Drive:", f"{os.environ['AROMA_OUT']}/aroma_pack.zip")
> ```

---

## STEP 3 — 로컬 압축 해제 위치

```
D:\project\aroma_dataset\_colab_pack\
```

**기존 로컬 파일을 바로 덮지 않는다.** 별도 디렉터리에 풀고, `bin_edges` · `matrix_symmetric` 정합을 확인한 뒤 반영 여부를 판단한다.

```
_colab_pack/
  MANIFEST.json
  synth_aroma_tobe/{ds}/annotations.json
  synth_random/{ds}/annotations.json
  roi/{ds}/clean_bg_selected.json
  roi/{ds}/clean_bg_summary.md
  roi/{ds}/roi_selected.json
  profiling/{ds}/compatibility_matrix.json
  profiling/{ds}/morphology_clusters.json
```

---

## STEP 4 — 정합 확인 (Colab 에서 미리 찍어두면 대조가 빠르다)

```python
import json
for DS in ALL:
    m = json.load(open(f"{S('profiling', DS)}/compatibility_matrix.json"))
    be = m.get('bin_edges') or {}
    r  = (m.get('matrix_symmetric') or {}).get('0') or {}
    top = sorted(r.items(), key=lambda t: -t[1])[:2]
    print(f"{DS:14s} bin_edges keys={len(be)}  sym cells={len(r):4d}  top={top}")
```

이 출력을 로컬 값과 대조한다. `sym cells` 나 `top` 셀이 다르면 **로컬 프로파일링과 Colab 프로파일링이 다른 판**이므로, 그 경우 `context_features.csv` 까지 가져와야 한다.

---

## 이 패키지로 돌리는 분석

로컬 `scratchpad/verify_synth_placement.py` — 동일 결함 · 동일 배경에서 **배치 규칙만** 갈라 대조한다.

| arm | 정의 |
|---|---|
| `ring` | 실제 붙은 좌표 (`synth_aroma_tobe` annotations) |
| `old` | 같은 배경에서 **footprint 평균 argmax 재산출** ← ring 기여 고립 |
| `rand` | 같은 배경에서 균등 무작위 (시드 42) |
| `rand_arm` | 실제 random arm (`synth_random` annotations 있을 때) |

**지표**

- **P1** = `mean_k JS( P_synth(k,·) ‖ P_real(k,·) )` — 낮을수록 실제 결함 문맥 분포에 가깝다
  - `P_real` = cluster k 결함의 **실제 인접 문맥** 셀 분포 (`defect_tiles.json` `adjacent_r1`)
  - `P_synth` = 붙은 자리 **둘레(ring)** 의 문맥 셀 분포
- **void 침범률** = footprint 에 void/결측 타일이 섞인 비율
- **배경 다양성** = 사용된 배경 이미지 수

> `P_real` 은 **국소 인접 분포**이고 ring 이 최적화한 대상은 `matrix_symmetric`(= `√(P_def_전역·P_clean)`) 이므로 **자기참조가 아니다.** 구방식(`old`)도 같은 `matrix_symmetric` 을 쓰되 footprint 평균 argmax 로 소비하므로, 두 arm 의 차이는 **score 계산 방식 하나**로 고립된다.

---

## 관련 문서

- `_SPEC.md` §0 — step3.5 / step5 정본 규격
- `step3_5_execute.md` — `--k_fit --site_selection ring` 실행·검증
- `step5_execute.md` — 생성 실행, 좌표 대조 검증 셀
- `.claude/.dev_note/aroma_adjacent_context_bg_selection.md` — 채택 근거·P1 정의
