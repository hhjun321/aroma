# Run#1 — Severstal 4조건 결정 실험 (baseline/random/casda/aroma)

> **목적**: 논문의 핵심 신호 "AROMA > Random" 및 "AROMA > CASDA"가 **실재하는가**를 multi-seed로 판정. 현재 유일한 AROMA>Random 결과는 Severstal n=1(degenerate) — 이 실험이 **프레임을 결정**한다(Frame B 불균형 multi-class detection 확정 여부).
> **런타임**: GPU(A100). **전제**: multi-class fix 커밋됨(bda5fd4, d59e8f3), CASDA synth 생성 완료(generate_casda + mask 매핑), AROMA synth는 **fix-이후 ROI로 재생성**.
> **이것이 paper의 main result가 될 실험.** 한 번에: CASDA 비교 + 검정력(multi-seed) + c2 fix 검증.

---

## 0. 왜 이 실험이 결정적인가

- 신뢰 데이터(MVTec 3-seed)는 AROMA≈/<Random → 일반 "AROMA>Random" 주장 반박됨.
- AROMA 우위 신호는 **불균형 multi-class(Severstal)서만** 출현(n=1). 이게 노이즈인지 실재인지 미판정.
- **이 run이 살아남으면** → Frame B(불균형 multi-class detection) 논문 확정. **죽으면** → 재설계 필요.
- 동시에 CASDA(시그니처 차별점, 데이터 0) + c2 fix(미검증) 한방 해결.

---

## 1. 전제 체크 (실행 전 필수)

```python
# fix 커밋 반영 확인
!cd /content/AROMA && git pull && git log --oneline -3
# bda5fd4 / d59e8f3 (multi class roi fix) 포함돼야 함
```

```python
import os, json
os.environ['DRIVE']        = os.environ.get('DRIVE','/content/drive/MyDrive/data/Aroma')
os.environ['AROMA_SCRIPTS']= '/content/AROMA/scripts/aroma'
os.environ['AROMA_OUT']    = f"{os.environ['DRIVE']}/aroma_output"
os.environ['AROMA_DATA']   = f"{os.environ['DRIVE']}/Aroma"
os.environ['RANDOM_SYNTH_DIR']= f"{os.environ['AROMA_OUT']}/synthetic_random"
os.environ['CASDA_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic_casda"
os.environ['AROMA_SYNTH_DIR'] = f"{os.environ['AROMA_OUT']}/synthetic"
os.environ['EXP4V2_OUT']      = f"{os.environ['AROMA_OUT']}/exp4v2/severstal_run1"   # fresh 폴더

# 3개 synth dir 모두 severstal/annotations.json 존재해야 함
for tag, d in [('random','RANDOM_SYNTH_DIR'),('casda','CASDA_SYNTH_DIR'),('aroma','AROMA_SYNTH_DIR')]:
    p = f"{os.environ[d]}/severstal/annotations.json"
    print(f"{'✅' if os.path.exists(p) else '❌ 없음'}  {tag}: {p}")
```

**필수 전제:**
- **AROMA synth = fix-이후 ROI로 재생성됐는가?** (step3 multi 게이트 통과 → step4 재합성). 안 됐으면 옛 monoculture synth → 무효. `severstal_exp4v2_multiclass_fix_rerun.md` §2-3 먼저.
- **CASDA synth 생성 완료** (generate_casda, mask 매핑 + `--local_staging`).
- random synth는 fix 무관(불변, 기존 재사용 OK).

> ⚠️ **옛 결과 캐시 주의**: `EXP4V2_OUT`을 fresh 폴더(`severstal_run1`)로 둔다. 기존 폴더 재사용 시 `--resume`이 옛(pre-fix/구버전 casda) 결과를 skip해 오염.

---

## 2. 4조건 실행 (multi-seed)

```python
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --condition all \
    --dataset_keys severstal \
    --class_mode multi \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --casda_synthetic_dir  $CASDA_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4V2_OUT \
    --seeds 42 1 2 3 4 \
    --val_frac 0.3 \
    --synth_ratio 1.0 \
    --imgsz 640 \
    --baseline_epochs 100 \
    --patience 20 \
    --batch 64 --cache ram --rect \
    --yolo_cache_dir $AROMA_OUT/yolo_cache \
    --resume
```

| 인자 | 값 | 이유 |
|------|----|------|
| `--seeds 42 1 2 3 4` | **n=5** | 검정력. critique: n=1→무효, CASDA도 n=2를 underpowered 지적. 최소 3, **권장 5** |
| `--condition all` | 4조건 | baseline/random/**casda**/aroma 자동 (ALL_CONDITION_KEYS) |
| `--synth_ratio 1.0` | 1.0 | 3 synth 조건 동량 보장 (공정성) |
| `--class_mode multi` | multi | per-class AP(c1~c4) |
| `--output_dir` | `severstal_run1` | 옛 결과와 분리 |

> **소요**: 4조건 × 5 seed = 20 학습 ≈ 수~십 시간. `--resume`으로 seed/조건 단위 재개. 세션 한도 대비 필수.
> GPU 장시간 — 사용자 판단 하 실행 (load 측정 아님, 효과 재현).

---

## 3. 공정성 검증 (결과 해석 전 게이트)

```python
import json, os
res = json.load(open(f"{os.environ['EXP4V2_OUT']}/exp4v2_results.json"))
arms = res["severstal"]["yolov8n"]

# (A) n_synth_train parity — random/casda/aroma 동일해야 공정
print("n_synth_train:", {c: arms[c].get("n_synth_train") for c in ['random','casda','aroma']})
# 다르면 cap 튜플 누락/생성량 불일치 → 비교 무효

# (B) per-class synth 분포 (CASDA c4/c2 starve 투명화)
for c in ['random','casda','aroma']:
    print(f"{c} n_synth_per_class:", arms[c].get("n_synth_per_class"))
# aroma: 균등(50/50/50/50×n_per_roi) 기대. casda: native라 c3 편중/c4 희소 가능(정당한 발견)
```

**합격**: random/casda/aroma `n_synth_train` 동일. 불일치면 멈추고 원인(cap 누락/생성량) 규명.

---

## 4. 결과 + paired 유의성 검정 (핵심)

단순 mean 비교 아니라 **seed별 paired test** (critique 요구).

```python
import json, os, numpy as np
from scipy import stats

base = os.environ['EXP4V2_OUT']
seeds = [42,1,2,3,4]
def per_seed_map50(cond):
    vals=[]
    for s in seeds:
        f=f"{base}/_seeds/seed{s}/exp4v2_results.json"
        if os.path.exists(f):
            vals.append(json.load(open(f))["severstal"]["yolov8n"][cond]["map50"])
    return np.array(vals)

CONDS=['baseline','random','casda','aroma']
for c in CONDS:
    v=per_seed_map50(c); print(f"{c:9s} map50 mean {v.mean():.4f} ± {v.std(ddof=1):.4f}  (n={len(v)})")

# paired Wilcoxon (소표본 robust) — AROMA vs Random, AROMA vs CASDA
for opp in ['random','casda']:
    a, b = per_seed_map50('aroma'), per_seed_map50(opp)
    if len(a)==len(b) and len(a)>=3:
        try:
            w,p = stats.wilcoxon(a,b)
            print(f"AROMA vs {opp}: Δ={a.mean()-b.mean():+.4f}, Wilcoxon p={p:.4f} {'(유의)' if p<0.05 else '(비유의)'}")
        except Exception as e:
            t,p = stats.ttest_rel(a,b)
            print(f"AROMA vs {opp}: Δ={a.mean()-b.mean():+.4f}, paired t p={p:.4f}")
```

per-class AP (c1~c4):
```python
arms = json.load(open(f"{base}/exp4v2_results.json"))["severstal"]["yolov8n"]
for cl in ['c1','c2','c3','c4']:
    row={c:((arms[c].get('per_class') or {}).get(cl) or {}).get('map50') for c in CONDS}
    print(cl, {k:(round(v,4) if isinstance(v,float) else None) for k,v in row.items()})
```

---

## 5. 판정 → 프레임 결정

| 결과 | 결론 |
|------|------|
| **AROMA > Random AND > CASDA, paired p<0.05 (또는 CI 비겹침)** | ✅ **Frame B 확정** — 불균형 multi-class detection 논문. c2 회복(c2 aroma≥random) + c4 starvation 없음도 확인 |
| AROMA > random/casda이나 비유의 (p>0.05) | ⚠️ 신호 약함 — seed 더(n=7~10) 또는 effect 작음 인정. 프레임 재고 |
| AROMA가 random/casda에 안 짐도 못 이김 | ❌ 핵심 주장 미성립 — Severstal n=1은 노이즈였음. **재설계 필요** (방법/프레임 근본 재검토) |

**추가 확인**:
- **c2 fix 검증**: c2 aroma map50가 random 이상으로 회복? (이전 0.2304<0.2821 해소?)
- **c4 starvation 해소**: c4 aroma > 0이고 비회귀?
- CASDA per-class: c3/c4서 AROMA ≥ CASDA (CASDA H4 주장 영역 직접 반박)

---

## 6. 주의

- **fix-이후 AROMA synth 필수** — 옛 monoculture synth로 돌리면 c2 회복 안 보임 = 무효.
- **paired test는 seed별 동일 split 전제** — exp4v2가 seed별 train/val split 결정 → aroma/random/casda가 같은 seed서 같은 val 봄(paired 타당).
- 결과는 `severstal_run1`에만. 옛 `severstal_multi`(pre-fix)와 혼동 금지.
- 이 결과가 **논문 §4.4 + Table 12를 채운다** (현재 FROZEN). 다른 fabricated 표(8/9/10/11)는 별도 실험(exp2 roi, exp3 fid) 필요 — 이 run은 supervised detection만.
- load 측정 아님(효과 재현) → load-test 정책 비해당. 단 GPU 장시간이라 사용자 승인 하.
```
