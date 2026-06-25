# ROI 품질 게이트 — Colab 사용 가이드 (addendum)

`scripts/aroma/roi_selection.py`에 추가된 **품질 게이트**(`--min_quality`,
`--background_type`)만 다루는 부록. 전체 4조건 exp4v2 재실행은 기존 가이드
`AROMA연구분석/colab_execute/severstal_4cond_rerun_guide.md`를 따르고, **Step 3(ROI 선택)에서
AROMA 조건에만** 아래 2-pass를 끼워 넣는다. 설계는 [[aroma_exp4v2_roi-quality-gate]] 참조.

> 기본값 `--min_quality 0.0` = 게이트 OFF = 기존 동작 동일. 다른 조건/데이터셋 영향 없음.
> origin의 Fix1–4(img_diversity_cap=1, class_floor, per_pair_cap, rarity_temp)는 그대로 유지.

---

## 전제: utils import + skimage

게이트는 `utils.suitability` / `utils.defect_characterization`(skimage 의존)에 의존.
- `AROMA_REF`가 repo 루트여야 import 성공 (아니면 게이트 no-op + 경고).
- `!pip install -q scikit-image` 필요(없으면 게이트 비활성).

```python
import os
os.environ['AROMA_REF'] = '/content/AROMA_PLUS'   # roi_selection의 utils import 루트
```

---

## Pass 1 — 게이트 OFF로 quality_score 분포 확인

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir $AROMA_OUT/profiling/severstal \
    --prompts_dir   $AROMA_OUT/prompts/severstal \
    --output_dir    $AROMA_OUT/roi/severstal \
    --sampling_strategy deficit_aware --top_k 200 \
    --class_floor \
    --min_quality 0 \
    --background_type directional
```
> `--class_floor` 등 나머지 인자는 기존 4조건 가이드의 AROMA 조건 설정 그대로 사용.

분포 확인:

```python
import json
from collections import Counter, defaultdict
cands = json.load(open(f"{os.environ['AROMA_OUT']}/roi/severstal/roi_candidates.json", encoding='utf-8'))
by_sub  = defaultdict(Counter)   # class -> subtype 분포
by_qs   = defaultdict(Counter)   # class -> quality_score 분포
for c in cands:
    ck = c.get('class_key', '_')
    by_sub[ck][c.get('defect_subtype')] += 1
    by_qs[ck][round(float(c.get('quality_score', 1.0)), 2)] += 1
for ck in sorted(by_sub):
    print(f"[{ck}] subtype={dict(by_sub[ck])}  qscore={dict(sorted(by_qs[ck].items()))}")
```

해석(directional 기준, 점수는 subtype별 이산값):
`linear_scratch=1.0 / elongated=0.9 / general=0.7 / compact_blob=0.4 / irregular=0.4`
- `--min_quality 0.5` → compact_blob·irregular(0.4) 제거, 나머지 통과 (시작점 권장)
- `--min_quality 0.7` → general 이상만
- c2(class2)를 어떤 subtype이 채우는지 보고, 저품질 subtype 비중이 높으면 그게 붕괴 원인.

---

## Pass 2 — 게이트 ON으로 실제 선택

```python
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir $AROMA_OUT/profiling/severstal \
    --prompts_dir   $AROMA_OUT/prompts/severstal \
    --output_dir    $AROMA_OUT/roi/severstal \
    --sampling_strategy deficit_aware --top_k 200 \
    --class_floor \
    --min_quality 0.5 \
    --background_type directional
```

로그 확인:
- `Quality gate: min_quality=0.500 → N/M candidates pass`
- 클래스별 `passing=… / …`
- `class=… has ZERO quality-passing ROIs` 경고가 있으면 컷오프 과도 → 낮춘다.

이후 Step 4(generate_defects) → exp4v2 학습은 **기존 4조건 가이드 그대로**. AROMA 조건의
입력 roi만 위에서 생성된 것을 가리키면 된다.

---

## 검증 포인트

- AROMA c2 합성/선택 수가 부적합 ROI 제외로 감소했는지.
- **새 baseline 대비** c2·전체 mAP50 회복 (Fix1–4가 이미 적용된 현재 AROMA 결과 대비 비교 —
  옛 0.3132/0.0675는 Fix 적용 전 값이므로 직접 비교 금지).
- 회귀 의심 시 `--min_quality 0`으로 게이트 끄면 기존 동작 재현.

## 참고
- [[aroma_exp4v2_roi-quality-gate]] — 설계·진단·재화해
- `AROMA연구분석/colab_execute/severstal_4cond_rerun_guide.md` — 전체 4조건 재실행
- `.claude/rules/colab-execution.md` — Colab 실행 규칙
