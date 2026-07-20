# Figure 14 — ROI_score composition (0.6·ctx_prior + 0.4·morph_prior) (spec)

## 목적
§3.2.5의 `ROI_score = 0.6·ctx_prior + 0.4·morph_prior` **산출 과정**을 실데이터로 시각화(결과=배치가 아님).
상위 후보 (cluster, cell) 쌍마다 점수를 **두 가중항으로 분해**(0.6·ctx = compatibility, 0.4·morph = cluster prior)해 어떻게 합산·서열되는지 보여준다.

## 대표셋
- severstal, aitex (한 figure에 2 패널). 이 figure는 score 구성이라 데이터 적합도 무관하게 둘 다 적합.

## 데이터 출처 (실측)
- `compatibility_matrix.json` matrix_symmetric[k][c] = ctx_prior
- `deficit_analysis.json` [k].prior = morph_prior = P(k)
- score = 0.6·ctx + 0.4·morph. 상위 12 후보(내림차순).

## 관찰 (검증됨)
- top-5 = 각 클러스터 peak(ctx=1.0) → 0.6 고정 + 0.4·morph로 서열.
  - severstal top5 score: 0.704/0.695/0.695/0.658/0.648 (모두 cell 0_0_0_1_0, morph만 차이 = funnel).
  - aitex top5: 0.735/0.693/0.690/0.649/0.633 (cell 다양).
- 6위부터 ctx<1 → score 하락.

## 구성 (단순)
- 2 패널(severstal | aitex) 가로 stacked barh, 상위 12 후보(위=최고점).
- 파랑 세그먼트 = 0.6·ctx_prior, 회색 = 0.4·morph_prior, 총길이 = ROI_score.
- y라벨 = 후보의 cluster "k{k}". 상단 5개(ctx=1.0)에 "cluster peaks (ctx=1.0)" 괄호 주석.
- 범례: 파랑=0.6·ctx_prior, 회색=0.4·morph_prior. suptitle = 공식.
- dpi=300, 가로 ~12in.

## Caption (초안)
**Figure 14.** Composition of ROI_score = 0.6·ctx_prior + 0.4·morph_prior for the top-ranked candidate defect–context pairs on Severstal and AITeX. Each bar splits into the weighted compatibility term (0.6·ctx_prior) and the weighted cluster prior (0.4·morph_prior): compatibility dominates the score and sets the ranking, while the cluster prior adds a small per-cluster offset that orders the compatibility-saturated peaks (ctx_prior = 1).
