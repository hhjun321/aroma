<!-- 출처: AROMA_DATASET/{profiling,roi,synth} + raw(mtd/leather/aitex_tiled) 로컬 CPU 검증. workflow aroma-copypaste-pivot-local-validation(wf_a911fbff-a0f, 4 agents). GPU 제외. 2026-07-11 -->

# 피벗 검증 — 로컬 CPU 실증 결과 (E1/E2 + Leather V1 보강)

## 1. 피벗 프레이밍

본 연구는 AROMA를 **다중 도메인(multi-domain) 데이터 기반 ROI 영역 검출 + 컨텍스트 인지형 배치(context-aware placement)를 복사-붙여넣기(copy-paste) 기반 위에서 학습 없이(training-free) 수행하는 증강 프레임워크**로 재정의한다. 이는 CASDA(단일 도메인 + ControlNet 생성)와 대비된다. ControlNet 생성 경로는 본 프레임워크의 핵심 기여에서 분리하여 **future work로 강등**하되, 앞선 검증에서 확인된 **결함 난이도 ⟂ 데이터 가용성(deficit⟂availability) 음의 상관 결과**는 유지하여 "생성이 왜 미래 과제로 남아야 하는가"의 근거로 삼는다. copy-paste 기판 위에서는 **위치(placement)가 유일한 조절 레버**이므로, 본 실증의 초점은 (a) clean-bg 선택이 랜덤 배경보다 나은가(E1), (b) 배치가 클래스별 실제 기하를 존중하는가(E2)에 맞춘다. 정직성 제약상, 컨텍스트 신호는 약하며(선행 V2: I/H 1–4%), 본 세션 결과는 aroma≈random이 지배적이고 aitex 단일 조건부 양성만 존재한다. **mAP 개선은 주장하지 않는다**(로컬에 GPU 학습 없음). 모든 결과는 기계론적(mechanistic)/프록시 증거로만 제시한다.

## 2. E1 — Clean-BG 매칭 신호 (clean-bg 선택이 랜덤 배경을 이기는가?)

소스 결함 배경 셀 분포 vs. 배포된 선택(chosen) vs. GOOD 풀 상의 random/best 의 히스토그램 교차(histogram-intersection). Leather 제외(context join 무효). 전 annotation non-dry, **join miss 0건**.

| ds | N | miss | sim_chosen | sim_random | sim_best | lift | %>rand |
|----|----|----|----|----|----|----|----|
| severstal | 600 | 0 | 0.1653 | 0.1632 | 0.6230 | **+0.0022** | 40.7% |
| mtd | 200 | 0 | 0.0762 | 0.0807 | 0.5021 | **−0.0045** | 41.0% |
| aitex | 600 | 0 | 0.8419 | 0.0594 | 0.8951 | **+0.7826** | 100.0% |

Pool(good/defect): severstal 5902/3620, mtd 956/388, aitex 7169/352. Distinct chosen backgrounds 567/186/461 — self-match·tiny-pool 아티팩트 없음. Aitex 셀 다양성(good 128 / defect 87 distinct) 확보 → 고 lift가 단일 셀 붕괴가 아님.
근거: `scratchpad/e1.py` (E1).

### 검증 (Verdict) — 신호는 도메인 분리적(domain-split), 균일하지 않음

- **severstal & mtd — 신호 없음.** lift ≈ 0(+0.0022, −0.0045), chosen이 random 바닥에 붙어 있고 ceiling에서 크게 아래(0.17 vs 0.62; 0.08 vs 0.50), annotation의 ~41%만 랜덤 배경을 이김(동전 던지기 이하). 배포된 clean-bg 선택은 **랜덤 정상 배경 선택보다 낫지 않다.** 선행 V2의 약한 컨텍스트 신호와 일치하며, 이 두 도메인에서 **aroma ≈ random**을 예측 → context-placement가 mAP를 끌어올릴 가능성 낮음.
- **aitex — 강한 진짜 신호.** lift +0.78, chosen(0.84) ≈ best(0.90) ≫ random(0.06), annotation 100%가 랜덤을 이김. 선택이 실제로 소스 결함 자신의 배경 분포에 배경을 매칭 → context-placement에 실질 신호 존재, aitex에서는 mAP 상승 *가능성* 있음.

**종합:** clean-bg 매칭 메커니즘은 설계대로 aitex에서만 작동하고 severstal/mtd에서는 랜덤 배경 선택과 구별 불가. 메커니즘 자체는 유효(aitex가 증명)하므로 severstal/mtd 실패는 코드 버그가 아니라 **해당 도메인의 컨텍스트 특징 분리성(feature separability) 문제**다.

## 3. E2 — 배치 기하(placement geometry) (배치가 클래스별 실제 기하를 존중하는가?)

실제 vs. 합성 edge+span 배치, 클래스별. 분류: w≥0.8·W 또는 h≥0.8·H → span; 아니면 bbox가 임의 경계 8% 이내 → edge; 아니면 surface. "e+s%"=(edge+span)/total, gap=placed−real. Path-read 실패 0건(전 11 class-row).

| ds | class | realN | plcN | real e+s% | placed e+s% | gap |
|----|-------|-------|------|-----------|-------------|-----|
| mtd | blowhole | 115 | 40 | 39.1 | 45.0 | +5.9 |
| mtd | break | 85 | 43 | 100.0 | 46.5 | **−53.5** |
| mtd | crack | 57 | 43 | 91.2 | 51.2 | **−40.1** |
| mtd | fray | 32 | 32 | 84.4 | 78.1 | −6.2 |
| mtd | uneven | 99 | 42 | 99.0 | 88.1 | −10.9 |
| leather | color | 19 | 60 | 0.0 | 41.7 | **+41.7** |
| leather | cut | 19 | 72 | 5.3 | 30.6 | **+25.3** |
| leather | fold | 17 | 48 | 35.3 | 52.1 | +16.8 |
| leather | glue | 19 | 27 | 5.3 | 33.3 | **+28.1** |
| leather | poke | 18 | 3 | 0.0 | 33.3 | **+33.3** |
| aitex | defect | 352 | 600 | 75.9 | 79.2 | +3.3 |

근거: `scratchpad/e2.py` (E2). Real: `D:/project/AROMA_DATASET/profiling/{profiling_mtd,profilng_leather,profiling_aitex}/morphology_features.csv`. Placed: `D:/project/AROMA_DATASET/synth/{mtd,leather,aitex}/annotations.json`. severstal 제외(raw pixel 없음).

### 검증 (Verdict) — 배치는 기하-무지(geometry-blind)

placed edge+span 분포가 실제 클래스별 패턴을 추종하지 않으며, 결정적으로 **데이터셋 간 반대 방향으로 어긋난다**:

- **mtd**: 실제 결함은 강하게 edge/span 구속(break 100%, crack 91%, uneven 99%) — 균열·파단이 판 가장자리까지 뻗기 때문. 합성은 이를 모두 ~45–50%로 평탄화 → edge 구속 결함을 표면 내부로 흩뿌림(break −53.5, crack −40.1). 실제 발생 위치를 재현하지 못함.
- **leather**: 실제 결함은 거의 전부 표면 내부(color/poke 0%, cut/glue 5%) — 소재 필드에 위치, 타일 경계 아님. 합성은 이 중 30–42%를 edge/span으로 밀어냄(전 클래스 양의 gap, +25~+42). 실제에 없는 edge 배치를 발명.
- **aitex**: 단일 "defect" 클래스, gap +3.3만 — placed ≈ real인 유일 케이스이나, aitex는 존중할 클래스별 기하가 없음(직물 타일 위 단일 병합 클래스) → 기하 인지의 증거가 아니라 약한 테스트일 뿐.

방향 뒤집힘(mtd는 edge에서 떼어냄, leather는 edge로 밀어붙임)은 **클래스 기하에 조건화하지 않고 위치를 샘플링하는 배치기**의 서명이다. copy-paste 하에서 위치가 전부인 레버이므로, 이는 전용 clean-bg/placement 모듈의 구체적·클래스별 타깃: 샘플된 bbox 위치를 각 class_key의 실제 edge/surface prior에 구속할 것.

**CAVEAT(정직):** 이는 배치-현실성 측정이지 downstream mAP 주장이 아니다. 이 gap을 닫으면 detector AP가 오른다는 확인은 GPU 학습이 필요(본 범위 밖). 또한 일부 행의 placed N이 작아(leather poke plcN=3) 해당 % 는 방향성으로만 취급.

## 4. Leather V1 — 결정론적 재클러스터 + V2 Colab 이월

**결과: 재적합이 n_clusters = 4 선택** (k=1..5 min BIC). 프로파일러 정확 복제: MORPH_FEATURES=[linearity,solidity,extent,aspect_ratio,eccentricity,circularity], `.../profilng_leather/morphology_features.csv`(92행, `defect_mask_path` 전부 unique), feature별 min-max 정규화(÷range+1e-6), `GaussianMixture(random_state=42, n_init=3)`, BIC로 k=1..5 선택, 92행 전체 예측·`defect_mask_path` 키.

**BIC by k:** k1=−665.40, k2=−1507.61, k3=−1554.37, **k4=−1661.82 (min→선택)**, k5=−1646.33
**Cluster↔class 정렬:** ARI = **0.2101**, NMI = **0.3335**, n_clusters = **4**

**Crosstab (defect_type × cluster):**
```
cluster     0   1   2   3
color       1   5   5   8
cut         0  11   8   0
fold        0   1  15   1
glue        8   2   3   6
poke       10   0   0   8
```

**DRIFT FLAG:** 커밋된 `.../profilng_leather/morphology_clusters.json`은 메타데이터에 `n_clusters: 4`를 보고하나 `cluster_assignments`에는 **19 instance / 2 distinct label(2,3)만 존재**(stem-collision 아티팩트, 19 vs 92). 현행 코드 재적합은 **92 instance / 4 populated cluster** 복원. 드리프트는 k값이 아니라 **assignment 커버리지**에 있음(커밋 19행/2 유효 클러스터, 재적합 92행/4 유효 클러스터). 커밋된 leather `cluster_assignments`는 무효로 취급, 위 재적합 사용.

**정렬 해석:** 약~중간 구조(ARI 0.21, NMI 0.33). `fold`→cluster 2(15/17) 등 일부 클래스 집중, `glue`/`poke`는 0/3 분산 — 깨끗한 class↔cluster bijection 없음. 형태 클러스터는 shape-driven이며 class-pure가 아님(기대대로).

**leather V2(context) — 로컬 BLOCKED.** `profilng_leather/context_features.csv`가 동일 stem-collision 버그(image_id=bare stem이 클래스 간 충돌) → context/cluster join 무효. **Colab phase0 재실행으로 이월.**
근거: `scratchpad/leather_recluster.py`.

## 5. 종합 — 피벗 thesis 실증 현황

| 축 | 선행/신규 증거 | 결론 | 근거 |
|----|----|----|----|
| ROI 영역 검출이 클래스 구조를 담는가 | V1 cluster↔class (선행), leather V1 재적합 ARI 0.21/NMI 0.33 | 약~중간 구조, class-pure 아님(shape-driven) | `class_vs_cluster_validation_20260711.md`; `scratchpad/leather_recluster.py` |
| 컨텍스트 신호 강도 | V2 context MI (선행, I/H 1–4%), E1 lift | 도메인 분리적 — aitex만 강함(+0.78), severstal/mtd≈0 | `class_vs_cluster_validation_20260711.md`; `scratchpad/e1.py` |
| clean-bg 선택 > 랜덤? | E1 %>rand | aitex 100%, severstal 40.7%, mtd 41.0% (2/3 도메인 랜덤 이하) | `scratchpad/e1.py` |
| 배치가 실제 기하 존중? | E2 gap | 아니오 — geometry-blind, 도메인 간 방향 뒤집힘 | `scratchpad/e2.py` |
| 실제↔선택 AR/scale 정합 | V3 (선행) | (선행 결과 참조) | `class_vs_cluster_validation_20260711.md` |
| 합성 ar_fallback | V4 (선행) | (선행 결과 참조) | `class_vs_cluster_validation_20260711.md` |
| deficit ⟂ availability | 선행 음의 상관 | 생성 경로 future-work 근거 유지 | `class_vs_cluster_validation_20260711.md` |

**scoped context-automation 주장:** 컨텍스트 자동화(clean-bg 매칭)는 **범용이 아니라 도메인-조건부**로만 실증된다. aitex는 메커니즘이 설계대로 작동함을 증명(E1 lift +0.78, %>rand 100%)하나, severstal/mtd에서는 랜덤과 구별 불가(E1 lift≈0, %>rand≈41%). 동시에 E2는 **배치 자체가 아직 클래스 기하를 조건화하지 못함**을 보여, copy-paste 유일 레버인 위치가 미최적화 상태임을 확인한다. 따라서 실증된 것은 "메커니즘 가능성(mechanistic capability, aitex)"과 "구체적 개선 타깃(E2 클래스별 edge/surface prior 구속)"이지, **범용 mAP 개선이 아니다**(GPU 부재, TBD).

## 6. AROMA.txt 반영 포인트 (모든 실험 후 적용)

- [근거: 세션 종합] 프레이밍을 "multi-domain 데이터 기반 ROI 검출 + context-aware placement on copy-paste substrate (training-free)"로 교체, CASDA(단일 도메인+ControlNet)와 대비 명시.
- [근거: `class_vs_cluster_validation_20260711.md` — deficit⟂availability] ControlNet 생성 경로를 future work로 강등하되, deficit⟂availability 음의 상관을 "생성이 미래 과제여야 하는 이유"로 유지.
- [근거: `scratchpad/e1.py` — E1] 컨텍스트 자동화(clean-bg 매칭)를 **도메인-조건부**로 서술: aitex lift +0.78·%>rand 100% (강한 신호) vs severstal/mtd lift≈0·%>rand≈41% (랜덤 이하). "범용 컨텍스트 매칭" 주장 삭제.
- [근거: `class_vs_cluster_validation_20260711.md` — V2 (I/H 1–4%)] 컨텍스트 신호가 전반적으로 약함을 E1 도메인 분리 결과와 정합되게 명시.
- [근거: `scratchpad/e2.py` — E2] 배치가 현재 geometry-blind임을 한계로 명시하고, **클래스별 edge/surface prior 구속**을 구체적 개선 타깃(전용 clean-bg/placement 모듈)으로 로드맵에 추가. mtd(edge에서 이탈)·leather(edge로 발명) 방향 뒤집힘을 근거로.
- [근거: `scratchpad/leather_recluster.py` — leather V1] leather morphology 클러스터 n_clusters=4, ARI 0.21/NMI 0.33(약~중간, shape-driven)로 기재. 커밋된 `morphology_clusters.json`의 19행/2클러스터 assignment는 stem-collision 무효 → 재적합 92행/4클러스터로 교체 명시.
- [근거: `scratchpad/leather_recluster.py` — leather V2 BLOCKED] leather context_features.csv의 stem-collision 버그와 V2 Colab phase0 재실행 이월을 known-limitation으로 명시.
- [TBD — GPU 필요] mAP/AP 개선 주장은 일절 넣지 않음. E1/E2는 mechanistic/proxy 증거로만 표기하고, 실제 detector 개선 확인은 GPU 학습 후 채움(현재 TBD).
- [근거: `scratchpad/e1.py` E1 pool 통계] 데이터 무결성 각주: E1 join miss 0건, distinct chosen 567/186/461, aitex 셀 다양성 확보(단일 셀 붕괴 아님) — 신호가 아티팩트가 아님을 명시.
- [근거: `class_vs_cluster_validation_20260711.md` — V1/V3/V4] 선행 검증 4종(V1 cluster↔class, V2 context MI, V3 real-vs-selected AR/scale, V4 synth ar_fallback) 요약을 실증 부록으로 유지, 각 수치는 커밋 문서에서 인용(재계산 금지).
