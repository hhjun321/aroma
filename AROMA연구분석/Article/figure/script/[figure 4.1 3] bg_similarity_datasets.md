# Figure — Background-selection compatibility (AROMA vs Random) across datasets

**스크립트**: `[figure 4.1 3] bg_similarity_datasets.py`
**출력**: `../image/[figure 4.1 3] bg_similarity_datasets.png`
**데이터 루트**: `D:/project/aroma_dataset` (환경변수 `AROMA_DATASET_ROOT`로 재지정 가능)

## 그림이 보여주는 것

**데이터셋별 그룹 violin/box 쌍** 1개씩(5종, Context Complexity Index(CCI) 내림차순 배치). 각 ROI에 배정된 normal 배경 이미지의 **배경 compatibility** 분포를 **AROMA**(파랑, compatibility 기반 선택) vs **Random**(주황, 균등 무작위)으로 비교한다. 데이터셋마다 Δ = mean(AROMA) − mean(Random), 단측 Mann-Whitney U p-value(H1: AROMA > Random), n(ROI 개수)을 함께 표기한다.

## copy-paste 하에서 이 측정이 유효한 이유

training-free copy-paste 엔진에서는 두 arm이 **같은 실 결함 픽셀**을 붙이고 **같은 결함 ROI**를 공유한다. 차이는 각 ROI에 **어느 normal 배경을 배정하는가** 뿐이다. 따라서 분포 충실도 지표(FID/KID/LPIPS, PRDC)는 *구조상* 두 arm에서 거의 동일하며 arm을 판별할 수 없다. 반면 배경 선택은 **실제로 다르다**. 이 그림은 바로 그 인자만 분리한다 — **엔진과 무관한(engine-independent)** AROMA 배치/선택 메커니즘 검정이며, 충실도 지표가 무력한 지점에서 정확히 의미를 가진다.

## 지표 (독립적, 데이터셋 단위 pooled 기준분포)

- **기준분포**(데이터셋별): 실 결함 이미지들의 배경 영역(mask 제외 픽셀) **텍스처 히스토그램** 평균을 pooling. class-agnostic.
- **ROI별**: 배정된 normal 이미지의 텍스처 히스토그램 ∩ 데이터셋 기준분포 (histogram intersection, 0–1).
- **텍스처 히스토그램** = `[intensity | gradient-magnitude | local-variance]` 히스토그램 연결(각 32 bins, 합 1로 재정규화). gradient·local variance가 AROMA의 symmetric compatibility 게이트가 쓰는 텍스처 단서를 포착한다. intensity 단독은 거의 균일한 강판에서 효과를 과소 표현한다.
- 이 지표는 **파이프라인 자신의 class_fit 점수와 독립**이다(AROMA는 구조상 class_fit을 최대화함). 따라서 Δ > 0은 compatibility 게이트가 *실제* 결함–배경 연관을 복원한다는 증거이며 순환논증이 아니다.

입력: `roi/<ds>/clean_bg_selected.json`(AROMA), `roi/<ds>/clean_bg_random_arm.json`(Random), 실 결함 이미지 + 마스크, `train/good` normal.

## 도출 근거 — 5종 전체 (실측 재계산 2026-07-29)

어떤 데이터가 어떤 경로로 이 그림의 숫자가 되는지 추적한다. 아래 수치는 `[figure 4.1 3] bg_similarity_datasets.py`의 함수(`build_ref`/`_texture_desc`/`_good_hist`/`_inter`/`_mwu_p`)를 그대로 import해 5종 전부 재실행한 값이며, 그림 생성 시와 동일한 seed(42)·동일 코드 경로다.

### 1. 산출 경로 (Severstal을 예로 든 공통 절차)

**기준분포 — `build_ref(root, scheme)`**
1. `test/<defect_type>/*`의 실 결함 이미지에서 `np.random.default_rng(42)`로 **최대 150장 서브샘플**(`REF_SAMPLE=150`, 결정론적).
2. grayscale 로드 → 대응 마스크를 읽어 `mask == 0`(결함 아닌 픽셀)만 선택 → **배경 영역만** 남김.
   - 마스크 경로 규약이 데이터셋마다 다름: Severstal은 `masks/{stem}.png`(`scheme="severstal"`), 나머지 4종은 `ground_truth/{defect_type}/{stem}_mask.png`(`scheme="gt_type"`). **5종 모두 마스크 적중률 100%** (아래 표 참조) — 마스크 누락으로 전체 이미지가 배경으로 오산입되는 경로는 발생하지 않았다.
3. `_texture_desc`로 `[intensity(0–256) | gradient-magnitude(0–400) | local-variance(0–2000)]` 각 32 bins → 연결 후 3으로 나눠 정규화 → **96차원 벡터, 합 = 1.0**(5종 전부 dim 96 / sum 1.0000 확인).
4. 서브샘플 히스토그램의 평균이 그 데이터셋의 기준분포. 결함 클래스 구분 없음(class-agnostic).

**ROI별 값 — `collect(...)`**
각 arm JSON 엔트리에서 **`assigned_normal_id` 단일 필드**만 읽는다(예: Severstal `roi_idx=0` → AROMA `_7576e7f81`, Random `_16585993c`). 선행 `_`를 떼고 `train/good/{stem}.*`를 찾아 **마스크 없이 전체 이미지**의 96차원 텍스처 히스토그램을 만들고, 기준분포와 **histogram intersection**(`Σ min(a,b)`, 0–1)을 취한다.

**arm 대칭성 검증**: 5종 전부에서 두 파일의 `roi_idx` + `defect_bbox`가 **엔트리 단위로 완전 일치**(AITeX/MTD/Kolektor/Leather 200/200, Severstal 1000/1000). 즉 ROI 집합·박스 크기·클래스 구성이 동일하고 `assigned_normal_id`만 다르다.

### 2. 데이터셋별 입력 규모 (실측)

| 데이터셋 | CCI | 결함 이미지 (기준분포 소스) | 마스크 적중 | `train/good` | void 게이트 통과 | top-K pool | ROI 수 |
|---|---|---|---|---|---|---|---|
| AITeX (`aitex_tiled`) | 0.440 | 352 (`defect` 단일 타입, 타일) | 352/352 | 7,169 | 6,287 | 315 | 200 |
| Severstal | 0.306 | 3,620 (class1 477 / class2 111 / class3 2,748 / class4 284) | 3,620/3,620 | 5,902 | 5,350 | 268 | 1,000 |
| MTD | 0.246 | 388 (blowhole 115 / break 85 / crack 57 / fray 32 / uneven 99) | 388/388 | 956 | 925 | 47 | 200 |
| Kolektor | 0.224 | 52 (`defect` 단일 타입) | 52/52 | 347 | 331 | 17 | 200 |
| MVTec Leather | 0.200 | 92 (color 19 / cut 19 / fold 17 / glue 19 / poke 18) | 92/92 | 245 | 243 | 13 | 200 |

- 기준분포는 `REF_SAMPLE=150` 상한이라 **Severstal만 서브샘플링이 실제로 작동**(3,620 → 150)하고, 나머지 4종은 결함 이미지 전량(352/388/52/92)을 쓴다.
- void 게이트 통과 수·pool 크기·가중치는 `roi/<ds>/clean_bg_summary.md`(파이프라인 산출)에서 인용. pool은 `pool_cut=p95` 적용 결과.

### 3. 데이터셋별 실측 결과

| 데이터셋 | n | AROMA mean±sd | Random mean±sd | Δ | p (단측 MWU) | Cohen's d | 판정 |
|---|---|---|---|---|---|---|---|
| MVTec Leather | 200 | **0.9343** ± 0.0232 | 0.8550 ± 0.0554 | **+0.0793** | 2.8e-42 | **1.87** | `***` |
| AITeX | 200 | **0.5903** ± 0.1053 | 0.5191 ± 0.1162 | **+0.0712** | 1.7e-11 | 0.64 | `***` |
| Severstal | 1,000 | **0.7555** ± 0.0670 | 0.7239 ± 0.0783 | **+0.0317** | 3.1e-23 | 0.43 | `***` |
| Kolektor | 200 | **0.9351** ± 0.0302 | 0.9201 ± 0.0382 | **+0.0151** | 1.2e-04 | 0.44 | `***` |
| MTD | 200 | 0.7634 ± 0.0866 | 0.7580 ± 0.0898 | +0.0054 | 0.254 | 0.06 | **n.s.** |

**"5종 중 4종 유의(MTD만 n.s.)" 주장은 재현됨.** 단 두 가지를 함께 읽어야 한다:

- **Δ는 CCI 순서와 단조 관계가 없다.** 그림의 x축이 CCI 내림차순 정렬이라 추세로 오독될 수 있으나, 실제 Δ 순위는 Leather(CCI 최저) > AITeX(최고) > Severstal > Kolektor > MTD다. 이 그림은 **CCI 조건부 서사의 근거가 아니다** — CCI/headroom 조건부는 §4.4 downstream 쪽 이야기다.
- **효과크기가 데이터셋마다 4배 이상 차이난다.** Leather d=1.87(큼) / AITeX 0.64(중) / Severstal·Kolektor 0.43~0.44(중소) / MTD 0.06(없음). p값만 보면 4종이 동일하게 `***`로 보이지만, Severstal·Kolektor의 `***`는 상당 부분 표본 수(1,000 / 좁은 sd)에서 온다.

### 4. 데이터셋별 개별 주석

- **MVTec Leather** — Δ와 d 모두 최대지만 **AROMA가 사용한 distinct 배경이 245장 중 11장**(Random 129장)이다. 유효 풀 243장에 `p95` 컷을 적용해 pool이 13장으로 줄어든 결과이며, 200 ROI가 평균 18회씩 같은 배경을 재사용한다. 즉 "가장 호환적인 배경을 고른다"는 목표는 달성했으나 **배경 다양성을 극단적으로 희생**한 형태다(과거 `img_diversity_cap` 문제와 같은 계열). Leather의 downstream Δ가 null/음수인 것(§4.4)과 함께 읽어야 하며, 이 그림 하나로 Leather에서 AROMA가 유리하다고 서술하면 안 된다.
- **AITeX** — 파이프라인 내부 `hist_intersection`이 mean 0.9194(median 1.0000)로 5종 중 유일하게 높고, 외부 지표와의 상관도 유일하게 뚜렷한 양(+0.318)이다. `clean_bg_summary.md`의 "aitex 강" 진술과 내·외부 지표가 **함께 일치하는 유일한 데이터셋**. 단 절대 수준은 가장 낮다(AROMA 0.590) — 타일 표면의 텍스처 변동 자체가 커서 기준분포와의 교차가 낮게 나온다.
- **Severstal** — Δ=+0.032, d=0.43. n=1,000이 유의성의 주된 힘. 외부 지표와 내부 `hist_intersection`의 상관이 **−0.292(음수)** 라 Δ의 원천이 히스토그램 랭킹(`w_src=0.5749`)이 아니라 `class_fit` 항(`w_class=0.4251`, 외부 상관 +0.351)쪽이다.
- **Kolektor** — 유의하나 Δ=+0.015로 최소 수준. Random조차 이미 0.920으로 **천장 근처**(good 347장이 균질한 표면)여서 개선 여지 자체가 좁다. AROMA distinct 배경 40장(Random 149장)으로 여기서도 집중이 관찰된다. 결함 이미지가 52장뿐이라 기준분포의 표본 기반도 5종 중 가장 얇다.
- **MTD** — 유일한 n.s.(p=0.254, d=0.06). 유효 풀이 956장 중 925장으로 거의 전부 남아 선별 여지가 크지 않고, Random 평균이 이미 0.758이다. 내부 `hist_intersection`도 mean 0.4736으로 5종 최저이며 외부 지표와 음의 상관(−0.107). `clean_bg_summary.md`의 "mtd ≈ random" 진술과 일치한다.

### 5. 이 Δ가 무엇 때문에 생기지 않았는지 (교란 배제 — 5종 공통)

- **void 게이트 차이가 아니다**: `clean_bg_selection.py:746 random_arm()`이 **AROMA와 동일한 `valid_ids` 풀**(void 게이트 통과분)에서 균등 추출한다. 두 arm 모두 void normal이 제거된 상태이므로 Δ는 유효 풀 **내부의 랭킹** 차이에서만 나온다.
- **ROI/크기/클래스 구성 차이가 아니다**: §1의 arm 대칭성 검증(5종 전부 `roi_idx`+`defect_bbox` 완전 일치).
- **마스크 누락 편향이 아니다**: 5종 모두 마스크 적중률 100% → 기준분포가 결함 픽셀을 삼켜 오염되는 경로 없음.
- **순환논증이 아니다**: 외부 지표 vs 파이프라인 자신의 점수 필드 상관 —

| 데이터셋 | corr(외부, 내부 `hist_intersection`) | corr(외부, `class_fit`) | 내부 `hist_intersection` mean |
|---|---|---|---|
| AITeX | **+0.318** | +0.596 | 0.9194 |
| Severstal | **−0.292** | +0.351 | 0.5307 |
| MTD | **−0.107** | +0.475 | 0.4736 |
| Kolektor | **−0.252** | +0.551 | 0.5670 |
| MVTec Leather | **+0.223** | +0.200 | 0.5217 |

  외부 지표는 AROMA가 최대화하는 내부 히스토그램 점수를 되받아 읽는 것이 아니다 — **5종 중 3종에서 음의 상관**이다. 반면 `class_fit`과는 5종 모두 양의 상관(+0.20 ~ +0.60)으로, 외부에서 관측되는 우위의 공통 성분은 **`class_fit` 항**이라는 해석이 데이터와 정합한다.

### 6. `clean_bg_summary.md`의 정직성 배너와의 관계

파이프라인 요약의 배너는 "histogram matching은 도메인 조건부 — aitex 강, severstal/mtd ≈ random"이라고 적고 있다. 이는 **파이프라인 내부 지표(`hist_intersection` lift)** 기준의 진술이고, 본 그림은 **외부 3채널 텍스처 지표**로 다른 것을 잰다. 두 진술은 §5 상관표로 오히려 서로 일관된다 — 내부 히스토그램 랭킹이 강한 곳은 AITeX 하나뿐이고(내부 0.9194, 외부 상관 +0.318), Severstal·Kolektor·MTD에서는 내부 랭킹이 약하거나 외부와 반대 방향이다. **이 그림을 "히스토그램 매칭이 Severstal/Kolektor/Leather에서 강하다"의 근거로 쓰면 안 된다.** 이 그림이 지지하는 명제는 "배경 선택 메커니즘이 균등 추출보다 실 결함 배경에 가까운 배경을 배정한다"까지다.

## 읽는 법

- **Δ > 0 이면서 \*\*\* 표기** ⟹ AROMA가 배정한 배경의 텍스처가 실 결함 배경과 Random보다 유의하게 잘 일치한다. n이 수백~수천 ROI 규모이므로 이 유의성은 **견고**하다 — n=3 seed의 downstream 비교(§4.4)가 방향성만 있고 검정력이 부족한 것과 대비된다.
- 실측 결과: 배경 선택 메커니즘은 **5종 중 4종에서 유의**(AITeX, Severstal, Kolektor, MVTec Leather. MTD는 n.s.).

### 결정적 해석 — 메커니즘 ≠ downstream 이득

이 그림은 **선택 메커니즘**을 측정하며 검출 성능 이득이 아니다. 둘을 혼동하면 안 된다. AROMA는 **MVTec Leather**와 **Kolektor**에서도 측정 가능하게 더 호환적인 배경을 선택하지만, 이 두 데이터셋의 *downstream* Δ(AROMA−Random)는 null/음수다(§4.4). 모순이 아니다:

- 메커니즘(AROMA가 텍스처상 더 가까운 배경을 고르는가?)은 거의 어디서나 작동한다 — "거의 균일한" 가죽 표면조차 compatibility 게이트가 균등 추출보다 가까운 매치를 고를 만큼의 텍스처 변동을 갖는다.
- downstream 이득(그 가까운 배경이 YOLO mAP를 올리는가?)은 **별도로 게이팅**된다: (i) baseline headroom — 천장에 가까운 데이터셋(MTD 0.91, Kolektor 0.97, Leather 0.83)은 개선 여지가 없음, (ii) 배경 변동이 실제로 검출을 교란하는지 여부 — 배경에 둔감한 표면에서는 배경을 잘 골라도 검출기가 배우는 내용이 바뀌지 않음.

따라서 이 그림은 **AROMA 배치 정책이 설계대로 동작한다(엔진 무관)** 는 증거로 읽어야 하며, §4.4의 CCI/headroom 조건부 *downstream* 서사를 중복이 아니라 **보완**한다. 호환 배경 선택은 정책이 유효하기 위한 필요조건이지만, 검출 이득의 충분조건은 아니다.

## 동반 자료 (정성, 이 스크립트에는 없음)

정성 몬타주(`bg_pool_compare`, 데이터셋별 대표 예시)가 같은 효과를 시각적으로 보여준다: AROMA의 top-K 배경 풀은 baseline 표면과 일관되게 일치하는 반면, Random의 풀은 비호환 표면(체커플레이트, void, 밝은 균열)을 섞는다. 그 몬타주는 예시용이고, **이** 그림이 전 표본 통계 집계다.

## 재현

```bash
python "[figure 4.1 3] bg_similarity_datasets.py"     # writes ../image/[figure 4.1 3] bg_similarity_datasets.png
```

`scipy`가 있으면 정확한 Mann-Whitney U를 쓰고, 없으면 1000회 permutation p-value로 폴백한다. 결정론적(기준분포 샘플링 seed 42 고정).
