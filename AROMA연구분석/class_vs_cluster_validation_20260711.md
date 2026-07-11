<!-- 출처: AROMA_DATASET/{profiling,roi} 4종(severstal/mtd/mvtec_leather/aitex) 로컬 실측. class-conditioned thesis 검증. 2026-07-11 -->

# class 라벨 vs 비지도 cluster — thesis 재포지셔닝 실증 검증 (4종)

> **목적**: "결함 그룹핑은 클래스 라벨 사용, 통계 자동화는 context(배경) 축에 한정"이라는 thesis 재포지셔닝의 실증 근거 확보.
> **데이터**: `AROMA_DATASET/{profiling,roi}` 4종. 원본 대용량 미사용 — profiling(morphology_features/context_features/compatibility_matrix/morphology_clusters) + roi(roi_selected)로 분석.
> **성격**: 진단 전용. 코드·파이프라인 변경 없음.

---

## V1 — cluster ↔ class 정렬 (다른 축인가)

| 데이터셋 | image_id 고유 | ARI | NMI | 해석 |
|---|---|---|---|---|
| **severstal** (4-cls) | ✓ | **0.020** | 0.081 | 거의 무관 — 클러스터가 클래스 미회복 |
| **mtd** (5-cls) | ✓ | **0.122** | 0.222 | 낮음 — 클러스터가 클래스를 가로지름 |
| **mvtec_leather** (5-cls) | ✗ | — | — | **측정 불가**: image_id 충돌(19/92) → 비지도 클러스터링 degenerate |
| **aitex** (1-cls) | ✓ | N/A | N/A | 단일클래스 |

→ **비지도 GMM 클러스터는 클래스 구조를 복원하지 못한다**(ARI 0.02–0.12). 라벨로 축을 바꾸는 것은 cosmetic이 아닌 실질적 재정의. leather는 클러스터링 자체가 깨져(고유키 버그) **라벨 사용이 사실상 필수**.

## V2 — context cell을 class vs cluster 중 무엇이 설명하나 (★ 스코프한 context 축)

I = 상호정보량(mutual info), H(cell) = context cell 총 엔트로피. I/H가 설명력.

| 데이터셋 | I(cell;class) | I(cell;cluster) | H(cell) | 설명력 | 해석 |
|---|---|---|---|---|---|
| **severstal** | 0.053 | 0.030 | 4.32 | ~1% | class ≳ cluster, **둘 다 무의미** |
| **mtd** | 0.188 | 0.179 | 4.79 | ~4% | class ≈ cluster, 둘 다 약함 |
| **mvtec_leather** | 0.009 | (불가) | 3.78 | ~0% | class 정보 0 |
| **aitex** | 0.000(단일) | **0.460** | 3.18 | ~14% | **cluster만** context 구조 포착 |

→ **결정적**: context cell은 **class·cluster 어느 grouping과도 거의 독립**(다중클래스 설명력 1–4%). 즉 **"class로 키잉하면 context가 개선된다"는 성립하지 않음** — context 조건부 자체가 약함(세션 전반 aroma≈random과 정합). 유일한 비지도 context 구조는 **단일클래스 aitex의 I=0.46** — 이는 **유보한 확장 (a) intra-class 세분화**이며, 하필 **arbiter(aitex)에서 가장 값짐**.

## V3 — 실제 vs 선택본 클래스별 AR/스케일 (선택이 baseline을 왜곡하나)

KS = 분포 거리(0~1), p = KS 유의확률. AR = regionprops aspect_ratio.

**severstal** (real 3620 → sel 200): 클래스 비율 **강제 균등화**
| class | real% | sel% | AR med (r→s) | KS_AR (p) |
|---|---|---|---|---|
| class1 | 13.2 | 25.0 | 2.71→5.67 | **0.37 (0.000)** |
| class2 | 3.1 | 25.0 | 11.69→10.64 | 0.15 (0.34) |
| class3 | **75.9** | 25.0 | 5.69→5.69 | 0.09 (0.77) |
| class4 | 7.8 | 25.0 | 2.19→2.80 | **0.27 (0.004)** |
→ 실제 **class3 76% 지배**를 선택이 **25% 균등**으로 붕괴(class2는 3%→25%, 8× 과대). class1/class4는 within-class AR도 **유의하게 왜곡**(더 세장 선택).

**mtd** (388→200): floor가 rare 부양(fray 8→16%, crack 15→22%; blowhole 30→20%). within-class AR KS 0.07–0.16(전부 비유의).

**mvtec_leather** (92→70): **poke 20%→1.4%(KS 0.94)**, cut 21→34%, glue 21→13%. within-class AR KS 0.21–0.27(비유의).

**aitex** (352→200): 단일클래스지만 선택이 **극단 세장 제거**(AR 5.13→3.75, **KS 0.16 p=0.003 유의**; area 630→278).

→ 현 선택은 클래스별 분포를 **광범위하게 왜곡**. 특히 severstal(76%→25%)·leather(poke 소멸)는 심각. within-class AR 유의 왜곡은 severstal class1/class4·aitex. 단 "baseline 비율 매칭"은 주장에서 제외(결정 2)했으므로 이는 진단이지 목표 아님.

---

## V4 — 합성 충실도 (synth annotations, leather 제외 severstal/mtd/aitex)

synth 산출(`AROMA_DATASET/synth/{ds}/annotations.json`)의 `method`·`class_key`로 최종 증강셋을 분석. `copy_paste_arfallback` = AR 게이트(`cn_ar_threshold` 2.5) 걸린 세장 결함이 ControlNet 대신 copy_paste로 처리된 것.

### ★ ControlNet 생성이 세장 결함에서 체계적으로 실패 (핵심 기제)
| 데이터셋 | 전체 ar_fallback | corr(real AR median, class ar_fallback%) | 극단 사례 |
|---|---|---|---|
| **severstal** | 64% (384/600) | **+0.91** | class2(AR 11.7) → **98% 폴백** (ControlNet 3/150) |
| **mtd** | 47% (94/200) | **+0.84** | crack(AR 11.2) → **81%**; blowhole(AR 1.6) → 10% |
| **aitex** | 38% (231/600) | (단일 클래스) | AR 5.13 → 38% 폴백 |

→ **"ControlNet arm"은 실상 "compact 결함=ControlNet, elongated 결함=copy_paste"**. 세장할수록 폴백률이 오름(corr +0.84~+0.91). 가장 세장한 클래스(severstal class2 98%, mtd crack 81%)는 사실상 전량 copy_paste. **프레임워크 우월성(Q1) 주장 시 "생성 엔진은 세장 결함을 못 만든다"를 반드시 병기.**

### 최종 증강셋이 baseline 클래스 분포를 강하게 왜곡
- **severstal**: synth **25%/25%/25%/25%** (강제 균등) vs real class3 **76%**·class2 3%. class_floor 효과가 합성 출력까지 전파.
- **mtd**: synth ~20% 균등 vs real blowhole 30%/fray 8%.
→ V3(선택)에서 본 왜곡이 합성 출력에 그대로 남음. baseline-유사 증강과는 반대 방향(단 이는 주장 제외 항목).

### aitex 세장 결함의 이중 불리 (arbiter 함의)
aitex 결함은 세장(AR 5.13): (a) V3에서 **선택이 극단 세장을 버리고**(AR IQR 32→11), (b) V4에서 **남은 것도 38% copy_paste 폴백**. → arbiter(aitex)에서 생성 엔진의 실제 기여가 제한적임을 정량 확인.

---

## 종합 판정 (정직)

1. **클래스 그룹핑은 방어 가능**: 비지도 클러스터가 클래스를 복원 못 함(V1 ARI 0.02–0.12) + leather 클러스터링 degenerate → **라벨 사용이 정직하고 실용적**. 논문에서 "라벨 있으면 라벨 사용, 없을 때만 비지도"로 서술하는 근거 확보.

2. **그러나 "class 키잉이 context를 개선"은 미지지**(V2): context는 class·cluster 어느 쪽과도 거의 독립. 자동화 기여를 context로 좁히되 **"context-plausible 배치 프레임워크"로 서술하고 "개선" 단정 금지**. rekeying 자체는 downstream 우위를 만들지 않음.

3. **비지도의 유일한 실측 가치 = 단일클래스 aitex intra-class context 구조(I=0.46)** = 유보한 확장 (a). **결정 1(b now / a for reviewer)은 합리적이나, (a)가 arbiter에서 가장 값지다는 점을 각주로 무장**(리뷰어가 정확히 이 지점을 찌를 수 있음).

4. **V3는 선택이 baseline 클래스 분포를 왜곡함을 보임**(severstal 76%→25% 등). 향후 baseline-유사 증강을 추구하면 현 class_floor가 역효과 — 단 이는 주장 제외 항목.

5. **(V4) 생성 엔진은 세장 결함을 못 만든다**(corr AR↔ar_fallback +0.84~+0.91): "ControlNet arm"은 compact만 생성·elongated는 copy_paste 폴백. severstal class2 98%·mtd crack 81% 폴백. → 프레임워크 우월성/생성 novelty 주장 시 **"세장 결함은 양 arm 모두 copy_paste"**를 필수 병기. arbiter(aitex)는 세장이라 생성 기여 제한적(V3 선택 드롭 + V4 38% 폴백 이중 불리).

## 논문 반영 (스코프 확정)
- 자동화 기여 = **배경 문맥 특성화 + context-plausible 배치**로 한정. V2 근거로 "개선" 대신 "프레임워크" 서술.
- 결함 그룹핑 = 라벨. **leather 클러스터링 degenerate**를 라벨 사용의 실증 근거로 인용.
- (a) intra-class 세분화 = future work, **aitex I(cell;cluster)=0.46**을 "단일클래스에서 비지도 세분화가 context 구조 회복" 예비 증거로 각주.

## 후속/데이터 정합 이슈
- **leather `image_id` 충돌 버그**: profiling이 stem(`000`)을 키로 써 5클래스가 19개로 붕괴 → `{defect_type}_{stem}` 복합키로 재빌드 필요. leather cluster측 V1/V2 유효화 조건.
- synth(annotations) 미사용 — 필요 시 배치 위치(bbox) 기반 placement 충실도(V4)로 확장 가능.
