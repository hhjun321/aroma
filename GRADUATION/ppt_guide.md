# 발표 PPT 가이드 — 논문별 핵심 수식·흐름 요약

학위논문 『Context-Aware 데이터 증강 기반 산업용 결함 검출 연구』를 구성하는 4편 논문의
증강법 핵심, 수식, 흐름을 발표용으로 간략 정리한다. (수식 출처: 원논문 docx OMML 직접 추출)

---

## 1. 객체탐지 모델기반 장비 결함검출 방법에 대한 연구 (제3장)

**한 줄 요약**
YOLOv8 계열 모델에 기본 증강 / Diffusion 기반 합성 증강을 적용해 장비 결함 검출 성능을 비교. 증강 효과를 Precision·Recall·mAP로 정량 평가하는 출발점 연구.

**수식**

| 번호 | 수식 | 설명 |
|---|---|---|
| (1) | $P = \dfrac{TP}{TP+FP}$ | 정밀도. 검출한 것 중 실제 결함 비율 |
| (2) | $R = \dfrac{TP}{TP+FN}$ | 재현율. 실제 결함 중 검출한 비율 |
| (3) | $AP = \int_0^1 P(R)\,dR$ | PR 곡선 아래 면적 (클래스별) |
| (4) | $mAP = \dfrac{1}{N}\sum_{i=1}^{N} AP_i$ | 전 클래스 AP 평균. N = 클래스 수 |

**흐름**

```
원본 결함 이미지 ─┬─ 기본 증강 (기하/광도) ──┐
                  └─ Diffusion 합성 증강 ────┤
                                            ▼
                                   YOLOv8 n/s/m 학습
                                            ▼
                              P / R / mAP50 / mAP50-95 평가
```

---

## 2. 결함검출 모델 성능 개선을 위한 결함데이터 증강 방법에 대한 연구 (제4장)

**한 줄 요약**
결함 세그멘테이션(EfficientNet·ResNet34 백본)에서 원본셋 vs 증강셋 혼합 학습 효과를 mIoU·Dice로 비교. 증강 비율을 높일수록 성능 상승 확인 → "증강 데이터를 어떻게 만드느냐"가 다음 과제로 이어짐.

**수식**
원논문·학위논문 모두 별도 수식 없음. 평가지표는 mIoU, Dice.

**흐름**

```
원본 데이터 ──► 분할비율 탐색 (6:4 / 7:3 / 8:2)
                      ▼
      증강 데이터 혼합 (증강 비율 0 → 0.5 …)
                      ▼
      세그멘테이션 학습 (EfficientNet / ResNet34)
                      ▼
              mIoU · Dice 비교
```

---

## 3. AROMA: Adaptive ROI-based Morphology-Aware Augmentation (제5장)

**한 줄 요약**
결함–배경 호환성 행렬을 **사람이 설계하지 않고 데이터에서 자동 도출**하는 copy-paste 증강 프레임워크. 결함 형태(MCI)와 배경 맥락(CCI)을 프로파일링 → 패치 단위 공동출현 통계로 대칭 호환성 게이트 학습 → 가장 어울리는 위치에 결함 배치. 5개 이종 데이터셋에 재설계 없이 적용.

**수식**

| 번호 | 수식 | 설명 |
|---|---|---|
| (1) | $CCI = \mathrm{Mean}(\text{TextureEntropy},\ \text{ContextClusterCount},\ \text{FreqComplexity},\ \text{OrientVariance})$ | 배경 맥락 복잡도. 4개 정규화 성분 평균. 값이 높을수록 context-aware 배치 효과 큼 |
| — | $K^{*} = \arg\max_k \mathrm{BIC}(k)$ | 결함 형태 클러스터 개수 자동 결정 (GMM + BIC). 수동 임계값 불필요 |
| (2) | $\mathrm{ctx\_prior}(k,c) \propto \sqrt{P_{def}(k,c)\cdot P_{clean}(c)}$ | **핵심**. 형태 클러스터 k ↔ 배경 셀 c 대칭 호환성. 결함 주변에서 실제 관측되고(P_def) 정상 이미지에도 존재하는(P_clean) 셀만 높은 점수. 기하평균 → 한쪽만 높은 후보 억제 |
| (3) | $\mathrm{ROI\_score} = \mathrm{ctx\_prior} + \mathrm{morph\_prior}$ | 소스 결함 순위화. morph_prior = 클러스터 사전확률 P(k). 상위 K개 채택 |
| (4) | $\cap(a,b) = \sum_c \min(a(c),\,b(c))$ | 히스토그램 교차. 아래 두 점수의 기본 연산 |
| (5) | $\mathrm{bg\_score} = \mathrm{src\_fit} + \mathrm{class\_fit} + \mathrm{size\_fit}$ | 배경 이미지 배정. 원본 주변 분포 일치·클래스 프로파일 일치·크기 수용 |
| (6) | $\mathrm{site\_score} = \cap(h_s,\ \mathrm{tgt}[k])$ | 배경 내 삽입 위치. 후보 위치 s의 ring 히스토그램 h_s와 클러스터 k 호환성 행의 교차 최대 지점 |

보조: 배경 텍스처 범주(백분위 자동 적응, 해석용)
- Smooth: LocalVariance ≤ P25
- Directional: OrientEntropy ≤ P25 ∧ FreqComplexity ≤ P25
- Periodic: FreqComplexity ≥ P75
- Organic: TextureEntropy ≥ P50 ∧ LocalVariance ≥ P75
- Complex: 그 외

**흐름**

```
데이터셋 프로파일링 ──► MCI / CCI 산출 (1)
        │
        ├─ 결함 형태: GMM 클러스터링, K* = argmax BIC
        └─ 배경 맥락: 특징별 삼분위(P33/P66) 셀 이산화
                      ▼
        대칭 호환성 게이트 ctx_prior (2)   ← 패치 공동출현 통계
                      ▼
        ROI 선택 ROI_score (3)  →  배경 배정 bg_score (5)  →  위치 확정 site_score (6)
                      ▼
        copy-paste 합성 (gradient-domain / alpha blending)
                      ▼
        YOLOv8n 다운스트림 평가 (Baseline vs Random-ROI vs AROMA)
```

**핵심 메시지**: 효과는 "baseline 여유(headroom) + 배경 복잡도(CCI)"가 함께 있을 때 발생. 없으면 Random과 동등(해롭지 않음).

### 스터디 보강 — 선택 3단계의 실제 수식 (출처: `AROMA연구분석/핵심이론스터디/AROMA/`)

논문 식 (3)·(5)·(6)은 코드 상 **"어떤 결함 → 어느 배경 이미지 → 그 안의 어느 자리"** 3단 선택 체인이다. 공통 재료는 phase0 프로파일링 산출물 하나(`matrix_symmetric`)이고, 픽셀은 다시 보지 않는다.

**① 어떤 결함을 쓸까** (`roi_selection.py`, step3)

```
ROI_score = 0.5·ctx_prior + 0.3·morph_prior + 0.2·quality_score     (realism 모드, 정본)
```
- ctx_prior = `matrix_symmetric[k][cell_key]` — 결함 자신의 소스 문맥 셀에서 읽은 **스칼라**
- morph_prior = P(k) = n_k / N (군집 크기 비율)
- quality_score = subtype × 데이터셋 도메인 라벨 매칭 점수 (상수 조건)
- 논문 식 (3)은 무가중 합. 코드 정본은 가중 0.5/0.3/0.2 (논문-코드 불일치, 미해결 논점)
- 선발 규칙: (k,c) pair quota + 클래스 floor + 소스 크롭 1회 사용(`img_diversity_cap=1`) → top_k (severstal 1000 / 나머지 200)

**② 어느 배경 이미지에 붙일까** (`clean_bg_selection.py`, step3.5 1단계)

```
∩(a,b) = Σ_c min(a_c, b_c)                      히스토그램 교집합 (논문 식 4)

src_fit   = ∩(h_g, p_src)     결함이 원래 살던 이미지의 표면과 닮았나   (개체 단위)
class_fit = ∩(h_g, p_cls)     이 클래스 결함들의 배경 집계와 닮았나     (클래스 단위)
k_fit     = ∩(h_g, tgt[k])    형태 군집 k의 목표 문맥을 지원하나        (군집 단위)
size_fit  = min(1, 0.95·W/w, 0.95·H/h)   왜곡 없이 들어가나             (기하)

λ_j = E_ROI[ max_g u_j(g) − median_g u_j(g) ]    cue 변별력 = best−median 갭
w_j = λ_j / Σλ                                   → U(g) = Σ_j w_j·u_j(g)
```
- 논문 식 (5)의 3-cue(src/class/size)에 코드는 **k_fit 추가, 가중은 lift로 자동 유도** (하드코딩 없음). 실측: w_size는 5종 중 4종에서 0으로 자동 소거, w_k는 5/5 생존
- 필터 아님, **랭킹**: void 게이트(P15 분산·엣지, 과반-void 컷)로 거른 뒤 U(g) 상위 pool(~16장) 배정
- 전역 질의 채택 근거: 결함 인접 타일 국소화 안은 self-retrieval MRR에서 5/5 열세(severstal 6.7배) → "이미지 선택은 전역, 자리 선택은 국소"

**③ 그 배경의 어느 자리에 붙일까** (`ring_sgm`, step3.5 2단계)

```
tgt[k]     = L1norm(matrix_symmetric[k])          군집 k의 목표 문맥 분포 (②의 k_fit과 공유)
h_s(c)     = 자리 s를 둘러싼 ring R(s) 타일들의 셀 정규화 히스토그램 (void 제외)
score(k,s) = ∩(h_s, tgt[k])                       (논문 식 6)
s*         = argmax_{valid s} score(k,s)
valid(s)  ⟺ ∀t∈F(s): observed ∧ ¬void          footprint 전 타일 admissible
```
- **tgt[k] 도출 4단계**: ① 인구조사 2번(P_def(k,·)=군집 k 결함 둘레 타일, P_clean=clean 전 타일) → ② AND 결합 √곱 (논문 식 2) → ③ 행 max=1 저장(히트맵) → ④ 사용 직전 합=1 재정규화. ③→④는 단조 rescale이라 모양 불변 → "논문 수식 무수정"
- **왜 ring**: footprint는 결함이 덮어 사라짐. 합성 후 경계 연속성을 결정하는 건 둘레
- **왜 분포 매칭(평균 ✗)**: 평균 argmax는 "흔한 셀이 많은 자리"로 수렴. min-겹침은 초과분을 잘라 흔한-셀 몰빵 차단. 실측: footprint→ring(평균 유지) 4/5 악화, 평균→분포 매칭 5/5 개선 → **개선의 원천은 score 정의**
- 성질: ∩(a,b) = 1 − TV(a,b). 교집합 최대화 ≡ total variation 거리 최소화 (표준 거리의 쌍대)
- 자리 quality 필터(옵션): CASDA 4-성분(blur 0.3 + contrast 0.3 + brightness 0.2 + noise 0.2)을 자리 단위 P15 분위로 적용. 절대 0.7 폐기(leather 자리 max 0.67)

**한 줄 암기**: 지도 1개(`matrix_symmetric`) ← 이미지가 h_g로, 자리가 h_s로 질의. 결함(①) → 배경(②, 전역) → 자리(③, 국소).

---

## 4. CASDA: Context-Aware Steel Defect Augmentation (제6장)

**한 줄 요약**
철강(Severstal) 전용 5단계 파이프라인. 결함 기하 특성과 배경 표면 특성을 분류 → 호환성 행렬(수작업) → 3채널 힌트 이미지 + 문맥 프롬프트로 ControlNet 조건부 생성 → 품질 게이트 통과 샘플만 학습셋에 통합. Class 2(희소 클래스) AP +22 pp.

**수식**

| 번호 | 수식 | 설명 |
|---|---|---|
| (1) | $\lambda = \dfrac{\lambda_{max}-\lambda_{min}}{\lambda_{max}+\lambda_{min}}$ | 선형성. 마스크 좌표 공분산 고유값. 1에 가까울수록 선형(스크래치) |
| (2) | $\sigma = \dfrac{A_{defect}}{A_{convex\_hull}}$ | 견고도(solidity). 볼록껍질 대비 면적. 낮으면 경계 불규칙 |
| (3) | $\varepsilon = \dfrac{A_{defect}}{A_{bbox}}$ | 면적비(extent). 직사각형성 |
| (4) | $\alpha = \dfrac{\text{major axis}}{\text{minor axis}}$ | 종횡비. 최소자승 타원 적합 |
| Table 1 | linear_scratch: λ>0.85 ∧ α>5.0 / irregular: σ<0.7 / compact_blob: α<2.0 ∧ σ>0.9 / general: 그 외 | 결함 4유형 분류. 임계값은 N=19,958 분포에서 log-scale Otsu로 도출 |
| (5) | $S_{suitability} = 0.5\,S_{matching} + 0.3\,S_{continuity} + 0.2\,S_{stability}$ | ROI 적합도. 매칭 품질·공간 연속성·배경 안정성 가중합. ≥0.7 고품질 템플릿 |
| — | τ = 14.09 (분산), ε = 0.607 (엣지 밀도) | clean 배경 패치 선택 임계값 (적합도용). log-scale Otsu 도출 |
| — | $\text{hint} = 0.5R + 0.3G + 0.2B$ | ControlNet 조건 입력. R=결함 기하, G=Sobel 표면 방향, B=7×7 국소분산 거칠기 |
| Table 7 | $Q = 0.3\,Q_{morph} + 0.3\,Q_{sharp} + 0.2\,Q_{artifact} + 0.2\,Q_{area}$, Q ≥ 0.7 채택 | 품질 게이트. 형태 일치·Laplacian 선명도·노이즈·면적비 |

**흐름**

```
[1단계] 기하 ROI 특성화 ── λ σ ε α (1)~(4) → 결함 4유형
        배경 분류 ── Canny 엣지밀도 + Sobel 방향 → Smooth/V-stripe/H-stripe/Textured/Complex
        ROI 적합도 S_suitability (5)
                      ▼
[2단계] ControlNet 구성 ── 3채널 힌트 (0.5R+0.3G+0.2B) + 문맥 프롬프트(F1~F4)
                      ▼
[3단계] 문맥 기반 합성 ── 호환성 행렬(결함유형 × 배경유형)로 배경 선택 → 생성
                      ▼
[4단계] 품질 게이트 ── Q ≥ 0.7 (83.0% 통과)
                      ▼
[5단계] 데이터셋 통합 → EB-YOLOv8 / YOLO-MFD 평가
```

### 스터디 보강 — 호환성 행렬의 실체 (출처: `AROMA연구분석/핵심이론스터디/CASDA/stage3.md`)

논문 §3.2.3 "결함타입×배경텍스처 공존 확률 matrix"는 코드 상 **손으로 적은 4×5 상수표** (`COMPATIBILITY_MATRIX`). 통계 참조 0회. 도출은 3단 수작업 체인:

```
단계 1  행 인덱스   결함 4지표(λ σ ε α) → 손임계값 5개 → subtype
        linearity>0.85 ∧ aspect>5.0 → linear_scratch
        solidity<0.7                → irregular
        aspect<2.0 ∧ solidity>0.9   → compact_blob
        else                        → general

단계 2  열 인덱스   64px 격자 → 손임계값 6개 → bg type 5종
        var<200 ∧ edge<0.02        → smooth
        x_ratio>0.65 ∧ edge>0.02   → vertical_stripe
        y_ratio>0.65 ∧ edge>0.02   → horizontal_stripe
        edge>0.15                  → complex_pattern
        var>500 ∨ edge>0.05        → textured

단계 3  셀 값       "결함이 잘 보이는가" 가시성 논증 → {1.0, 0.8, 0.5, 0.2}
        예) compact_blob: smooth 1.0 / vertical_stripe 0.8 / textured 0.5 / complex 0.2
```

- 배경 선택 = `rng.choices(names, weights=호환점수)` 확률 추출. Poisson blending = `cv2.seamlessClone`(dilation 8px)
- **단위 불일치**: 통계는 결함 centroid가 든 64px 셀 1개로 잡히는데, 적용은 이미지 1장 = 라벨 1개(중앙 256열만 crop). 위치 인자 `roi_x_center`는 계산 후 폐기 → **AROMA가 채운 축이 정확히 여기**
- 셀 값 의미는 "보이면 좋다"(detectability)인데 논문은 "공존 확률"로 서술. 실측 P(bg|subtype)와 방향 반대

### CASDA → AROMA 대조표 (발표 핵심 슬라이드 후보)

| 축 | CASDA | AROMA |
|---|---|---|
| 행 k (결함) | 규칙 subtype 4종, 임계값 5개 손설정 | GMM + BIC 클러스터. 개수·경계 데이터 결정. 잔여 버킷 없음 |
| 열 c (배경) | 규칙 bg type 5종, 임계값 6개 손설정 | 5특징 P33/P66 tertile → 최대 3⁵=243 cell key |
| 셀 값 | 손으로 적은 {1.0, .8, .5, .2} | ctx_prior(k,c) ∝ √(P_def·P_clean), row-max=1 |
| 대칭성 | 비대칭 (결함→배경 일방) | 대칭 (결함측·배경측 기하평균) |
| 배경 단위 | 이미지 1장 = 라벨 1개 → **이미지 선택** | 64px 타일 → **이미지 안 좌표 선택** |
| 위치 | 계산 후 폐기 | ring 히스토그램 매칭으로 s* 확정 |
| 생성 | ControlNet 조건부 생성 | training-free copy-paste (배치 효과 분리) |
| 데이터셋 이식 | 임계값 11개 + 셀 20개 재설계 | 재설계 0 |

AROMA의 "automatic re-derivation"은 matrix **값**만이 아니라 **행·열 인덱스 정의까지** 자동화한 것.

---

## 연구 흐름 한 장 요약 (전체 서사)

```
3장  증강이 효과 있나?        → YOLOv8 + 기본/Diffusion 증강, mAP 평가
4장  얼마나 섞어야 하나?       → 증강 비율 ↑ → mIoU·Dice ↑
6장  어디에 어떤 결함을 넣나?  → CASDA: 수작업 호환성 행렬 + ControlNet (철강 전용)
5장  그 규칙을 자동화하자      → AROMA: 데이터 기반 호환성 게이트 (5개 데이터셋 범용)
```

> 발표 순서는 논문 순(3→4→5→6) 또는 서사 순(3→4→6→5) 중 택일. 서사 순이 CASDA→AROMA 발전 관계 설명에 유리.

---

## 비고 — 학위논문 본문 수식 점검 결과 (2026-10-28)

- 3장: 원논문 식 (1)~(4) 누락 (지표 이름만 언급)
- 5장: ctx_prior 식 본문 탈락(양식적용 과정 소실), CCI 식 누락, 소제목 CCI/MCI 오기
- 6장: 식 (10)·(11) "Smooth/Complex = 14.09/0.607"은 원논문에 없음 (적합도 임계값을 분류 수식으로 오기), λσεα·S_suitability 식 누락
- 전 장: `(식 N)` 본문 참조 0건
