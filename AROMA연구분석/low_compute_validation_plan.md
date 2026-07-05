# AROMA 저비용 검증 전략 — 최종 보고서

## 0. 요약

9개 각도의 제안 16건에 대한 적대적 비판 결과, **생존 6건 + 조건부 생존 3건 + 기각 7건**으로 정리된다. 핵심 판정 기준은 두 가지였다.

1. **순환성**: AROMA 자신의 cluster/cell 라벨 위에서 정의된 지표로 AROMA를 평가하면 "옵티마이저가 자기 목적함수에서 랜덤을 이긴다"는 동어반복이 된다 → exp2 계열 통계 제안 전부가 이 문제로 원안 기각.
2. **채널 절단**: copy-paste 합성에서 aroma/random은 동일 real 결함 crop pool을 쓰므로, crop 단위 분류 프록시(linear probe류)는 AROMA의 작용 변수(결함×배경 pairing)를 측정 전에 잘라낸다 → probe 계열 3건 기각.

또한 중요한 사실 확인: **exp4v2 full run(300ep×3seed)은 아직 수행 전**이다(20260704 run은 epochs=1 스모크, mvtec_leather 합성 미생성). 따라서 "기존 full 결과를 gold로 삼는" 류의 제안은 전제가 무너지며, 전략의 목표는 "exp4v2 재실행 절감"이 아니라 **"exp4v2 1회 실행을 최소 구성으로 설계 + 나머지 입증 부담을 CPU/저GPU 축으로 이관"**이다.

---

## 1. 생존 제안 우선순위 로드맵 (비용 대비 입증력 순)

### 1순위. PRDC — real 결함 manifold 커버리지 정량화 (dist-P1, 수정판)

**무엇을 입증하는가**
AROMA의 핵심 주장("부족한 형태×배경 조합을 타겟팅해 훈련 분포의 빈 곳을 메운다")을 **외부 feature 좌표계**에서 반증 가능한 형태로 검증한다. 예측: 동일 copy-paste 엔진이므로 **Precision/Density는 두 조건 동등**(TOST 동등성 검정), **Recall/Coverage만 aroma가 유의하게 상승**. FID는 fidelity/diversity를 한 스칼라로 뭉개 이 분리 논증이 불가능하므로 P&R/D&C가 필수. exp2의 순환성(자체 라벨)과 n=1 약점을 동시에 해소하는 가장 강한 CPU급 증거.

**필수 수정사항 (비판 반영)**
- 기준 분포는 반드시 **held-out test real 결함 패치** — train 결함을 넣으면 합성이 자기 소스의 복사본이라 자명하게 높아짐
- 조건 간 표본 수 n 엄격 동일화 + k sensitivity(k=3,5,10) 보고
- Wilcoxon-on-bootstrap(pseudo-replication) 금지 → 두 synth 풀을 섞어 재분할하는 **permutation test**로 교체

**Colab 실행**
```python
!pip install prdc
!python $AROMA_SCRIPTS/experiments/exp5_prdc.py \
    --real_data_dir $AROMA_DATA \
    --aroma_synthetic_dir $AROMA_OUT/synthetic \
    --random_synthetic_dir $AROMA_OUT/synthetic_random \
    --dataset_keys severstal mvtec_leather aitex mtd \
    --reference test --nearest_k 3 5 10 \
    --permutation_reps 1000 \
    --output_dir $AROMA_OUT/exp5_prdc
```
exp3_generation_quality.py의 마스크 리졸버·패치 로더를 import/복제해 신설(exp5_prdc.py). feature는 DINOv2 ViT-S/14 또는 InceptionV3, 추출 후 .npy로 Drive 캐시(후속 실험과 공유).

**예상 자원**: feature 추출 T4 기준 데이터셋당 1–3분, permutation은 CPU. 전체 30분 이내.

**exp4v2 대비 한계**: coverage↑ → mAP↑ 인과는 주지 못한다. 이 다리는 아래 5순위(최소 exp4v2)가 놓는다.

---

### 2순위. 임베딩 kNN retrieval coverage — test 결함 커버리지 기제 증거 (proxy-P2, 수정판)

**무엇을 입증하는가**
copy-paste 증강이 검출기를 돕는 기제 = "테스트 분포에 가까운 결함×배경 조합의 밀도 증가"를 지표(1-NN distance, coverage curve)와 1:1 대응으로 측정. real test 각 결함 인스턴스에 대해 학습 풀(real / +random / +aroma)까지의 최근접 cosine distance 분포 비교. 비판에서 **evidence_valid=true**로 생존한 몇 안 되는 제안이며, 1순위와 임베딩을 공유하므로 추가 비용이 거의 0이다.

**필수 수정사항**
- 검정은 인스턴스 단위 Mann-Whitney가 아니라 **이미지 단위 clustered bootstrap**(같은 이미지 내 crop 의존성으로 p 과소평가 방지)
- **diversity 항 병기**(합성 간 pairwise distance) — 중복/노이즈 합성이 coverage를 부풀리는 것 차단
- 논문에 "AROMA 선택은 test를 보지 않는다"를 명문화 — test 기준 평가와 선택 알고리즘의 정보 분리를 선제 기술 (leakage 오해 차단)

**Colab 실행**: 1순위와 동일 세션에서 캐시된 임베딩 로드 후 numpy/sklearn CPU 셀. 결함 bbox crop + context 포함 crop 두 버전.

**예상 자원**: 임베딩 공유 시 CPU only 수 분.

**exp4v2 대비 한계**: 기제(mechanism) 증거이지 downstream 인과가 아니다. 합성 crop이 소스 ROI의 근사 복사라 이 지표는 사실상 "ROI 선택의 좋음"을 외부 좌표계에서 재기술하는 것 — 단독으로는 못 서고 5순위와 패키지로 제시한다.

---

### 3순위. Rare-mode coverage — 독립 클러스터 기반 조합 타겟팅 검증 (dist-P2, 수정판)

**무엇을 입증하는가**
논문의 "부족한 조합 타겟팅" 문구와 가장 직결되는 이산 모드 증거. real 결함 패치의 **독립 임베딩(DINOv2) k-means 클러스터**로 결함 모드를 정의하고, rare 모드(빈도 p25 이하)에 대한 synth의 hit rate를 aroma vs random 30-seed null 분포 대비로 검정. random 재선택은 ROI 메타데이터(roi_selected.json)만으로 CPU 밀리초에 가능해 30-seed 반복이 사실상 공짜 — exp2의 n=1 약점을 **비순환 좌표계에서** 해소한다.

**필수 수정사항**
- **JSD-to-real 항 삭제** — real 빈도 분포에 가까운 것을 성공으로 놓으면 "rare 과대표집"이라는 AROMA 목적과 내부 모순. rare-mode hit rate만 유지 (또는 deficit-보정 목표 분포로 재정의)
- k=8,10,12,15 × clustering seed 5의 **sensitivity 필수 보고**
- **rare 모드가 test 결함에도 존재하는지 필터** 추가 — "train에서 rare"가 test에서 무의미한 모드일 가능성 차단, downstream 관련성 공백 보강

**Colab 실행**: 1·2순위 임베딩 캐시 재사용, exp2_roi_quality.py의 candidates/selected 로딩·_equalize_budget 재사용, random 전략 코드 seed만 바꿔 30회 호출.

**예상 자원**: 임베딩 5–10분(GPU) 공유, 클러스터링·null 분포는 CPU 수 분.

**exp4v2 대비 한계**: 역시 mechanism evidence. mAP 대체 불가.

---

### 4순위. epochs·patience 실측 재산정 — 전 GPU 실험 공통 절감 (cheap4-4, 수정판)

**무엇을 입증하는가**
입증력 손실이 거의 없는 horizontal 절감. 300ep+patience 50은 보수적 상한이며, 실측 수렴점 기반으로 깎으면 이후 모든 run(5·6순위)에 30–50% 절감이 곱해진다. 조기종료 기준이 전 조건 동일하므로 비교 공정성 유지.

**필수 수정사항**
- 기존 full results.csv가 없으므로 **파일럿 필수**: severstal 1셀이 아니라 **"큰 것 1종(severstal) + 작은 것 1종(mtd)" 2셀 × 3조건** — 소규모 데이터셋은 epoch당 스텝이 적어 더 늦게 수렴할 수 있어 외삽 위험
- 상한 = 조건별 best_epoch의 max × 1.3, `--patience 30`
- **사후 게이트**: best_epoch가 새 상한의 90% 이내인지 results.csv에서 확인, **위반 셀은 자동 재실행**을 규칙으로 사전 명시 — synth를 많이 받은 arm이 늦게 수렴해 Δ가 조건-비대칭으로 왜곡되는 리스크 방어

**Colab 실행**: 코드 수정 0. 파일럿 300ep run 후 CPU 셀에서 `_seeds/seed1/{ds}/yolov8n/{cond}/results.csv`의 argmax(mAP50-95) epoch 분석. 이후 `--baseline_epochs`/`--patience`만 조정, epochs 변경 시 fresh output_dir(skip-cache 규칙).

**예상 자원**: 파일럿 GPU 1–3시간(이후 실험의 첫 run으로 겸용하면 순증 0에 가까움).

**한계**: 절감 장치일 뿐 증거를 생산하지 않음.

---

### 5순위. 최소 exp4v2 = label-efficiency 커브 + 혼합 seed 정책 (cheap4-3 + lit-4 병합 + cheap4-2 반전)

**무엇을 입증하는가**
downstream 인과의 마지막 고리. 단일 full-budget 점 비교 대신 **real 25/50/100% × 3조건 커브(severstal + mtd 2종)**로 재구성 — "real이 희소할수록 AROMA-선택 합성이 gap을 더 메운다"는 증강 연구의 표준 논증 축을 얻으면서, 25/50% 지점은 데이터가 적어 epoch당 시간·조기종료가 빨라 **총 GPU가 4종 flat 3seed보다 적다**. 도메인 폭(aitex, mvtec_leather)은 100% 지점 single-point로만 커버해 "커브 2종 + 도메인 폭 4종"으로 역할 분리.

**필수 수정사항 (비판 반영)**
- **주 주장은 Δ(aroma−random) 커브로 한정** — 합성 ROI 소스가 제외된 75% real에서 올 수 있는 leakage는 양 arm에 대칭이라 A-vs-R은 유효하지만, vs-baseline "few-shot 해결" 서사는 부풀려짐. ROI/합성 pool 구성을 논문에 투명 기술하고, 가능하면 25% 지점 1개에서 ROI 소스를 retained real로 제한한 strict 변형을 추가해 leakage 크기를 각주로 실측
- 10% 지점 제외(mtd에서 결함 수 한 자릿수로 split 불안정), AUBC 단일 스칼라 대신 **지점별 Δ + 단조성 검정**
- 지점별 epochs는 4순위 수렴 실측 기반 재산정 (epochs 고정 시 수렴도가 budget과 교락)
- **seed 정책 반전**(cheap4-2 비판의 fix): 큰 severstal은 seed 분산이 작으니 1–2 seed + val bootstrap CI 보조, **작은 데이터셋(mtd/aitex/leather)은 3 seed 유지** — 작은 쪽 3seed는 GPU 시간이 얼마 안 되므로 애초에 절감 대상이 아니다. bootstrap CI는 시각화 보조로만 사용(학습 확률성 분산을 포착하지 못하므로 유의성 주 근거 금지)

**Colab 실행**
```python
# --real_frac 인자 패치 (10–20줄: _get_real 이후 서브샘플, synth cap은 축소된 n_real_train 기준 재계산 — 기존 산식 재사용, 캐시 키에 real_frac 포함)
!python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
    --dataset_keys severstal mtd --condition all \
    --real_frac 0.25 --seeds 1 2 43 \
    --baseline_epochs <실측상한> --patience 30 \
    --output_dir $EXP4V2_OUT/frac025 ...
# 0.5 동일, 1.0 지점은 도메인 폭 run과 겸용
```
사전 조치: 스모크 분석 문서의 mvtec_leather 합성 생성, aitex schema v3 재빌드 병행.

**예상 자원**: GPU 총 **5–8시간** 추정 (커브 2종 × 2신규지점 + 100% 지점 4종 혼합 seed 정책 + epoch 절감 적용). 원래 계획(4종×3조건×300ep×3seed, 9–15시간+) 대비 40–60% 절감하면서 논증 축은 오히려 추가.

**한계**: 커브는 2종에 한정 — 다중도메인 일반성은 100% 지점 single-point + 1~3순위 CPU 증거로 분담. 리뷰어가 aitex/leather 커브를 요구할 수 있으나 aitex는 val 17장으로 판정력이 낮아(스모크 분석 §3.6) 제외 손실이 작다는 근거 제시 가능.

---

### 6순위. 저예산 YOLO 앵커 병행 기록 — 미래 절감 보험 (proxy-P4 + lit-2 재포지셔닝)

**무엇을 입증하는가**
5순위 full run과 **병행하여** 단축 설정(yolov8n·40–60ep·해상도는 결함 최소 크기가 수 픽셀 유지되는 선까지만 축소)을 기록해, "단축 런이 조건 순위를 보존한다"를 부록으로 문서화. 이후 revision에서 리뷰어가 ablation·추가 데이터셋을 요구할 때 proxy 비용으로 대응하는 **보험**. 실패 기준(부호 불일치 시 proxy 기각)을 사전 등록하는 점이 과학적으로 정직.

**필수 수정사항**: 단축 런도 **seed 3개**(런당 5–15분이라 부담 적음 — seed 분산을 모르면 anchor 불일치의 원인 귀속 불가). 판정은 12점 Spearman(CI가 무의미하게 넓음)이 아니라 **사전 등록한 "데이터셋별 Δ(A−R) 부호 일치율"**을 1차 기준으로.

**예상 자원**: GPU 1–2시간. **한계**: 본실험 대체가 아니라 미래 대응 장치 — 현 시점 입증력에 기여하지 않음. 우연 부호 일치(4개면 1/16) 한계도 부록에 명시.

---

### 7순위 (조건부). exp2 MC null — "manipulation check"로 격하 유지 (stats-1·2, 수정판)

원안("AROMA가 낫다의 증거")은 순환성으로 기각됐지만, **폐기가 아니라 격하**한다: "선택 알고리즘이 의도한 목적함수를 실제로 최적화한다"는 **구현 건전성 확인(manipulation check)**으로 명시 보고하면 논문에 여전히 자리가 있다.
- z-score·Stouffer 결합 금지(정규성 위반 + p 하한 클리핑), **empirical p=(r+1)/(K+1) + null 95% interval만** 보고
- 같은 MC null 프레임을 **1~3순위의 외부 임베딩 지표에 적용하면 순환성이 해소**되어 "4/4 도메인 방향 일치 + 도메인별 empirical p" 수준의 겸손한 일반화 문장 생성 가능 (지표 간 상관 명시, K=10⁴)
- **자원**: CPU 수 분~수십 분.

**budget sweep(stats-3)**은 다음 조건을 모두 충족할 때만 부활: (a) AROMA 선택 파이프라인을 budget n으로 **실제 재실행**(서브샘플 희석은 straw-man), (b) 포화 이전 구간(5–25%)으로 제한, (c) 외부 임베딩 지표 사용. severstal budget 불일치(1690 vs 200) 논란 해소용으로 가치 있음. CPU only.

---

## 2. 기각 제안 및 사유

| 제안 | 기각 사유 (한 줄) |
|---|---|
| exp3 FID/KID bootstrap (stats-4) | 가짜 pairing(다른 이미지 집합 간 인덱스 공유) + full-image 지표가 배경에 지배되어 선택 신호 소실 → 결함 패치 단위로 고치면 PRDC와 수렴하므로 **1순위에 흡수** |
| DINOv2 patch linear probe (proxy-P1) | 합성 결함이 real crop의 픽셀 복사본이라 frozen 공간에서 3조건 포화 예상 + post-hoc bbox 라벨 노이즈 + anchor 검증 부재 |
| PatchCore/PaDiM defect bank (proxy-P3) | score = d(normal)−d(defect)는 즉석 고안 지표(표준 프로토콜 아님) + 수학적으로 2순위 kNN coverage의 재포장이라 독립 증거 아님 |
| MMD + linear probe (dist-P3) | ΔMMD 유의성은 two-sample permutation이 검정하지 않는 대상 + test 근접을 목표로 삼는 순간 leakage 반박 개방 + probe는 배경 맥락 절단 |
| 2단계 스크리닝→선택적 확증 (cheap4-1) | 스크리닝 통과 셀만 확증 = winner's curse/selection bias + aitex 판정불가로 절감률 붕괴 → 파일럿 겸용으로 **4순위에 흡수** |
| seed→bootstrap 전면 대체 (cheap4-2) | 지배 분산원(학습 확률성)을 bootstrap이 못 잡음 + per-image AP 정의 불가·aitex val 17장 검정력 0 → 반전된 형태로 **5순위에 흡수** |
| Linear probe 대체 다운스트림 (lit-1) | AROMA의 작용 채널(결함×배경 pairing)을 crop이 절단 — 양방향으로 정보량 없는 측정 |
| Low-fidelity rank-correlation (lit-2) | gold(full 결과)가 아직 없어 전제 붕괴 + 12점 τ는 CI 무의미 → 보험으로 **6순위에 흡수** |
| PatchCore few-shot (lit-3) | 합성 데이터가 모델 파라미터에 들어가지 않는 평가로는 "학습 가치" 주장 불가 + threshold 1스칼라 병목 |

---

## 3. 최종 "논문 증거 스택" 구성과 리뷰어 방어 논리

```
[L1] Intrinsic 건전성        exp2 + MC null (manipulation check로 명시)          CPU 수 분
[L2] 외부 좌표계 커버리지     PRDC (Recall/Coverage↑, Precision/Density 동등 TOST)  GPU 30분
                             + kNN test-coverage (기제) + rare-mode hit rate     (임베딩 공유)
[L3] Downstream 인과         real_frac 커브 (severstal+mtd, Δ(A−R) 주 주장)       GPU 5–8h
                             + 도메인 폭 4종 100% 지점 (혼합 seed 정책)
[L4] 보험/부록               단축 YOLO anchor 부호 일치 기록                        GPU 1–2h
```

**예상 리뷰어 공격과 방어선:**

- *"exp2 지표는 자기 목적함수 아닌가?"* → 맞다고 인정하고 L1을 manipulation check로 명시 포지셔닝. 실질 주장은 L2의 **독립 임베딩 좌표계**에서 성립하며, AROMA는 DINOv2 feature를 선택에 사용하지 않으므로 순환 없음.
- *"coverage가 좋다고 검출이 좋아지나?"* → L2 단독으로 주장하지 않는다. L3가 동일 태스크(YOLO mAP50) direct evidence를 제공하고, L2는 **왜** 좋아지는지의 기제 설명으로 기능. 지표 예측("Recall만 상승, Precision 동등")이 사전 등록된 반증 가능 가설이라는 점이 사후 합리화 반박을 차단.
- *"n=1/seed 부족 아닌가?"* → L2는 permutation/clustered bootstrap/30-seed null로 인스턴스·선택 수준 통계를 확보. L3는 분산이 큰 소규모 데이터셋에 3 seed를 유지하고, 분산이 작은 severstal만 축소 — "절감을 분산이 작은 곳에서만 했다"는 원칙으로 방어.
- *"왜 커브는 2종뿐인가?"* → 도메인 폭은 4종 100% 지점 + L2의 4종 전체 커버리지 증거로 분담. aitex는 val 17장으로 어느 설계에서도 판정력이 낮음을 데이터로 제시.
- *"test 정보 leakage?"* → "AROMA 선택은 test를 보지 않는다" + real_frac 실험의 ROI 소스 구성 + strict 변형 각주를 논문에 명문화.
- *"ablation이 부족하다"* (revision 단계) → L4 anchor가 사전 등록된 부호-일치 기준을 통과했다면 추가 ablation을 proxy 비용으로 수행.

**총 GPU 예산**: 약 **7–11시간** (파일럿 겸용 시 하한). 원래 exp4v2 full 3seed 단독(9–15시간+, ablation 별도) 대비 절감하면서, 증거의 층은 1층(mAP 점 비교)에서 4층으로 늘어난다.

**실행 순서 권고**: L1·L2(CPU/T4, 즉시 착수, mvtec_leather 합성 생성과 병행) → 4순위 파일럿(epoch 상한 확정) → L3 본실험(단축 anchor 병행 기록) → L4 부록 정리.