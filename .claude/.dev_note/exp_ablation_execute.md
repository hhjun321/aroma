# exp_ablation_execute — 3단계 leave-one-out Ablation Study 실행 문서

## (사용할 skills: 없음 — 실험 실행 문서. 코드 수정 없음)

## (성격: 리뷰어 대응 예비 실험 — **로컬 proxy 단계 검증 완료 (2026-08-18, severstal).** mAP 단계는 Colab GPU 잔여)

### 검증 결과 (2026-08-18 로컬 CPU, severstal)

- [x] **smoke (top_k=20, arm당 40장)**: 3 arm 분리·전 단계 지표 산출 성공, 전체 ~5분
- [x] **k200 (top_k=200, arm당 400장)**: smoke 결론 유지, p-value 강화, 전체 ~35분 (합성 3 arm 병렬)
- [x] **sanity**: ring arm의 저장 `site_score` 완전 재현 (평균 오차 2.5×10⁻⁷) — 분석 스크립트가 파이프라인 스코어링과 수치 동일
- [x] **A1 (random ROI + AROMA bg/site) arm 완성 (2026-08-19)**: clean_bg ring+qf 200/200 (ring 3139 positions, fallback 1.9%) → 합성 400장 (`position_source: ring=397/400`, used=400 fallback=0 mismatch=0) → site_score **0.120** (n=398, full 0.131 근접) — site 메커니즘이 ROI 선정과 독립적으로 작동함을 확인. A1의 변별 지표는 Stage 1 (ctx_prior 0.013 vs 0.183)
- [x] **Colab mAP 실행 가이드 작성 (2026-08-19)**: `AROMA연구분석/colab_execute_new/exp_ablation_mAP_execute.md` — arm 3종 Colab 재생성(remap 불필요) → exp4v2 `--condition aroma` 단독 + `--aroma_synthetic_dir` 교체 + arm별 output_dir × 3 seeds → 취합표
- [x] **Colab 1차 합성 실행 (2026-08-19)**: A2 2000장·A1 400장 생성 — **parity 결함 발견**: Drive sym_final은 1000 ROI 스케일인데 가이드 STEP 1이 로컬값 top_k=200을 사용 → A1만 400장. 가이드 수정 완료(top_k를 기존 roi_selected 수에 동적 일치 + STEP 4 진입 전 parity 검수 셀). ~~A1은 top_k=1000으로 재선정·재합성 필요~~ → **A1 재실행 완료 (07:30 로그)**: 1000 ROI → 2000장, `position_source: ring=1949 fallback=51/2000` (2.6%, 정상 규모), `used=2000 fallback=0 mismatch=0`. ~~4 arm parity 2000장 달성~~ **→ parity 결함 2호 발견 (parity 검수 셀 실측: ablation 3 arm=2000 vs full=3000)**: full run은 `n_per_roi=3`(step5 NREP=3, 1000×3=3000)인데 가이드 STEP 0이 로컬값 `N_PER_ROI=2` 사용. exp4v2 cap@1.0=2534라 full은 2534 소비·ablation은 2000 소비 → 훈련 수량 confound. **처방: 3 arm 전부 `--n_per_roi 3`으로 STEP 3 셀 재실행** (STEP 1·2 재실행 불필요. copy_paste는 캐시 없이 전량 재생성하나 파일명 superset(`_00`·`_01` 덮어씀+`_02` 추가)+annotations 전체 재작성이라 디렉터리 비우기 불필요 — sidecar 캐시는 ControlNet 전용임을 코드로 확인). 가이드 수정 완료(N_PER_ROI=3, T=3000, parity 셀 주석). **→ 3-rep 재합성 완료 (2026-08-19 Colab, 07:45–09:55)**: A1 3000 (`ring=2919 fallback=81`, 2.7% — ring 풀 2919 ≥ cap 2534라 exp4v2 ring-우선 fallback=0 성립), A2 3000 (`fallback=3000`, exhausted 42/3000=1.4% — 동일 14개 ROI × 3 rep, gate 결정론 정합), A3 3000 (`fallback=3000`, repick 1). 3 arm 전부 `used=3000 fallback=0 mismatch=0`. **잔여: parity 셀(4 arm=3000 확인) → STEP 4 GPU.** A2는 2000장 정상(재사용 가능), **A3도 2000장 완료 확인** (06:57 완료 로그: `Generated 2000 (0 skipped)`, `position_source: fallback=2000/2000`, `used=2000 fallback=0 mismatch=0`, placement repick=0 — fitness bg 사전배정이라 compat 재추첨 무발생, A2 exhausted 28건과 대조로 게이트 작동 입증)
- [x] **downstream mAP 1차 실행 (2026-08-19/20 Colab GPU, 3 arm × 3 seed)** — 결과 로컬 업로드(`D:/project/aroma_dataset/ablation/exp4v2_{arm}/seed{42,1,2}/`). ⚠️ **프로토콜 결함 3호**: 가이드 STEP 4 커맨드에 그룹 A 하이퍼파라미터가 없어 **exp4v2 기본값으로 학습됨** (args.yaml 실측: epochs 50·batch 16·imgsz 256·patience 0·rect off·synth cap 미적용 3000장 vs 정본 100·128·640·25·rect·cap 2534). 기존 Table 8(full .5197/baseline .5033)과 비교 무효, full−Ax 절대 분해 불가. **arm 간 상대 비교는 유효** (3 arm 동일 조건·동일 seed) — 분석은 아래 "downstream 상대 결과" 절
- [x] **그룹 A 프로토콜 재실행 완료 (2026-08-21)** — 가이드 STEP 4 개선(그룹 A 플래그 내장·`--seeds 42 1 2` 단일 호출·`exp4v2_ga_{tag}` 분리·args.yaml 검증 셀·해석 가정 5개 명문화) 후 9 run 실행. 프로토콜 검증 ALL OK(epochs 100·imgsz 640·batch 128·patience 25·rect·cap 2534). 결과 로컬: `.claude/.etc/ablation/20260821/exp4v2_ga_*/`. **분석은 아래 "그룹 A downstream 결과" 절 — 기본값 판 순위가 반전됨(프로토콜 아티팩트 확정)**
- [ ] 타 데이터셋 확장 (mtd 우선 — D1 이질성-비례 가설 검증, kolektor 2순위) / D2 site-score 층화 arm (별도 결정)
- [ ] 타 데이터셋 확장 (aitex / mvtec_leather 우선)

---

## 개요

리뷰어가 "AROMA 3단계 파이프라인(§3.2.4)의 각 단계 기여를 분리하라(ablation)"고 요구할 경우의 대응 실험. **기존 CLI 플래그만으로 3개 leave-one-out arm이 분리되므로 코드 수정이 불필요**하다는 것이 핵심 발견이다.

| arm | Stage 1 (ROI) | Stage 2 (BG) | Stage 3 (Site) | 구현 방법 |
|---|---|---|---|---|
| full AROMA | compat | fitness | ring | 기본 (step3→3.5→5 체인) |
| A1: ROI-random | **random** | fitness | ring | `roi_selection --sampling_strategy random` |
| A2: BG-random | compat | **random** | (position 無→랜덤) | `clean_bg_selection --emit_random_arm` 산출물을 `--clean_bg_json`으로 소비 |
| A3: Site-random | compat | fitness | **random** | `clean_bg_selection --site_selection off` → generate가 `_random_paste_position` 폴백 |
| all-random | random | random | random | 기존 Random arm (§4 기측정) |

§4.1은 Stage 1(coverage 통계)·Stage 2(hist∩ 통계)의 mechanism 수준 비교만 있고, **단계별 downstream 분해는 부재** — 이 문서가 그 갭의 실행 절차다.

---

## 로컬 실측 결과 (severstal, k200 = top_k 200, arm당 400장)

### Stage 1 — ROI selection (roi_selected.json 비교, n=200)

| | ctx_prior | roi_score | clusters | cells | pairs |
|---|---|---|---|---|---|
| AROMA(compat) | **0.183** | 0.168 | 2 | 2 | 2 |
| random | 0.013 | 0.092 | 5 | 78 | 151 |

- AROMA 200개가 (cluster×cell) 2개 pair에 집중 — severstal 근균일 표면의 지배 cell 구조(Figure 3.2.4-2 서술) 그대로. §4.1 coverage↔compatibility 트레이드 재현.

### Stage 2 — BG assignment (Fig 4.1-3 지표: 배정 배경 vs pooled real defect bg hist∩, n=200)

| | hist∩ mean | distinct bg |
|---|---|---|
| AROMA bg | **0.411** | 96/200 |
| random bg | 0.248 | 196/200 |

- Δ=+0.163, Mann-Whitney(one-sided) **p=6.7×10⁻⁵³** (smoke Δ+0.171과 일치)
- 부수 관찰: AROMA distinct bg 96/200 — Leather(15/400)만큼 극단은 아니나 절반 collapse. §4.3 diversity-loss 메커니즘이 severstal에도 약하게 존재

### Stage 3 — Site resolution (합성 annotations의 실제 배치 위치에서 ∩(h_s, tgt[k]) 재채점)

| arm | site_score mean | footprint void | vs ring p |
|---|---|---|---|
| ring (full) | **0.131** (n=394) | 5/400 | — |
| A1 roi-random | 0.120 (n=398) | 3/400 | (site 유지 arm — 변별은 Stage 1 지표) |
| site-off (A3) | 0.081 (n=391) | 66/400 | 1.1×10⁻⁶⁰ |
| bg-random (A2) | 0.061 (n=373) | 110/400 | 3.2×10⁻⁸⁶ |

- **누적 분해**: 0.131 (full) → 0.081 (−site) → 0.061 (−site−bg). site-off > bg-random (Δ+0.019, p=8.7×10⁻¹³) — Stage 2와 Stage 3 기여가 **독립적·누적적**으로 분리됨
- **void 노출**: 랜덤 위치는 66~110/400건이 void 타일 위 — §3.2.6 void 배제가 site resolution에 내장된 효과의 직접 수치
- generate 텔레메트리: ring arm `position_source: ring=391 fallback=9` (clean_bg의 ring-무효 ROI 폴백과 정합), 3 arm 전부 `used=400 fallback=0 mismatch=0`

---

## 그룹 A downstream 결과 (severstal, 2026-08-21 Colab GPU — **정본**)

> 프로토콜 검증 ALL OK: 9 run 전부 epochs 100 · imgsz 640 · batch 128 · patience 25 · rect · cap 2534 (args.yaml assert 통과) — Table 8 병기 성립. 결과: `.claude/.etc/ablation/20260821/exp4v2_ga_{a1_roirand,a2_bgrand,a3_siteoff}/exp4v2_results.json`

### 수치 (mAP@0.5, 기존 Table 8 병기)

| arm | per-seed (42/1/2) | mean ± std | full 대비 | 부호 일치 |
|---|---|---|---|---|
| **full AROMA** (A+A+A) | .5219 / .5337 / .5035 | **.5197 ± .0152** | — | — |
| all-random (R+R+R) | .5141 / .5185 / .4870 | .5065 ± .0171 | −1.3pt | — |
| baseline (real-only) | .5285 / .4939 / .4875 | .5033 ± .0221 | −1.6pt | — |
| A2 BG-random (A+R+R연쇄) | .4808 / .5340 / .4843 | .4997 ± .0298 | −2.0pt | 2/3 (seed1 동률) |
| A3 Site-random (A+A+R) | .5035 / .5278 / .4661 | .4991 ± .0311 | −2.1pt | 3/3 |
| **A1 ROI-random (R+A+A)** | .4788 / .4919 / .4743 | **.4817 ± .0091** | **−3.8pt** | **3/3 (최소 격차 +2.9pt)** |

### 발견 4개

1. **기본값 판 순위는 프로토콜 아티팩트로 확정.** 기본값(256px) A1>A3>A2 → 그룹 A(640px) **A2≈A3>A1 완전 반전**. 기본값 판의 "배경 단독 +2.1pt"(A3−A2)는 그룹 A에서 −0.06pt ≈ 0으로 소멸. 256px가 severstal 1600×256 스트립을 뭉개 위치/crop 신호를 왜곡한다는 감사 경고가 실측으로 입증됨 — **기본값 판 수치는 어떤 서술에도 인용 금지** (프로토콜 민감성 사례로만).
2. **full이 전 arm 위 — leave-one-out 전부 full 아래.** full−A1 +3.8pt (per-seed +.0431/+.0418/+.0292 — 3/3, 최소 +2.9pt로 이번엔 견고), full−A3 +2.1pt (3/3: +.0184/+.0059/+.0374), full−A2 +2.0pt (2/3, seed1 −.0003 동률). **crop 선택 제거가 최대 손상** — 기본값 판 결론의 정반대.
3. **가산 모형 기각 — 단계는 상보적 체인.** 가산이면 all-random이 synth arm 최저여야 하나 실제 full 다음 2위(.5065). 특히 **A1 < all-random이 3/3 견고**(−.0353/−.0266/−.0127): "random crop + AROMA 배치" < "전부 random". 해석: AROMA 배치는 배경을 소수 집중시키는데(로컬 실측 distinct bg 96/200 — §4.3 diversity-loss 메커니즘) 그 집중이 compat crop과 정합할 때만 이득, random crop과 결합하면 다양성 손실만 잔존. A2≈A3 = 위치가 random이 되는 순간 배경 fitness 가치도 소멸 — 배경 선택의 가치가 ring 채점을 경유해 실현됨(상호작용).
4. **부분 적용 arm은 baseline(.5033)마저 하회** — 비정합 합성은 없느니만 못함. full만 baseline +1.6pt.

### 서술 가능 주장 (논리 감사 통과분)

- **핵심**: "3단계는 통합 체인으로만 가치를 낸다 — 어느 하나를 빼도 full 대비 2.0~3.8pt 하락, 부분 적용은 all-random·baseline보다도 낮다." leave-one-out이 개별 기여 분해가 아니라 **설계 정합성(coherence) 입증** — R1-3·R3-3 "왜 3단계인가"에 직접 대응.
- full−A1 +3.8pt: 3/3 seed·최소 +2.9pt — 방향성 견고 (가장 강한 단일 수치).
- full−A2/A3 +2.0~2.1pt: 2~3/3 — 방향성 수준으로만.
- **금지**: "유의" 단어(n=3, sign test 최소 p=0.125), c1/c2/c3 개별 가산 분해(가산 모형 기각됨), A2·A3 std(~.03) 간 비교. seed1이 전 arm 공통 고점(seed 상관) — 보고는 paired 차분으로.

---

## (구판 — 인용 금지) downstream 상대 결과 (severstal, 2026-08-19/20 Colab GPU — ⚠️ exp4v2 기본값 프로토콜)

> **2026-08-21 그룹 A 재실행으로 대체됨.** 아래 순위(A1>A3>A2)는 그룹 A에서 반전 — 프로토콜 아티팩트 확정. 프로토콜 민감성 기록 목적으로만 보존.

> **프로토콜 주의 (결함 3호)**: 가이드 STEP 4 커맨드에 그룹 A 플래그 미포함 → 기본값 학습 (epochs 50 · batch 16 · imgsz 256 · patience 0 · rect off · synth 3000 cap 미적용). severstal 1600×256 스트립을 rect 없이 256²로 리사이즈한 영향이 절대 수치 하락(~0.30 vs Table 8 ~0.52)의 지배 요인 추정. **Table 8과 비교 금지, full−Ax 절대 분해 불가** (이 프로토콜의 full arm 부재). **arm 간 상대 비교는 유효** — 3 arm 동일 조건(real 2534 + synth 3000, 동일 seed 42/1/2). 결과 위치: `D:/project/aroma_dataset/ablation/exp4v2_{a1_roirand,a2_bgrand,a3_siteoff}/seed{42,1,2}/exp4v2_results.json`

### 수치 (mAP@0.5, YOLOv8n)

| arm (제거 단계) | per-seed (42/1/2) | mean ± std | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| A1 ROI-random ([1] crop 선택 제거) | .3007 / .3026 / .3150 | **.3061 ± .0078** | .1224 | **.4960** | .3217 |
| A3 Site-random ([3] 위치 선택 제거) | .2970 / .3008 / .3122 | .3033 ± .0079 | .1225 | .4472 | **.3358** |
| A2 BG-random ([2] 배경 선택 제거 + [3] 연쇄 붕괴) | .2845 / .3001 / .2631 | **.2826 ± .0186** | .1122 | .4411 | .3149 |

### 단계별 판독

**A1 (crop 선택 제거) — 손실 최소.** random crop이라도 [2][3] 체인이 살아 있으면 crop 기준으로 배경을 재선발·재배치해 궁합 결손을 하류에서 복구 (`ring=2919/3000` — site 체인이 random crop에도 정상 작동). copy-paste에서 crop은 어차피 real 결함이라 클래스 신호는 진짜 — H1(재조합 무정보)과 정합. precision 최고(.496) = 그럴듯한 자리에만 붙어 FP 유발 표본이 적음.

**A2 (배경 선택 제거) — 최저, 분산도 최대.** ⚠️ 순수 [2] 제거가 아님: random 배경엔 ring 정보가 없어 [3]도 연쇄 붕괴 (`fallback=3000/3000`) — A2 = [2]+[3] 동시 제거. placement gate exhausted 42/3000 = random 배경이 compat τ=0.1381 미달하는 조합 실재 (AROMA 배경 arm에선 0건). 엉뚱한 배경 위 결함 = test 분포에 없는 (배경,결함) 조합을 학습 → 성능·재현성(±.0186, 타 arm 2.4배) 동시 훼손. 로컬 proxy와 정합 (site_score 0.061 최악, void 110/400).

**A3 (위치 선택 제거) — 소폭 손실.** 배경 자체는 fitness 상위라 대부분 위치가 "덜 최적이지만 치명적이지 않음" (severstal 근균일 표면: 이미지 내 위치 간 차이 < 이미지 간 차이). recall 최고(.336)·precision −4.9pt 패턴이 메커니즘 노출: 무작위 위치가 출현 문맥을 넓혀 recall↑, void 경계·비정형 자리 표본이 FP 성향 키워 precision↓, mAP 순손실.

### 기여 서열 (A1 ≈ 준-full 가정 — [1] 기여 ≈ 0이므로)

| 기여 추정 | 계산 | 크기 | 부호 일치 |
|---|---|---|---|
| [2] 배경 선택 단독 | A3 − A2 | **≈ +2.1pt** | 3/3 seed |
| [3] 위치 선택 단독 | A1 − A3 | ≈ +0.3pt (노이즈 경계) | 3/3 seed |
| [2]+[3] 합산 | A1 − A2 | ≈ +2.4pt (단독 합과 정합 — 가산적) | 3/3 seed |
| [1] crop 선택 | full−A1 필요 — 측정 불가 | ? | — |

**핵심 결론: "어디에 붙이느냐(배경 ≫ 위치)가 무엇을 붙이느냐(crop)보다 결정적."** 로컬 proxy 누적 분해(site_score 0.131→0.081→0.061, 독립·가산)와 downstream 서열 완전 일치 — 메커니즘(proxy)→성능(mAP) 방향 일관성 확보.

### 주장 가능 범위 (정직성 원칙)

- **가능**: leave-one-out 상대 순위에서 BG assignment 최대 기여(부호 3/3), proxy 분해와 방향 일치 — 리뷰어 대응(R1-3/R3-3) 보조 증거. 단 R3-3(ROI 스코어링 0.6+0.4 입증) 관점에선 이 결과는 placement 체인 기여 입증에 강하고 Stage 1 crop 선택의 severstal downstream 기여는 약하다는 정직한 한계 포함.
- **불가**: full 대비 절대 기여량, Table 8 병기, "AROMA가 X pt 기여" 정량 주장. A1↔A3 0.3pt는 방향성 이상 금지 (n=3).
- 정본 수치가 필요하면 그룹 A 프로토콜 재실행 (9 GPU run + 가이드 STEP 4 플래그 삽입 선행).

---

## 실행 절차 (로컬 CPU — proxy 단계)

전 단계 CPU. 산출 루트: `D:/project/AROMA_DATASET/ablation_k200/severstal/` (smoke는 `ablation_smoke/severstal/`).

### 0. 사전 조건 (전부 로컬 확인됨)

| 입력 | 경로 |
|---|---|
| profiling (matrix_symmetric 포함) | `D:/project/AROMA_DATASET/profiling/profiling/severstal/` |
| complexity | `D:/project/AROMA_DATASET/complexity/severstal/` |
| good 이미지 (5,902장) | `D:/project/AROMA_DATASET/severstal/train/good/` |
| defect 이미지·GT 마스크 | `D:/project/AROMA_DATASET/severstal/test/class{1..4}/`, `severstal/masks/{stem}.png` |
| τ (step4c 확정값) | `D:/project/AROMA_DATASET/compat_gate/severstal/compat_tau_prescan_severstal.json` → **ds_tau=0.1381** |

### 1. prompts (1.5초) — roi_selection이 파일 존재를 강제 (내용은 메타데이터만)

```
python scripts/aroma/prompt_generation.py \
  --profiling_dir D:/project/AROMA_DATASET/profiling/profiling/severstal \
  --complexity_dir D:/project/AROMA_DATASET/complexity/severstal \
  --output_dir <SM>/prompts
```

### 2. Stage 1 두 arm (~14초/arm)

```
python scripts/aroma/roi_selection.py --profiling_dir <PROF> --prompts_dir <SM>/prompts \
  --output_dir <SM>/roi_aroma --sampling_strategy compatibility --top_k 200 --seed 42 \
  --img_diversity_cap 1 --class_mode multi --class_floor --min_quality 0.0

python scripts/aroma/roi_selection.py --profiling_dir <PROF> --prompts_dir <SM>/prompts \
  --output_dir <SM>/roi_random --sampling_strategy random --top_k 200 --seed 42
```

### 3. ⚠️ 경로 remap (Colab산 profiling 필수 절차)

`morphology_features.csv`가 defect `image_path`를 `/content/drive/MyDrive/data/Aroma/...`로 기록 → 미처리 시 **합성 전량 skip** (`Defect image not found`). `roi_selected.json`(양 arm)에서:

1. `image_path`: `/content/drive/MyDrive/data/Aroma/severstal` → `D:/project/AROMA_DATASET/severstal` 치환
2. `defect_mask_path`: `.../defect_masks/{class}_{stem}.png` → `D:/project/AROMA_DATASET/severstal/masks/{stem}.png` (class prefix 제거 stem 매핑 — k200 실측 200/200 hit)

원본 profiling·파이프라인 코드는 무수정. remap 스니펫은 세션 로그(2026-08-18) 참조.

### 4. Stage 2+3 (ring+qf ~2분 / off ~40초)

```
# full AROMA + A2 대조군 동시 방출
python scripts/aroma/clean_bg_selection.py --profiling_dir <PROF> --roi_dir <SM>/roi_aroma \
  --output_dir <SM>/roi_aroma --k_fit --site_selection ring --emit_random_arm \
  --site_quality_filter --image_dir D:/project/AROMA_DATASET/severstal/train/good

# A3: 같은 bg 배정, position=None 방출
python scripts/aroma/clean_bg_selection.py --profiling_dir <PROF> --roi_dir <SM>/roi_aroma \
  --output_dir <SM>/cbg_siteoff --k_fit --site_selection off --emit_random_arm
```

### 5. 합성 3 arm (n_per_roi=2 → 400장/arm, 2~15분/arm, 병렬 가능)

공통 플래그: `--method copy_paste --n_per_roi 2 --seed 42 --blend_mode seamless --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 --compat_mode symmetric --compat_threshold 0.1381 --compat_matrix_json <PROF>/compatibility_matrix.json`

```
# full:      --roi_dir <SM>/roi_aroma                                             --output_dir <SM>/synth_ring
# A3(site):  --roi_dir <SM>/roi_aroma --clean_bg_json <SM>/cbg_siteoff/clean_bg_selected.json --output_dir <SM>/synth_siteoff
# A2(bg):    --roi_dir <SM>/roi_aroma --clean_bg_json <SM>/roi_aroma/clean_bg_random_arm.json  --output_dir <SM>/synth_bgrand
```

검수 포인트: `clean_bg resolve: used=T fallback=0 mismatch=0`, ring arm `position_source: ring≈T`, A2/A3 `fallback=T`.

### 6. proxy 지표 분석

```
python D:/project/AROMA_DATASET/ablation_smoke/severstal/analyze_smoke_arms.py <SM>
```

- `clean_bg_selection.py` 내부 함수(`_tile_grid`/`_ring_keys`/`_target_by_cluster`/`_hist_intersection`)를 그대로 import — 스코어링 재구현 없음
- 산출: `<SM>/proxy_metrics.json` (stage1 coverage/ctx_prior, stage2 pooled hist∩, stage3 site_score 재채점)
- ⚠️ 조인 규약 2건: annotations `bbox`는 **[x,y,w,h]** (x1y1 아님) / good `image_id`는 `_{stem}` prefix
- 유의성: scipy `mannwhitneyu` one-sided (세션 로그의 인라인 블록 참조)

---

## Colab mAP 연결 (잔여 — 리뷰어가 downstream 수치까지 요구할 때)

1. 위 arm별 합성 산출물을 exp4v2 입력 규약(`composed_to_exp4v2.py` 경유)으로 변환
2. arm당 YOLOv8n 3-seed (exp4v2 프로토콜 동일 — severstal batch128 유지, **소형셋 확장 시 kolektor/leather는 batch16+patience0** [[project_exp4v2_batch128_collapse]])
3. 비교표: Baseline / all-random / A1 / A2 / A3 / full — §4.3 Table 8에 열 추가 형태
4. **우선순위**: A2를 mvtec_leather에서 — §4.3의 diversity-loss 진단(15/400 collapse)을 직접 검증하는 가장 강한 서사

비용: arm 3개 추가 × 3 seeds = 9 YOLO run/데이터셋. severstal 단독이면 Colab Pro 세션 1~2회 규모.

---

## 산출물 위치

- `D:/project/AROMA_DATASET/ablation_smoke/severstal/` — smoke (n=20) + `analyze_smoke_arms.py` + `proxy_metrics.json`
- `D:/project/AROMA_DATASET/ablation_k200/severstal/` — k200 (n=200) + `proxy_metrics.json`

## 미확정 사항

- ~~TODO: 논문 반영 위치~~ → **결정·작성 완료 (2026-08-21)**: `Article/text/section4_4.txt` 신설 (§4.4 Ablation Study, Table 11). 그룹 A 정본 수치만 사용, 상보적 체인 서사, n=3 한계 명기("directional rather than inferential"), A2 연쇄 제거 명시. 잔여 연결 작업: §4 도입부(section4_1.txt 첫 문단)의 섹션 안내에 4.4 언급 추가 여부, §5 Discussion 반영 여부 — 별도 결정
- TODO: aitex(신호 강함)·mvtec_leather(A2 서사) 확장 시 remap 규칙 데이터셋별 재확인 (마스크 파일명 규약 상이 가능)
- <결정 필요> proxy 지표를 논문에 실을 경우 §4.1과의 수치 차이 설명 필요 — smoke/k200의 Δ(+0.163)는 top-K 집중 선택분이라 §4.1 전체분포 Δ(+0.032)보다 큼. 동일 지표·다른 모집단임을 명기할 것
