# exp_ablation_execute — 3단계 leave-one-out Ablation Study 실행 문서

## (사용할 skills: 없음 — 실험 실행 문서. 코드 수정 없음)

## (성격: 리뷰어 대응 예비 실험 — **로컬 proxy 단계 검증 완료 (2026-08-18, severstal).** mAP 단계는 Colab GPU 잔여)

### 검증 결과 (2026-08-18 로컬 CPU, severstal)

- [x] **smoke (top_k=20, arm당 40장)**: 3 arm 분리·전 단계 지표 산출 성공, 전체 ~5분
- [x] **k200 (top_k=200, arm당 400장)**: smoke 결론 유지, p-value 강화, 전체 ~35분 (합성 3 arm 병렬)
- [x] **sanity**: ring arm의 저장 `site_score` 완전 재현 (평균 오차 2.5×10⁻⁷) — 분석 스크립트가 파이프라인 스코어링과 수치 동일
- [x] **A1 (random ROI + AROMA bg/site) arm 완성 (2026-08-19)**: clean_bg ring+qf 200/200 (ring 3139 positions, fallback 1.9%) → 합성 400장 (`position_source: ring=397/400`, used=400 fallback=0 mismatch=0) → site_score **0.120** (n=398, full 0.131 근접) — site 메커니즘이 ROI 선정과 독립적으로 작동함을 확인. A1의 변별 지표는 Stage 1 (ctx_prior 0.013 vs 0.183)
- [x] **Colab mAP 실행 가이드 작성 (2026-08-19)**: `AROMA연구분석/colab_execute_new/exp_ablation_mAP_execute.md` — arm 3종 Colab 재생성(remap 불필요) → exp4v2 `--condition aroma` 단독 + `--aroma_synthetic_dir` 교체 + arm별 output_dir × 3 seeds → 취합표
- [x] **Colab 1차 합성 실행 (2026-08-19)**: A2 2000장·A1 400장 생성 — **parity 결함 발견**: Drive sym_final은 1000 ROI 스케일인데 가이드 STEP 1이 로컬값 top_k=200을 사용 → A1만 400장. 가이드 수정 완료(top_k를 기존 roi_selected 수에 동적 일치 + STEP 4 진입 전 parity 검수 셀). **A1은 top_k=1000으로 재선정·재합성 필요** (STEP 1→2(a)→3 a1만 재실행, 기존 synth_a1_roirand 디렉터리 비우고). A2는 2000장 정상(재사용 가능), A3는 1000-ROI 소비라 2000장 예상 — 완료 로그로 확인할 것
- [ ] downstream mAP (exp4v2 YOLO 3-seed) — Colab GPU에서 위 가이드 실행. **proxy 결과만으로 mAP 주장 금지** (기존 정직성 원칙)
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

- TODO: 논문 반영 위치 — §4에 ablation 소절 신설 vs 리뷰어 응답문서 전용 (리뷰 코멘트 수신 후 결정)
- TODO: aitex(신호 강함)·mvtec_leather(A2 서사) 확장 시 remap 규칙 데이터셋별 재확인 (마스크 파일명 규약 상이 가능)
- <결정 필요> proxy 지표를 논문에 실을 경우 §4.1과의 수치 차이 설명 필요 — smoke/k200의 Δ(+0.163)는 top-K 집중 선택분이라 §4.1 전체분포 Δ(+0.032)보다 큼. 동일 지표·다른 모집단임을 명기할 것
