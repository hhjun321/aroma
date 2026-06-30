# AROMA 핵심 연구 — domain-general ROI thesis & compounding 실험 설계

## (성격: 연구 전략·설계 노트 — 코드 패치 노트 아님)

> 이 노트는 코드 구현 노트가 아니라 AROMA 핵심 연구의 전략·설계·방어성 노트다. [[aroma_severstal_flat_diagnosis_and_direction]]와 같은 성격이며, 실험 결과 해석·논문 주장 범위·남은 핵심 실험(compounding) 설계를 정리한다. 관련: [[aroma_exp4v2_foreground-void-rejection]], [[aroma_exp4v2_aroma-underperformance-diagnosis]], [[aroma_severstal_flat_diagnosis_and_direction]].
>
> 출처: `aroma-thesis-compounding-devnote` 워크플로우(9 에이전트: grounding 3 + 분석 3 + redteam 2 + 종합 1). redteam reviewer 판정 = "현 framing은 REJECT, major revision 시 salvageable".

## 개요

exp4v2(yolov8n, fix+gate 합성, n=3)의 cross-domain 결과를 코드·결과 파일 수준에서 재검토한 결과, AROMA의 기여를 "baseline 상시 격파"로 framing하면 자기 반례(severstal)로 즉시 reject된다. 정정된 thesis는 **AROMA = 생성과 직교하는 결함×배경 type 인지 ROI 선택 컴포넌트**이며 CASDA(단일 도메인 특화 end-to-end 프레임워크)와 경쟁이 아니라 직교·상보 관계다. 본 노트는 (i) thesis를 결과가 실제로 지지하는 형태로 scoping하고, (ii) 각 claim의 방어 가능/불가를 증거에 묶고, (iii) 미입증 핵심 gap인 compounding 실험을 redteam이 짚은 엔진 결함까지 반영해 설계한다.

핵심 caveat (반복 명시): **exp4v2의 'casda' arm은 진짜 CASDA가 아니다.** `generate_casda.py:122`가 `method="copy_paste"`를 하드와이어하고 `controlnet_synthesis`는 `NotImplementedError` 스텁이다. 어떤 arm에서도 diffusion은 실행되지 않았다. 따라서 'casda'는 **CASDA-style suitability ROI 선택(CASDA-ROI)**일 뿐이며, 'CASDA'를 숫자 옆에 적으면 안 된다.

## ① 정정된 thesis

AROMA는 "baseline을 항상 이기는 증강 프레임워크"가 아니라 **생성(how to render)과 직교하는 결함×배경 type 인지 ROI 선택(where/what to paste) 컴포넌트**다. CASDA는 단일 도메인(Severstal) 특화 end-to-end 프레임워크(ControlNet 프롬프트 생성 + Poisson 블렌딩; `extract_rois→roi_metadata.csv`가 구조적으로 Severstal 전용)로 baseline 상회를 입증하는 **생성 품질 축**이다. AROMA의 기여는 이와 직교한다:

- **(a)** 동일 합성 엔진(copy-paste, 동일 blend_mode) 아래에서 ROI 선택이 random보다 일관 우수하거나 최소한 열위가 아님,
- **(b)** steel/texture/object 도메인을 가로질러 붕괴 없이 작동(robustness),
- **(c)** CASDA가 구조적으로 못 가는 컬러·다중 도메인으로 확장.

AROMA는 생성/합성 품질에 개입하지 않는다(copy-paste 사용). 현 증거는 이 thesis를 **"domain-general win"이 아니라 "데이터 희소 regime에서 가치 + 그 외 robustness"**라는 scoped 형태로만 지지한다.

## ② Cross-domain 증거표

| 도메인 | ratio | 순위 (map50) | 부호 | 신뢰도 |
|---|---|---|---|---|
| severstal 4-way | 1.0 | baseline 0.390 > aroma 0.357 > random 0.343 > casda-ROI 0.338 | **aug < baseline** | n=3, aroma>random 2/3 seed (유의 아님) |
| severstal 4-way | 0.4 | baseline 0.390 > random > casda-ROI > aroma 0.318 | **aug < baseline**, aroma 최악 | **n=1 (일화적)**, thesis 내러티브와도 상충 |
| mvtec carpet | 0.5 | aroma 0.916 > random 0.897 > baseline 0.822 | aug > baseline, aroma > random | delta 0.019 < 1 SD |
| mvtec leather | 0.5 | random 0.807 > aroma 0.782 > baseline 0.759 | aug > baseline, **aroma < random (역전)** | n=3 |
| visa | 0.5 | aug ≈ baseline, aroma 무붕괴 | map50 0.06–0.21 (노이즈) | robustness only |

**정직한 독법:** aug>baseline는 MVTec texture에서만 성립하고, severstal에서는 양 ratio·전 seed에서 **부호가 reverse**(증강이 해를 끼침)한다. aroma>random은 carpet에서만 유의 근처(<1 SD)이고 leather에서 역전한다. synth set이 seed 간 동일하므로 error bar는 selection 분산을 미반영 → **selection 효과의 유효 독립 복제 = 1**. 따라서 'domain-general'·'consistently beats random'은 과대주장이다.

## ③ 방어성 (claim matrix)

**방어 가능:**
- AROMA = 생성과 직교하는 ROI 선택 컴포넌트. (`score_roi` L242가 합성 전 소비, 모든 arm이 동일 `copy_paste_synthesis` 통과)
- 데이터 희소 도메인에서 aug>baseline + 신호 약한 곳 무붕괴. (MVTec carpet/leather > baseline; VisA 무회귀 PASS — 단 VisA는 robustness datapoint, positive 증거 아님)
- domain-general이 아니라 **cross-regime robustness** + CASDA가 못 가는 컬러·다중 도메인 확장. (roi_selected.json 스키마 공통, `_reinhard_transfer` Lab-space domain-agnostic; CASDA는 roi_metadata.csv가 Severstal 전용)

**방어 불가 (반드시 scoping/수정):**
- ❌ "AROMA가 baseline을 항상 이긴다" — severstal 양 ratio 반례.
- ❌ "AROMA가 진짜 CASDA(ControlNet+Poisson) 프레임워크를 격파" — casda arm은 copy-paste(`generate_casda.py:122`), diffusion 미실행. 'CASDA-ROI'로만 표기. 추가로 어댑터가 deficit=0.0/morph=0.0/ctx=0.0 하드코딩 + class_floor 부재 → aroma>casda-ROI는 부분적으로 class-balance/gating 효과(순수 score 비교 아님).
- ❌ "domain-general benefit" — MVTec texture 단일 regime 근거. 부호 reverse 도메인이 있는 집합에 'general'은 부적절.

**미입증 (개념적 기여의 핵심 risk — redteam):**
- ⚠️ "이득의 원천 = 결함×배경 type 인지(morph×ctx)" — 코드상 morph/ctx는 pair 내 tiebreak로만 진입하고 rarity_temp=1.0에서 후보가 score로 tie → 실제 승자는 `_img_jitter`(경로 해시). 선택 집합을 실제로 바꾸는 것은 `img_diversity_cap=1` + coverage floor + deficit quota + class_floor이며, random arm은 이를 전혀 안 받는다. **type-awareness ablation(morph/ctx zeroed) + fair baseline(random + 동일 diversity/coverage/class machinery) 없이는 개념적 기여를 attribute 불가.**

## ④ ★ Compounding 실험 설계

**목적:** 미입증 핵심 gap — ROI 선택이 *좋은 생성* 위에서도 효과 있는가(직교 + compound)?

**설계 (단일 인자 swap):** 생성기 G를 고정하고 ROI 리스트만 교체.
- arm1 baseline=real-only · arm2 random-ROI+G · arm3 CASDA-ROI+G · arm4 AROMA-deficit-ROI+G
- load-bearing 비교 = **arm4 vs arm3**(생성 동일, 선택만 다름) 및 arm4 vs arm2.
- exp4v2는 condition-agnostic(synthetic dir만 교체)이라 harness 변경 불필요.

**'G' 정의 고정 필수:** 레포 `stage4_diffusion_synthesis.py`는 ControlNet-**Inpaint**(마스크 영역 디노이징, Poisson 없음)이고 grounding의 'ControlNet+Poisson'과 다른 합성이다. 권장은 stage4 ControlNet-Inpaint(레포 존재 + novel morphology 생성 가능 → copy-paste 한계 해소). 발표 전 한 엔진으로 확정·문서화하지 않으면 'framework 비교' 오해로 미끄러진다.

**⚠️ 선행 수정 (redteam — 엔진이 음성결과에 구조적 편향, 수정 전 null 보고 금지):**
- **치명결함1 (morphology 미흐름):** stage4가 run당 `control_pil`+`seed_profile`을 단 1개 `seed_path` 결함에서 ONCE 계산(L249-250) → ROI별 결함 형태가 생성기로 거의 안 흐름 → roi_selected.json의 defect 내용 무시 → ROI leverage 구조적 ≈0. **수정:** ROI마다 자신의 crop+mask에서 per-ROI Canny control + per-ROI inpaint mask 생성(`_canny_control_image`를 entry마다 호출).
- **치명결함2 (seed binding):** `seed=seed+i`(L276)가 latent noise를 placement 순서에 묶음 → 동일 (defect,bg) 쌍이 arm마다 다른 latent로 디노이즈 → ROI-vs-noise confound가 ~0.02 map50 마진과 동급. **수정:** `seed = stable_hash(source_roi_image_id + paste_x + paste_y + rep_idx)` content-hash 바인딩, synthetic SET을 도메인·arm당 1회 생성·캐시(YOLO seed마다 재생성 금지).

**도메인:** carpet 우선(aug>baseline & aroma>random 둘 다 성립하는 유일 regime). severstal = falsification arm(high risk, thesis 걸지 말 것). VisA = robustness only. ControlNet은 Severstal 강판 튜닝이라 carpet/VisA 생성 fidelity 미검증 → 검출 sweep 전 **도메인별 시각 pilot로 gate-in**.

**지표:** 1차 YOLO val map50, **n≥5 seed**(이상 8-10), paired per-seed delta(arm4−arm3) + sign/Wilcoxon(겹치는 error bar 금지). 2차 map50-95. MVTec/severstal per-class AP(minority starvation 노출). selection seed도 resample.

**Confound 통제:**
- C2 물량 parity: exp4v2 synth_ratio cap **두 위치(L2100, L2145) 튜플에 새 arm key 둘 다 추가**(누락 시 uncapped 자동 부당 승). n_ok/n_skip 로깅, 학습 전 각 arm >=cap 보장, skip 시 top-up. 0.4·1.0 양 ratio서 방향 일치 요구.
- C3 downstream parity: blend/reject_clean_bg/foreground/class_mode/epochs/COCO init 동일, `roi_dir`만 차이임을 프로그램적으로 assert.
- C4 class-balance vs deficit score 분리: **arm3b = CASDA-ROI + AROMA class_floor** 추가. thesis가 'deficit SCORE'면 arm4 vs arm3b가 load-bearing.

**음성결과 해석 (redteam — broken engine vs true null 구분 필수):** 선행 control 통과 전 null 보고 금지.
- POSITIVE control: 극단 arm(최희소 클래스만 deficit oversample vs quality-top only)으로 per-class AP가 예측대로 shift하는지 — shift 없으면 엔진 inert, null 무의미.
- NEGATIVE control: 동일 ROI 리스트 2회 실행이 content-hash seed로 동일 map50 재현(seed 결정성).
- 사전등록: (i) arm4≥arm3 ∧ arm4>arm2 → **compounding SUPPORTED**(누락 기둥). (ii) arm4≈arm3, 둘 다 >arm2 → 'deficit-aware가 좋은 생성서 worse 아님'(약한 방어). (iii) arm4≈arm3≈arm2 → 좋은 생성 위에선 ROI 무관 = copy-paste-era 효과(thesis (a) 일반화 실패, **정직 보고**). (iv) severstal ControlNet에도 aug<baseline → 생성품질이 원인 아님(데이터 충분/포화), 주장 bound.

## ⑤ 구현 scope

'좋은 생성' 엔진은 외부 hhjun321/CASDA 클론 불필요 — 레포에 디퓨전 스택 존재(`stage4_diffusion_synthesis.py`, `scripts/train_controlnet.py`). (이전 대화의 '컨트롤넷은 stub, AROMA에 생성 없음'은 부정확 — 별도 stage4 파이프라인이 있음.) 단 stage4는 ControlNet-Inpaint이지 ControlNet+Poisson 생성이 아님이 실제 차이.

**옵션 A (권장)** — `generate_defects.py` `controlnet_synthesis` 스텁(L777-790) 직접 구현, stage4 코어 재사용. run()이 고정 kwargs로 호출하고 스텁이 `**kwargs` 수용 → 시그니처 충돌 없음. 반환 계약 `{'bbox':[x,y,w,h],'mask_path':<str|None>}`만 충족. ~150-250줄: roi_entry 언팩 → per-ROI Canny control(치명결함1 수정 동반) → defect_bbox→inpaint mask → `StableDiffusionControlNetInpaintPipeline`(`_synthesize_one` 거의 그대로 + content-hash seed로 교체, 치명결함2 수정 동반) → mask PNG 저장 + bbox 반환. `_SYNTHESIS_METHODS`에 controlnet 키 이미 등록 → Colab서 `generate_defects.run(method='controlnet', ...)` 한 줄로 exp4v2 합류. ROI는 roi_dir만 다르면 되므로 직교.

**옵션 B (비권장)** — 외부 CASDA Stage B/C/D 클론. placement_map.json↔roi_selected.json 양방향 어댑터 필요, ROI만 교체하려면 CASDA 코어 수정 필요('코어 수정 금지' 충돌), 크로스레포 디버깅. 동일 엔진이 레포에 있어 이점 없음.

**단계적 3-tier:**
- **Tier 0 (GPU 0원, 즉시):** `blend_mode='seamless'`(cv2.seamlessClone NORMAL_CLONE, L206-294)로 copy_paste 기반 AROMA-ROI vs CASDA-ROI vs random-ROI. **단 generate_casda/random 기본 blend_mode='alpha'이므로 양 arm 모두 `--blend_mode seamless` 강제**(누락 시 합성품질 confound). 'ROI 선택이 Poisson compositing 위에서 우수한가'에 싸게 답하나, copy-paste+Poisson은 원본 결함 복사라 **novel morphology 생성 gap 미해결 → 디퓨전 실험 대체 불가.**
- **Tier 1 (GPU 소):** 사전학습 sd-controlnet-canny 가중치만 다운로드(학습 0) + 옵션A 스텁 → zero-shot 디퓨전 위 ROI 비교. 비용 대비 입증력 균형 최선.
- **Tier 2 (GPU 중~대):** 도메인별 ControlNet 파인튜닝(`train_controlnet.py`). 최강 주장이나 비용 최대 + `train.jsonl` 빌더 부재 → 추가 작업.

**선행조건:** 학습된 `.safetensors` 체크포인트 실재 여부 미확인(코드 참조만, Drive 의존). GPU 비용은 load-test 정책상 실측 안 함(추정만): 30 steps·512px·fp16, 도메인·arm당 1회 생성·캐시. Colab 규칙(CLAUDE.md pytest 금지·Colab 직접검증, colab-execution.md `!python $VAR`) 정합.

## ⑥ TODO · 미확정

- [ ] **type-awareness ablation**: morph/ctx zeroed vs full, 그리고 'random + 동일 diversity/coverage/class machinery' fair baseline — 미실행 시 개념적 기여 attribute 불가.
- [ ] **n≥5 + selection seed resample**: 현 n=3 + 동일 synth set은 selection 분산 미반영. severstal r0.4 n=1은 인용 전 n≥3로 승격.
- [ ] **compounding 선행 수정 2건**(per-ROI control/mask, content-hash seed) + positive/negative control 통과 후에만 null 해석.
- [ ] **'좋은 생성' 엔진 확정**: ControlNet-Inpaint vs ControlNet+Poisson — 발표 전 문서화.
- [ ] **ControlNet 도메인 fidelity pilot**: Severstal 튜닝 가중치의 carpet/VisA zero-shot 출력 시각 검증.
- [ ] **power analysis**: carpet 마진 0.019가 3-seed std 수준 — compounding delta 측정 가능 도메인 존재 여부 사전 확인.
- [ ] **claim 언어 scoping**: abstract/intro/figure caption/ALL_CONDITION_KEYS legend 전부 — 'CASDA' 무자격 사용 금지, 'general' → 'cross-regime robustness', 'no diffusion was run' 명시.
- [ ] **blend parity 검증**: 발표된 exp4v2 run config에서 모든 arm이 동일 blend_mode였는지 확인(alpha/seamless 혼용 시 compositing confound).

---

## ⑦ 구현 (compounding 엔진) — 2026-06-30

**엔진 확정 (⑤ 옵션 A 폐기)**: dev_note ⑤는 in-repo `stage4_diffusion_synthesis.py`(ControlNet-**Inpaint**) 구현을 제안했으나, 코드 추적 결과 **학습된 모델은 CASDA `test_controlnet.py`(StableDiffusionControlNetPipeline, SD-v1.5+canny) + Stage C Poisson** regime으로 학습됨(04-StageB/C). Inpaint pipeline(SD-inpainting base)에 넣으면 **regime 불일치(OOD)**. → 엔진 = **CASDA 실제 추론(ControlNetPipeline+Poisson), CASDA 스크립트 무수정 호출**. 트레인 모델: `Severstal/controlnet_training_v5.5/best_model`(save_pretrained: config.json+fp16 safetensors).

**파이프라인 (단일인자 swap = roi_metadata.csv만 차이)**:
```
roi_selected.json(arm) → [어댑터] → roi_metadata.csv(CASDA 스키마)
  → prepare_controlnet_data.py (hint 3채널 + train.jsonl 자동 — csv prompt 무시·재생성)
  → test_controlnet.py(best_model, 30steps/cond0.7/512)
  → compose_casda_images.py (Poisson; 출력 images/+masks/full-frame+metadata.json)
  → [§3d 변환] composed → exp4v2 annotations.json
  → exp4v2 4-way(baseline/random/casda/aroma) severstal, --class_mode multi, seeds 42 1 2
```

**신규 코드**: `scripts/aroma/aroma_to_casda_roi.py` (AROMA roi_selected.json + morphology_features.csv → CASDA roi_metadata.csv). py_compile OK, severstal 로컬 검증(헤더 byte-exact, morphology join 100%).
- 필드: defect_bbox[x,y,w,h]→"(x1,y1,x2,y2)", class_id←class_key, region_id per-image, background_type=complex_pattern, linearity/solidity/extent/aspect←morphology join(defect_mask_path key), prompt="" (packager 재생성), roi_image_path/roi_mask_path←crop-aligned PNG(`--make_crops`).

**확정 결정 (reviewer 대응)**:
| 결정 | 값 | 근거 / reviewer 방어 |
|------|-----|------|
| **MAX_ROIS** | **1383** (= CASDA 가용 ROI, binding constraint; aroma 1690·random 1690을 trim) | **보고 하이퍼파라미터 아님.** 보고 budget = synth:real **0.4 (1013장/arm)**, 전 조건 동일+이전 run 동일. 1383=생성 풀(최소 조건에 맞춘 엄격 parity), ≥1013이라 학습 budget 커버. 본문 아닌 부록. |
| **패키징 필터 3종 균일 무력화** | aroma/random(어댑터): roi_bbox==defect_bbox(edge skip)+area 큼+score 0.7. casda(cap셀): roi_bbox=defect_bbox, area.clip(≥100), stability/matching=0.7, rec=acceptable | CASDA 패키징은 **edge(경계마진)+area(≥100)+quality(stability≥0.3·matching≥0.5)** 3필터. AROMA full-strip 결함(bbox 큼·경계닿음)+저스케일 roi_score → 적용 시 차등 컷(aroma 우회·casda 1383→465). generation 엔진 아닌 selection-side 필터이고 각 arm 선택은 상류 확정 → **3 arm 균일 무력화 = 단일인자 swap 정합**. `prepare` CLI 미노출+코어 무수정 → csv 컬럼으로만 가능. ⚠️ compose가 roi_bbox로 placement 결정 시 영향 — severstal full-strip이라 무해 추정, 실행 시 확인. |
| **compositions-per-roi=1** | 풀=MAX_ROIS×1=1383 | synth_ratio 0.4 trim 1013과 정합, ×5 폭증 회피 |
| **synth_ratio=0.4** | exp4v2 cap=1013 | 최근 severstal cleanbg 0.4 run(real 2534/synth 1013)과 직접 비교 |

**CASDA CLI 확정**: `prepare_controlnet_data.py --roi_metadata --output_dir --train_images --train_csv --skip_validation --workers`(default 0=순차→`-1` 병렬 필수); `test_controlnet.py --model_path --jsonl_path --output_dir --num_inference_steps --guidance_scale --controlnet_conditioning_scale --resolution`; `compose_casda_images.py --generated-dir --hint-dir --metadata-csv --summary-json --clean-images-dir --train-csv --output-dir --compositions-per-roi`.

**경로/성능**: CASDA 데이터 루트(`/content/drive/MyDrive/data/Severstal`)≠AROMA(`.../Aroma`). CN_ROOT(hints+generated) **로컬(/content)** 권장(Drive PNG write 병목). `CSV_CASDA_FULL`(원본 읽기전용)≠`CSV_CASDA`(capped 출력, 원본 보호).

**가이드**: `AROMA연구분석/colab_execute/compounding_controlnet_execute.md` (§0 env → §1 ROI → §2 어댑터+cap → §3a prepare → §3b generate[GPU] → §3c compose → §3d convert → §4 exp4v2[GPU] → §5 결과).

**prompt 결정 (최종, 실측 기반)**: AROMA step2 자유서술 prompt를 추론에 쓰려 했으나(§3a.5 patch), **smoke 실측 결과 OOD로 생성 비현실** — A/B 확인: 동일 aroma hint에 CASDA식 prompt=그럴듯, AROMA step2 prompt=깨짐. casda smoke(CASDA hint+prompt)=정상이라 엔진·hint는 OK → **원인=prompt 분포 불일치**. → **결정 (i): §3a.5 patch 미사용**, packager 재생성 **CASDA식 prompt**(AROMA `defect_subtype`+background_type → CASDA 템플릿) 사용 = in-distribution + AROMA 형태결정 반영("AROMA-informed prompt"). AROMA step2 자유서술 prompt-generation은 **논문 별도 컴포넌트로 기술**하되 생성엔 미사용. (어댑터 csv prompt 컬럼은 packager 무시 → 무해, prompt-ablation 참조용.) train.jsonl 이미 patch됐으면 §3a prepare 재실행으로 복원.

**남은 검증**: GPU 실행(best_model 추론); §3d 변환 후 exp4v2 annotations 로드 확인; composed→exp4v2 normal_image 매핑(metadata.json) 선택 보강; AROMA prompt OOD 영향 점검(생성물 육안).
