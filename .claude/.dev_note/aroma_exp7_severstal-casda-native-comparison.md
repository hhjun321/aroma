# exp7 — severstal baseline vs aroma vs CASDA-native 3-arm downstream 비교

---

## ⛔ 상태: 선행 대기 (BLOCKED — 미착수)

**exp7은 severstal `sym_final` 전체 체인이 끝나야 시작 가능하다.** 현재(2026-07-10) severstal sym_final 미실행 → graft 소스(aroma-CN/baseline/random)가 없어 착수 불가. 본 문서는 **선행 완료 후 진행할 계획서**다.

### 선행 필수 체인 (순서대로, 완료 전 exp7 착수 금지)

```text
prepare_datasets(severstal) → phase0 → step1 → step2 → step3
  → step5 (ControlNet 학습 + τ 사전스캔) → step4 (aroma-CN 생성 + random 생성)
  → exp4v2 (severstal baseline/random/aroma per-seed)   ← 여기까지가 graft 소스
  ─────────────────────────────────────────────────────
  → [exp7 착수] 어댑터 → casda arm 학습 → 판정
```

- 선행 = `colab_execute_new/`의 severstal 파이프라인(`_SPEC` 규약). aroma arm = **aroma-CN**(ControlNet+seamless), severstal multi nc=4, imgsz 640/rect/100ep/seeds 42 1 2.
- 선행 산출 확인점: `sym_final/exp4v2/…severstal…` per-seed(baseline/random/aroma) 존재 → 그때 exp7 STEP 2 graft 가능.
- **exp7 자체는 GPU 1트랙(casda arm 3-seed)만 추가**. 무거운 부분(CN 학습·생성)은 전부 선행 체인에서 소진.

### exp7 착수 시 체크리스트 (선행 완료 후)

- [ ] severstal sym_final exp4v2 per-seed(baseline/random/aroma-CN) 실재 확인
- [ ] CASDA composed 이미지+마스크 실물 drive 존재 확인 (`$CASDA_COMPOSED/{images,masks}`)
- [ ] 어댑터(`casda_composed_adapter.py`) 구현 → `synth_casda/severstal/annotations.json`
- [ ] `exp7_execute.md` 가이드 작성
- [ ] exp4v2 casda arm 학습(파라미터 = 선행과 동일 잠금)

> 아래 설계(어댑터·가이드·파라미터)는 **확정본**이다. 선행이 끝나면 그대로 구현 착수한다.

---

## (사용할 skills: feature-dev)

## 개요

논문 갭 **"CASDA 프레임워크 비교 미실행"**(project_paper_state_gaps)을 해소한다. 기존 `casda` 조건(`generate_casda.py` → shared copy-paste 엔진, ROI 선택 전략만 격리한 ablation)과 **별개**로, CASDA의 **네이티브 풀파이프라인 산출물**(ControlNet 생성 + Poisson Blending composed)을 그대로 casda arm으로 주입해 **AROMA 프레임워크 vs CASDA 프레임워크** 정면 비교를 수행한다.

**공정 비교 원칙(사용자 정정)**: 두 프레임워크 모두 **ControlNet 생성** 기반이다. AROMA arm은 copy-paste가 아니라 **sym_final의 `--method controlnet` + `--blend_mode seamless`**(seamlessClone) 산출을 쓴다. 차이 = **blend 방식**(CASDA=Poisson normal clone / AROMA=seamlessClone) + **AROMA의 ROI modeling**(deficit/compat/symmetric 선택). copy-paste arm(aroma-CP, 20260705)으로 붙이면 AROMA를 약체로 대표해 불공정 → 배제.

severstal 단독, downstream = `exp4_v2_supervised_detection.py`(YOLOv8n supervised). 3 arm = **baseline / aroma(=aroma-CN, sym_final ControlNet+seamless, 이식) / casda-native(신규 학습)**. exp4v2에는 `casda` 4번째 조건이 이미 완성돼 있어 **신규 코드는 어댑터 1개뿐**이고, exp4v2 하네스는 무수정 재사용한다.

효과 주장은 3-seed(42 1 2) per-class AP + Δ로 확정한다.

---

## 확정 결정 (사용자)

1. **casda arm = CASDA 네이티브 composed** (별개 real basis 허용 — 진짜 프레임워크-대-프레임워크). shared-engine ablation(`generate_casda`) 아님.
2. **CASDA synth 소스 = drive 기존 `augmented_dataset`(v5.5 composed, 30,300장) 재사용**, GPU 불요. `suitability_score` **top-N pool**로 매칭.
3. **baseline/aroma = `sym_final` severstal exp4v2에서 graft(이식)**. aroma arm = **aroma-CN**(sym_final ControlNet + seamless blend). severstal **multi nc=4**. casda arm만 fresh 학습.
   - ⚠️ **선결**: severstal sym_final 전체 체인(phase0→step1→2→3→step5 CN학습+τ→step4 생성→exp4v2)이 완료돼 baseline/aroma per-seed가 존재해야 graft 가능.

## 프레이밍 (overclaim 방지)

이 비교는 **프레임워크 전체 비교**다. **두 프레임워크 모두 ControlNet 생성 기반** — 차이는 (a) blend 방식(CASDA=Poisson normal clone / AROMA=seamlessClone), (b) AROMA의 ROI modeling(deficit/compat/symmetric 선택 + clean-bg 게이트). casda의 real basis(CASDA `/data/Severstal` train_images 배경)가 aroma(`Aroma/severstal` ROI+배경)와 다르다 — **설계상 허용**(각 프레임워크 고유 파이프라인 재현). 유효 비교 조건은 **동일 downstream real train/val/test split + 동량(synth_ratio cap) + 누수 dedup** 세 가지뿐. 논문 표기: "CASDA (ControlNet + Poisson blend) vs AROMA (ControlNet + seamless blend + ROI modeling)".

---

## 영향도 분석

### 이 기능이 변경하는 상태
- 신규 `synth_casda/severstal/annotations.json` 생성 (drive).
- `exp7` 출력 `exp4v2_results.json`에 `casda`(=native) 조건 map50/per_class/n_synth_train 기록.

### 그 상태를 전제로 동작하는 기존 로직
- `exp4_v2_supervised_detection.py`의 `casda` 조건 처리: **이미 완성·검증**(`--casda_synthetic_dir`, `ALL_CONDITION_KEYS`, cap 공정성 튜플 `("random","casda","aroma")`, per-class n_synth 로깅, `_resolve_synth_class` cluster_id fallback). **무수정**.
- baseline/random/aroma 이식 로직: `exp4v2_symmetric_downstream_execute.md` STEP 2 준용. 불변.

### 공정성 (load-bearing)
- casda arm은 이미 cap 튜플 양쪽(L2332 절대 cap / L2425 synth_ratio cap)에 포함됨 → synth 무제한 불공정 없음.
- **exp4v2 cap = `rng_sub.sample(anns, cap)` = seeded RANDOM**(L2332/2425), top-k 아님. 따라서 "suitability top-k" 의도는 **exp4v2 cap이 못 지킨다** → 어댑터가 pool 단계서 top-N으로 미리 잘라야 확정된다.
- count parity: 어댑터가 top-N(N ≥ aroma labelable 600) pool을 emit하면, exp4v2 `synth_ratio` cap이 전 arm 동일 random trim → 학습 synth 수 동일.

### 누수 (correctness, 비협상)
- CASDA composed `source_background` stem이 `Aroma/severstal/test/class*`에 존재하면 **train/test 누수** → 어댑터에서 제외.

---

## 수정 내용

### 1. `scripts/aroma/casda_composed_adapter.py` — 신규 스크립트 (CPU, GPU 불요)

CASDA `augmented_dataset/metadata.json`(30,300 composed) → AROMA 타깃 스키마 `{output_dir}/severstal/annotations.json`.

**타깃 스키마** (`synth_aroma_sym/annotations.json` 실측 기준):
`bbox`[x,y,w,h]px · `class_key`(class1..4) · `cluster_id`(1-4) · `image_path` · `mask_path` · `normal_image`(옵션) · `source_roi`(옵션) · `dry_run:false`.

- **class 매핑**: CASDA `class_id` 0/1/2/3 → `class_key` class1/2/3/4 + `cluster_id` 1-4 (severstal class = `class_id + 1`).
  - metadata 분포: c0=11762, c1=950(**rare = severstal class2**), c2=5971, c3=11617.
- **bbox 변환**: metadata `bbox_format="yolo"` 정규화 `[cx,cy,w,h]` → px `[x,y,w,h]`. `W=1600, H=256`. `x=(cx-w/2)*W`, `y=(cy-h/2)*H`, `w*W`, `h*H`. 음수/경계 clamp.
- **경로 resolve**: metadata `image_path`("images/…png") / `mask_path`("masks/…_mask.png") → CASDA drive 절대경로 prefix(`$CASDA_COMPOSED` = `/content/drive/MyDrive/data/Severstal/augmented_dataset/casda_composed`). 두 드라이브 모두 Colab 마운트 → **복사 불요**, 절대경로 참조.
- **suitability top-N pool**: metadata `suitability_score` desc 정렬 후 top-N emit. `--top_n`(기본값 = aroma arm labelable 이상, 예 800~1000). exp4v2 cap이 이후 동일 random trim으로 count parity 보장.
- **누수 dedup**: composed `source_background`(예 "310c89662.jpg")의 stem이 `--test_stems`(= `Aroma/severstal/test/class*` 파일 stem 집합)에 있으면 제외.
- **진단 로그**: `n_total / n_dedup_leak / n_missing_image / n_missing_mask / n_class_unmapped / per-class(c1..c4) emit 카운트`.

**CLI(안)**:
```
--metadata_json   $CASDA_COMPOSED/metadata.json
--composed_root   $CASDA_COMPOSED           # image_path/mask_path prefix
--output_dir      $SYNTH_CASDA               # /{severstal}/annotations.json 작성
--dataset_key     severstal
--test_dir        $AROMA_DATA/severstal/test # 누수 dedup용 stem 수집
--top_n           1000
--img_w 1600 --img_h 256
```

> **재사용 안 함**: 기존 `casda_roi_adapter.py`는 Stage A ROI-CSV → copy-paste 엔진 입력용(roi dict)이다. 여기서 필요한 건 **이미 composed된 전체 이미지 → annotations.json**이라 별도 어댑터가 맞다.

### 2. `AROMA연구분석/colab_execute_new/exp7_execute.md` — 신규 colab 가이드

- **STEP 0 환경**: `_SPEC §1` 공통 셀 그대로 + CASDA drive env 추가
  (`CASDA_DRIVE=/content/drive/MyDrive/data/Severstal`, `CASDA_COMPOSED=$CASDA_DRIVE/augmented_dataset/casda_composed`, graft 소스 `EXP4V2_REF=$(S('exp4v2'))`(= `sym_final/exp4v2`, severstal 결과), `SYNTH_CASDA=$AROMA_OUT/sym_final/synth_casda`, `EXP_OUT=$AROMA_OUT/sym_final/exp7`).
- **STEP 1 전제확인**: composed 이미지+마스크 실물 drive 존재, `sym_final/exp4v2` 또는 `_seeds/seed{42,1,2}/exp4v2_results.json`에 severstal baseline/random/aroma(=aroma-CN) 존재, real data 존재.
- **STEP 2 graft**: `sym_final/exp4v2`에서 severstal **baseline/random/aroma(aroma-CN)** per-seed 이식 → `$EXP_OUT/_seeds/seed{s}/exp4v2_results.json`. `casda` 셀은 비워둠(STEP 4에서 학습).
  (`exp4v2_symmetric_downstream_execute.md` STEP 2 준용. aroma = sym_final ControlNet+seamless 그대로 headline.)
- **STEP 3 어댑터 실행**: `casda_composed_adapter.py` → `$SYNTH_CASDA/severstal/annotations.json`. 진단 로그로 per-class·dedup 확인.
- **STEP 4 exp4v2 실행(GPU)**: `--condition all --dataset_keys severstal --class_mode multi --casda_synthetic_dir $SYNTH_CASDA` (+ random/aroma dir는 이식분 로드용). `--resume`가 이식된 baseline/random/aroma skip, **casda만 학습**.
  파라미터 = 이식 소스와 동일(아래 잠금).
- **STEP 5 판정(CPU)**: map50 + per-class AP(c1-4, 특히 **rare c2**) + per-seed Δ(**casda−aroma**, **casda−baseline**, **aroma−baseline**). 사전 등록 판정표.

---

## 파라미터 잠금 (graft 유효조건 — 비협상)

casda arm 학습은 이식 소스(`sym_final/exp4v2` severstal, `_SPEC §4` 트랙 A)와 **완전 동일**:

| 항목 | 값 |
|------|----|
| class_mode | `multi` (nc=4) |
| imgsz | 640 |
| rect | `--rect` |
| baseline_epochs | 100 |
| patience | 50 |
| seeds | 42 1 2 |
| val_frac | 0.3 |
| synth_ratio | 1.0 |
| batch | 64 |
| cache | ram |

하나라도 다르면 이식 비교 무효. 파라미터 변경 시 skip 조건에 걸려 재학습 안 됨 → fresh `--output_dir` 또는 casda 항목 삭제로만 재실행.

---

## 수정 대상 파일

- **신규** `scripts/aroma/casda_composed_adapter.py`
- **신규** `AROMA연구분석/colab_execute_new/exp7_execute.md`
- **무수정 재사용** `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (casda 조건 완성됨)

---

## 암묵적 요구사항 / 엣지

| 상황 | 처리 |
|------|------|
| composed `source_background`가 test와 겹침 | dedup 제외 (누수 차단, 하드) |
| composed 이미지/마스크 파일 disk 부재 | 해당 entry skip + `n_missing_*` 카운트 |
| class_id가 0-3 밖 | skip + `n_class_unmapped` 경고 |
| yolo bbox 정규화값이 경계 초과 | px 변환 후 [0,W]/[0,H] clamp, w/h ≤ 1px면 skip |
| top_n < aroma labelable(600) | 경고 — count parity 깨질 수 있음, top_n 상향 권장 |
| exp4v2 cap이 random이라 top-k 무시 | 어댑터 pool 단계서 top-N 확정으로 해결 |
| CASDA vs AROMA drive 경로 상이 | 둘 다 마운트, 절대경로 참조로 무해 |
| mask 있는데 bbox 없음(빈 마스크) | exp4v2 GT-mask 경로가 union bbox 회수(기존 동작) |

---

## 테스트 (Colab, pytest 금지)

1. **어댑터 sanity**: `annotations.json` entry 수 = top_n − dedup − missing. per-class(c1..c4) 카운트 로그, rare c2 존재 확인. `n_dedup_leak` 보고.
2. **경로 유효성**: 샘플 entry의 `image_path`/`mask_path` disk 존재(Colab).
3. **공정성 parity**: exp4v2 결과 JSON에서 random/casda/aroma `n_synth_train` **동일** 확인.
4. **이식 정합**: STEP 4 로그에 baseline/random/aroma `RESUME skip … (cached map50=…)`, casda만 실제 학습.
5. **결과**: per-condition map50 + per-class AP(c1..c4) + 3-seed.

**성공 기준 해석** (사전 등록):

| 관측 | 해석 |
|------|------|
| aroma > casda-native, 전 seed | AROMA(ROI modeling) > CASDA 프레임워크 |
| aroma ≈ casda-native | 두 프레임워크 downstream 동급 (합성 novelty·품질 상쇄) |
| casda-native > aroma | CASDA(Poisson blend)가 AROMA(seamless+ROI modeling)보다 유리 → 정직 보고 |
| 전 arm > baseline | 합성 증강 자체가 이득 |
| rare c2 AP | CASDA H4(소수클래스) 주장 영역 — 양 프레임워크 c2 AP 대조 |

---

## 진행 전 미검증 전제 (Colab 확인)

- composed 이미지/마스크 실물 drive 경로(`.etc`엔 `metadata.json`만 있음). `$CASDA_COMPOSED` 실제 레이아웃(`images/`, `masks/`) 확정 필요.
- **★핵심 선결: severstal `sym_final` 전체 체인 완료 여부.** aroma-CN(ControlNet+seamless) + baseline + random per-seed가 `sym_final/exp4v2`에 있어야 graft 가능. 미완이면 exp7 전에 severstal sym_final(phase0→…→step5 CN학습+τ→step4 생성→exp4v2)을 먼저 돌려야 함 — GPU 다수 세션. graft 대신 fresh 3-arm으로 갈지 재검토 가능.

---

## 미확정 / 후속 (TODO)

- `--top_n` 최종값: aroma arm labelable(≈600) 대비 여유분. STEP 3 진단 후 확정.
- composed `metadata.json`의 `source_background` 필드 정확한 키명(스캔 결과 `source_background` 확인됨) — 어댑터 dedup에서 fallback 키(`source_image` 등) 방어.
- reviewer가 "real basis 불일치" 지적 시 → 동일 ROI 풀 재정렬(shared-engine) ablation을 보조로 제시(기존 `generate_casda` 결과 재활용).
