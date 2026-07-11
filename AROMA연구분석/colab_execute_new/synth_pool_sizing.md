# 합성 pool 사이징 — 데이터셋별 유효 top_k / n_per_roi 분석

> 목적: **synth:real 비율을 exp4v2에서 자유롭게 스윕**할 수 있도록, 데이터셋별로 합성 pool을
> "미리 충분히" 생성해두기 위한 `--top_k` / `--n_per_roi` / `--pool_k` 권고값을 제시한다.
> STEP 3B(copy_paste) · STEP 4(random) 재생성 시 이 표를 소비한다.
>
> 작성 근거(2026-07-11 로컬 실측): `AROMA_DATASET/roi/<ds>/roi_candidates.json`(후보 수),
> `roi_selected.json`(현재 선택 수), 로컬 real `test/<type>` 결함 수, severstal은 exp4v2 Colab
> 로그값(real_train=2534). 관련: `synth_pool_sizing` ↔ leather 배경 붕괴 이슈.

---

## 0. 원리 — 무엇이 상한을 정하는가

- **합성 pool 크기 = `top_k`(선택 ROI 수) × `n_per_roi`(ROI당 변형 수).**
  단, per-class parity / per-pair 캡에 의해 실제 출력이 이보다 작을 수 있으므로 **parity 로그로 실측 확인**한다.
- **exp4v2가 실제로 쓰는 양(cap) = `real_train × synth_ratio`** (`--synth_ratio`) 또는 `--max_synth_per_ds`.
  - `real_train = real_defect × (1 − val_frac)`, 기본 `val_frac=0.3`.
  - synth pool이 cap보다 작으면 **available 전량 사용(no trim)** — 즉 원하는 비율에 못 미친다(현재 severstal이 이 상태).
- 따라서 **생성 목표 = `real_train × (테스트할 최대 ratio)` 이상**으로 오버슈트 → exp4v2에서 seed 결정론적 subsample로 비율만 낮춰가며 스윕(재생성 불필요).

### top_k를 올릴지 n_per_roi를 올릴지
- **1순위 `top_k` ↑**: 서로 다른 결함 crop이 늘어 **진짜 다양성** 증가. 단 하위 랭크(낮은 roi_score/deficit 꼬리) 편입 → 후보 수가 한계.
- **2순위 `n_per_roi` ↑**: copy_paste는 *같은 crop을 다른 배경/위치에* 붙인 변형이라 **near-duplicate** 위험. 무한정 올리면 다양성 포화.
- ⚠️ **`n_per_roi`는 `pool_k` 이하까지만 배경이 실제로 달라진다**: generate 루프가 `_cbg_pairs[rep_idx % len(pool)]`로 소비 → `n_per_roi > pool_k`면 같은 배경 재사용(위치 지터만). 그러므로 **`--pool_k ≥ n_per_roi` 명시**. leather는 유사도-랭킹 붕괴로 배경이 18종으로 collapse된 전례가 있어 특히 pool_k 명시가 중요.

---

## 1. 데이터셋별 현황 (실측)

| 데이터셋 | 후보 수 | 현재 top_k | 현재 synth(관측) | real_defect | real_train@val0.3 | ratio 1.0 cap | 현재 대비 |
|---|---:|---:|---:|---:|---:|---:|---|
| **severstal** | 266,624 | 200 | 400 | 3,620 | **2,534** | 2,534 | ❌ **6.3× 부족** |
| **mtd** | 16,877 | 200 | 400 | 388 | 271 | 271 | ✅ 충분 |
| **aitex** | 6,168 | 200 | 400 | 352 | 246 | 246 | ✅ 충분 (1.6×) |
| **mvtec_leather** | 968 | 200 | 600 | 92 | 64 | 64 | ✅ 9× 여유 |

> 관측 synth로 역산한 실효 n_per_roi: leather 600/200=**3**, mtd 400/200=**2**, severstal 400/200=**2**.
> step5 명령은 `--n_per_roi 3`이므로 mtd/severstal이 2로 나온 것은 **parity/per-class 캡으로 트림**됐을
> 가능성 — 재생성 시 parity 로그(`used/fallback`, arm 총량 동수)로 반드시 확인.

**핵심 결론**: cap을 결정하는 것은 각 데이터셋의 **real_train**이다. real 결함이 압도적으로 많은
**severstal만 ratio 1.0에서 심각하게 부족**하고, 나머지(leather/mtd/aitex)는 real_train이 작아
현재 pool로도 ratio 1.0을 이미 충족한다.

---

## 2. 권고값 (테스트할 최대 ratio = 1.0 기준, 여유율 1.2×)

목표 synth = `ceil(real_train × 1.0 × 1.2)` (real_frac 축소·seed 변동 흡수용 20% 헤드룸).

| 데이터셋 | 목표 synth | 권장 top_k | 권장 n_per_roi | top_k×n_per_roi | 권장 pool_k | 비고 |
|---|---:|---:|---:|---:|---:|---|
| **severstal** | ~3,050 | **1,000** | 3 | 3,000 | ≥3 (4~6 권장) | 후보 266k라 top_k↑ 여유 충분(상위 0.4%). **유일하게 재생성 필요.** |
| **mtd** | ~330 | 200(유지) | 3 | 600 | ≥3 | 현재로도 충분. n_per_roi 3 통일 시 600. |
| **aitex** | ~300 | 200(유지) | 3 | 600 | ≥3 | 현재 400으로 ratio 1.0 충족. n_per_roi 3 통일 시 600. |
| **mvtec_leather** | ~80 | 200(유지) | 3 | 600 | **명시 ≥3** | 이미 9× 초과. 배경 18종 collapse 있으니 pool_k 명시로 배경 다양성 확보. 후보 968뿐이라 top_k↑ 여지 적음. |

> **더 큰 ratio(예 2.0)까지 볼 계획이면** 목표 synth를 그 배수로 재산정. severstal ratio 2.0 → ~5,000
> 필요 → top_k 1,700×3 등(후보 여유 있음).

### 왜 severstal만 top_k를 크게 올리나
- real_train 2,534는 real 결함(3,620)이 많기 때문. 1:1 증강엔 2,534장 필요.
- 후보 266,624개 중 top_k=1,000은 **상위 0.4%** — 품질 꼬리 유입 미미. n_per_roi=3과 곱해 3,000 확보.
- leather는 반대: real_train 64라 목표가 80, 현재 600으로 이미 과잉 → **올릴 이유 없음**(오히려 exp4v2 subsample로 줄여 씀).

---

## 3. 공정성 · 재현성 체크리스트

- [ ] **`--top_k` · `--n_per_roi`는 데이터셋 내에서 aroma·random arm 동일값** (둘 다 인자 보유). 데이터셋 *간* 다른 건 정상(ratio가 정규화).
- [ ] **`--pool_k ≥ n_per_roi`** 로 명시 (aroma 경로 배경 다양성; random_arm은 배경 배정 방식이 달라 무관하나 총량 parity는 확인).
- [ ] 재생성 후 **parity 로그로 arm 총량 동수** 및 `clean_bg used≈total, fallback/mismatch≈0` 확인.
- [ ] **한 번 최대 크기로 생성** → exp4v2에서 `--synth_ratio 0.3 0.5 1.0` 스윕으로 비율만 조절(재생성 금지, seed 결정론적 subsample).
- [ ] severstal 대량 재생성은 copy_paste(CPU) + `--local_staging`(스테이징 병렬화 반영됨)로 수행.

---

## 4. exp4v2 ratio 스윕 (생성 후)

동일 pool을 재사용해 비율만 바꾼다. **frac/ratio 지점마다 fresh `--output_dir`** (resume 충돌 방지):

```python
for RATIO in (0.3, 0.5, 1.0):
    !python $AROMA_SCRIPTS/experiments/exp4_v2_supervised_detection.py \
        --dataset_keys severstal mvtec_leather mtd aitex \
        --condition baseline random aroma \
        --random_synthetic_dir $RANDOM_SYNTH_DIR \
        --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
        --real_data_dir        $AROMA_DATA \
        --output_dir           $EXP4_OUT/ratio_$RATIO \
        --yolo_cache_dir       $EXP4_OUT/yolo_cache \
        --synth_ratio $RATIO --seeds 42 1 2
```

> `--synth_ratio` 지정 시 `--max_synth_per_ds`는 무시된다. cap = `max(1, int(real_train × ratio))`.
> pool이 cap보다 크면 subsample, 작으면 전량(경고 로그) — **severstal이 3,000이면 ratio 1.0(2,534)까지 subsample로 커버**된다.

---

## 5. severstal 재생성 명령 (§2 권고 적용 — 유일하게 재생성 필요)

> top_k 200→1000은 **ROI 재선택부터 cascade**가 필요하다. `clean_bg_selected.json`은 roi_idx로
> 매칭되므로 ROI를 재선택하면 clean-bg도 **반드시 함께 재생성**해야 `clean_bg resolve mismatch`가
> 안 난다. **진행 중인 exp4v2 종료 후** 실행하고, severstal exp4v2는 fresh `--output_dir`로 재실행.
> step3 env(`S()`, `AROMA_SCRIPTS`, `TAU_BY_DS`, `normal_dir()`)가 세팅돼 있다고 가정.

```python
DS = "severstal"   # 이 재생성은 severstal 한정
os.environ['DS']      = DS
os.environ['PROF']    = S('profiling', DS)
os.environ['PROMPTS'] = S('prompts', DS)
os.environ['ROI']     = S('roi', DS)
os.environ['NORMAL']  = normal_dir(DS)
os.environ['COMPAT']  = f"{S('profiling', DS)}/compatibility_matrix.json"
os.environ['TAU']     = str(TAU_BY_DS[DS])        # step4c 확정 τ (재사전스캔 금지)

# ── (1) ROI 재선택: top_k 200 → 1000 (severstal = multi) ────────────────
#     roi_selected.json 덮어씀. per_pair_cap_frac 0.05 → pair당 상한 50 (1000×0.05).
!python $AROMA_SCRIPTS/roi_selection.py \
    --profiling_dir     $PROF \
    --prompts_dir       $PROMPTS \
    --sampling_strategy deficit_aware --score_mode realism \
    --top_k 1000 --img_diversity_cap 1 \
    --class_mode multi --class_floor --per_pair_cap_frac 0.05 \
    --output_dir        $ROI

# ── (2) clean-bg 재선정: 새 1000 ROI에 매칭 + random arm + pool_k 명시 ──
#     pool_k=6 (≥ n_per_roi 3) → n_per_roi 변형이 서로 다른 배경에 앉도록 보장.
!python $AROMA_SCRIPTS/clean_bg_selection.py \
    --profiling_dir  $PROF \
    --roi_dir        $ROI \
    --pool_k 6 \
    --emit_random_arm

# ── (3) AROMA arm (copy_paste): n_per_roi 3 → 1000×3 = 3,000 (목표 ≥2,534) ─
os.environ['OUT'] = S('synth_aroma', DS)
!python $AROMA_SCRIPTS/generate_defects.py \
    --roi_dir     $ROI \
    --normal_dir  $NORMAL \
    --output_dir  $OUT \
    --method      copy_paste \
    --n_per_roi 3 --seed 42 --blend_mode alpha \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0 \
    --compat_mode symmetric --compat_threshold $TAU \
    --compat_matrix_json $COMPAT \
    --local_staging

# ── (4) random arm (통제군): top_k·n_per_roi·blend 동일 (1000×3, alpha) ──
os.environ['OUT_R'] = S('synth_random', DS)
!python $AROMA_SCRIPTS/generate_random.py \
    --candidates_json $ROI/roi_candidates.json \
    --normal_dir      $NORMAL \
    --output_dir      $OUT_R \
    --top_k 1000 --n_per_roi 3 --seed 42 \
    --blend-mode alpha \
    --local_staging \
    --reject-clean-bg --min-bg-quality 0.7 --bg-blur-threshold 100.0
```

### 검증 포인트
- [ ] **cascade 순서**: (1)→(2)→(3)/(4). (2) 생략 시 `clean_bg resolve mismatch` 급증(다른 결함용 배경에 붙음).
- [ ] **출력 수 ≥ 2,534 확인**: 기존 severstal은 `--n_per_roi 3`인데도 400(=200×2)으로 나온 트림 이력이 있다. 재생성 후 `Generated N images`가 ≥2,534(cap@1.0)인지 확인 — 부족하면 top_k를 1,300~1,500으로 올려 재실행(후보 266k라 여유 충분).
- [ ] **arm 총량 parity**: (3) aroma와 (4) random의 `Generated N`이 동수.
- [ ] **blend_mode 대칭**: 위는 alpha 통일. seamless로 갈 거면 (3) `--blend_mode seamless` + (4) `--blend-mode seamless`를 **함께** 변경(`generate_random`도 `--blend-mode` 지원, default alpha).
- [ ] **재생성 후 exp4v2**: severstal만 fresh `--output_dir` → `--synth_ratio` 스윕에서 cap 2,534까지 subsample 커버.
