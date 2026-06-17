# AROMA Exp 4 — Real 이미지 로컬 캐시로 학습 I/O 개선

## (사용할 skills: micro-fix)

## 개요

Colab에서 `_prepare_ad_dataset_with_masks()`가 생성하는 symlink는 Google Drive를 가리킴.
학습 중 DataLoader가 Drive에서 이미지를 직접 읽어 I/O 병목 발생.
Real 이미지(train_normal, test_good, test_defect, masks)를 dataset당 1회 `/tmp`에 복사하고
이후 모든 조건/모델 루프는 local 경로를 사용하도록 개선.

isp_LSM_1 기준: 5243 real images × 12조건(4모델×3조건) = Drive I/O 12→1회로 단축.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- 학습 중 DataLoader I/O 경로: Drive symlink → `/tmp/aroma_exp4_cache/{ds}/` local
- 중간 상태: `/tmp/aroma_exp4_cache/` (Colab 세션 종료 시 자동 소거)

### 그 상태를 전제로 동작하는 기존 로직
- `_prepare_ad_dataset_with_masks()`: local 경로를 받아도 동일하게 동작 (symlink/copy 무관)
- `--resume`: local cache 재구축은 idempotent (파일 존재 시 skip) → resume 호환

### Synth 이미지는 대상 외
- 600장, 조건당 1회 read → Drive I/O 허용 범위
- 조건별 tmpdir에 기존 방식(symlink/copy) 유지

---

## 수정 내용

### 1. `scripts/aroma/experiments/exp4_downstream_ad.py` — `_local_cache_real_images()` 신규 함수

`_run_ad_mode()` 직전에 삽입.

- `train_normal`, `test_good`, `test_defect`, masks를 `/tmp/aroma_exp4_cache/{ds}/`에 병렬 복사
- `_copy_if_missing()`: 파일 존재 시 skip (idempotent)
- `mask_map` 재구축: 원본 Drive 경로 키 → local 경로 키로 교체 (함수 반환 lists에 반영)
- 반환: 기존 lists dict에서 경로만 local로 교체한 dict

### 2. `_run_ad_mode()` — 데이터셋 루프 내 local cache 호출

```python
# model 루프 시작 전
if os.name != "nt":
    lists = _local_cache_real_images(ds, lists, num_workers=num_workers)
```

Linux(Colab)에서만 실행. Windows 로컬 실행 시 skip.

---

## 버그 수정 (리뷰에서 발견)

### Fix 1 — mask_map 재구축 O(n²) → O(n)

```python
# 수정 전 (CRITICAL: O(n²) + duplicate path 시 mask 충돌)
for orig_p, local_p in zip(orig_td, local_td):
    mask_dst = dirs["masks"] / f"mask_{orig_td.index(orig_p):05d}.png"

# 수정 후
for i, (orig_p, local_p) in enumerate(zip(orig_td, local_td)):
    mask_dst = dirs["masks"] / f"mask_{i:05d}.png"
```

### Fix 2 — incremental save 순서 (ds 내 이전 model 결과 소실 방지)

```python
# 수정 전 (HIGH: 이전 model 결과가 저장에서 누락)
results[ds] = dict(ds_results)         # ds_results에 current model 없음
results[ds][model_name] = dict(model_results)

# 수정 후
ds_results[model_name] = dict(model_results)  # 먼저 ds_results에 반영
results[ds] = dict(ds_results)                 # 전체 저장
```

crash 후 `--resume` 시 같은 dataset의 이전 model 조건 불필요 재실행 방지.

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_downstream_ad.py`

---

## 예상 효과

| 항목 | 기존 | 개선 후 |
|------|------|---------|
| real image Drive 읽기 횟수 | 조건×모델 수만큼 | dataset당 1회 |
| isp_LSM_1 (5243 images) | 12회 Drive read | 1회 copy → 12회 local read |
| DataLoader random access | Drive ~50ms/image | local ~2ms/image |

---

## 테스트

CLAUDE.md: pytest 금지. Colab 직접 검증.

```python
# 로그에서 확인할 것
# "Local cache: copying N real images for {ds} -> /tmp/aroma_exp4_cache/{ds}"
# "Local cache ready for {ds}: {elapsed}s  (train=N test_good=N test_defect=N)"
# 두 번째 dataset부터: elapsed 시간이 첫 번째보다 짧으면 skip 동작 확인

!python $AROMA_SCRIPTS/experiments/exp4_downstream_ad.py \
    --model patchcore \
    --condition baseline \
    --dataset_keys isp_LSM_1 \
    --random_synthetic_dir $RANDOM_SYNTH_DIR \
    --aroma_synthetic_dir  $AROMA_SYNTH_DIR \
    --real_data_dir        $AROMA_DATA \
    --output_dir           $EXP4_OUT \
    --seed 42
```
