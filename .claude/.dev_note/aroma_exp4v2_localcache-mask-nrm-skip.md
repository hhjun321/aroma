# AROMA exp4v2 — LocalCache 스테이징에서 mask·nrm 복사 제거 (Drive 잔류 + key-rewrite)

## 사용할 skills: feature-dev

> **근거**: `_local_cache_for_yolo` 리팩터(mask staging 제거 + key-rewrite 유지 + nrm 조건부 스킵) + 다운스트림 mask 읽기 경로(Drive 경로 소비) 정합 확인. 단일 함수지만 로컬-캐시 계약(경로 rewrite) 변경 + 정확성 경계(label 빌드·val GT) → micro-fix보다 **feature-dev**.

---

## 개요

exp4v2 다운스트림 검출 학습은 시작 전 `_local_cache_for_yolo`(`exp4_v2_supervised_detection.py`)로 Drive 이미지를 `/tmp`로 복사한다. 실측 로그(severstal, seed 42): **10434 파일 복사에 568.1s** 소요.

복사 구성:
| 항목 | 장수 | epoch당 read | 스테이징 정당성 |
|------|------|--------------|----------------|
| real test_defect 이미지 (`td_`) | 3620 | ✅ | 정당 |
| real merged mask (`mask_`) | 3620 | ❌ (label 1회) | **불필요** |
| real train_normal (`tn_`, bg negative) | 792 | ✅ | 정당 |
| synth 이미지 (`syn_`) ×2 cond | 800 | ✅ | 정당 |
| synth normal_image (`nrm_`) ×2 | ~800 | mask 있으면 미read | **낭비** |
| synth mask (`msk_`) ×2 | ~800 | ❌ (label 1회) | **불필요** |

`/tmp` 복사 = "Drive read 1 + /tmp write 1", label 빌드가 `/tmp` read. mask는 `_mask_to_bboxes`로 **1회만** 읽혀 `yolo_cache_dir`에 label.txt로 캐시되고(YoloCache), 이후 학습 루프는 이미지+label만 읽는다. YoloCache **hit이면 mask는 0회** read. 즉 mask를 `/tmp`로 복사해도 가속할 read가 없다 → **staging이 net 손해**(write만 추가).

**수정**: merged mask(real+synth)는 `/tmp`로 복사하지 않고 **Drive 원본 경로 유지**, `mask_map` **key만 `/tmp` 이미지 경로로 rewrite**. synth `normal_image`(`nrm_`)는 **해당 annotation에 mask가 있으면 스테이징 스킵**(bbox는 mask가 authoritative, normal_image는 mask-less fallback에만 read).

**효과(장수 산수, 부하 실측 아님)**: 10434 → ~5212 복사(**~50% 감소**). 남는 바닥 = 매 epoch 읽는 이미지(td 3620 + tn 792 + syn 800). 실제 초 단축은 Colab 재실행으로 확인.

---

## 근거 코드 (확증)

- `_write_yolo_labels`(`~1140-1152`): `mask_present`면 `_mask_to_bboxes(mask_path)`가 **authoritative**; `normal_image`는 **`not mask_present`일 때만** `_extract_defect_bboxes` fallback(`1151`). AROMA/random synth는 mask 영속화 → `normal_image` 미read.
- `_split_defects`(`~195`, `~2362`): `mask_map.get(p)` — **키 존재 확인만**, 파일 미read.
- YoloCache(`~1792` hit / `~1870` built+saved): mask는 label 빌드 시 **≤1회** read, 이후 label.txt 캐시.
- **선례**: `class_mask_map`(`~1385-1398`) per-class PNG는 이미 **Drive에 두고 key만 rewrite** — 정상 동작. merged mask에 동일 패턴 적용.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `/tmp` LocalCache에 mask PNG를 더 이상 생성하지 않음. `mask_map`/synth `mask_path` **value = Drive 원본 경로**, key/`image_path`는 `/tmp`.

### 그 상태를 전제로 동작하는 기존 로직
- **label 빌드(`_build_real_labels`/`_write_yolo_labels`, `~1562`/`~1142`)**: 이제 mask를 Drive에서 read. label 빌드는 캐싱 직후(초반) 실행 → Drive 마운트 유지 시점. yolo_cache hit이면 mask read 자체 없음.
- **`mask_map` key lookup(`~2359`, `_get_real_test_images_and_labels`)**: key는 여전히 `/tmp` 이미지 경로로 rewrite되어야 lookup 성립 → **key-rewrite 유지 필수**(value만 Drive).
- **synth `mask_path` 소비(`_write_yolo_labels`, `1125`,`1140-1150`)**: Drive 경로 read로 동작. `Path(mask_path).exists()` 체크 통과 필요(Drive 유효 경로).
- **synth `normal_image` fallback(`1151`)**: mask 있는 synth는 이 분기 미도달 → `nrm_` 미스테이징이어도 무해. **mask 없는 synth(legacy/CASDA copy-paste)** 는 `normal_image`가 fallback으로 필요 → 이 경우엔 스테이징 유지해야 함.

### delete/skip 계열 — "없음(0개)" 상태
- synth annotation에 mask 없음 + normal_image 있음 → **`nrm_` 스테이징 유지**(스킵하면 bbox fallback 깨짐). 스킵 조건 = `mask_path && Path(mask_path).exists()`일 때만.
- Drive 언마운트 중 label 빌드 → mask read 실패. 완화: 빌드는 캐싱 직후 실행(마운트 유지), yolo_cache가 label 영속화. 잔여 위험은 기존 class_mask_map(Drive 잔류)과 동일 수준.

---

## 수정 내용

모두 `scripts/aroma/experiments/exp4_v2_supervised_detection.py`, `_local_cache_for_yolo`.

### 1. real merged mask — 복사 제거, key만 rewrite (`~1371-1383`)
- 현재: `mask_futs`로 `_stage_one(mask_src, mdst)` 복사 후 `local_mask_map[local_p] = str(mdst)`.
- 변경: 복사 스킵. `local_mask_map[local_p] = lists["mask_map"][orig_p]` (**Drive 원본 경로**). key는 `local_p`(=/tmp td_ 경로)로 rewrite 유지.
- `class_mask_map` 처리(`~1392-1398`)와 동일 정책(이미 value=Drive)이므로 정합.

### 2. synth `nrm_` — mask 있으면 스테이징 스킵 (`~1422-1424`, `~1440-1442`)
- 현재: `ann.get("normal_image")` 있으면 무조건 `_stage_one`.
- 변경: `has_mask = bool(ann.get("mask_path") and Path(ann["mask_path"]).exists())`. `if ann.get("normal_image") and not has_mask:` 일 때만 스테이징. mask 있으면 `new_ann["normal_image"]`는 **원본 값 유지**(또는 그대로 carry) — fallback 미사용이므로 경로 정확성 무관하나, stale 방지 위해 원본 유지 권장.

### 3. synth mask (`msk_`) — 복사 제거, Drive 경로 유지 (`~1428-1445`)
- 현재: `mask_src` 있으면 `_stage_one(mask_src, mask_dst)` 후 `new_ann["mask_path"] = str(mask_dst)`.
- 변경: 복사 스킵, `new_ann["mask_path"] = mask_src`(Drive 원본). `_write_yolo_labels`가 Drive에서 1회 read.

### 4. 로그 정합 (`~1461-1469`)
- `[LocalCache] ... staged` 카운트와 `masks=%d`가 실제 복사 수를 반영하도록 조정. mask/nrm 미복사분을 "kept-on-Drive"로 별도 표기(관측성) 권장.

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`

## 테스트 (Colab, pytest 금지 — 프로젝트 규칙)

1. **주 경로 (severstal multi, mask 존재)**: `--local_staging`(LocalCache). 로그 복사 수 10434 → ~5212 확인. label 빌드(`YoloCache built+saved`) train/val 라벨 수가 수정 전과 **동일**(2534/1086) 확인 → mask Drive-read 정합.
2. **yolo_cache hit 재실행**: 2회차에서 mask read 0회, 산출 라벨 동일.
3. **mask-less synth(legacy/CASDA)**: mask 없는 annotation에서 `nrm_` 스테이징 유지 + `_extract_defect_bboxes` fallback 정상 → bbox 생성 확인.
4. **mvtec_leather/mtd(single mode)**: mask_map Drive-잔류 + key-rewrite로 real 라벨 정상 빌드 확인.
5. **val GT**: `val map50 != 0`(로그 `val GT check`) — val 라벨이 정상 생성되는지 확인.

> 스테이징 초 단축 실측은 load-test 정책상 자동 실행 안 함 — Colab 재실행 결과로만 확인.
