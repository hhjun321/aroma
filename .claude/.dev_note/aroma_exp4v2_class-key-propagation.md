# AROMA 라벨 상속성 — class 문자열 명시 전파 (경로 재파싱 의존 제거)

---

## (사용할 skills: feature-dev)

> **진단 근거**: `.claude/.etc/roi_approve/roiCheck.md` 진단 2-1(라벨 상속성). 다중-에이전트 코드 조사로 class_key 전파 단절 지점 3곳 확인 (2026-07-08).
> **관련**: [[project_multi_defect_type]](결함유형별 seed_dirs), [[aroma_exp4v2_severstal-synth-multiclass]](severstal 멀티클래스).

## 개요

멀티클래스(`--class_mode multi`) YOLO 라벨링에서 synth 결함의 클래스를 `_resolve_synth_class`가 `source_roi` 경로의 `/test/{type}/` 세그먼트를 **정규식 재파싱**해 복원한다. 원본 `defect_type` 문자열(`class_key`)은 `roi_selection.py` 후보 dict → `roi_selected.json`까지 살아있으나, `generate_defects.py`의 `annotations.json`에서 **누락**되고, 이후 오직 `source_roi`(경로)와 `cluster_id`(형태학 군집)만 남는다. 경로가 `test/{type}` 패턴이 아니거나(local_staging으로 stale해진 경로, CASDA ROI-crop 경로) severstal이 아니면 **class 0으로 강등**된다.

목표: class 문자열을 annotation **필드로 명시 전파**해 경로 재파싱 의존을 제거하되, 기존 경로 재파싱은 **fallback으로 유지**(하위호환). single 모드·구 annotations.json은 라벨 byte-identical 보존.

## 영향도 분석

### 이 기능이 변경하는 상태
- `annotations.json`에 `class_key` 필드 1개 추가 (dry-run + 실제 record 양쪽).
- multi 모드에서 클래스 해결 경로: 우선순위 0(명시 필드) → 실패 시 기존 경로 재파싱/severstal fallback.
- **YOLO 라벨(.txt) 출력**: class_key가 정확히 복원되면 기존과 동일(경로 재파싱이 성공하던 케이스). 경로 재파싱이 실패해 class 0으로 강등되던 케이스만 **올바른 클래스로 교정**됨 → 이 경우 라벨이 의도적으로 변함(버그 수정).

### 그 상태를 전제로 동작하는 기존 로직
- **`_load_synth_annotations`(exp4_v2:852) 필드 스트립 — 결정적 함정**: 로드 시 새 dict를 재구성하며 `image_path/normal_image/source_roi/mask_path/cluster_id` **5개 키만 통과**시킨다. generate_defects가 class_key를 써도 이 로더가 버려서 `_write_yolo_labels`까지 도달 못 함. **로더 패스스루가 필수** (수정 2).
- **class_mode 기본 single**(exp4_v2:3255): single이면 `cls_id=0` 고정, `_resolve_synth_class` 미호출(1153-1164). **개선 실효 범위 = multi 모드 전용**. single 실험은 annotations.json에 필드만 추가될 뿐 라벨 불변 → 재현성 보존.
- **name_to_id 정합**: `class_key`=`defect_type` 문자열. severstal은 `test/classN` 폴더명과 동일, mvtec/aitex/mtd는 raw 타입명과 동일 → `name_to_id` 키와 직접 정합, 별도 매핑 불필요.

### delete/remove/bulk 계열
- 해당 없음 (annotation 필드 추가 + 클래스 해결 우선순위 변경).

## 수정 내용

### 1. `scripts/aroma/generate_defects.py` — annotations.json에 class_key 기록

두 곳 모두 `annotations.append({...})` dict에 키 추가:
- dry-run record (근처 2351-2365)
- 실제 record (근처 2408-2426)

```python
"class_key": roi_entry.get("class_key"),
```

- aroma/random 아암: `roi_selected.json` 항목(=`roi_entry`)에 `class_key` 실재 → 값 채워짐.
- CASDA copy-paste 아암: roi_entry에 class_key 없음 → `None` → fallback이 처리.

### 2. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `_load_synth_annotations` 패스스루 (근처 852)

`valid.append({...})` dict에 추가:

```python
"class_key": e.get("class_key"),
```

- 구 annotations.json(필드 없음)은 `None` 통과 → 하위호환.

### 3. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — `_resolve_synth_class` 우선순위 0 (근처 1049-1070)

기존 (1)경로 재파싱 → (2)severstal cluster_id fallback **앞에** 삽입:

```python
# Priority 0: explicit class_key annotation field (path-reparse independent)
ck = ann.get("class_key")
if name_to_id and isinstance(ck, str) and ck in name_to_id:
    return name_to_id[ck]
```

기존 (1)(2)는 fallback으로 그대로 유지.

### 4. (선택) `scripts/aroma/casda_roi_adapter.py` — CASDA copy-paste도 채우기 (근처 599)

roi_dict에 `"class_key": f"class{class_id}",` 추가 시 CASDA copy-paste 아암도 경로 재파싱 없이 해결. **미추가 시 기존 severstal cluster_id fallback(우선순위 2)이 동작하므로 필수 아님.** severstal 외 데이터셋 CASDA 비교 계획이 있으면 추가 권장. `composed_to_exp4v2.py`(severstal 전용·cluster_id 기반)는 우선순위 2가 이미 커버 — 실익 낮음.

## 수정 대상 파일

- `scripts/aroma/generate_defects.py` (2곳: dry-run/실제 record)
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` (2곳: `_load_synth_annotations`, `_resolve_synth_class`)
- `scripts/aroma/casda_roi_adapter.py` (선택)

## 암묵 요구사항 (엣지케이스/하위호환)

- **`class_key == "_"`** (aroma defect_type 없을 때 기본값): `name_to_id`에 `"_"` 없음 → 우선순위 0 미스 → 경로 재파싱 fallback. 정상.
- **키 부재/None** (CASDA copy-paste, 구 roi_selected.json): `.get` → None → 우선순위 0 스킵. 하위호환.
- **구 annotations.json** (기존 실험 산출물): 로더 None 통과 → 경로 재파싱 → 기존과 동일 결과. **라벨 byte-identical 보존**.
- **single 모드**: `_resolve_synth_class` 미호출 → 영향 없음. single 재현성 완전 보존.
- **opt-in**: 별도 플래그 없이 자연 하위호환(있으면 우선, 없으면 fallback). annotations.json 파일 내용에 키 1개 추가되나 **라벨(.txt) 출력 불변**(재파싱 성공 케이스) → 실험 결과 재현성 보존. annotations.json 자체의 byte-identical은 깨짐(무해).

## 테스트 (Colab 검증 — 테스트 코드 작성·pytest 금지)

1. **multi 모드 교정 확인**: 경로 재파싱이 실패하던 케이스(비-test 경로/비-severstal)에서 class 0 강등 → 올바른 클래스로 교정되는지 라벨 .txt 확인.
2. **multi 정상 케이스 회귀**: 경로 재파싱이 성공하던 케이스는 우선순위 0가 동일 결과 반환 → 라벨 불변 확인.
3. **single 모드 불변**: `--class_mode single` 라벨이 기존과 동일.
4. **구 annotations.json 하위호환**: 필드 없는 구 산출물로 라벨 재생성 → 기존과 동일.

## TODO / 후속

- 필드명: `class_key`(roi_selection 컨벤션 일치) 채택. 로더에서 `e.get("class_key") or e.get("defect_type")` 별칭 수용 여부 결정.
- CASDA copy-paste(수정 4) 이번 포함 여부 — severstal 외 CASDA 비교 계획에 종속.
- annotations.json 스키마 변경을 "재현성 영향 없음(라벨 불변)"으로 처리 확정 — opt-in 플래그로 감쌀 필요는 없다고 판단.
