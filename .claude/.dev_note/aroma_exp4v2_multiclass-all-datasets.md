# Exp4v2 — multi-class per-class 측정을 확정 4종 전체로 일반화

## (사용할 skills: feature-dev)

> 단일 파일(`exp4_v2_supervised_detection.py`)이지만 여러 함수(class 열거·mask 해소·YAML 빌더·gate)에 걸친 50줄+ 변경 + 설계 결정(클래스 정의 정본·id 안정성) 포함 → feature-dev.
> 계열: [[aroma_exp4v2_per-class-metrics]](결과 추출/집계 배관 — 이미 구현), [[aroma_multidomain_aitex-mtd-integration]](데이터셋 통합 — 이미 구현), [[aroma_research-core_self-contained-multidomain-design]](breadth 목표).

## 개요

`dataset_config.json`은 확정 4종(severstal·mvtec_leather·aitex·mtd) **모두** `class_mode:"multi"`로 선언돼 있으나, `exp4_v2_supervised_detection.py:2108`이

```python
ds_class_mode = class_mode if ds == "severstal" else "single"
```

로 **`--class_mode multi`를 severstal에만 적용**한다. 나머지 3종은 플래그를 줘도 single(`nc=1 'defect'`)로 강제되어 per-class AP가 관측되지 않는다.

연구 방향이 **4종 모두 per-class 측정 필요**로 개선됨(rare-class별 AROMA 기여 = 연구 crux, severstal c2뿐 아니라 aitex 희소코드·mtd fray 등에서도 확인 필요). 이에 multi-class 학습 경로를 데이터셋-일반적으로 리팩터링한다.

**이미 있는 것 (재사용)**: per_class 결과 추출부(`_run_yolo_condition`)는 `val_results.names` 기반이라 YAML에 names만 동적으로 채우면 **추가 4종 지원이 거의 자동**(fallback만 severstal 전용). 데이터셋 통합(config 엔트리·prepare·mask 해소)도 완료.
**없는 것 (본 작업)**: (1) gate가 severstal 하드코딩, (2) YAML 빌더가 `nc=4`/`c1~c4` 하드코딩, (3) real/synth 라벨의 class id가 severstal 전용(0 또는 class{N}-1), (4) 클래스 열거 로직 부재.

### ★ 코드 조사로 확정된 핵심 메커니즘 (2026-07-03)

라벨의 class id는 두 경로에서 나오며, **둘 다 `test/{type}` 폴더명으로 일반화 가능**:

- **real 라벨** (`_get_real_test_images_and_labels`): 결함 이미지는 `test/{type}/{stem}` 아래 존재 → 그 이미지의 **부모 폴더명 = type = 단일 클래스**. 현재 generic은 merged mask + `class_id=0`(:1411) 하드. severstal만 `class_mask_map`(1-based `class{c}`, :1383 `cls-1`)로 다중클래스.
- **synth 라벨** (`_write_yolo_labels` → `_resolve_synth_class`): 합성 annotation의 `source_roi`가 **원본 real 결함 경로**(`.../test/{type}/{stem}.jpg`, staging 무관 원본 유지 — generate_defects.py:1171,1216)를 가리킴. 현재 `.../class{N}/` 세그먼트만 파싱(severstal). → **`test/{type}` 세그먼트 파싱 + `name_to_id` 매핑으로 severstal 포함 전 데이터셋 통합**.

annotation 스키마(실측): `{image_path, source_roi, image_id, normal_image, cluster_id, cell_key, prompt, method, blend_mode, roi_score, deficit, mask_path, bbox, dry_run}`. **`defect_type`/`class` 필드는 없음** → 클래스는 `source_roi` 경로 파싱이 유일·정본 경로(cluster_id는 aroma에선 morphology 클러스터라 class 아님, CASDA-severstal에서만 class=cluster_id).

---

## 클래스 정의 정본 (데이터셋별)

| 데이터셋 | 클래스 수 | 클래스 | mask 레이아웃 | 정본 소스 |
|----------|----------|--------|--------------|----------|
| severstal | 4 | class1~4 | `masks/class{c}/{stem}.png` (이미지당 다중 클래스 가능) | `range(1,5)` (기존) |
| mvtec_leather | 5 | color/cut/fold/glue/poke | `ground_truth/{type}/{stem}_mask.png` | 디스크 `test/{type}` |
| aitex | 12 | 002/006/010/016/019/022/023/025/027/029/030/036 | `ground_truth/{ddd}/{stem}_mask.png` | `aitex_manifest.json` `defect_codes` |
| mtd | 5 | blowhole/break/crack/fray/uneven | `ground_truth/{class}/{stem}_mask.png` | `mtd_manifest.json` `classes` |

- **일반(generic) 데이터셋**(leather/aitex/mtd): 결함 이미지는 **정확히 한 개** `test/{type}` 폴더에 속함 → 이미지당 클래스 1개. `_resolve_masks_generic`가 반환하는 merged mask가 곧 그 단일 클래스의 mask.
- **severstal**: 이미지당 다중 클래스 가능(`class{c}` 별도 mask) → 기존 특수 경로 유지.

### 설계 결정 — 클래스 열거 순서(= YOLO class id)  ★확정 (Colab 실측 2026-07-04)

**열거는 `test_defect` 경로의 부모 폴더에서 뽑는다** — `{ds}/test` 추측 스캔 금지.

> **실측 교훈**: 검증 셀이 `{AROMA_DATA}/mvtec_leather/test`를 스캔해 nc=0(빈 리스트)이 나왔다. 실제 디스크 경로는 `{AROMA_DATA}/mvtec/leather/test`(mvtec 하위 중첩) — **데이터셋 키 ≠ 디스크 경로**. `_get_image_lists`는 이미 데이터셋별 경로 규칙(isp/mvtec/visa/severstal/aitex/mtd)을 해석해 `test_defect` 절대경로를 반환하므로, 그 경로들의 부모 폴더명에서 클래스를 뽑으면 경로 규칙 문제가 전부 사라진다.

1. **열거 = `sorted(set(Path(p).parent.name for p in test_defect_paths))`**. `_get_image_lists`가 해석한 실제 결함 경로 기반 → mvtec_leather 포함 전 데이터셋 정확.
2. class id = 정렬 리스트의 **0-based index**. 데이터셋당 **1회 계산해 real/val/합성 라벨 전 경로에서 공유** — id drift 방지(load-bearing).
3. **`sorted()`가 4종 모두 자연 순서와 일치**(확인): severstal `class1..4`→0~3(기존 `class{c}-1`과 동일 ★회귀 0), aitex 3자리 zero-pad `002..036` 사전순=수치순, mtd/leather 알파벳순.
4. per_class 출력은 **이름 기반**이라 id 순서 무관(reporting 불변).
5. **synth `source_roi` 파싱과 정합**: source_roi = `.../{디스크경로}/test/{type}/{stem}` → 정규식 `[\\/]test[\\/]([^\\/]+)[\\/]`로 뽑은 `{type}`이 위 부모폴더 집합과 동일 → `name_to_id` 매핑 100% 성립(severstal/aitex/mtd 실측 파싱=폴더일치=n 확인).

> ⚠️ **id 안정성**: 열거는 `_get_image_lists`가 반환한 **전체 결함 경로(test_defect)의 부모폴더 집합** 기준으로 데이터셋당 1회 고정. val GT에 없는 클래스가 학습셋에만 있어도 class 리스트/id 불변 → seed·split·resume 재현성 유지.

### 적용 범위 — 데이터셋-일반 capability (4종 하드게이트 아님)

multi-class는 특정 데이터셋에 매인 기능이 아니라 **`test/{type}` 구조를 가진 모든 데이터셋에 동작하는 일반 기능**으로 구현.

- `--class_mode multi` → 실행에 포함된 **모든 데이터셋**에 적용(4종뿐 아니라 mvtec_cable 8종·isp area/points·visa 등도 그 자리에서 per-class).
- **유일한 특례 = severstal**: `masks/class{c}`(이미지당 다중 클래스). 그 외 전부 generic(`test/{type}` 부모폴더 = 이미지당 단일). 특례는 `dataset_key=="severstal"`로 **mask 해소 단계에서만**, gate/열거/라벨은 공통.
- **단일 test 폴더**(visa `anomaly`, toothbrush `defective`): nc≤1 → single 축퇴 → byte-identical.
- **연구 범위(실행 대상)는 확정 4종 그대로**. 본 일반화는 코드가 4종에만 하드-게이트되지 않게 하는 것.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `exp4_v2_supervised_detection.py`
  - gate(:2108): severstal 하드코딩 + config 게이트 제거 → `ds_class_mode = class_mode`(플래그 종속, 전 데이터셋).
  - 클래스 열거: **신규 헬퍼** `_enumerate_defect_classes(test_defect_paths)` → `(class_names, name_to_id)`.
  - real 라벨: generic은 merged-mask 분기에서 `class_id=name_to_id[부모폴더]`(class_mask_map 신설 안 함, Option A). severstal은 기존 class_mask_map 경로 유지.
  - YAML 빌더(:1440~): `nc`·`names` 동적화.
  - multi 라벨 생성(:969~/:1717~): class id를 동적 매핑으로.

### 그 상태를 전제로 동작하는 기존 로직 (하위호환 / 깨지면 안 됨)
- **severstal multi 경로 무회귀**: `_resolve_severstal_masks`의 `class{c}` 레이아웃·이미지당 다중클래스 유지. 일반화 후에도 severstal 결과가 기존과 동일해야 함(회귀 테스트 필수).
- **`--class_mode` 미지정(기본 single) 시 전 데이터셋 byte-identical**: gate `ds_class_mode=class_mode`라 플래그가 single이면 모든 데이터셋이 기존 single 경로. (단 `--class_mode multi`를 주면 mvtec_cable 등도 이제 multi로 전환됨 — 의도된 동작, §적용 범위.)
- **per_class 추출/집계**([[aroma_exp4v2_per-class-metrics]]): `val_results.names` 우선이라 동적 names로 자동 대응. 단 fallback `{0:c1..3:c4}`는 severstal 전용 → **generic에서 names 누락 시 오작동 가능** → names를 YAML/모델에 확실히 싣고, fallback도 방어적으로.
- `plot_exp4v2_results.py`·reader: `results[ds][model][cond]` 스칼라 + `per_class` 키만 읽음 → 클래스 수·이름이 데이터셋마다 달라도 안전(비파괴적).

### 회귀 위험 지점
- **nc 변동**: aitex nc=12(희소 코드 다수, 019=38 vs 027/029/036=1) → 극심한 불균형. 학습은 되나 희소 클래스 AP가 0/노이즈일 수 있음(정직히 보고, per_class에 등장분만).
- **class id ↔ name 매핑 불일치**: 열거 순서가 train/val/합성 라벨에서 달라지면 라벨 오염 → 전 경로 단일 소스 공유로 차단.
- **열거 = test_defect 부모폴더 기준**: prepare가 일부 코드를 skip해도 `_get_image_lists`가 반환한 실재 결함 경로에서만 클래스가 나오므로 항상 정합(manifest 대조 불요). aitex 실측 nc=12 전부 존재 확인.

---

## 수정 내용

### `scripts/aroma/experiments/exp4_v2_supervised_detection.py`

#### 1. gate 정정 (:2108) — 플래그 종속(전 데이터셋 일반), config 게이트 제거

```python
# 기존: ds_class_mode = class_mode if ds == "severstal" else "single"
ds_class_mode = class_mode   # --class_mode 를 전 데이터셋에 동일 적용
```
- **severstal 하드코딩 + config 하드게이트 둘 다 제거.** multi-class는 데이터셋-일반 capability(§"적용 범위" 참조).
- `--class_mode` 기본값 `single` 유지 → 플래그 미지정 시 전 데이터셋 기존과 byte-identical(회귀 0).
- 실제 열거 결과 nc≤1이면(단일 test 폴더 데이터셋) `_enumerate_defect_classes`가 single로 축퇴 → 안전.
- config `class_mode:multi`는 하드게이트에서 **문서/힌트로 격하**(코드가 참조하지 않음).

#### 2. 신규 — `_enumerate_defect_classes(test_defect_paths)`

반환 `(class_names: List[str], name_to_id: Dict[str,int])`. 로직:
- `class_names = sorted({Path(p).parent.name for p in test_defect_paths})` — `_get_image_lists`가 반환한 실제 결함 경로의 부모 폴더(= defect type). 디스크 경로 추측 없음(mvtec_leather=`mvtec/leather` 등 경로 규칙 무관).
- severstal → `["class1".."class4"]`(id 0~3 = 기존 `class{c}-1`과 동일 ★회귀 0), aitex → 12 code, mtd/leather → 5.
- id = index. `name_to_id`로 역매핑. manifest 미사용(런타임 부재·불필요 확정).
- **nc≤1 이면 single 축퇴**(단일 test 폴더 데이터셋: visa `anomaly`, mvtec_toothbrush `defective` 등 → `defect` 단일 클래스로 byte-identical) + 경고.
- `_get_image_lists` 내부에서 multi일 때 계산해 반환 dict(`class_names`,`name_to_id`)에 실어 하위 전 경로 공유.

#### 3. YAML 빌더 (:1440~) — 동적 nc/names (+ severstal 표시명 매핑)

```python
# 기존: multi → nc=4, names=['c1','c2','c3','c4']
if class_mode == "multi" and class_names:
    disp = _display_class_names(dataset_key, class_names)  # severstal→[c1..c4], 그외 폴더명
    nc = len(disp)
    names_line = "names: [" + ", ".join(f"'{n}'" for n in disp) + "]\n"  # ★따옴표 필수
else:
    nc, names_line = 1, "names: ['defect']\n"
```
- ⚠️ **YAML 따옴표 필수**: aitex 코드 `002` 등이 unquoted면 YOLO가 정수(2)로 파싱 → 반드시 `'002'`로 렌더. 전 클래스명 quote.
- `class_names`·`dataset_key`를 빌더까지 인자로 전달.
- 신규 `_display_class_names(dataset_key, class_names)`: `["c%d"%(i+1) for i in range(len)]` if severstal else `list(class_names)`.

#### 4. real 라벨 class id — generic은 부모폴더 매핑 (Option A, 단순)

**class_mask_map을 generic에 신설하지 않는다**(regression 면적 최소화). `_get_real_test_images_and_labels`의 merged-mask 분기(:1397~1411)에서, multi + generic일 때 `class_id = name_to_id[Path(img_p).parent.name]`로 부여(현재 하드코딩 `class_id=0` → 동적). generic은 이미지당 단일 클래스라 merged mask 하나면 충분.
- **severstal 경로 무수정**: `class_mask_map`(1-based, :1383 `cls-1`) 그대로 유지 → severstal 회귀 0.
- `name_to_id`를 함수 인자로 전달(기본 None → 기존 `class_id=0` 동작 = single/타 데이터셋 byte-identical).

#### 5. synth 라벨 class id — `_resolve_synth_class` 일반화

```python
def _resolve_synth_class(ann, name_to_id, dataset_key):
    src = str(ann.get("source_roi") or ann.get("normal_image") or "")
    m = re.search(r"[\\/]test[\\/]([^\\/]+)[\\/]", src)   # test/{type}
    if m and m.group(1) in name_to_id:
        return name_to_id[m.group(1)]                     # severstal: class1→0 포함 통합
    if dataset_key == "severstal":                         # CASDA fallback (severstal 한정)
        try:
            cid = int(ann.get("cluster_id")) - 1
            if 0 <= cid <= 3: return cid
        except (TypeError, ValueError): pass
    return None   # 미해소 시 호출부에서 0 폴백 + 경고
```
- 기존 `_parse_severstal_class`(`.../class{N}/` 파싱)는 `test/{type}` 파서가 흡수(severstal test 폴더 = `test/class{N}`) → 제거하거나 내부 재사용.
- `_write_yolo_labels` 시그니처에 `name_to_id`/`dataset_key` 전달.

#### 6. per_class fallback 방어 ([[aroma_exp4v2_per-class-metrics]] §1)
- 추출부 fallback `names or {0:"c1",...}`을 generic 대응으로: `val_results.names` 우선, 없으면 `{i:n for i,n in enumerate(class_names)}`. severstal 전용 c1~4 하드 fallback 제거.

---

## 수정 대상 파일

- `scripts/aroma/experiments/exp4_v2_supervised_detection.py`
  - :2108 gate (config 기반) / 신규 `_enumerate_defect_classes` / `_get_image_lists`가 class_names·name_to_id를 lists에 실어 전달 / :1440 YAML 동적 nc·names / :1411 real 라벨 class_id(generic 부모폴더 매핑) / `_resolve_synth_class` + `_write_yolo_labels` 일반화(:936,:1024) / per_class fallback(:? per-class-metrics 도입부)
- (문서) `AROMA연구분석/colab_execute/exp4v2_execute.md` — ⚠️ class_mode 섹션·부록 A를 "4종 multi 지원"으로 갱신(구현 완료 후)

> **manifest 위치**: 정본 `aitex_manifest.json`/`mtd_manifest.json`은 현재 `.claude/.etc/dataset_config/`(레퍼런스 업로드). 런타임에는 prepare 스크립트가 데이터셋 디렉토리(`{real_data_dir}/{ds}/`)에 생성 → 스크립트는 데이터셋 디렉토리에서 조회, 없으면 디스크 폴더 스캔 fallback.

---

## 암묵적 요구사항 (엣지)

- **클래스 0개**(test 폴더 없음/빈 경우): generic 열거가 빈 리스트면 해당 데이터셋 multi 불가 → single로 폴백 + 경고 로그(silent 금지).
- **manifest ↔ 디스크 불일치**: 디스크 실재 폴더 우선, manifest는 순서만. 디스크에 있으나 manifest에 없는 코드 → 리스트 뒤에 `sorted()` 부착.
- **희소 클래스(aitex 027/029/036=1장)**: 학습/val split에서 한쪽에만 존재 가능. id는 고정 유지, per_class엔 val 등장분만(강제 0-채움 금지 — "검출 0"과 "GT 없음" 구분 보존).
- **severstal 회귀**: 일반화 후 severstal multi 결과가 기존 c1~4 수치와 동일해야 함(id 매핑·mask 경로 불변 확인).
- **resume 호환**: nc/names는 데이터셋 정의 기준 고정이라 seed/split 무관 → 기존 resume 판정(`map50` 유효성) 불변.
- **imgsz 32배수**: 기존 제약 유지(무관).

---

## 테스트 (Colab 전용, 새 테스트코드·pytest 금지)

- 로컬 정적: `python -m py_compile scripts/aroma/experiments/exp4_v2_supervised_detection.py`.
- Colab 스모크(`--class_mode multi --baseline_epochs 1`):
  1. **mtd**: `per_class`에 blowhole/break/crack/fray/uneven(등장분) 존재, YAML `nc=5`·names 정상.
  2. **aitex**: `nc=12`, per_class에 등장 defect code, 희소 코드 노이즈 허용.
  3. **mvtec_leather**: `nc=5`(color/cut/fold/glue/poke).
  4. **severstal 회귀**: 기존 c1~4 수치와 동일(중대 확인).
  5. **single/config-non-multi(mvtec_cable)**: per_class 키 부재 = byte-identical.
- macro 정합성: per_class map50들의 평균 ≈ top-level `map50`(반올림 오차 내).

---

## 미확정 사항

### 해소됨 (코드 조사, 2026-07-03)
- ~~**합성 라벨의 class id**~~ → **해소**: annotation `source_roi`가 원본 `test/{type}` 경로를 담음(스키마·generate_defects.py:1171,1216 확인). `test/{type}` 파싱 + name_to_id로 severstal 포함 통합(수정 #5). `defect_type`/`class` 필드는 없으나 불필요.

### Colab 실측 완료 (2026-07-04)
1. ~~**generic source_roi 실측**~~ → **확정**: severstal aroma/random 5070, aitex 600, mtd aroma494/random464 **모두 파싱=폴더일치=n(미파싱 0)**. `test/{type}` 파싱 메커니즘 전 데이터셋 성립.
2. ~~**aitex 실재 test/{ddd}**~~ → **nc=12 전 코드 존재**(002/006/010/016/019/022/023/025/027/029/030/036). 1장짜리 코드도 폴더 유지.
3. ~~**manifest 위치**~~ → severstal X / aitex·mtd O로 혼재 → **manifest 미사용 확정**(열거는 test_defect 부모폴더). mtd nc=5, leather는 경로중첩(`mvtec/leather`)이라 test_defect 기반 열거 필수(위 설계 §실측 교훈).
4. **⚠️ 데이터-state**: mvtec_leather **aroma 합성 미생성**(`annotations.json 없음`). 전체 multi run 전 leather aroma 합성 생성 필요(exp2/exp3 합성 단계) — 코드 blocker 아님, 데이터 준비 항목.

### 구현 중 확정
4. **aitex nc(≤12) 학습 안정성**: 극단 불균형 + 일부 1장 → 희소 클래스 사실상 미학습. per_class 등장분만 정직 보고. code→명칭 매핑 표기는 breadth 보고용 별도.
5. **class 열거 전달**: `_get_image_lists`에서 1회 계산해 lists dict(`class_names`, `name_to_id`)에 실어 real/synth/YAML 전 경로 공유(권장) — id drift 차단.
6. **severstal name 표기** → ★확정 (사용자 결정 2026-07-04): **`c1~c4` 유지**(per_class-metrics 정합·기존 severstal 결과 호환). 구현: `name_to_id`(source_roi 파싱·real 폴더→id)는 **폴더명(class1~4) 기반 유지** → id 정합 불변. **display 이름 분리** — 신규 `_display_class_names(dataset_key, class_names)`: severstal이면 `[c1..c4]`(id 순서 동일), 그 외 폴더명 그대로. 이 display 이름을 **`_write_yolo_yaml`의 `names:`에만** 사용 → `val_results.names`→per_class 키가 severstal에서 c1~c4로 나옴. generic은 폴더명 그대로(color/blowhole/019…).
   - downstream 하드코딩 없음 확인(plot_exp4v2_results.py·기타 reader에 `c1~c4` 미참조, 변경 4줄 :1450/:1989에만 존재) — (b) 선택은 기존 저장 결과 호환 목적.
