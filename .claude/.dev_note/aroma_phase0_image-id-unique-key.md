# AROMA Phase 0 — distribution_profiling image_id 고유키 + context 실제 dim 컬럼

## 사용할 skills: micro-fix

> 이 dev_note는 **두 개의 독립적이나 같은 워커·같은 재실행을 공유하는 phase0 수정**을 담는다.
> - **작업 A** — `image_id` 클래스-충돌 고유키 수정 (아래 §개요·§수정내용 1·2)
> - **작업 B** — `context_features.csv`에 실제 이미지 dim(`image_w`/`image_h`) 컬럼 추가 (§작업 B)
>
> 둘 다 `_context_worker`를 건드리고, 둘 다 4종 phase0 전체 재실행에 흡수된다. 함께 머지·재실행하면 1회 재생성으로 둘 다 반영된다.

---

## 개요 (작업 A — image_id 고유키)

MVTec 계열(`mvtec_leather`)은 `defect_type` 하위폴더(color/cut/fold/glue/poke)마다 동일 stem(`000`,`001`…)을 가진 파일이 존재한다. `distribution_profiling.py`가 `image_id = image_path.stem`으로 잡아(morph worker ~330, context worker ~380) **클래스 간 stem이 충돌**한다. `step5_morphology_clustering`(~926 `self.cluster_assignments[row["image_id"]] = ...`)이 image_id를 dict 키로 쓰므로 **last-wins 덮어쓰기** → leather 92개 결함이 19개로 붕괴. GMM은 92행에 적합(22/46/24=92)되나 저장된 per-instance assignment는 19개로 붕괴.

결과: (a) leather 비지도 클러스터링이 사실상 degenerate, (b) `roi_selection.py`의 `assignments.get(image_id)`가 동일 stem 5개 결함에 **단일 클러스터**만 반환 — 실제 파이프라인 결함(leather ROI 선정이 잘못된 클러스터 사용). 논문 재포지셔닝 검증(`AROMA연구분석/class_vs_cluster_validation_20260711.md`)에서 leather V1/V2 측정 불가로 발견됨.

수정: image_id를 `f"{defect_type}_{image_path.stem}"` 고유키로 생성(morph·context 두 워커 동일 규칙). mtd/severstal/aitex는 stem이 이미 고유라 논리 영향 없으나 image_id 문자열이 바뀌므로 4종 프로파일링 재생성 필요 — **어차피 `colab_execute_new` 전체 재실행 예정이라 이 재생성은 그 안에 흡수됨**(구/신 profiling 혼용 위험 없음).

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `morphology_features.csv` / `context_features.csv`의 `image_id` 문자열 값 (전 데이터셋)
- `morphology_clusters.json` `cluster_assignments`의 키 및 **길이**(leather 19→92; mtd/severstal/aitex 불변)
- `compatibility_matrix.json`의 `matrix`/`P_def_patch`(leather는 클러스터 조인이 정상화되어 값이 바뀜)

### 그 상태를 전제로 동작하는 기존 로직
- `_build_symmetric`(~585 `cluster_assignments.get(r.get("image_id"))`) — context patch를 소스 이미지 클러스터에 조인. **morph/context가 글자 단위 동일 키를 emit해야 유지됨(락스텝 불변식).**
- `roi_selection.py` `assignments.get(image_id)`(~353) — 후보의 cluster_id 결정. leather는 현재 붕괴된 매핑을 쓰므로 이 수정으로 **정상화**(회귀가 아니라 결함 해소).
- `context_features.csv` image_id 기반 P(cell|cluster) 집계 — 동일 고유키로 정합.

### delete/remove/revoke/bulk 계열 확인
- 해당 없음(파괴적 작업 아님). 단 **image_id 스키마 변경**이므로 기존 커밋 profiling과 문자열 불일치 → 전체 재생성 필요(아래 롤아웃).

---

## 수정 내용

### 1. `scripts/distribution_profiling.py` — `_morph_worker` (~330)

```python
# 현재
"image_id": image_path.stem,
# 수정 후
"image_id": f"{defect_type}_{image_path.stem}",
```
(`defect_type`은 해당 워커 스코프에 이미 존재 — task["defect_type"])

### 2. `scripts/distribution_profiling.py` — `_context_worker` (~380)

```python
# 현재
"image_id": image_path.stem,
# 수정 후
"image_id": f"{defect_type}_{image_path.stem}",
```
**두 곳을 반드시 동일 규칙으로** — 한쪽만 바꾸면 `_build_symmetric`(~585) 조인이 전부 깨진다.

### 구분자·규칙 근거
- `{defect_type}_{stem}`는 이미 이 스크립트가 마스크를 `defect_masks/{defect_type}_{p.stem}.png`(~730)로 저장하는 **기존 unique 네이밍 컨벤션**과 정합 → image_id와 mask 파일명이 논리적으로 정렬됨. 신규 위험 미도입.
- 파싱 위험: `scripts/` 전체 grep 결과 profiling image_id를 되파싱(split/replace/경로 재구성)하는 코드 **없음**. 유일 히트 `prepare_severstal.py:243 Path(image_id).stem`은 데이터 준비 단계의 파일명 로컬 변수로 **별개 스코프**(profiling 조인키와 무관) — composite id가 다운스트림 파서를 깨뜨릴 위험 없음.

---

## 수정 대상 파일

- `scripts/distribution_profiling.py` — `_morph_worker`, `_context_worker` 각 1줄 (총 2줄)
- (재생성 산출물, 코드 아님) 4종 `morphology_*`/`context_features.csv`/`compatibility_matrix.json` — `colab_execute_new/phase0_execute.md` 재실행으로 생성
- (선택) `colab_execute_new/phase0_execute.md` — 필요 시 image_id 스키마 변경 메모 추가

---

## 작업 B — context_features.csv 실제 이미지 dim 컬럼 추가

### B-0. 개요/동기

`clean_bg_selection.py`의 `_image_dim()`은 `context_features.csv`의 `patch_xy`(64px stride 타일 top-left) 최대값 + tile로 이미지 dim을 **추정**한다. `_context_worker`의 타일링이 `for i in range(h // gs)`(~368) 즉 **비-overlap truncate**라, 이미지 우/하단 부분 타일(0~63px)이 잘려 실제보다 **항상 작게** 추정된다.

측정(로컬, mtd good 956장): `patchgrid − actual` → **879/956 이미지에서 과소추정, mean −31~−32px, max −63px**(과대추정 0건, 단방향).

이 dim은 clean_bg의 두 소비 지점 기준이 된다:
- **(a) 옵션1 size-fit 신호** — `_scale_to_fit`의 fit-rescale 계수(crop↔bg 면적비) → scale_factor.
- **(b) 옵션2/Phase3 기하 위치** — `_place_position`/`_effective_wh`의 좌표 계산.

과소추정은 **clamp-safe**(위치가 실제 이미지 안쪽에 보수적으로 들어가 generation clamp를 유발하지 않음)이나, **edge-flush 위치가 실제 가장자리보다 ~30px 안쪽**에 놓여 geometry prior(E2) 품질을 저하시킨다. 정밀 해소 = phase0가 실제 dim을 방출하면 clean_bg가 patch 격자 추정 대신 실제값 사용. (참조: `aroma_step3_5_clean-bg-selection.md`, clean_bg 커밋 `b2ebd76`의 "알려진 한계".)

### B-1. `scripts/distribution_profiling.py` — `_context_worker` (~380) dim 방출

`_context_worker`는 이미 이미지를 열어 `h, w = img.shape`(~361)를 보유 — **재읽기 없이** patch row에 실어 방출.

```python
# 현재 (~380)
feats.update({
    "image_id": f"{defect_type}_{image_path.stem}",
    "image_type": image_type,
    "patch_xy": f"{x1}_{y1}",
})
# 수정 후 — 실제 이미지 dim(px) 두 컬럼 추가 (같은 image_id 모든 패치 row에 동일값)
feats.update({
    "image_id": f"{defect_type}_{image_path.stem}",
    "image_type": image_type,
    "patch_xy": f"{x1}_{y1}",
    "image_w": w,
    "image_h": h,
})
```

- CSV 컬럼 순서: 기존 컬럼(`image_id,image_type,patch_xy,local_variance,edge_density,texture_entropy,frequency_energy,orientation_consistency`) **뒤에** `image_w,image_h` 추가(append). CSV writer가 `feats` dict 키 순서/`fieldnames`를 어떻게 잡는지 확인해 신규 컬럼이 헤더에 포함되도록 할 것(구현자: writer의 fieldnames 소스 확인).
- `w`/`h`는 `img.shape`가 `(h, w)`이므로 **image_w=w, image_h=h**(축 혼동 주의).

### B-2. `scripts/aroma/clean_bg_selection.py` — `_image_dim()` 실제값 우선 + fallback

`_image_dim(rows)`이 row에 `image_w`/`image_h`가 있으면 **실제값을 우선 사용**, 없으면(구 profiling 산출물) 현행 patch_xy max+tile 추정으로 fallback:

```python
# 우선 경로: 첫 유효 row의 image_w/image_h가 숫자면 그대로 반환
# fallback: 현행 max(patch_x)+tile, max(patch_y)+tile
```

- **하위호환**: 컬럼 없는 기존 CSV도 그대로 동작(추정 fallback). 신규 재생성분은 실제값 사용.
- 적용 범위: `good_dim`(good 이미지)과 `src_dim_by_img`(결함 소스 이미지, `defect_by_img` 기반) **양쪽 경로 모두** `_image_dim`을 거치므로 한 함수 수정으로 둘 다 반영됨.
- `image_w`/`image_h`가 결측·비숫자·0이면 fallback(엣지 (1)).

### B-3. 영향도 (작업 B)

- **변경 상태**: `context_features.csv` 스키마에 2개 컬럼 추가(append-only).
- **전제 로직**: `context_features.csv`를 읽는 소비처가 신규 컬럼에 깨지지 않아야 함. `csv.DictReader` 사용처(clean_bg, 기타)는 컬럼 추가에 **안전**(키 접근, 위치 무관). 구현자: `context_features.csv`를 `read_csv`/수동 인덱스로 파싱하는 곳이 없는지 최종 grep.
- **clean_bg 산출 변화**: 실제 dim 사용 시 scale_factor·position·edge-flush가 실제 가장자리 기준으로 재계산됨 → E2 geometry prior 품질 개선(회귀 아님).

## 롤아웃 (colab_execute_new 전체 재실행 맥락)

- 사용자가 `colab_execute_new`를 **전체 재실행** 예정 → image_id 수정을 **재실행 전에** 머지하면 phase0→step1~5→exp가 처음부터 교정된 고유키로 일관 생성됨.
- 따라서 "4종 profiling 재생성"은 별도 비용이 아니라 전체 재실행에 흡수되고, **구/신 profiling 혼용(drift-mixing) 위험 없음**. phase0 STEP1 백업·STEP3-3 drift 체크는 "이전과 다름" 확인 용도로만 유지.

---

## 테스트 / 검증

CLAUDE.md 정책: pytest 금지 — Colab phase0 재실행 + 로컬 재측정 셀.

1. **phase0 재실행** (4종, drift check STEP1 backup / STEP3-3 병행)
2. **(a)** `mvtec_leather/morphology_clusters.json`의 `cluster_assignments` 길이 == **92** (현재 19)
3. **(b)** leather V1 cluster↔class ARI 재계산 가능 (로컬 복구값 ARI≈0.21 참고; 현행 코드 재적합은 n_clusters=4로 커밋본 3과 drift → 재생성으로 통일)
4. **(c)** leather V2 `I(cell;class)`/`I(cell;cluster)` 산출 가능해짐 (현재 context_features image_id도 stem-충돌로 차단)
5. **(d)** mtd/severstal/aitex — image_id **문자열만** 변경, 클러스터 구조·`cluster_assignments` 길이 불변(회귀 없음) 확인

### 작업 B 검증

6. **(e)** 재생성된 `context_features.csv` 헤더에 `image_w,image_h` 존재, good 이미지 행에 실제 픽셀 dim 기록.
7. **(f)** clean_bg 재실행 시 `_image_dim`이 실제값 사용 → mtd 로컬 `patchgrid − actual` 갭(mean −31px)이 **0으로 수렴**, edge-flush 위치가 실제 가장자리(margin_frac 밴드)에 붙는지 로컬 재측정(E2 prior 대비).
8. **(g)** 구 profiling(컬럼 없음)으로 clean_bg 실행 시 **fallback 추정으로 무오류 동작**(하위호환).
9. **(h)** `scale_factor`가 실제 dim 기준으로 재계산되어 옵션1 size-fit 신호가 실제 면적비 반영.

---

## TODO / 미확정

- **파싱 위험 최종 재확인(구현자)**: `scripts/` 외부(`roi_selection` 소비처, `colab_execute_new/` 노트북)에서 image_id를 `split`/`Path(...)`로 되파싱하거나 경로 재구성하는 곳이 없는지 최종 grep.
- **구분자 방어**: 향후 image_id 파싱 코드가 생기면 `defect_type`/stem 내 `_`로 경계 모호 가능 → 그때 `rsplit("_", 1)` 등 방어 규칙 별도 결정. (현재는 파싱 없음 → 무해)
- **drift 문서 갱신**: leather n_clusters 3→4 통일 시, 커밋된 3-클러스터 결과에 의존하는 문서/분석이 있으면 함께 갱신할지 확인.
- **(작업 B) CSV writer fieldnames**: `context_features.csv`를 쓰는 지점이 `feats` dict 키에서 fieldnames를 유도하는지, 고정 리스트인지 확인 후 신규 2컬럼이 헤더에 포함되게 조정.
- **(작업 B) `_morph_worker` dim 불요**: 실제 dim은 clean_bg 위치/스케일에만 쓰이고 이는 context 경로(`good_dim`/`src_dim_by_img`)에서만 소비 → `morphology_features.csv`에는 dim 컬럼 **불필요**(추가하지 않음). 필요성 재확인.
- **(작업 B) morph defect_bbox 좌표계**: Phase3 `_class_edge_prior`가 `iid_to_bbox`(morphology의 `defect_bbox`)와 `src_dim_by_img`(context 추정 dim)를 조합 — dim이 실제값으로 바뀌면 이 prior의 edge/span 분류도 실제 기준으로 정정됨(개선). morph bbox는 원본 픽셀 좌표라 정합 확인.
