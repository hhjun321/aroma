# AROMA Phase 0 — distribution_profiling image_id 클래스-충돌 고유키 수정

## 사용할 skills: micro-fix

---

## 개요

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

---

## TODO / 미확정

- **파싱 위험 최종 재확인(구현자)**: `scripts/` 외부(`roi_selection` 소비처, `colab_execute_new/` 노트북)에서 image_id를 `split`/`Path(...)`로 되파싱하거나 경로 재구성하는 곳이 없는지 최종 grep.
- **구분자 방어**: 향후 image_id 파싱 코드가 생기면 `defect_type`/stem 내 `_`로 경계 모호 가능 → 그때 `rsplit("_", 1)` 등 방어 규칙 별도 결정. (현재는 파싱 없음 → 무해)
- **drift 문서 갱신**: leather n_clusters 3→4 통일 시, 커밋된 3-클러스터 결과에 의존하는 문서/분석이 있으면 함께 갱신할지 확인.
