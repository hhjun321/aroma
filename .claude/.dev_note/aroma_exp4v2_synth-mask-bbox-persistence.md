# AROMA Exp4v2 — 합성 시점 GT 마스크·bbox를 annotations.json에 영속화

---

## (사용할 skills: feature-dev)

## 개요

exp4v2 진단 workflow(wf_929b5b7c-a84) 결과, 합성 데이터가 baseline을 못 이기는 근본 원인이 **합성 GT 라벨 품질 붕괴**로 확정됨. `generate_defects.py`가 합성에 사용한 타원 마스크와 paste 좌표를 `annotations.json`에 저장하지 않아, downstream `exp4_v2_supervised_detection.py`의 정확한 mask 우선 경로가 실전에서 미사용된다. 대신 부정확한 image-diff 휴리스틱(`_extract_defect_bboxes`)이 주 경로가 되어 bbox가 실제 결함보다 사방 ~6px 부풀고 over-segmentation 발생 → YOLO가 잘못된 localization을 학습 → precision 붕괴(mvtec_cable baseline P=0.833 → aroma 0.19 / random 0.09).

본 작업은 합성 시점에 이미 메모리에 존재하는 타원 마스크와 paste 좌표를 PNG + annotation 키로 영속화하여, exp4_v2가 정확한 GT를 직접 사용하도록 만든다. `generate_random.py`도 `generate_defects.run()`에 위임하므로 한 곳 수정으로 random·aroma 양쪽에 적용된다.

---

## 영향도 분석

### 이 기능이 변경하는 상태
- `annotations.json` 스키마: 정상 분기 항목에 `mask_path`, `bbox` 키 추가 (additive — 기존 키 보존)
- 새 산출물: `{output_dir}/{dataset}/masks/syn_{roi_idx:05d}_{rep_idx:02d}.png` (마스크 PNG)
- `copy_paste_synthesis` 반환 타입: `bool` → `Optional[Dict[str, Any]]`

### 그 상태를 전제로 동작하는 기존 로직
- `exp4_v2_supervised_detection.py` `_load_synth_annotations`(L502-523): 이미 `mask_path` 키를 인식(`e.get("mask") or e.get("roi_mask") or e.get("mask_path")`, L506)하고 valid dict에 보존(L522) → **로직 변경 불필요, 새 키 자동 인식**
- `_write_yolo_labels`(L637-686): mask 우선 경로(L651-660)가 그대로 작동. mask_path 채워지면 diff 휴리스틱(L661-662) 우회
- YOLO 캐시 `schema_version`: synth 라벨 산출 방식이 바뀌므로 재빌드 필요 가능성
- `run()`의 local_staging push 로직(L500-502): 현재 `img_dir`만 Drive로 push → `masks/`도 push하도록 확장 필요 (누락 시 exp4_v2가 마스크 못 찾아 휴리스틱으로 회귀)

### delete/remove/revoke/bulk 계열 여부
- 해당 없음. additive 스키마 변경. 단 **전체 재합성 필요** (마스크/좌표는 합성 시점에만 존재 → 기존 합성 이미지에서 복원 불가)

---

## 이론적 근거 (왜 mask_path가 핵심, bbox는 메타)

코드 확인 결과 (opus 분석, file:line 근거):
- **exp4_v2는 annotation의 `bbox` 키를 직접 읽는 경로가 없음** (Q3 확인). valid dict는 `image_path / normal_image / source_roi / mask_path` 4키뿐(L518-523). bbox는 항상 mask→bbox(L652) / diff→bbox(L662) / template(L668)로 *계산*함.
- → `mask_path`만 채우면 `_mask_to_bboxes`(L652)가 정확 bbox를 산출. **mask_path가 라벨 정확도의 핵심**, bbox 키는 디버깅/검증 메타로만 기록(additive).

마스크 좌표계 정합:
- exp4_v2의 mask→bbox는 **합성 이미지 좌표계** 기준(L652-660, mask 크기≠이미지 크기 시 스케일링).
- → 저장 마스크를 **normal 이미지 전체 크기 캔버스**로 만들고 paste position에 ellipse 배치하면 (mw,mh)==(img_w,img_h)가 되어 스케일링 우회 → 가장 정확.
- 저장 마스크는 **feather 적용 전 원본 ellipse(L210)** 기준 → scipy 유무와 무관하게 bbox 안정.

---

## 수정 내용

### 1. `scripts/aroma/generate_defects.py` — 마스크/bbox 영속화 (주 수정)

**`copy_paste_synthesis` 반환 타입 변경** (L165 시그니처, L218/L222 반환):
- `-> bool` → `-> Optional[Dict[str, Any]]`
- 성공 시 `{"mask_path": <경로>, "bbox": [x, y, cw, ch]}`, 실패 시 `None`
- 실패 반환부(L182/192/196/222 `return False`) → `return None`

**마스크 저장 + bbox 산출** (L210 직후, L213 position 확보 후):
- L203 `cw, ch = defect_img.size` (crop 크기 — 스코프 내 가용)
- L210 ellipse 마스크 `mask` (PIL `L`) — 스코프 내 가용
- L213 `position = _random_paste_position(...)` → `(x, y)`
- normal 이미지 전체 크기 캔버스(`L` 모드, 0으로 초기화) 생성 → position에 ellipse 마스크 paste → `masks/syn_{roi_idx:05d}_{rep_idx:02d}.png`로 PNG(무손실) 저장
- bbox = `[x, y, cw, ch]` (ellipse가 crop 전체를 채우므로 crop bbox = paste bbox)

**`run()` 정상 분기 수정** (L472-495):
- L472 `ok = synthesis_fn(...)` → `meta = synthesis_fn(...)`; `if meta:` 분기
- annotation dict(L482-495)에 `"mask_path": meta["mask_path"]`, `"bbox": meta["bbox"]` 추가
- 마스크 경로는 기존 `image_path`(=`final_out_path`, 절대경로)와 **동일 규약**으로 기록 (exp4_v2 `_resolve_path`와 정합)

**파일명 정합 (견고성)**:
- 합성 이미지: `syn_{roi_idx:05d}_{rep_idx:02d}.jpg` (L449)
- 마스크: `masks/syn_{roi_idx:05d}_{rep_idx:02d}.png` — **이미지와 동일 stem**
- → 명시 `mask_path` 경로(L507) **및** fallback 첫 후보(`masks/{stem}.png`, L511) 양쪽 매칭

**local_staging push 확장** (L500-502):
- 현재 `img_dir`만 Drive push → `masks/`도 push (누락 시 휴리스틱 회귀)

### 2. `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — 검증 + 주석 갱신

- **L498-501 주석 갱신**: "AROMA's generate_defects.py currently writes no mask" → 사실과 달라짐, 갱신
- `_load_synth_annotations`(L506)·mask 우선 경로(L651-660) **로직 변경 불필요** — 새 `mask_path` 자동 인식
- (선택) bbox 키 실효화: valid dict에 `"bbox": e.get("bbox")` 추가 + `_write_yolo_labels`에서 직접 읽기 경로 — **미확정, 우선순위 낮음** (mask_path만으로 정확)

### 3. `scripts/aroma/generate_random.py` — 수정 불필요

- `generate_defects.run()`에 전적 위임(L143-151), 자체 annotation 미기록 → 자동 적용

### 4. (선택) min_area 정합

- synth mask 경로 `_mask_to_bboxes` default=50(L378)이나 `_write_yolo_labels`가 `min_area=200`(L623/L652)으로 호출 → 현재 mask도 200 필터.
- val GT는 min_area=50. 통일하려면 mask 경로만 50으로 분리 — **작은 결함 누락 trade-off, 별도 판단** (필수 아님)

---

## 수정 대상 파일

- `scripts/aroma/generate_defects.py` — 주 수정 (`copy_paste_synthesis` L165-222, `run` annotation L472-495, push L500-502)
- `scripts/aroma/experiments/exp4_v2_supervised_detection.py` — 주석 갱신 L498-501 (+ 선택적 bbox 읽기)
- `scripts/aroma/generate_random.py` — 수정 불필요 (자동 적용)

---

## 암묵적 요구사항 (엣지케이스)

- **합성 실패 항목** (`return None`): annotation 미기록(기존 동작 유지). `mask_path` 키 부재가 다른 경로를 깨지 않는지 확인
- **dry_run 분기** (L454-469): 합성 안 함 → mask/bbox 산출 불가. exp4_v2가 dry_run=True를 skip(L490-492)하므로 **수정 불필요**
- **다중 결함**: 현 copy_paste는 ROI 1개 단일 paste → 단일 ellipse 마스크. `_mask_to_bboxes`는 다중 contour 지원하므로 향후 다중 paste 확장 시 자동 대응
- **scipy 부재**: feather(L129 `HAS_SCIPY`)는 엣지만 부드럽게. 저장 마스크는 feather 전 ellipse 기준 → 무관
- **PNG 강제**: 이미지는 `.jpg`(lossy)이나 마스크는 반드시 **PNG(무손실)** — JPEG 압축이 경계 흐리면 bbox 재부정확
- **경로 충돌**: 마스크 파일명 이미지 stem과 1:1 → 충돌 없음

---

## 테스트 (Colab — CLAUDE.md: 새 테스트 코드·pytest 금지, Colab 직접 검증)

1. mvtec_cable `generate_defects.py` 재실행 → `annotations.json`에 `mask_path`/`bbox` 키 채워졌는지, `masks/syn_*.png` 생성됐는지 확인
2. 마스크 PNG 1-2개 합성 이미지 위 오버레이 → ellipse가 paste 결함 위치와 일치 확인
3. `exp4_v2` 실행 → 로그에서 mask 우선 경로(L651-660) 발동, diff 휴리스틱 미사용 확인. YOLO 라벨 bbox가 ~6px 부풀지 않는지 확인
4. **재측정**: mvtec_cable precision 회복 (aroma 0.19 / random 0.09 → baseline P=0.833 방향)
5. local_staging=True 경로로도 마스크가 Drive `output_dir/masks/`에 push되는지 확인

---

## 미확정 사항 (TODO)

- [ ] **bbox 키 실효화 여부**: exp4_v2는 현재 bbox 미사용(Q3). (a) 메타로만 기록 / (b) exp4_v2에 직접 읽기 추가 — 코드 근거상 (a)로 충분(mask_path만으로 정확). 구현 시 사용자 결정
- [ ] **min_area 통일 범위**: mask 경로만 50 vs synth 전체 200 유지 — 작은 결함 누락 trade-off (선택)
- [ ] **mask_path 기록 형식**: 절대경로(기존 `image_path` 규약) vs synth_base 상대 — 절대 권장, 구현 시 확정
- [ ] **재합성 범위**: mvtec_cable 단독 검증 후 4개 데이터셋(random+aroma) 전체 재합성 → exp4v2 재측정
