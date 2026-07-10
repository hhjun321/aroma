# clean-bg 선택 로직 육안검증 절차

`generate_defects.py`의 개선된 **clean 배경 선택 + positive placement** 로직을 로컬
`mtd` / `mvtec_leather` 실데이터로 재현·육안검사하는 절차. 실 이미지로 동일 검사
반복 시 이 문서 참조.

---

## 1. 검증 대상 — 커밋 3계층

`git log --oneline -- scripts/aroma/generate_defects.py` 기준, clean-bg 선택 로직은
아래 3커밋이 쌓여 완성됨. 각 계층마다 별도 육안검사 산출물 존재.

| # | 커밋 | 함수 (generate_defects.py) | 검증 스크립트 | PNG 산출물 |
|---|------|---------------------------|--------------|-----------|
| L1 | `362e7dd` positive placement (Scenario A) | `_positive_place` | `viz_positive_place.py` | `positive_place_{ds}.png` |
| L2 | `23bd373` image-level compat-ranked 선택 | `_rank_normals`, `_image_compat_score`, `_normal_tile_cells` | `verify_image_rank.py`, `viz_defect_vs_cleanbg.py` | `image_rank_{ds}_top20.png`, `defect_vs_cleanbg_{ds}.png` |
| L3 | `8056130` D_v-specific bg 분포 매칭 | `_dv_bg_hist`, `_cell_hist`, `_hist_intersection`, `_rank_normals` | `smoke_dv_rank.py` | `defect_vs_cleanbg_dv_{ds}_c{cid}.png` |

기반 커밋(직접 시각화 대상 아님): `6b027f7` 64px footprint 타일링, `6c8658f`
clean-grounded SGM `matrix_symmetric`.

검증 스크립트 4종은 `scripts/` 하위에 보존됨. (원본은 세션 scratchpad → 유실 방지 복사본)

---

## 2. 사전 요건 (로컬)

### 2.1 데이터셋 (MVTec 레이아웃)
```
.claude/.etc/{leather,mtd}/
  train/good/*            ← clean 배경 풀 (선택 후보)
  test/<defect_type>/*    ← 결함 소스 이미지 (D_v)
  ground_truth/<defect_type>/<id>_mask.png   ← 결함 마스크 (L3 배경 제외용)
```
> 디렉토리명 주의: 데이터셋 키 `mvtec_leather` ↔ 로컬 폴더 `leather` (스크립트 `ds_dir()`가 매핑).

### 2.2 프로파일링 산출물 (Stage 0/프로파일링 결과)
```
.claude/.etc/profiling_tobe/{mvtec_leather,mtd}/
  compatibility_matrix.json   ← matrix_symmetric + bin_edges (필수)
  morphology_clusters.json    ← cluster_assignments
  morphology_features.csv     ← image_id, defect_type, defect_bbox, image_path
```

### 2.3 파이썬 환경
- `numpy`, `opencv-python`(cv2), `Pillow`, `matplotlib` 로컬 설치.
- pytest 불필요 (프로젝트 정책: 새 테스트/빌드테스트 금지 — 이건 육안검사용 1회성 스크립트).
- 스크립트가 `D:/project/aroma/scripts` + `scripts/aroma`를 `sys.path`에 넣고
  `generate_defects` 모듈 함수를 **그대로 재사용** → 코드-검증 일치 보장.

---

## 3. 실행

`.claude/.etc/positive_place_viz/scripts/`에서 (인자 없이 leather+mtd 둘 다 처리):

```bash
cd D:/project/aroma/.claude/.etc/positive_place_viz/scripts

# L1 — positive placement (footprint 배치 위치)
python viz_positive_place.py

# L2 — image-level compat 랭킹 (top vs bottom, 결함↔선택 clean 대응)
python verify_image_rank.py       # 수치(lift %) + image_rank_*_top20.png
python viz_defect_vs_cleanbg.py   # defect_vs_cleanbg_{ds}.png

# L3 — D_v-specific 배경분포 매칭 (히스토그램 교차)
python smoke_dv_rank.py           # cluster별 PNG + RESULTS JSON (new vs old 비교)
# 단일 데이터셋만: python smoke_dv_rank.py leather
```
전 산출물은 `.claude/.etc/positive_place_viz/*.png`에 저장.

---

## 4. 각 PNG 읽는 법 (육안검사 포인트)

### L1 `positive_place_{ds}.png`
- 행별: crop 소/중/대 표본. **좌** = 결함 소스 + 원본 bbox(빨강),
  **우** = clean-bg + `_positive_place`가 고른 배치 위치(녹색).
- 라벨: `mean_compat`, `nonvoid` 타일 수. `ALL-VOID -> repick`이면 게이트가
  전 위치 거부(검은/평탄 배경) → repick 폴백 동작.
- **확인**: 녹색 박스가 검은/평탄 void가 아닌 실제 텍스처 영역에 놓이는가.

### L2 `image_rank_{ds}_top20.png`
- 상단 = compat TOP-20 clean(녹색), 하단 = BOTTOM-20(빨강). 썸네일에 score.
- **확인**: TOP이 결함 cluster의 문맥과 어울리는 배경인가, BOTTOM이 확연히
  이질적인가. `verify_image_rank.py` 콘솔의 `lift=+NN%`(ranked vs uniform 200표본)
  가 양수면 랭킹이 무작위보다 유효.

### L2 `defect_vs_cleanbg_{ds}.png`
- cluster별 1행: **좌** = 실제 결함 이미지(+bbox), **우** = 해당 cluster의
  TOP-10 선택 clean-bg.
- **확인**: 선택된 clean 배경이 결함이 원래 살던 배경과 시각적으로 정합하는가.

### L3 `defect_vs_cleanbg_dv_{ds}_c{cid}.png`
- **좌** = D_v 결함 소스(+bbox, GT mask 썸네일), **우** = D_v **배경 분포**와
  히스토그램 교차 최상위 clean top-10 (sim 값).
- L2(cluster-aggregate)와의 차이가 핵심: L3는 그 결함 개체의 실제 배경분포에
  맞춤 → `smoke_dv_rank.py` JSON의 `new_inter_mean`(L3) > `old_inter_mean`(L2)이면
  이질감 감소, `new_lv_cv`(local-variance 변동계수) 낮을수록 선택 배경 균질.
- `seed_diversity`: seed 0/1/2/42에서 top-K 샘플이 서로 다르면 argmax 붕괴 없이
  다양성 확보(고정 seed 재현성 + 풀 다양성 양립).

---

## 5. 실 이미지로 반복 시 체크리스트

1. `.claude/.etc/<ds>/`에 `train/good` + `test/<type>` + `ground_truth` 존재 확인.
2. `profiling_tobe/<ds>/`에 3개 프로파일링 파일 존재, `matrix_symmetric` 비어있지
   않은 cluster 있는지 확인 (`compatibility_matrix.json`).
3. 새 데이터셋이면 스크립트 상단 `DATASETS` / `ds_dir()` 매핑에 항목 추가.
4. L1→L2→L3 순서로 실행, 위 §4 포인트로 PNG 육안 판정.
5. 정량 근거: `verify_image_rank.py` lift %, `smoke_dv_rank.py` new vs old inter/cv.
