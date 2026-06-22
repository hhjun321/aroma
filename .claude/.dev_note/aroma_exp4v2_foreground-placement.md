# 결함 합성 — Foreground 제약 배치 (객체 영역 내 paste)

## (사용할 skills: feature-dev)

## 개요

B안(결함 bbox/mask 영속화) 적용 후 합성 검증 결과: 크롭·마스크·bbox는 정확하나 **결함이 객체 밖 배경(void)에 paste되는 문제** 확인. visa_pcb 샘플(`syn_00032_01`)에서 bbox `[49,337,237,162]` → PCB(중앙)가 아닌 하단 검은 배경에 결함이 떨어짐.

원인: `_random_paste_position`(generate_defects.py L145-155)이 normal 이미지 **전체에서 균등 무작위** 샘플. visa_pcb·mvtec처럼 객체가 배경 위에 놓인(object-centric) 데이터셋에서 결함이 배경으로 감. 검출 학습에 무의미(배경=결함 오학습) + 비현실적.

본 작업: paste 위치를 **normal 이미지의 foreground(객체) 영역 내부로 제한**. 진단 워크플로가 지적한 "맥락-인식 배치 미구현"의 1차 실용 해법(foreground 제약). DEFICIT 기반 context-aware 배치(논문 thesis)는 후속 TODO.

## 영향도 분석

### 변경 상태
- `copy_paste_synthesis`의 paste position 산출 로직 (무작위 → foreground 제약).
- 합성 이미지 픽셀 위치 + annotations `bbox` 값 (객체 영역 내로 이동).
- 신규 함수: foreground mask 산출 헬퍼.

### 그 상태를 전제로 동작하는 기존 로직 (모두 안전)
- `_alpha_composite`(L103-142): position만 받음 — 산출 방식 무관, 변경 없음.
- B안 mask 저장(full_mask.paste at position): 새 position 그대로 사용 — 정합 유지.
- annotations bbox = `[x_paste, y_paste, crop_w, crop_h]`: position이 foreground 내로 바뀌어도 동일 공식.
- exp4_v2 `_mask_to_bboxes`: mask PNG에서 재유도 — position 무관, 자동 정합.

### 회귀 위험
- **객체가 프레임 전체를 채우는 데이터셋**(mvtec_cable 등): foreground ≈ 전체 → 무작위와 동등. 회귀 없음.
- foreground 감지 실패(균일 이미지, 객체 불명확) → **무작위 fallback**(기존 동작 보존).

## 이론적 근거

- 결함 검출 학습은 결함이 **객체 표면 위**에 있을 때만 유효. 배경 void의 결함은 "배경에 결함이 있다"는 잘못된 사전분포 주입.
- visa/mvtec 다수 데이터셋이 단일 객체 + 균일 배경 구조 → foreground 분리가 Otsu로 충분히 가능.
- foreground 제약은 context-aware 배치(DEFICIT)의 **필요조건**: 어떤 context cell에 놓든 일단 객체 위여야 함. 따라서 A(foreground)가 B(context)의 선행 단계.

## 수정 내용

### scripts/aroma/generate_defects.py

**1. 신규 헬퍼 `_foreground_mask(normal_img) -> Optional[np.ndarray]`**
- normal 이미지 grayscale → Otsu 이진화.
- **폴라리티 자동 판정**: 4 코너 픽셀의 다수 클래스 = 배경. 반대 클래스 = foreground (밝은 객체/어두운 객체 양쪽 대응).
- 최대 연결요소(connected component)만 객체로 채택 (노이즈 제거). `cv2.connectedComponentsWithStats` 또는 morphology.
- **객체 비율 가드**: foreground가 이미지의 (예) >90% 또는 <2%면 분리 무의미 → `None` 반환(무작위 fallback).

**2. 신규 헬퍼 `_foreground_paste_position(fg_mask, bg_size, crop_size, mask_crop, rng, max_tries=20) -> Optional[Tuple[int,int]]`**
- crop이 이미지 안에 들어가는 유효 (x,y) 범위에서 무작위 샘플.
- **결함 실제 영역(mask_crop의 foreground 중심 또는 다수 픽셀)이 fg_mask 내부**에 들어가도록 검증. crop 전체가 아니라 **결함 마스크 영역 기준**(crop 박스엔 배경 여백 포함되므로).
- max_tries 내 실패 → `None`(호출부에서 무작위 fallback).

**3. `copy_paste_synthesis` 본체 수정 (position 산출부)**
- 현재: `position = _random_paste_position(normal_img.size, defect_crop.size, rng)`
- 변경:
  ```python
  fg = _foreground_mask(normal_img)
  position = None
  if fg is not None:
      position = _foreground_paste_position(fg, normal_img.size, defect_crop.size, mask, rng)
  if position is None:
      position = _random_paste_position(normal_img.size, defect_crop.size, rng)  # fallback
  ```
- real-mask 경로·ellipse 경로 **공통 적용** (배치는 크롭 방식과 독립). `mask` 변수(real 또는 ellipse)를 결함 영역 판정에 사용.

**4. 재현성**: `rng`로 샘플 → seed 고정 시 동일 결과. fg_mask 산출은 결정적(Otsu).

### 수정 불필요 (자동 적용)
- `generate_random.py`: `generate_defects.run()` 위임 → 자동.
- `exp4_v2`: mask PNG에서 bbox 재유도 → position 변경 자동 정합.

## 수정 대상 파일
- `scripts/aroma/generate_defects.py`

## 암묵적 요구사항 (엣지)
- **폴라리티**: 밝은 객체/어두운 배경(visa_pcb) AND 어두운 객체/밝은 배경 양쪽. 코너 다수결로 배경 판정.
- **객체 프레임 전체**(mvtec_cable): fg≈전체 → 무작위와 동등, 회귀 없음. 가드(>90%)로 명시 fallback.
- **객체 없음/균일**: fg 가드(<2% 또는 >90%) → None → 무작위 fallback.
- **결함 crop > foreground 영역**: max_tries 실패 → 무작위 fallback (또는 crop>normal 기존 skip).
- **다중 객체**: 최대 연결요소만 — 보조 객체 무시(단순화, 회귀 아님).
- **mask 영역 판정**: crop 박스 중심이 아닌 **결함 마스크 픽셀 중심/다수**가 fg 내부여야 — 박스 여백이 fg 밖이어도 결함 본체는 객체 위.
- **재현성**: 동일 seed → 동일 배치. fallback 분기도 결정적.
- **성능**: foreground Otsu는 normal당 1회, 경미. max_tries 루프 상한 20.

## 테스트 (Colab, pytest 금지)
1. visa_pcb 재합성 → `syn_*.jpg` 육안: 결함이 **PCB 위**에 paste되는지 (배경 void 아님). annotations bbox가 객체 영역 좌표인지.
2. mvtec_cable 재합성 → 회귀 확인: 객체가 프레임 전체라 무작위와 유사하게 정상 배치, 실패율 0.
3. foreground 가드 동작: 로그에 fallback 빈도(무작위로 떨어진 비율) 확인 — 과도하면 폴라리티/가드 재검토.
4. mask PNG + bbox 정합 유지(B안 회귀 없음): bbox가 mask foreground 영역과 일치.
5. exp4_v2 재측정 → 배경 결함 제거로 precision 추가 개선 기대.

## 미확정 TODO
- foreground 감지: Otsu 단독으로 visa/mvtec 충분한지 — 실패 데이터셋 있으면 adaptive threshold/morphology 보강.
- 결함 영역 판정 기준: 마스크 centroid in fg vs 마스크 픽셀 N% in fg — 후자가 견고하나 비용↑. 1차 centroid.
- context-aware 배치(DEFICIT 기반, 논문 thesis): cell_key→공간 좌표 매핑 필요. 별도 dev_note로 후속.
- foreground mask를 profiling에서 미리 산출·캐싱할지(반복 합성 시 재계산 회피) — 1차는 on-the-fly.
