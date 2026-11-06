# 스터디 노트 5_features.md — 졸업논문(20261102) 기준 동기화

## (성격: 문서 작업 — 코드·논문 미수정)

대상: `AROMA연구분석/핵심이론스터디/AROMA/5_features.md`
기준: `GRADUATION/20261102_졸업논문.md` 제5장 제2절 1~4항
일자: 2026-11-03

관련: [[aroma_table3_background_descriptor_definitions]](Table 3 개명·불일치 기록), [[aroma_step1_complexity-analysis]](CCI/MCI)

---

## 1. 배경

기존 노트(2026-10-30)는 영문 Article `section3_2.txt` 번호(식 1, Table 2/3/4, Figure 3.2.2-x)에 묶여 있었다. 스터디는 졸업논문으로 진행하므로 식·표·그림 번호와 지표 이름을 졸업논문 체계로 재매핑했다.

## 2. 변경 내용

| 항목 | 변경 |
|---|---|
| 헤더·등장위치표 | 졸업논문 식 (5-1)~(5-6)·표 5-1/5-2 번호로 교체. 영문 Article은 보조 기준으로 강등 |
| §0.1 신설 | 이름 3중 대응표: 메모 구명 → 졸업논문 표 5-2(구명 유지) → 영문 Table 3(개명). 졸업논문이 산문으로 계열 구분하는 문단 인용 |
| §2 | 식 (5-1) 표기, 졸업논문 성분 국문명, 본문 인용 실측값 굵게 표시. "주파수 복잡도가 구분" 서술은 raw 순위이지 CCI 기여도 아님을 명시 |
| §3 | 표 5-2·식 (5-3)~(5-6) 매핑. 디스크립터 표를 졸업논문 (1)~(4) 번호·국문명으로. Example Surface 열 성격 추가 |
| §4 | 졸업논문이 "데이터셋마다 자동으로 계산된다"라고 더 강하게 주장함을 기록(심사 리스크). 해소 항목(MCI Entropy 표현, ppt argmax) 분리 표기 |
| §5 | 식 (5-2)(5-7)(5-8), 그림 5-3/5-4, 표 5-3 매핑. SAM 단순화 서술 vs 코드 GT→SAM→Otsu 차이. argmin 정정 완료 반영. 단일클래스 ClassDiversity=0 추가 |
| §6 신설 | 소비처 지도: 6특징 → 맥락 셀 / morph_prior / ctx_prior / ROI_score / bg_score / site_score. 졸업논문 "품질 게이트"(픽셀 4항목, `generate_defects.py:440`)와 `suitability.py` subtype 매칭(논문 미서술)을 분리 |
| §8 신설 | 스터디 자문 7문항 (+ §0.2 관련 8번 추가) |
| §0.2 신설 (스터디 중 추가) | 개념 모델: 국소 연산자 / 디스크립터(자) / filter(디스크립터+부등호+임계+순서) / 라벨 / 벡터(raw) 용어 고정. 디스크립터 1축이 반대 방향으로 두 범주에 쓰임 표. 벡터 갈래(GMM·tertile·CCI/MCI, 실제 배치)와 라벨 갈래(메타데이터·warping, 옆길) 분리. 라벨은 어느 벡터에도 미포함. CCI·GMM엔 filter→범주 그림 미적용 |

## 2-1. 스터디 중 추가 (2026-11-06)

| 항목 | 변경 |
|---|---|
| §1.1 신설 | row 저장 구조(raw 5값+메타, 셀 코드 미저장), bin 규칙(searchsorted right), 자리 순서, 243 셀, 퇴화 경계, 패치/이미지 단위 구분 |
| §1.2 신설 | §1→§2 브리지: 5→4 매핑 그림, CCI는 셀 코드 미사용, 평균/분산/군집수 선택 이유, 이름 꼬임 |
| §3.0 신설 | 표 5-2 호출처 전수: `classify_patch`는 CASDA stage1만, AROMA `roi_selection` background_type은 데이터셋 상수, `build_train_jsonl`은 제3규칙. 라벨 실사용처 = CASDA stage3/prompt |
| §4-5, §4-6 | 논문 4항 "표 5-2로 라벨 부여"·3항 "elastic-warping" 모두 AROMA 코드 대응물 없음(CASDA 것). subtype 실제 소비처 = 단일클래스 class_fit + 꺼진 quality gate. subtype_mode 기본 fixed |
| §5.6 확장 | cascade 순서 의존, mask=꼬리표 2개(k·subtype 독립), morph_label vs defect_subtype 출처 차이, 실행 모드, 소비처 실측 표 |
| §6 보강 | bg_score 행에 class_fit 클래스 축 주석, "주 흐름은 k" + "k는 행 번호(군집 단위 값, mask 단위 조회)" + 군집/mask 단위 입력 구분 |
| §6.1 신설 | mask C(k=3) 추적: [0] P_clean/P_def/기하평균 → [1] ROI_score·쿼터 → [2] bg_score 3적합도(단위 표) → [3] site_score ring 분포 매칭 → [4][5] 게이트·블렌딩. 군집/mask 기여 표. ROI_score 단순합 = 의도적 단순화 주석 |
| §8 확장 | 자문 9~14번 추가 |
| §9 신설 | 스터디 진행 기록(통과 범위·다음 지점) |

## 3. 검증한 사실

- 코드 라인 참조 현행 유지 확인: `distribution_profiling.py:213/262/876`, `compute_complexity.py:450/580`, `background_characterization.py:93`.
- `roi_selection.py:249`에서 `SuitabilityEvaluator.matching_score(subtype, background_type)` 호출 — `background_type` 소비처 확인.
- `ppt_guide_21.md:202,308` argmin BIC — 과거 argmax 표기 정정됨.
- 졸업논문 §5.2 2항 본문 "P25, P50, P75는 … 데이터셋마다 자동으로 계산된다" — 코드 고정 임계(100.0/1.0/0.15/2.5)와 불일치 잔존.

## 4. 미결 (코드·논문 미수정)

- [ ] 불일치 A(percentile vs 고정 임계) 처리 방향 — 졸업논문 정정(a) 또는 코드 개조(b). 심사 전 (a) 권장.
- [ ] 식 (5-4) Directional AND 조건 — 코드 대응물 없음. A 처리 시 동반 검토.
