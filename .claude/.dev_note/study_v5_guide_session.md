# 스터디 — v5 페이지별 가이드 (졸업논문 20261102 기준)

## (성격: 문서 작업 — 코드·논문 미수정)

대상: `AROMA연구분석/핵심이론스터디/v5_페이지별_스터디가이드.md`
기준: `GRADUATION/20261102_졸업논문.md` (제4장 L384–464, 제5장 L615–686, 제6장 L815–962)
코드: aroma 저장소 + CASDA 원본 `D:\project\CASDA\CASDA`
일자: 2026-11-11

관련: [[study_5_features_thesis_sync]](제5장 특징 스터디, §4-7 추가)

---

## 1. 배경

가이드는 `학위논문_발표_v5.pptx`(46장) 기준. 저장소에는 39장 `학위논문_발표.pptx`만 있어 슬라이드 번호가 아닌 가이드 본문 + 졸업논문으로 진행. 사용자 요청 슬라이드만 순서 무관 처리.

## 2. 변경 내용

| 슬라이드 | 추가 |
|---|---|
| 15 | 식 (4-2)(4-3) 변수 의미·식 설명·두 식 관계·제5장 연결을 한 블록에. 제4장 코드 본 저장소 없음 표기 |
| 28 | P_def "둘레(ring)" → "결함 이미지의 비결함 패치 전부"로 정정. `_build_symmetric` 5단계·출력 키·ε·행 정규화·장난감 예·k=행 번호. 불일치: `roi_selection.py:429` 구형 `matrix` 사용 |
| 29 | ∩ 정의, h_g, src_fit/class_fit/size_fit 의미, 논문 단순합 vs 코드 lift 가중 |
| 30 | grid/footprint/후보 자리/h_s ring/tgt[k]/argmax/forced_xy |
| 40 | S 세 성분 식(continuity 0.6/0.4, stability 0.4/0.3/0.3), 행렬 = 수작업 `MATCHING_RULES`(표 6-2 일치), 활용 1·2, 불일치: 3단계 `compose_casda_images.py:91 COMPATIBILITY_MATRIX` ≠ 표 6-2 (irregular·general 키 없음 → 균등 랜덤) |
| 41 | R/G/B 생성 규칙·값·역할, 그레이 복제 이유, 3채널 이유. 논문 G "0"·B "0.5배" vs 코드 ×0.3·상수 20 |
| 42 | F1~F4 출처(detailed 스타일), 힌트=공간/프롬프트=의미, pipeline 인자(0.7/7.5/30)와 잔차·cross-attention 작용 |
| 42-1 신설 | 증강 방법 5단계: 생성(`test_controlnet.py:344`, 클래스별 수 차등) → 배경 선택 → 합성(`compose_casda_images.py:768`: 원본 bbox 좌표 + jitter ±100 + scale 0.875~1.0, 8px 팽창 seamlessClone) → 품질 게이트(`validate_augmented_quality.py:202` 5항 ≥ 0.7) → 통합(`merge_datasets.py` RLE). 불일치: 논문 (6-18) 4항 0.3/0.3/0.2/0.2 vs 코드 5항 |
| 부록 B | P33/P66 모집단, G 채널 항목 해소 체크 |
| 진행 기록 | 신설 |
| `5_features.md` | §4-7(구형 matrix), §9 행 추가 |

## 3. 검증한 사실

- `distribution_profiling.py:551 _build_symmetric` — P_def_patch는 `image_type==defect` row 전부(결함 과반 패치는 `_context_worker`에서 이미 제외). ring 없음.
- `roi_selection.py:429` `compat_json.get("matrix")`; `matrix_symmetric` grep 0건.
- `clean_bg_selection.py:598 _ring_keys`, `:611 _best_ring_site`(8이웃 ring, 1/len 균등, footprint void 배제), `:525 _target_by_cluster`.
- CASDA `src/analysis/roi_suitability.py:24 MATCHING_RULES` = 표 6-2 값. `:142` 0.5/0.3/0.2.
- CASDA `scripts/compose_casda_images.py:91` 별도 COMPATIBILITY_MATRIX, `:627/631 rng.choices(weights)`.
- CASDA `src/preprocessing/hint_generator.py` G else 분기 ×0.3, B smooth 상수 20.
- CASDA `scripts/test_controlnet.py:246` pipeline 인자.

## 4. 미결 (코드·논문 미수정)

- [ ] 28번 불일치 — ROI_score ctx_prior를 `matrix_symmetric`으로 교체(코드) 또는 논문 (5-9) 설명 정정.
- [ ] 40번 불일치 — 3단계 COMPATIBILITY_MATRIX를 표 6-2로 통일(코드) 또는 논문 문장 정정.
- [ ] 41번 — 논문 G "0"·B "0.5배" 문구 vs 코드.
- [ ] 15번 — 제4장 코드(별도 저장소) 커널 크기·k 값.
- [ ] 42-1 — 논문 식 (6-18) Q 4항 vs 코드 5항 가중 불일치.
