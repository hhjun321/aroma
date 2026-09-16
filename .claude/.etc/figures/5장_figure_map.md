# 제5장 그림 매핑 표

- 대상 장: 제5장 AROMA: Adaptive ROI-based Morphology-Aware Augmentation
- 원문: `GRADUATION/articles/3.AROMA_ Adaptive ROI-based Morphology-Aware.docx`
- 상태: **STEP 1~6 완료** (2026-09-16)
- **원문 그림 19개 전량 삽입** (D-03 승인으로 Figure 10·11 포함) → 학위논문 그림 **17개**(5-4, 5-5는 각 2패널)
- 원문은 `Figure N ↔ word/media/imageN.png`가 1:1로 대응한다.

## 매핑

| 학위논문 | 캡션 | 원문 | media | 예약 경로 |
|---------|------|------|-------|----------|
| 그림 5-1 | AROMA 파이프라인 | Figure 2 | `image2.png` | `figures/fig5-1.png` |
| 그림 5-2 | 데이터셋 복잡도 지형(MCI vs. CCI) | Figure 1 | `image1.png` | `figures/fig5-2.png` |
| 그림 5-3 | Severstal·AITeX의 데이터 기반 형태 클러스터 | Figure 3 | `image3.png` | `figures/fig5-3.png` |
| 그림 5-4 | 배경 맥락 특성 분포와 tertile 경계 — (a) Severstal, (b) AITeX | Figure 4·5 | `image4.png`, `image5.png` | `figures/fig5-4a.png`, `figures/fig5-4b.png` |
| 그림 5-5 | 결함 형태 특성 분포와 표 5-3 경계값 — (a) Severstal, (b) AITeX | Figure 6·7 | `image6.png`, `image7.png` | `figures/fig5-5a.png`, `figures/fig5-5b.png` |
| 그림 5-6 | 데이터셋별 대칭 호환성(ctx_prior) 히트맵 | Figure 9 | `image9.png` | `figures/fig5-6.png` |
| 그림 5-7 | ROI 선택 및 호환성 기반 배치 흐름 | Figure 8 | `image8.png` | `figures/fig5-7.png` |
| 그림 5-8 | 실제 이미지에서의 배경 할당 | Figure 10 | `image10.png` | `figures/fig5-8.png` |
| 그림 5-9 | 원본 결함의 ring 맥락과 확정된 배치 위치 비교 | Figure 11 | `image11.png` | `figures/fig5-9.png` |
| 그림 5-10 | ROI 배치 커버리지 비교 | Figure 12 | `image12.png` | `figures/fig5-10.png` |
| 그림 5-11 | 정성적 ROI 배치 비교 | Figure 13 | `image13.png` | `figures/fig5-11.png` |
| 그림 5-12 | 배경 선택 호환성 비교 | Figure 14 | `image14.png` | `figures/fig5-12.png` |
| 그림 5-13 | AITeX ROI 비교 | Figure 15 | `image15.png` | `figures/fig5-13.png` |
| 그림 5-14 | Kolektor ROI 비교 | Figure 16 | `image16.png` | `figures/fig5-14.png` |
| 그림 5-15 | Severstal ROI 비교 | Figure 17 | `image17.png` | `figures/fig5-15.png` |
| 그림 5-16 | MTD ROI 비교 | Figure 18 | `image18.png` | `figures/fig5-16.png` |
| 그림 5-17 | MVTec Leather ROI 비교 | Figure 19 | `image19.png` | `figures/fig5-17.png` |

- **순서 주의**: 학위논문과 원문의 순서가 두 곳에서 뒤바뀐다 — 5-1↔Figure 2 / 5-2↔Figure 1, 5-6↔Figure 9 / 5-7↔Figure 8.
- 원문 media 19개 전부 사용. 제외 항목 없음.

## 본문 변경 (도형 삽입에 수반)

1. **번호 재부여**: Figure 10·11 삽입으로 기존 그림 5-8~5-15가 **5-10~5-17로 이동**.
2. **5.2 방법론에 2개 문단 신설** (원문 3.2.4 Background Assignment / Site Resolution 발췌):
   - 배경 할당: `bg_score = src_fit + class_fit + size_fit` (히스토그램 교차 기반) → 그림 5-8
   - 자리 확정: ring 맥락 히스토그램 `h_s`와 `tgt[k]`의 교차 `site_score = ∩(h_s, tgt[k])` 최대 지점 → 그림 5-9
   - 두 단계 모두 기존 초안에 서술이 없었던 내용으로, 원문에서 새로 발췌해 추가하였다.
3. **캡션 범위 축소 (D-05 적용)**: 그림 5-4·5-5의 원문 그림은 Severstal·AITeX 2종만 수록하므로, 학위논문 캡션의 "데이터셋별"을 제거하고 `(a) Severstal, (b) AITeX` 패널 표기로 대체. 본문 그림 5-5 참조에도 "(그림 5-5는 Severstal·AITeX 사례)"를 명시.

## 검증 결과 (STEP 6)

- [x] 제5장 `\[그림 ...\]` 플레이스홀더 0건
- [x] 캡션 17개 (`그림 5-1` ~ `그림 5-17`), 결번·중복 없음
- [x] 이미지 참조 19개 (5-4·5-5 각 2패널 포함)
- [x] 본문 인라인 참조 17개, 캡션과 1:1 대응
- [x] STEP 5 — `GRADUATION/figures/` 에 이미지 파일 배치 완료 (docx `word/media/`에서 복사·개명, 무가공)
