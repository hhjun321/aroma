# 제4장 그림 매핑 표

- 대상 장: 제4장 결함검출 모델 성능 개선을 위한 결함데이터 증강 방법에 대한 연구
- 원문: `GRADUATION/articles/2.결함검출 모델 성능 개선을 위한 결함데이터 증강 방법에 대한 연구.docx`
- 상태: **STEP 1~6 완료** (2026-09-16)
- **번호 변경**: 결정(D-02 승인)에 따라 원문 그림 1~4를 모두 삽입 → 기존 그림 4-1·4-2(성능 그래프)가 **4-5·4-6으로 이동**

## 매핑

| 학위논문 | 캡션 | 원문 그림 | docx media | 예약 경로 | 원문 해설 절 |
|---------|------|----------|-----------|----------|------------|
| 그림 4-1 | 결함이미지와 마스크이미지 | 그림 1 | `word/media/image12.jpg` | `figures/fig4-1.jpg` | 3-1 |
| 그림 4-2 | 생성된 결함이미지와 마스크이미지 | 그림 2 | `word/media/image5.jpg` | `figures/fig4-2.jpg` | 3-2 |
| 그림 4-3 | 유효 영역 추출 | 그림 3 | `word/media/image10.jpg` | `figures/fig4-3.jpg` | 3-3 |
| 그림 4-4 | Poisson blending을 이용한 데이터 합성 | 그림 4 | `word/media/image8.jpg` | `figures/fig4-4.jpg` | 3-4 |
| 그림 4-5 | 원본셋 모델과 증강셋 모델의 mIoU 성능비교 | 그림 5 | `word/media/image11.jpg` | `figures/fig4-5.jpg` | Ⅳ |
| 그림 4-6 | 원본셋 모델과 증강셋 모델의 Dice 성능비교 | 그림 6 | `word/media/image15.jpg` | `figures/fig4-6.jpg` | Ⅳ |

- 원문 그림 3의 영문 캡션은 `Fig. 3. Adaptive Context Matching`으로 국문 캡션("유효 영역 추출")과 문구가 다르다. 학위논문은 국문 캡션을 따른다.

## 제외 media (학위논문 미사용)

| media | 사유 |
|-------|------|
| `image19.jpg` | 저널 로고 |
| `image18.jpg` | Open Access 라이선스 배너 |
| `image2.jpg` | 저자 약력 사진 (한호준) |
| `image16.png` | 저자 약력 사진 (문일영) |
| `image1.png`, `image3.png`, `image4.png`, `image6.png`, `image7.png`, `image9.png`, `image13.png`, `image14.png`, `image17.png` | 수식 이미지 — 학위논문은 수식을 LaTeX로 작성 |

- 원문 media 19개 중 6개가 학위논문 그림에 대응, 13개 제외.

## 본문 반영 위치

| 그림 | 삽입 위치 | 인라인 참조 |
|------|----------|------------|
| 4-1 | 표 4-1 직후 | `(표 4-1, 그림 4-1)` — 4.2 데이터셋 문단 |
| 4-2 | 표 4-3 직후 | `(그림 4-2)` — ControlNet 조건/출력 문장 |
| 4-3 | ROI 추출 문단 직후 | `(그림 4-3)` |
| 4-4 | Poisson blending 문단 직후 | `(그림 4-4)` |
| 4-5 | 4.3 ResNet34 비율별 성능 문단 직후 | `(그림 4-5)` |
| 4-6 | 4.3 Dice 서술 문단 직후 | `(그림 4-6)` |

## 검증 결과 (STEP 6)

- [x] 제4장 `\[그림 ...\]` 플레이스홀더 0건
- [x] 캡션 6개, 결번·중복 없음
- [x] 이미지 참조 6개
- [x] 본문 인라인 참조 6개, 캡션과 1:1 대응
- [x] STEP 5 — `GRADUATION/figures/` 에 이미지 파일 배치 완료 (docx `word/media/`에서 복사·개명, 무가공)
