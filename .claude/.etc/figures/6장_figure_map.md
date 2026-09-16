# 제6장 그림 매핑 표

- 대상 장: 제6장 CASDA: Context-Aware Steel Defect Augmentation
- 원문: `GRADUATION/articles/4.CASDA_ Enhancing Steel Defect Detection Through Context-Aware Data Augmentation Framework.docx`
- 상태: **STEP 1~6 완료** (2026-09-16)
- **원문 Figure 1~9 전량 삽입**(D-04 승인) → 학위논문 그림 **9개**

## 매핑

| 학위논문 | 캡션 | 원문 | media | 예약 경로 |
|---------|------|------|-------|----------|
| 그림 6-1 | CASDA 파이프라인 | Figure 1 | `image12.jpg` | `figures/fig6-1.jpg` |
| 그림 6-2 | 기하학적 ROI 특성화 파이프라인 | Figure 2 | `image3.jpg` | `figures/fig6-2.jpg` |
| 그림 6-3 | 형태 특성 분포 — (1) linearity(λ), (2) aspect ratio(α), (3) solidity(σ) | Figure 5 | `image5.png` | `figures/fig6-3.png` |
| 그림 6-4 | 배경 텍스처 분류 파이프라인 | Figure 3 | `image6.jpg` | `figures/fig6-4.jpg` |
| 그림 6-5 | 정상 배경 패치에서 계산한 패치별 배경 분산 분포 | Figure 6 | `image8.png` | `figures/fig6-5.png` |
| 그림 6-6 | 동일 패치 집합에서 계산한 배경 엣지 밀도 분포 | Figure 7 | `image10.png` | `figures/fig6-6.png` |
| 그림 6-7 | CASDA 1단계에서 분류된 결함 하위유형별(상단)·배경 유형별(하단) 대표 ROI 샘플 | Figure 4 | `image11.jpg` | `figures/fig6-7.jpg` |
| 그림 6-8 | CASDA 2단계의 3채널 힌트 이미지 구성 | Figure 8 | `image2.jpg` | `figures/fig6-8.jpg` |
| 그림 6-9 | CASDA 3단계에서 ControlNet으로 생성한 결함 샘플 | Figure 9 | `image9.jpg` | `figures/fig6-9.jpg` |

- **순서 주의**: 학위논문 번호는 본문 등장 순서를 따르므로 원문 순서와 어긋난다 — 원문 Figure 5→그림 6-3, Figure 3→그림 6-4, Figure 6→그림 6-5, Figure 7→그림 6-6, Figure 4→그림 6-7.
- **그림 6-2 / 6-4 판정 근거**: 원문 Figure 2의 행이 4개 결함 하위유형, Figure 3의 행이 5개 배경 유형으로, 학위 본문의 "4개 유형으로 분류한다"·"5개 유형으로 분류한다" 서술과 정확히 대응한다. 원문 Figure 4(대표 샘플 모음)는 두 내용을 합친 별도 그림이므로 표 6-1 뒤에 그림 6-7로 배치하였다.

## 제외 media

| media | 사유 |
|-------|------|
| `image4.png` | CC BY 라이선스 로고 |

- 원문 media 10개 중 9개 사용, 1개 제외.

## 본문 변경 (도형 삽입에 수반)

1. **6.2 도입부**: "전체 파이프라인은 그림 6-1과 같이 5단계로 구성된다." 문장 추가.
2. **임계값 도출 근거 서술 신설** (원문에서 새로 발췌, 기존 초안에 없던 내용):
   - 결함 유형 임계값(λ=0.85, σ=0.9)은 수작업이 아니라 전체 결함 인스턴스 분포(N=19,958)에서 로그 스케일 Otsu 분할로 도출 → 그림 6-3
   - 배경 적합도 임계값(τ=14.09, ε=0.607)도 정상 배경 패치 분포에서 동일 방식으로 도출 → 그림 6-5·6-6
   - 저분산·저엣지 배경만 선택해 합성 결함과 주변 배경의 도메인 불일치를 줄인다는 근거 문장 추가
3. **번호 재부여**: 기존 그림 6-1·6-2·6-3·6-4 → 각각 6-2·6-4·6-8·6-9로 이동.

## 검증 결과 (STEP 6)

- [x] 제6장 `\[그림 ...\]` 플레이스홀더 0건
- [x] 캡션 9개, 본문 등장 순서와 번호 일치, 결번·중복 없음
- [x] 이미지 참조 9개
- [x] 본문 인라인 참조 9개, 캡션과 1:1 대응
- [x] STEP 5 — `GRADUATION/figures/` 에 이미지 파일 배치 완료 (docx `word/media/`에서 복사·개명, 무가공)
