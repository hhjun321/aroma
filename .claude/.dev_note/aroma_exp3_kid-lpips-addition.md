# AROMA Exp 3 - KID/LPIPS 지표 추가

## (사용할 skills: feature-dev)

## 개요

논문 Exp 4(Synthetic Quality Verification)에서 FID, KID, LPIPS 세 가지 지표를 요구하지만
현재 exp3_generation_quality.py는 FID만 구현됨. KID와 LPIPS를 추가하여 논문 지표 완성.

- KID (Kernel Inception Distance): 소표본(n<50)에서 FID보다 신뢰성 높음. 편향 없는 추정량.
- LPIPS (Learned Perceptual Image Patch Similarity): 지각적 유사도. 낮을수록 더 유사.

## 영향도 분석

### 이 기능이 변경하는 상태
- exp3_results.json: fid 외에 kid, lpips 필드 추가
- exp3_summary.md: KID/LPIPS 열 추가

### 그 상태를 전제로 동작하는 기존 로직
- 없음 (평가 전용 스크립트)

## 수정 내용

### 1. scripts/aroma/experiments/exp3_generation_quality.py

#### 추가 패키지
- from torchmetrics.image.kid import KernelInceptionDistance
- import lpips

#### _compute_kid_score() 신규 함수
- KernelInceptionDistance(feature=2048, subset_size=50) 사용
- n_synth < 50이면 kid_unstable=True
- ThreadPoolExecutor 기반 이미지 로딩 (기존 _compute_fid_score 패턴 동일)

#### _compute_lpips_score() 신규 함수
- lpips.LPIPS(net="alex") 사용
- real patch 1개 vs synth 전체 평균 LPIPS 계산
- n_real < 10이면 lpips_unstable=True

#### 출력 형식 변경
{
  "isp_LSM_1": {
    "fid": {"random": {"fid": 62.1}, "aroma": {"fid": 44.3}},
    "kid": {"random": {"kid": 0.012, "kid_unstable": false}, "aroma": {"kid": 0.008}},
    "lpips": {"random": {"lpips": 0.45}, "aroma": {"lpips": 0.32}},
    "n_real_patches": 95
  }
}

### 2. AROMA연구분析/colab_execute/exp3_execute.md

패키지 설치에 lpips 추가:
  !pip install torchmetrics[image] anomalib lpips -q

## 수정 대상 파일
- scripts/aroma/experiments/exp3_generation_quality.py
- AROMA연구분析/colab_execute/exp3_execute.md

## 엣지 케이스

| 상황 | 처리 |
|------|------|
| KID subset_size > n_synth | kid_unstable=True, 값 기록 |
| LPIPS n_real < 10 | lpips_unstable=True |
| GPU 없음 | LPIPS CPU fallback (느리지만 동작) |

## 선행 조건
- Step 4 재실행 완료 (source_roi 경로 복원)

## 테스트
CLAUDE.md: pytest 금지. Colab 직접 검증.
이미 구현된 --mode fid와 동일 실행 명령으로 KID/LPIPS 결과 추가 확인.
