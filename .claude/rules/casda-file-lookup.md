# CASDA 프로젝트 파일 탐색 규칙

CASDA 프로젝트에서 파일·스크립트·모듈 위치를 찾아야 할 때는 **Glob/Grep 직접 탐색보다 먼저** 아래 문서를 참조합니다.

## 참조 문서 위치

`D:\project\CASDA\CASDA\CASDA\` 하위 노트 파일들:

| 문서 | 참조 시점 |
|------|----------|
| `00-INDEX.md` | 경로 변수(PROJ, SCRIPTS, DRIVE_DATA 등) 확인 |
| `01-Overview.md` | 전체 아키텍처·리포지토리 구조 파악 |
| `02-Pipeline-StageA.md` | Stage A 스크립트·입출력 경로 확인 |
| `03-Pipeline-StageB.md` | Stage B 스크립트·입출력 경로 확인 |
| `04-Pipeline-StageC.md` | Stage C 스크립트·입출력 경로 확인 |
| `05-Pipeline-StageD.md` | Stage D 스크립트·입출력 경로 확인 |
| `06-Scripts-Reference.md` | 핵심 14개 스크립트 입출력 매핑·계산 로직 |
| `07-Models.md` | 모델 구조·파일 위치 확인 |
| `08-Dataset-Groups.md` | 실험 그룹·데이터셋 구성 확인 |
| `09-Experiments.md` | 실험 결과·설정 확인 |

## 적용 규칙

1. 스크립트 파일 위치 → `06-Scripts-Reference.md` 먼저 확인
2. 경로 변수(ROI_DIR, CN_DATASET 등) → `00-INDEX.md` 먼저 확인
3. 특정 Stage 스크립트·파라미터 → 해당 Stage 노트 먼저 확인
4. `src/` 모듈 위치 → `06-Scripts-Reference.md` 하단 `src/ 모듈 구조` 섹션 확인
5. 문서에서 찾지 못한 경우에만 Glob/Grep 직접 탐색

## 이 규칙의 목적

문서에 이미 정리된 정보를 반복 탐색하는 비용을 줄이고,
파일 위치·경로 변수를 일관되게 참조합니다.
