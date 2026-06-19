# aroma-plus — 2026-06-05

## Session `cef5eb11` — 2026-06-05T01:31:25.918Z

### #1 · 10:31:52

> 👤 **User**
>
> /devnote

🤖 **Claude**

- 🔍 Glob `**/*.md`
- 🔍 Glob `**/*.md`
- 📖 Read `Adaptive ROI Modeling via Complexity-Aware Morphology and Context Analysis copy.md`
- 📖 Read `Adaptive ROI Optimization via Morphology-Aware Analysis.md`

Args가 파일명이 아닌 경로 분석 요청이므로, `.etc` 디렉토리 확인 후 devnote 목록 표시합니다.

두 파일 분석 완료. 내용 요약합니다.

---

## `.claude/.etc` 문서 분석

**파일 현황:**
- `Adaptive ROI Modeling via Complexity-Aware Morphology and Context Analysis copy.md`
- `Adaptive ROI Optimization via Morphology-Aware Analysis.md`

두 파일 **내용 동일** — 제목/파일명만 다름, `copy` 파일은 중복본.

---

## AROMA 문서 핵심 요약

**목적:** CASDA(규칙 기반 ROI) 한계 극복 → 데이터 구동형 ROI 정책 학습

### 핵심 아키텍처
```
Target Dataset
  → [MCI/CCI 복잡도 분석]
    → 1단계: 위상학적 라우팅 (Multimodal→GMM / Unimodal→Percentile)
    → 2단계: Bayesian/Optuna 세부 최적화
```

**대리 목적 함수:**
$$\max_\Theta f(\Theta) = \text{Silhouette}(\mathcal{M}) + \text{Silhouette}(\mathcal{C}) + \lambda \cdot I(\mathcal{M};\mathcal{C})$$

### 5단계 파이프라인

| Step | 내용 |
|------|------|
| 1 | MCI/CCI 복잡도 분석 + 하이브리드 정책 라우팅 |
| 2 | 동적 클러스터링 + Deficit-Aware ROI 샘플링 |
| 3 | ROI 메타데이터 → 시맨틱 프롬프트 → 디퓨전 인페인팅 |
| 4 | 잠재공간 블렌딩 + 숏컷 방지 + YOLO 어노테이션 자동화 |
| 5 | YOLOv8n TSTR 프로토콜 평가 (mAP50, F1) |

### 타깃 데이터셋 (2×2 복잡도 매트릭스)

|  | CCI 낮음 | CCI 높음 |
|--|---------|---------|
| **MCI 낮음** | MVTec AD | VisA (PCB) |
| **MCI 높음** | Severstal Steel | 극단 환경 |

### Step 4 핵심 기법
- **잠재공간 마스크 블렌딩:** 에지 보존하며 배경 조화
- **구조 보존형 질감 조화:** 고주파 분리 후 저주파만 매칭
- **Anti-Shortcut Augmentation:** 마스크 3~7px 랜덤 Dilation

---

**이 문서로 무엇을 하길 원하시나요?** devnote 작성, 구현 계획, 추가 분석 등 말씀해주세요.

---

### #2 · 10:42:43

> 👤 **User**
>
> 두 문서는 AROMA 연구 주제에 대한 정리내용이다. 이를 바탕으로 연구를 진행한다.

🤖 **Claude**

- 🔧 Skill
- 🔍 Glob `**/*`
- 📖 Read `README.md`

프로젝트 구조 먼저 파악합니다.

레포 확인 완료 — 빈 프로젝트, 코드 없음. `.claude/rules/colab-execution.md` 있으니 Google Colab 환경.

---

첫 번째 질문:

"연구를 진행한다" — 구체적으로 어떤 단계가 목표입니까?

**A.** 5단계 파이프라인 전체를 코드로 구현 (scripts/ 작성)
**B.** 특정 Step에 집중 (예: Step 1 복잡도 분석만 먼저)
**C.** CASDA 기존 코드 위에 AROMA 컴포넌트를 올리는 확장
**D.** 실험 결과 재현 (기존 구현 있음, Colab 노트북 정리)

---

### #3 · 10:48:53

> 👤 **User**
>
> A 인데, B 부터 순차적으로 진행하자

🤖 **Claude**

Step 1부터 순차 구현. 다음 질문:

CASDA 기존 코드가 있습니까? (`.claude/rules/colab-execution.md`에 `github.com/hhjun321/CASDA` 클론 명령 있음)

**A.** CASDA 코드 그대로 재사용 가능 — AROMA는 그 위에 추가
**B.** CASDA 참고용만, AROMA는 독립 구현
**C.** CASDA 없음 / 접근 불가 — 완전 새로 작성