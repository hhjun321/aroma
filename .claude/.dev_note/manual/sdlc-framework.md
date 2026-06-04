# igloo_workflow SDLC 프레임워크 사용 설명서

이 문서는 `.claude/` 하위에 구성된 AI-assisted 개발 사이클(SDLC) 프레임워크의 개념과 사용법을 설명합니다.
Claude Code를 사용하는 누구든지 이 문서를 읽고 프레임워크를 온전히 활용할 수 있도록 작성했습니다.

---

## 1. 프레임워크 개요

이 프레임워크의 목적은 **"무의식적인 즉흥 구현"을 줄이고, 모든 코드 변경이 계획 → 구현 → 검증 → 명세화 사이클을 거치도록** 유도하는 것입니다.

Claude Code가 단순한 코딩 도구가 아니라 **프로젝트 전반의 개발 프로세스를 함께 운영하는 파트너**로 동작합니다.

### 핵심 구성 요소

```
.claude/
├── rules/              ← Claude가 항상 지켜야 할 규칙 (자동 로드)
├── skills/             ← /슬래시 명령으로 호출하는 작업 파이프라인
├── .dev_note/          ← 작업 계획 파일 저장소 (이 사이클의 중심)
│   └── .completed      ← 완료된 dev_note 파일 목록 (자동 관리)
└── .spec/              ← 프로젝트 명세 문서 (자동 갱신)
```

---

## 2. 구성 요소 상세

### 2.1 Rules — 항상 작동하는 가드레일

Claude Code가 프로젝트를 열면 `.claude/rules/` 하위의 모든 `.md` 파일을 자동으로 컨텍스트에 로드합니다. 사용자가 호출하지 않아도 항상 적용됩니다.

| 파일 | 목적 |
|------|------|
| `devnote-guard.md` | 구현 요청 감지 → dev_note 없이 진행할지 확인 |
| `implementation.md` | 아키텍처 제약 (Flowise 코어 수정 금지, TypeORM 패턴 등) |
| `rest-api.md` + `rest-api-stack.md` | REST API 구현 표준 (페이지네이션, 정렬, 에러 응답 등) |
| `validation.md` | 입력 밸리데이션 기준 |
| `nodejs-hotpath.md` | 이벤트 루프 블로킹 위험 자동 점검 |
| `production-readiness.md` | 운영 환경 대비 체크 (OOM, 타임아웃, 에러 로깅) |
| `load-test-policy.md` | 부하 테스트 자동 실행 금지 (토큰 비용 방지) |

### 2.2 Skills — /명령으로 호출하는 파이프라인

Skills는 Claude에게 내리는 단계별 지시문입니다. `/스킬명`으로 호출하거나, 다른 스킬이 내부에서 자동으로 호출합니다.

| 스킬 | 명령 | 역할 |
|------|------|------|
| `devnote` | `/devnote` | dev_note 파일 목록 표시 및 로드 |
| `devnote-write` | `/devnote-write` | 대화 맥락으로 dev_note 파일 작성 |
| `feature-dev` | `/feature-dev` | 전체 구현 파이프라인 (신규 기능, 복합 개선) |
| `micro-fix` | `/micro-fix` | 경량 구현 파이프라인 (소규모 수정, 50줄 미만) |
| `post-verify` | `/post-verify` | 구현 후 검증 |

### 2.3 Dev Notes — 작업 계획 파일

`.claude/.dev_note/` 하위에 마크다운 파일로 작업을 계획합니다. 이 파일이 **사이클의 시작점**입니다.

**파일명 패턴:**
```
1.0.12_ver.md          ← 버전 단위 묶음 (여러 개선 사항 통합)
1.0.11_bugfix_1.md     ← 버그 수정 (원인 분석 포함)
1.0.11_hotfix_1.md     ← 긴급 소규모 수정
1.0.11_<slug>.md       ← 단일 기능 개선
```

**파일 구조:**
```markdown
# <버전> <작업 제목>

## (사용할 skills: feature-dev)   ← 어떤 스킬로 구현할지 명시

## 개요
왜 이 작업이 필요한지

## 수정 내용
### 1. `패키지/경로/파일.ts` — 변경 요약
구체적인 변경 내용

## 수정 대상 파일
- `패키지/경로/파일.ts`

## 테스트
- 테스트 명령 또는 시나리오
```

**완료 추적:** `feature-dev`/`micro-fix` 스킬이 작업 완료 시 `.completed` 파일에 파일명을 자동 기록합니다. `/devnote` 목록에서 완료/미완료로 구분해서 보여줍니다.

### 2.4 Spec — 자동 갱신되는 프로젝트 명세

`feature-dev`/`micro-fix` 완료 시 자동으로 갱신됩니다. 명세를 별도로 관리하지 않아도 됩니다.

| 파일 | 내용 |
|------|------|
| `.spec/README.md` | 변경 이력 요약, 환경변수 주요 키 |
| `.spec/features.md` | 사용자 관점 기능 명세 (§번호 체계) |
| `.spec/api.md` | REST 엔드포인트 상세 명세 (§번호 체계) |

---

## 3. SDLC 사이클

```
① 구현 요청
      ↓
② [Guard] devnote-guard.md 자동 발동
      ↓
   ┌──────────────────────────────────┐
   │ "dev_note 없이 진행할까요?"      │
   │  1. 그냥 진행  2. dev_note 먼저  │
   └──────────────────────────────────┘
        ↓                    ↓
   [1번 선택]           [2번 선택]
        ↓                    ↓
   ④로 바로            ③ /devnote-write
                            ↓
                       dev_note 파일 생성
                            ↓
③ /devnote → 파일 선택 → 브리핑
      ↓
④ /feature-dev 또는 /micro-fix
      │
      ├── STEP 1: 계획 수립 (planner 에이전트)        [feature-dev 전용]
      ├── STEP 2: 코드베이스 탐색 (code-explorer)     [feature-dev 전용]
      ├── STEP 3: 질의응답
      ├── STEP 4: 아키텍처 설계 (code-architect)      [feature-dev 전용]
      ├── STEP 5: 구현 + TDD
      ├── STEP 6: 코드 리뷰 (code-reviewer + 언어별 리뷰어)
      ├── STEP 7: 보안 리뷰 (security-reviewer 에이전트)
      ├── STEP 8: 최종 점검 + 테스트 실행
      └── STEP 9: 명세 문서 갱신 + devnote 완료 기록
            ↓
      .spec/ 자동 갱신
      .dev_note/.completed 자동 기록
```

> `micro-fix`는 STEP 1(계획), STEP 2(탐색), STEP 4(아키텍처)가 없는 경량 버전입니다.

---

## 4. 언제 무엇을 쓸까

### dev_note를 써야 하는 경우
- 새 기능 추가
- 버그 수정 (원인 분석이 필요한 중간 규모 이상)
- 리팩토링
- 환경변수 추가, 설정 변경
- DB 마이그레이션

### dev_note 없이 바로 진행해도 되는 경우 (Guard 예외)
- 오타 수정
- 줄 1-2개 단순 수정
- 코드 설명, 질문 답변
- `.claude/` 파일 수정 (스킬, 규칙, 이 문서 등)

### feature-dev vs micro-fix

| 판단 기준 | feature-dev | micro-fix |
|----------|------------|-----------|
| 변경 규모 | 크거나 복합적 | 단일 관심사, 50줄 미만 |
| 새 파일/추상화 | 있음 | 없음 |
| 원인 분석 필요 | 있음 | 없음 |
| 예시 | 신규 API, 복잡한 버그 수정 | 환경변수화, 상수 조정, 간단한 오작동 수정 |

---

## 5. 전형적인 사용 시나리오

### 시나리오 A: 계획된 작업 (권장)

```
1. /devnote-write          ← 대화로 작업 계획 논의 후 파일 생성
2. /devnote                ← 목록에서 파일 선택, 브리핑 확인
3. "진행시켜"              ← feature-dev 또는 micro-fix 자동 실행
4. 단계별 승인             ← 각 STEP에서 사용자 확인 후 진행
5. 완료                    ← .spec/ 갱신, .completed 기록
```

### 시나리오 B: 즉흥 요청 → 가드 → dev_note 작성

```
1. "이 버그 고쳐줘"        ← Guard 발동
2. "2번 선택"              ← /devnote-write 자동 호출
3. dev_note 파일 생성 확인
4. /devnote → 진행         ← 이후 시나리오 A와 동일
```

### 시나리오 C: 즉흥 요청 → 가드 → 바로 진행

```
1. "이 버그 고쳐줘"        ← Guard 발동
2. "1번 선택 / 그냥 해줘"  ← 바로 feature-dev 실행
3. 완료                    ← .spec/ 갱신 (devnote 완료 기록 제외)
```

---

## 6. 에이전트 모델 배정

각 스킬이 내부적으로 서브에이전트를 호출할 때 사용하는 모델입니다.

| 역할 | 모델 | 이유 |
|------|------|------|
| `planner` (계획 수립) | opus | 전체 영향 범위·위험 요소 깊은 추론 필요 |
| `code-explorer` (코드 탐색) | haiku | 파일 탐색 위주 — 비용 절감 |
| `code-architect` (설계) | opus | 설계 트레이드오프 다각도 판단 |
| `code-reviewer` (코드 리뷰) | sonnet | 코딩 품질 체크 최적 |
| `security-reviewer` (보안 리뷰) | opus | 놓쳤을 때 비용이 크고, 전체 패턴 꿰뚫어야 함 |

---

## 7. 알아두면 좋은 동작

- **`.completed` 자동 관리**: dev_note를 통해 완료된 작업은 자동으로 완료 표시됩니다. 직접 편집할 필요 없습니다.
- **명세 자동 갱신**: `.spec/` 파일은 feature-dev/micro-fix가 알아서 갱신합니다. 갱신이 필요 없는 내부 변경은 skip을 명시적으로 보고합니다.
- **부하 테스트 자동 실행 없음**: `load-test-policy.md` 규칙에 따라 성능 측정·벤치마크는 자동 실행하지 않습니다. 필요하면 명시적으로 요청합니다.
- **Auto mode에서도 Guard 작동**: Auto mode를 켜도 devnote-guard의 확인 절차는 유지됩니다.

---

## 8. 파일 위치 한눈에 보기

```
.claude/
├── rules/
│   ├── devnote-guard.md          ← 구현 요청 가드
│   ├── implementation.md         ← 아키텍처 제약
│   ├── rest-api.md               ← REST API 공통 표준
│   ├── rest-api-stack.md         ← REST API 스택 특화
│   ├── validation.md             ← 입력 밸리데이션
│   ├── nodejs-hotpath.md         ← 이벤트 루프 점검
│   ├── production-readiness.md   ← 운영 환경 대비
│   └── load-test-policy.md       ← 부하 테스트 금지
│
├── skills/
│   ├── devnote/SKILL.md          ← dev_note 읽기/목록
│   ├── devnote-write/SKILL.md    ← dev_note 작성
│   ├── feature-dev/SKILL.md      ← 전체 구현 파이프라인
│   ├── micro-fix/SKILL.md        ← 경량 구현 파이프라인
│   └── post-verify/SKILL.md      ← 구현 후 검증
│
├── .dev_note/
│   ├── .completed                ← 완료 파일 목록 (자동 관리)
│   ├── manual/
│   │   └── sdlc-framework.md     ← 이 문서
│   └── 1.0.xx_*.md              ← 작업 계획 파일들
│
└── .spec/
    ├── README.md                 ← 변경 이력 요약
    ├── features.md               ← 기능 명세
    └── api.md                    ← API 명세
```
