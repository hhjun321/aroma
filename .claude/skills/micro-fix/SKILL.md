---
name: micro-fix
description: >
  feature-dev의 경량 버전. Research(STEP 2: 코드베이스 탐색)와 Plan(STEP 1: 계획 수립, STEP 4: 아키텍처 설계)을 제거합니다.
  변경 범위와 방향이 이미 확정된 소규모 수정에 사용합니다.
  조건: 변경이 단일 관심사, 새 파일·추상화·의존성 없음, 50줄 미만.
  위 조건을 하나라도 충족하지 않으면 /feature-dev를 사용하세요.
  Trigger phrases: "micro-fix", "/micro-fix"
---

# 구현 파이프라인 — Micro-Fix

**feature-dev에서 Research(코드베이스 탐색) + Plan(계획 수립·아키텍처 설계)을 제거한 경량 파이프라인입니다.**
변경 범위와 방향이 이미 확정된 소규모 수정에 사용합니다.

구현 요청: **사용자 메시지 또는 args에서 파악합니다.**

---

## 모델 사용 정책

| 단계 | 에이전트 | 모델 | 이유 |
|------|---------|------|------|
| STEP 2 | `code-reviewer`, 언어별 리뷰어 | **sonnet** (기본값) | 코딩 품질 체크는 Sonnet이 최적, reasoning보다 코딩 능력이 중요 |
| STEP 3 | `security-reviewer` | **opus** | 놓쳤을 때 비용이 크고, 전체 코드 패턴을 꿰뚫어야 함 |

---

## STEP 1 — 질의응답

구현 요청을 검토하여 모호하거나 미확정된 사항을 정리합니다.

**CRITICAL: 이 단계를 건너뛰지 않습니다.**

- 관련 파일을 직접 읽어 현재 상태를 파악합니다
- edge case, 에러 처리, 통합 포인트, 범위 경계, 하위 호환성, 성능 요건 포함
- 질문 목록을 명확하게 정리하여 사용자에게 제시
- 운영 환경 제약 (Rate Limit, 타임아웃, 대용량 파일/메모리 한계) 고려 여부
- **사용자의 답변을 기다린 후 STEP 2로 진행합니다**

사용자가 "알아서 해줘"라고 하면 권장안을 명시하고 명시적 확인을 받습니다.

---

## STEP 2 — 구현

**사용자 승인 없이 구현을 시작하지 않습니다.**

1. 이전 단계에서 식별된 관련 파일 전체를 직접 읽습니다
2. 확정된 방향에 따라 구현합니다
3. **프로젝트 컨벤션을 엄격히 준수합니다** (CLAUDE.md 및 활성화된 rules 참조)
4. TDD 접근법 적용:
   - 테스트 파일을 먼저 작성 (RED 상태)
   - 구현하여 테스트 통과 (GREEN 상태)
   - 리팩토링 (IMPROVE)
5. 구현 완료 후 **해당 테스트 파일만** 실행하여 GREEN 상태를 확인합니다
   - 루트에서 `pnpm test`를 실행하지 않습니다 — turbo가 모든 패키지 테스트를 재실행하여 불필요하게 오래 걸립니다
   - 대신 변경이 발생한 패키지 디렉토리에서 테스트 파일을 직접 지정합니다
   - 예: `cd packages/server && pnpm test -- --testPathPattern=IglooPhoneValidation`

---

## STEP 3 — 코드 리뷰 (`code-reviewer` + 언어별 리뷰어 에이전트 병렬 실행, model: **sonnet**)

### 언어 감지

**에이전트 호출 전, 변경 파일의 확장자를 확인하여 언어 리뷰어를 결정합니다.**
확장자가 먼저이고, 불분명하면 프로젝트 루트 파일로 재확인합니다.

| 변경 파일 확장자 | 루트 파일 (fallback) | 언어 리뷰어 |
|---|---|---|
| `.ts` `.tsx` `.js` `.jsx` | `package.json` | `typescript-reviewer` |
| `.py` | `requirements.txt` `pyproject.toml` `setup.py` | `python-reviewer` |
| `.go` | `go.mod` | `go-reviewer` |
| `.rs` | `Cargo.toml` | `rust-reviewer` |
| `.java` | `pom.xml` `build.gradle` (Java) | `java-reviewer` |
| `.kt` `.kts` | `build.gradle` (Kotlin) | `kotlin-reviewer` |
| `.dart` | `pubspec.yaml` | `flutter-reviewer` |
| `.cpp` `.cc` `.cxx` `.h` `.hpp` | `CMakeLists.txt` | `cpp-reviewer` |
| `.cs` | `*.sln` `*.csproj` | `csharp-reviewer` |

확장자가 섞인 경우 주요 언어(변경 파일 수 기준) 리뷰어를 사용합니다.
감지 불가 시 `typescript-reviewer`를 기본값으로 사용합니다.

### Lint-fix

**에이전트 호출 전, 변경된 파일에 한해서만 lint-fix를 실행합니다.**

감지된 언어에 맞는 linter를 사용합니다:

| 언어 | lint-fix 명령 |
|---|---|
| TypeScript / JavaScript | `pnpm exec eslint --fix $(git diff --name-only HEAD \| grep -E '\.(ts\|tsx\|js\|jsx)$' \| tr '\n' ' ')` |
| Python | `ruff check --fix $(git diff --name-only HEAD \| grep -E '\.py$' \| tr '\n' ' ')` |
| Go | `golangci-lint run --fix ./...` |
| Rust | `cargo clippy --fix` |
| Java / Kotlin | `./gradlew spotlessApply` (Gradle 프로젝트) 또는 skip |
| 기타 | linter 미사용 프로젝트면 이 단계를 skip |

- 변경 파일이 없거나 해당 확장자가 없으면 skip
- fix 후 경고만 남아 있으면 그대로 진행

### 리뷰 에이전트 호출

**반드시 두 서브에이전트를 병렬로 호출합니다. model은 생략하여 기본값(sonnet)을 사용합니다.**

- **`code-reviewer`**: 코드 품질, 패턴, 에러 처리, 불변성 원칙, 중복 로직
- **감지된 언어 리뷰어**: 언어 특화 타입 안전성, async 패턴, 관용구, 보안 이슈

**각 에이전트에 반드시 다음 지시를 포함합니다:**

> 변경된 파일 각각에 대해, 그 파일이 시스템 내 다른 파일과 맺고 있는 암묵적 계약을 추론하고, 해당 계약이 유지되는지 관련 파일을 직접 탐색하여 교차 검증하라.
> 계약 추론의 예시 (문서화 여부와 무관하게 적용):
> - ORM 엔티티(`*.entity.ts`) → 마이그레이션 SQL과 일치 여부, 컬럼 타입이 모든 지원 DB에서 유효한지
> - 라우터 → 연결된 서비스/타입과 시그니처 일치 여부
> - 권한 enum/상수 → 이를 참조하는 맵, UI 컴포넌트, 가드와의 일관성
> - 인터페이스/타입 정의 → 이를 구현하거나 소비하는 모든 파일
> - 환경변수 추가/변경 → .env.example 반영 여부
>
> 이 목록은 예시일 뿐이며, 변경된 파일의 역할로부터 관련 파일을 스스로 도출해야 한다.
> 사전에 문서화되지 않은 의존 관계도 발견하면 반드시 보고한다.
> 운영 대비 체크: DB 커넥션 누수, 무거운 동기 처리 로직, 로깅 누락(에러 발생 시 추적 가능한 컨텍스트가 있는지), 우아한 종료(Graceful Shutdown) 저해 요소가 없는지 중점적으로 검증하라.

**`*.entity.ts` 또는 `migrations/` 경로 파일이 변경된 경우, `database-reviewer` 에이전트를 추가로 병렬 실행합니다. model은 생략합니다.**

두 에이전트의 결과를 취합하여 CRITICAL/HIGH 이슈를 즉시 수정합니다.
재검토가 필요한 경우 에이전트를 재호출합니다.

---

## STEP 4 — 보안 리뷰 (`security-reviewer` 에이전트, model: **opus**)

인증/권한/사용자 입력 처리 코드가 포함된 경우 **반드시 `security-reviewer` 서브에이전트를 호출합니다. `model: "opus"`를 명시합니다.**

CRITICAL 이슈는 반드시 수정 후 다음 단계로 넘어갑니다.

---

## STEP 5 — 최종 점검

1. CLAUDE.md의 관련 체크리스트 및 프로젝트 컨벤션 확인
2. 변경이 발생한 **패키지 단위**로 테스트를 실행합니다
   - 루트에서 `pnpm test`(= `turbo run test`) 실행 금지 — 전체 패키지 재실행으로 수분 소요
   - 변경된 패키지 디렉토리로 이동 후 해당 패키지 테스트만 실행합니다
   - 예: `cd packages/server && pnpm test`
3. 변경된 파일 목록 최종 요약
4. 후속 작업 또는 알려진 제한사항 기록

---

## STEP 6 — 명세 문서 갱신 (`.claude/.spec/`)

**모든 micro-fix 건은 이 단계를 거칩니다.** 단, 영향도 분류 결과 명세에 영향이 없으면 6.1에서 명시적으로 skip 보고합니다. micro-fix는 정의상 작은 변경이므로 다수가 skip되는 것이 정상입니다.

대상 파일:
- `.claude/.spec/README.md` — 하단 "변경 이력 요약" 섹션과 "환경변수 관련 주요 키" 표
- `.claude/.spec/feature-list.md` — 사용자/관리자 행위 목록 (`§N` 번호 체계)
- `.claude/.spec/functional-spec.md` — 시스템 동작 명세 (`§N.M` 번호 체계)
- `.claude/.spec/api.md` — REST 엔드포인트 단위 스펙 (`§N.M` 번호 체계)

### 6.1 영향도 분류

| 변경 유형 | `README.md` 변경 이력 | `feature-list.md` | `functional-spec.md` | `api.md` 본문 | Postman 컬렉션 |
|---|---|---|---|---|---|
| **새 사용자 행위 신설** (신규 메뉴·기능 추가) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **사용자 행위 제거** (기능 삭제·메뉴 제거) | ✅ | ✅ | ✅ | ✅ | ✅ |
| REST 엔드포인트 동작 변경 (응답 envelope·status·검증 규칙·기본값, 행위는 동일) | ✅ | ❌ | ✅ | ✅ | ✅ |
| 사용자 가시 동작 변경 (메시지·강제 조건·rate limit 정책) | ✅ | ❌ | ✅ | 해당 시 | 해당 시 |
| 환경변수 추가/기본값 변경/제거 | ✅ | ❌ | 해당 시 | ❌ | ❌ |
| 권한 매핑·검증 정책 변경 | ✅ | 해당 시 | ✅ | 해당 시 | 해당 시 |
| 보안 정책 변경 (마스킹·`Cache-Control`·검증 규칙) | ✅ | ❌ | ✅ | 해당 시 | ❌ |
| 알고리즘/구현 결정 (PBKDF2 iteration, chunk 크기 등) | ✅ (1줄) | ❌ | ❌ | ❌ | ❌ |
| 내부 리팩토링·테스트 추가·주석·로깅·타입 정리 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 외부 계약 불변 버그 수정 | 선택(눈에 띄는 사용자 영향이 있으면 ✅) | ❌ | ❌ | ❌ | ❌ |

표 결과 모든 칸이 ❌이면 STEP 6를 종료합니다 (보고에 `명세 영향 없음 — STEP 6 skip` 명시).

### 6.2 `README.md` 변경 이력 추가

기존 패턴을 그대로 따릅니다 (파일 하단 "변경 이력 요약" 섹션 마지막에 한 줄 추가):

```
- **YYYY-MM-DD (버전 식별자)**: 한 줄 핵심 요약 — 영향 받은 영역(features.md §X.Y / api.md §Z 등) 명시.
```

- **날짜**: 사용자 컨텍스트의 `currentDate`를 `YYYY-MM-DD`로 사용
- **버전 식별자**: 기존 `1.0.x` 패턴이 누적 중이면 다음 후보 번호를 제시하고 **사용자 확인**을 받습니다 (micro-fix는 patch 단위가 적절). 사용자가 부여 원치 않으면 생략
- **요약**: 1–2문장. micro-fix는 변경 범위가 작으므로 한 줄이 일반적

### 6.3 `feature-list.md` 본문 갱신

영향도 표에서 `feature-list.md` ✅로 분류된 경우:

- 사용자/관리자가 **할 수 있는 행위** 관점에서 한 줄짜리 항목으로 추가 또는 제거
- 기존 섹션(`§1 인증`, `§2 내 계정 관리` 등)에 위치
- **들어가는 것**: 행위 이름 (예: "사용자 일괄 추가 (JSON Import)")
- **안 들어가는 것**: 권한 키, HTTP 메서드, 검증 규칙, 알고리즘, 기술 상세
- **L1/L2 경계 기준**: 사용자가 할 수 있는 행위가 추가·제거될 때만 갱신. 동작 조건·검증 규칙·권한 변경은 `functional-spec.md`만 갱신.

### 6.3b `functional-spec.md` / `api.md` 본문 갱신

영향도 표에서 `functional-spec.md` 또는 `api.md` ✅로 분류된 경우, 다음 원칙으로 갱신합니다 (`feature-dev` skill의 STEP 9.3b / 9.4와 동일):

- 기존 섹션 번호 체계(§N.M) 유지
- 모드별 동작 차이·권한·rate limit·호환성 깸 여부 명시
- 본문 변경 폭은 micro-fix 범위에 비례하게 최소화 (보통 한 단락 추가/수정)
- **안 들어가는 것**: private helper 이름, iteration 수·chunk 크기 같은 구현 상세 (dev_note에 위임)
- list 엔드포인트가 영향 받았다면 `.claude/rules/rest-api.md` §1.4 체크리스트 항목과 명세가 일치하는지 확인

### 6.4 Postman Collection 갱신

`api.md` 본문이 ✅인 경우, `.claude/.spec/Aw REST API.postman_collection.json`도 함께 갱신합니다.
`api.md`가 ❌인 경우(내부 변경·리팩토링 등)에는 이 단계도 skip합니다.

**절대 수정하지 않는 필드:**
- `info._postman_id`, `info._exporter_id`
- 컬렉션 루트 `auth` (OAuth 2.0 설정 전체)
- `event[]`

**폴더 구조:** api.md 섹션 그룹과 동일하게 유지
(`"1. 인증 (Auth)"`, `"2. 조직 (Orgs)"`, `"3. 워크스페이스 (Workspaces)"`, `"4. 권한 (Permissions)"`, `"5. 사용자 관리 (Users)"`, `"8. 감사 로그 (Audit Logs)"` 등)

**케이스별 처리:**

| 케이스 | 처리 |
|--------|------|
| 신규 엔드포인트 추가 | 해당 폴더의 `item[]` 끝에 아이템 객체 추가 |
| 기존 엔드포인트 변경 (경로·메서드·바디·설명 등) | 기존 아이템 필드 수정 |
| 엔드포인트 삭제 | 해당 아이템 객체 제거 |
| 신규 섹션 그룹 추가 | 루트 `item[]`에 폴더 객체 추가 |

**아이템 구조 패턴 (기존 아이템을 실제 템플릿으로 참조할 것):**
- 이름: `"{섹션번호} {한국어 이름}"` (예: `"8.2 감사 로그 상세 조회"`)
- `auth`: public 엔드포인트만 `{"type": "noauth"}`. 인증 필요 시 생략 (컬렉션 OAuth 상속)
- `url.raw`: `"{{baseUrl}}/api/v1/aw/..."` 형식
- Path variable: URL에 `{{varName}}` + `url.variable[]` 배열에 `{"key": "...", "value": "{{varName}}"}` 추가
- Query parameter: `url.query[]` 배열에 `disabled: true`로 예시 추가
- Body (POST/PUT만): `"mode": "raw"` + `"options": {"raw": {"language": "json"}}`
- `response`: 항상 `[]`
- `description`: api.md 기준 한국어 설명. 인증 요건·rate limit·주요 응답 필드 포함

### 6.5 교차 검증

- `README.md`의 "환경변수 관련 주요 키" 표 갱신 여부 (신규 환경변수 또는 기본값 변경)
- `.env.example`·코드 내 상수와 명세의 기본값 일치 여부
- 권한 키 변경 시 `AwPermissions.ts`와 일치 여부
- `feature-list.md`의 항목 중 `functional-spec.md`에 대응 섹션이 없는 것이 있는지 (L1 ↔ L2 매핑 누락 검출)

### 6.6 보고 형식

```
명세 갱신 결과
- README.md 변경 이력: <한 줄>
- feature-list.md: §N <갱신/신규> — <한 줄 요약> (또는 "해당 없음")
- functional-spec.md: §X.Y <갱신/신규> — <한 줄 요약>
- api.md: §Z <갱신/신규> — <한 줄 요약>
- Postman 컬렉션: <변경된 엔드포인트 목록> (또는 "api.md 없음으로 skip")
```

영향 없음으로 분류된 경우: `명세 영향 없음 — STEP 6 skip`만 보고.

### 6.7 devnote 완료 기록

args의 첫 줄이 `devnote:<파일명>` 형식인 경우, 해당 파일명을 `.claude/.dev_note/.completed`에 기록합니다:

1. `.claude/.dev_note/.completed` 파일을 Read합니다 (없으면 빈 목록으로 처리)
2. 해당 파일명이 이미 있으면 중복 추가하지 않고 종료합니다
3. 없으면 파일 끝에 파일명을 한 줄 추가합니다 (예: `1.0.11_bugfix_1.md`)

이 단계는 명세 영향 없음 여부와 무관하게 항상 수행합니다.

---

위 요청을 바탕으로 STEP 1부터 시작합니다. 관련 파일을 직접 읽으며 질의응답을 진행합니다.
