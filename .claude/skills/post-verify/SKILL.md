---
name: post-verify
description: >
  절차 없이 이미 작성된 코드를 사후 검증합니다.
  feature-dev / micro-fix를 거치지 않고 직접 구현한 경우, 이 스킬로 거쳤어야 할
  코드 리뷰 · 보안 리뷰 · 최종 점검을 소급 적용합니다.
  구현 단계(STEP 5)는 이미 완료된 것으로 간주하고 건너뜁니다.
  Trigger phrases: "post-verify", "/post-verify", "사후 검증", "검증해줘", "리뷰 돌려줘"
---

# 사후 검증 파이프라인 (Post-Verify)

절차 없이 작성된 코드를 소급하여 검증합니다.
feature-dev의 STEP 6 · 7 · 8을 그대로 적용합니다.

args에 전달하는 내용에 따라 세 가지 방식으로 검증 범위를 지정할 수 있습니다.

> **[방식 1] 기능 설명** — 구현한 내용을 자연어로 설명
> `/post-verify 전화번호 밸리데이션 로직을 IglooUserRouter에 추가했어`
>
> **[방식 2] 경로 지정** — 파일 또는 디렉토리 경로를 직접 입력
> `/post-verify packages/server/src/igloo/users/IglooUserRouter.ts`
> `/post-verify packages/server/src/igloo/users/`
>
> **[방식 3] 자동 감지** — args 없이 호출하면 git diff로 변경 범위 추출
> `/post-verify`

---

## 모델 사용 정책

| 단계 | 에이전트 | 모델 | 이유 |
|------|---------|------|------|
| STEP 2 | `code-reviewer`, 언어별 리뷰어 | **sonnet** (기본값) | 코딩 품질 체크는 Sonnet이 최적 |
| STEP 3 | `security-reviewer` | **opus** | 놓쳤을 때 비용이 크고, 전체 코드 패턴을 꿰뚫어야 함 |

---

## STEP 1 — 검증 범위 확정

args 내용을 보고 아래 세 가지 방식 중 하나를 적용합니다.

---

### [방식 1] 기능 설명이 args로 전달된 경우

args가 자연어 설명인 경우 (파일 경로 형태가 아닌 문장 또는 기능 묘사):

1. 설명에서 관련 파일과 모듈을 추론합니다
   - 언급된 클래스명, 라우터명, 기능명 등을 단서로 코드베이스에서 대상 파일을 탐색합니다
2. 추론된 파일 목록을 사용자에게 보여주고 확인합니다:
   - "다음 파일을 검증 대상으로 파악했습니다. 맞나요? [목록]"
   - 누락된 파일이 있으면 추가합니다
3. **확인을 받은 후** 해당 파일을 직접 읽고 STEP 2로 진행합니다

---

### [방식 2] 파일/디렉토리 경로가 args로 전달된 경우

args가 경로 형태인 경우 (`/`, `.ts`, `.tsx`, `.js` 등 포함):

1. 지정된 경로의 파일을 직접 읽어 현재 상태를 파악합니다
2. 파악한 내용을 요약합니다: "검증 대상: [파일 목록]. 검증을 시작합니다."
3. 바로 STEP 2로 진행합니다 (별도 확인 불필요)

---

### [방식 3] args 생략 시 (자동 감지)

1. `git diff --name-only HEAD`를 실행하여 변경된 파일 목록을 추출합니다
2. 추출된 목록을 사용자에게 보여주고 다음을 확인합니다:
   - 이 중 검증할 파일이 어떤 것인지
   - 목록 외에 추가로 포함할 파일이 있는지
3. **사용자의 확인을 받은 후** 해당 파일을 직접 읽고 STEP 2로 진행합니다

---

## STEP 2 — 코드 리뷰 (`code-reviewer` + 언어별 리뷰어 에이전트 병렬 실행, model: **sonnet**)

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
>
> 추가로, 변경된 로직에 대한 테스트가 존재하는지 확인한다. 테스트가 없으면 HIGH 이슈로 보고한다.

**`*.entity.ts` 또는 `migrations/` 경로 파일이 변경된 경우, `database-reviewer` 에이전트를 추가로 병렬 실행합니다. model은 생략합니다.**

두 에이전트의 결과를 취합하여 CRITICAL/HIGH 이슈를 즉시 수정합니다.
재검토가 필요한 경우 에이전트를 재호출합니다.

---

## STEP 3 — 보안 리뷰 (`security-reviewer` 에이전트, model: **opus**)

인증/권한/사용자 입력 처리 코드가 포함된 경우 **반드시 `security-reviewer` 서브에이전트를 호출합니다. `model: "opus"`를 명시합니다.**

CRITICAL 이슈는 반드시 수정 후 다음 단계로 넘어갑니다.

---

## STEP 4 — 최종 점검

1. CLAUDE.md의 관련 체크리스트 및 프로젝트 컨벤션 확인
2. 변경이 발생한 **패키지 단위**로 테스트를 실행합니다
   - 루트에서 `pnpm test`(= `turbo run test`) 실행 금지 — 전체 패키지 재실행으로 수분 소요
   - 변경된 패키지 디렉토리로 이동 후 해당 패키지 테스트만 실행합니다
   - 예: `cd packages/server && pnpm test`
3. 변경된 파일 목록 최종 요약
4. 후속 작업 또는 알려진 제한사항 기록

---

위 요청을 바탕으로 STEP 1부터 시작합니다. args를 확인하여 방식 1 · 2 · 3 중 하나를 적용합니다.
