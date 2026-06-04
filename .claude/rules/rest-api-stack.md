# REST API 구현 규칙 — 프로젝트 스택 특화

이 문서는 [rest-api.md](rest-api.md)의 공통 REST API 규약을 **이 프로젝트의 기술 스택**(Express + TypeORM + Aw 인증 레이어) 위에서 구현할 때 적용하는 규칙을 정의합니다. 공통 원칙은 [rest-api.md](rest-api.md)에서 먼저 확인한 뒤 본 문서를 참고하세요.

> 입력 밸리데이션 규칙은 [validation.md](validation.md)와 함께 적용합니다.

---

## 1. 라우트 등록 순서 (Express)

Express 라우팅은 **등록 순서대로 매칭**되므로, 정적 path는 동적 `:param` 보다 **항상 먼저** 등록합니다.

```ts
// 올바른 순서
router.get('/users/export', ...)   // 정적
router.get('/users/import', ...)   // 정적
router.get('/users/:userId', ...)  // 동적 — 가장 마지막

// 잘못된 순서 — '/:userId'가 'export'를 잡아버림
router.get('/users/:userId', ...)
router.get('/users/export', ...)   // unreachable
```

새 정적 경로를 추가할 때는 `'/:param'` 위에 위치했는지 반드시 확인합니다.

---

## 2. 인증·인가 (AwAuthGuard)

- 모든 비공개 엔드포인트는 **router 단에서 `AwAuthGuard` 적용** (`router.use(AwAuthGuard)`)
- 라우터 핸들러는 `req.awUser!.id`로 요청자 식별
- 권한 체크는 **서비스 레이어**에서 수행 — 라우터는 thin
- 권한 부족 시 403, 인증 실패 시 401 (Guard가 자동 처리)
- 워크스페이스/조직 등 scope 리소스는 path에 포함하지 않고 `req.awUser.activeWorkspaceId` 등 인증 컨텍스트로 결정 (본 프로젝트 컨벤션)
- **권한 체크가 누락된 엔드포인트는 보안 결함** — 신규 라우트 추가 시 반드시 검토

---

## 3. Express 입력 처리 패턴

- **Body**: `req.body ?? {}`로 destructure (body 미전송 시 undefined 방어)
- **Query**: 모든 값이 `string | string[] | undefined` — 명시적으로 `asString`/`Number()`/boolean 변환 후 사용
- **Path param**: `req.params.<name>` — 빈 문자열 가능성 거의 없으나 길이/형식 검증 권장

다중 값 query는 CSV(`?ids=a,b,c`) **및** 반복(`?ids=a&ids=b`) 양식 모두 수용:

```ts
// canonical: AwUserRouter export endpoint
let userIds: string[] | undefined
const raw = req.query.userIds
if (Array.isArray(raw)) userIds = raw.filter((v): v is string => typeof v === 'string')
else if (typeof raw === 'string' && raw.length > 0) userIds = raw.split(',').map(s => s.trim()).filter(Boolean)
```

---

## 4. TypeORM 패턴

### 4.1 페이지네이션 + total 카운트

페이지네이션의 total 카운트는 `getManyAndCount()`로 한 번에 조회합니다:

```ts
// canonical: AwUserService.listUsers
const page = Math.max(1, query.page ?? 1)
const limit = Math.min(100, Math.max(1, query.limit ?? 20))
const skip = (page - 1) * limit

const [data, total] = await qb.skip(skip).take(limit).getManyAndCount()
return { data, meta: { total, page, limit } }
```

### 4.2 정렬 (화이트리스트 → enum/literal)

`ORDER BY` 절에 사용자 입력 문자열을 직접 보간하면 SQL 인젝션 표면이 됩니다. 화이트리스트 통과 후 enum/literal로 사용:

```ts
// canonical: AwUserService.listUsers
const sortBy = resolveSortColumn(query.sortBy) // 화이트리스트 통과 또는 fallback
const sortDir = query.sortDir === 'asc' ? 'ASC' : 'DESC'
qb.orderBy(`u.${sortBy}`, sortDir)
if (sortBy !== 'createdAt') qb.addOrderBy('u.createdAt', 'DESC')
```

### 4.3 파라미터 바인딩

- **모든 사용자 입력은 TypeORM 파라미터 바인딩**으로 전달 (`:param` placeholder 또는 Repository API)
- `qb.where(\`u.field = '${userInput}'\`)` 같은 문자열 보간 금지
- `LIKE` 패턴의 `%`는 서버에서 wrapping (사용자가 `%`를 입력해도 표면적으로는 부분일치로 동작)

### 4.4 트랜잭션

- 여러 테이블에 걸친 mutation은 **반드시 트랜잭션** 안에서 실행 (`AwDataSource.transaction(...)`)
- 외부에 보이는 `*Member` 데이터를 만들기 전에 부모 리소스(orgs, workspaces) 먼저 INSERT — FK 위반 방지

---

## 5. 에러 응답 (서비스 ↔ 라우터 분리)

- 서비스 레이어에서는 `Object.assign(new Error(msg), { statusCode: status })` 패턴으로 status를 함께 throw
- 라우터 catch 블록은 `awErrorResponse(res, err, '[태그]')` 헬퍼로 위임 (`src/aw/utils/awErrorResponse.ts`)
- 5xx: `logger.error` 기록 후 응답은 `'Internal server error'`로 마스킹
- 인라인 4xx 거부(입력 검증 실패 등): `awErrorJson(res, statusCode, message)` 사용 (4xx 전용)
- 응답 shape: `{ statusCode, success: false, message, code? }` — Flowise 코어 에러 포맷과 동일한 키

```ts
// canonical: AwUserRouter
} catch (err: any) {
    awErrorResponse(res, err, '[AwUser] GET /users')
}
```

---

## 6. Rate Limiting (express-rate-limit)

다음 카테고리에는 rate limiter 적용이 **권장 또는 필수**:

| 카테고리 | 정책 권장값 |
|---------|------------|
| 인증 (`/auth/login`, `/auth/refresh`) | IP/계정당 분당 5–10회 |
| 비밀번호 초기화 / 변경 | IP당 분당 5회 |
| Bulk import / export | IP당 분당 30회 (canonical: `importExportLimiter`) |
| 관리자 전용 mutation | 필요 시 적용 |

- `express-rate-limit` 사용
- 환경변수로 윈도우/최대 횟수 노출 (`AW_*_RATE_WINDOW_MS`, `AW_*_RATE_MAX`)
- 거부 시 429와 한국어 에러 메시지 반환

---

## 7. Body 크기 제한 (expressJson)

- 기본 JSON 파서로 처리되는 일반 mutation은 그대로 사용
- **Bulk import / 대용량 페이로드** 라우트에는 **별도 `expressJson({ limit: '<X>mb' })` 파서 명시**
- 한도는 서버 배치 한도와 정합되게 산정 (예: 100명 × 평균 row 크기 → 2MB)

```ts
const importBodyParser = expressJson({ limit: '2mb' })
router.post('/import', limiter, importBodyParser, handler)
```

---

## 8. 리소스 경로 컨벤션

- 본 프로젝트의 Aw 레이어 리소스는 **`/aw/<resource>`** prefix를 사용 (예: `/aw/users`, `/aw/organizations`, `/aw/users/import-export`)
- 이는 Flowise 코어 라우트와의 충돌을 피하기 위한 컨벤션
- 신규 Aw 엔드포인트는 반드시 이 prefix 아래에 등록

---

## 9. 테스트 위치 / 실행

- 테스트 위치: `packages/server/test/unit/aw/` 하위에 `*.test.ts`
- 실행: 변경된 패키지 디렉토리에서 해당 파일만 지정
  - 예: `cd packages/server && pnpm test -- --testPathPattern=AwUserService`
- 루트 `pnpm test` 금지 — turbo 전체 재실행이 됨

---

## 10. 신규 엔드포인트 스택-특화 체크리스트

[rest-api.md](rest-api.md)의 공통 체크리스트에 **추가로** 확인:

- [ ] 정적 경로가 `'/:param'` 보다 위에 등록되어 있음 (§1)
- [ ] `AwAuthGuard` 적용 + 서비스 레이어 권한 체크 (§2)
- [ ] Aw 리소스는 `/aw/` prefix 아래에 등록 (§8)
- [ ] 서비스 레이어에서 `statusCode`를 함께 throw, 라우터가 분기 (§5)
- [ ] TypeORM 파라미터 바인딩 사용, ORDER BY는 화이트리스트 (§4)
- [ ] 멀티-테이블 mutation은 트랜잭션 (§4.4)
- [ ] 인증/Bulk 라우트에 `express-rate-limit` (§6)
- [ ] Bulk 라우트에 `expressJson({ limit })` (§7)
- [ ] 테스트는 `packages/server/test/unit/aw/`에 작성 (§9)

---

## 참고: Canonical 구현체

본 규칙의 모든 패턴은 다음 구현체를 표준으로 합니다. 새 엔드포인트 작성 시 이 파일들을 템플릿으로 참고하세요.

- `packages/server/src/aw/users/AwUserRouter.ts` — 라우터 패턴 (정적/동적 순서, 에러 처리, rate limiter)
- `packages/server/src/aw/users/AwUserService.ts` — `listUsers`: 페이지네이션 + 필터 + 정렬 통합 구현
- `packages/server/src/aw/users/AwUserTypes.ts` — `ListUsersQuery`, `SortableColumn` 화이트리스트 선언
- `packages/server/test/unit/aw/AwUserService.test.ts` — 목록 조회 테스트 시나리오
