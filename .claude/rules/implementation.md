# Igloo 구현 규칙

이 프로젝트(igloo_workflow)에서 코드를 작성할 때 반드시 준수해야 하는 구현 규칙입니다.

---

## 아키텍처 제약

- **Flowise 코어 파일 직접 수정 금지**: igloo 동작은 `src/igloo/`와 `src/index.ts`의 두 미들웨어 블록으로만 확장합니다. Flowise 원본 파일을 수정하면 upstream 업데이트 시 충돌이 발생합니다.
- **igloo 로직은 `src/igloo/` 하위에서만 작성**: 위 원칙의 구체적 위치 규칙입니다.
- **`enterprise/` 경로 import 금지**: 반드시 igloo 자체 구현체를 사용합니다.
  - 권한 체크: `IglooPermissionCheck` 사용 (enterprise의 PermissionCheck 사용 금지)
  - 타입: `IglooAuthTypes`의 `IglooLoggedInUser` 사용
- **`repo.save()` 대신 `repo.update()` 사용**: `select:false`로 설정된 필드(예: `passwordHash`, `passwordSalt`)가 덮어써지는 것을 방지합니다. 신규 엔티티 INSERT는 `repo.save()` 그대로 사용합니다.

---

## TypeORM 엔티티 컬럼 타입

igloo는 SQLite / PostgreSQL / MySQL / MariaDB 4개 엔진을 모두 지원합니다. 엔티티에 컬럼 타입을 명시할 때 반드시 **크로스-DB 호환 타입**만 사용합니다.

| 용도 | 올바른 방법 | 잘못된 예 |
|------|-----------|----------|
| 날짜/시간 (수동 관리) | `@Column({ type: 'timestamp', nullable: true })` | `type: 'datetime'` — PostgreSQL 미지원 |
| 날짜/시간 (타입만 생략) | ❌ 사용 금지 — `Date \| null` 유니온은 reflect-metadata가 `Object`로 emit하여 런타임 오류 발생 | `@Column({ nullable: true })` (type 없이) |
| 생성/수정 시각 (자동) | `@CreateDateColumn()` / `@UpdateDateColumn()` | 직접 `type` 명시 |
| 문자열 | `type: 'varchar'` | `type: 'nvarchar'` 등 특정 DB 전용 타입 |

- `type: 'datetime'` — MySQL/SQLite에서만 동작, **PostgreSQL에서 런타임 오류 발생**
- `type` 생략 + `Date | null` 유니온 — reflect-metadata가 `Object`로 emit → **PostgreSQL에서 런타임 오류 발생**
- **날짜 컬럼은 반드시 `type: 'timestamp'`를 명시** — 4개 DB 모두 지원
- 엔티티 컬럼 타입과 마이그레이션 SQL이 논리적으로 일치하는지 교차 확인할 것

---

## DB 마이그레이션

igloo 엔티티에 필드를 추가하거나 변경할 경우 **4개 DB 엔진 모두** 마이그레이션 파일을 작성해야 합니다:

- `packages/server/src/igloo/database/migrations/sqlite/`
- `packages/server/src/igloo/database/migrations/postgres/`
- `packages/server/src/igloo/database/migrations/mysql/`
- `packages/server/src/igloo/database/migrations/mariadb/`

기존 `1774828800000-IglooInit.ts` 파일을 템플릿으로 참조하세요.

---

## 권한 추가 시 동시 수정 항목

새로운 igloo 권한(`igloo.menu.*` 등)을 추가할 때는 다음 파일을 **반드시 동시에 수정**해야 합니다:

1. `src/igloo/rbac/IglooPermissions.ts` — `IGLOO_PERM` enum에 상수 추가
2. `src/igloo/rbac/IglooPermissions.ts` — `PERM_MAP`에 Flowise 리소스 권한 매핑 추가

---

## 사이드바 메뉴 추가 체크리스트

새 사이드바 메뉴 항목을 추가할 때 **반드시 동시에 수정**합니다:

1. `src/menu-items/dashboard.js` — 항목에 `iglooMenuId: 'igloo.menu.<name>'` 추가
2. `src/igloo/rbac/IglooPermissions.ts` — `IGLOO_PERM` enum에 상수 추가 (`igloo.menu.*` prefix이면 `ALL_MENU_PERMISSIONS`에 자동 포함됨)
3. `src/igloo/rbac/IglooPermissions.ts` — `PERM_MAP`에 Flowise 리소스 권한 매핑 추가

**Bundled read permissions**: 새 페이지가 다른 메뉴 권한 소유의 API를 호출해야 하는 경우:
- 해당 read-only Flowise 권한을 호출하는 쪽 메뉴의 `PERM_MAP` 항목에 추가
- Write 권한(create/update/delete)은 공유하지 않음 — 소유 메뉴에만 부여

---

## `igloo.workspace.manager` 역할 규칙

`igloo.workspace.manager`는 **반드시 `igloo.menu.org_management`와 함께 부여/해제**합니다.
`IglooOrgService.setWsMemberRole()`이 이 두 권한을 원자적으로 처리합니다.
직접 DB를 조작하거나 별도 API를 만들 때도 반드시 둘을 함께 처리해야 합니다.

---

## 테스트

- 테스트 파일 위치: `packages/server/test/unit/igloo/` 하위에 `*.test.ts` 형태로 작성
  - jest.config.js의 `roots: ['<rootDir>/test']` 설정으로 `src/` 하위는 스캔되지 않음
- 테스트 실행: 변경된 패키지 디렉토리에서 해당 파일만 지정하여 실행 (루트 `pnpm test` 금지 — turbo 전체 재실행)
  - 예: `cd packages/server && pnpm exec jest --testPathPattern=AwUserService`
  - ⚠️ `pnpm test -- --testPathPattern=X` 형태 사용 금지 — pnpm이 `--`를 jest에 그대로 전달해 path 필터가 무시됨

---

## 보안 체크리스트

인증/권한/사용자 입력 처리 코드를 작성한 경우 다음 항목을 반드시 확인합니다:

- [ ] JWT 검증 로직이 올바른지 (`IglooAuthGuard`를 경유하는지)
- [ ] 권한 체크가 누락된 엔드포인트가 없는지
- [ ] TypeORM 파라미터 바인딩을 사용하는지 (SQL 인젝션 방지)
- [ ] 모든 사용자 입력값이 검증되는지
- [ ] 에러 메시지에 민감 정보가 노출되지 않는지
- [ ] 민감한 응답(임시 비밀번호 등)에 `Cache-Control: no-store` 헤더가 설정되어 있는지

---

## 기능 추가 최종 체크리스트

기능 구현 완료 후 다음 항목을 확인합니다:

- [ ] 사이드바 메뉴를 추가한 경우 위 "사이드바 메뉴 추가 체크리스트" 3단계를 충족하는지
- [ ] DB 마이그레이션이 필요한 경우 4개 엔진 모두 파일이 존재하는지
- [ ] 기존 테스트가 모두 통과하는지 (`pnpm test`)
