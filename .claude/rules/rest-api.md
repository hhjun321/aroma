# REST API 구현 규칙 — 공통

스택-특화 구현 패턴(Express + TypeORM + Aw 인증)은 [rest-api-stack.md](rest-api-stack.md)를 참고합니다.
입력 밸리데이션은 [validation.md](validation.md)와 함께 적용합니다.

---

## 1. List 엔드포인트 필수 기능

다수 데이터를 반환하는 모든 GET 엔드포인트는 **요청 여부와 무관하게** 포함:

**페이지네이션**: `page`(1-base) + `limit`. 기본값 `page=1` / `limit=20`. `limit ≤ 100` 클램프.  
응답 envelope: `{ data: [...], meta: { total, page, limit } }`

**필터링**: 단일 `search` 토큰 또는 컬럼별 필터. 빈 문자열·공백만 입력 → 미지정으로 처리. 날짜 형식 오류 → 400.

**정렬**: `sortBy`(화이트리스트 필수) + `sortDir`. 비허용 값 → 기본값 fallback. 보조 정렬 키 추가.  
⚠️ `ORDER BY`에 사용자 입력 직접 보간 금지.

### 체크리스트
- [ ] `page`, `limit` 기본값·클램프 적용
- [ ] `{ data, meta: { total, page, limit } }` envelope
- [ ] 빈 문자열·공백 미지정으로 정규화
- [ ] `sortBy` 화이트리스트, 비허용 값 fallback
- [ ] 보조 정렬 키 지정
- [ ] 권한 scope에 따른 결과 필터링

---

## 2. HTTP 메서드 / 상태 코드

| 메서드 | 시맨틱 | 성공 코드 |
|--------|--------|----------|
| GET | 조회 | 200 |
| POST | 생성/액션 | 201 (생성) / 200 (액션) |
| PUT / PATCH | 수정 | 200 |
| DELETE | 삭제 | 200 + `{ ok: true }` |

| 코드 | 의미 |
|------|------|
| 400 | 입력 오류 / 비즈니스 규칙 위반 |
| 401 | 인증 없음/만료 |
| 403 | 권한 부족 |
| 404 | 리소스 없음 |
| 409 | 중복 (이메일 등) |
| 429 | Rate limit 초과 |
| 500 | 서버 오류 (메시지 마스킹) |

---

## 3. 리소스 네이밍

- 복수 명사 + kebab-case: `/users`, `/organizations`, `/users/import-export`
- 동사는 액션이 명확할 때만: `/auth/login`, `/users/:id/reset-password`

---

## 4. 에러 응답 형식

```json
{ "error": "Human-readable message" }
```
- 4xx: 사용자에게 보일 메시지 그대로
- 5xx: 항상 `"Internal server error"` 마스킹, 서버 로그에만 원본 기록

---

## 5. 보안 헤더

- **민감 정보 응답** (임시 비밀번호, 토큰, 개인정보 export 등): `Cache-Control: no-store`
- **파일 다운로드**: `Content-Disposition: attachment; filename="..."` + 적절한 `Content-Type`
- **에러 응답**: 스택 트레이스, 파일 경로, DB 쿼리, 환경변수 노출 금지

---

## 6. 구현 세부 (rest-api-stack.md 참고)

입력 처리, Rate limiting, Body 크기 제한, 트랜잭션, SQL 인젝션 방지, 라우팅 충돌 회피의 구체적 구현은 [rest-api-stack.md §3–§9](rest-api-stack.md) 참고.

---

## 7. 신규 엔드포인트 최종 체크리스트

- [ ] (List API) §1 체크리스트 충족
- [ ] HTTP 메서드·상태 코드 시맨틱 정합 (§2)
- [ ] 리소스 경로 복수 명사 + kebab-case (§3)
- [ ] 에러 응답 `{ error: string }` envelope, 5xx 마스킹 (§4)
- [ ] 민감 응답에 `Cache-Control: no-store` (§5)
- [ ] [validation.md](validation.md) table-stakes 밸리데이션 적용
- [ ] [rest-api-stack.md §10](rest-api-stack.md) 스택-특화 체크리스트 추가 충족
