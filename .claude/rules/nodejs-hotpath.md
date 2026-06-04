# Node.js Hot-path 점검 규칙

이 프로젝트(igloo_workflow)는 Node.js + Express 기반의 단일 이벤트 루프 환경에서 동작합니다. 다음 트리거 조건에 해당하는 작업을 **신규 작성·수정·리뷰**할 때는, 사용자가 명시적으로 요구하지 않더라도 **동기 API의 이벤트 루프 블로킹 가능성을 반드시 점검**합니다.

본 룰의 목적은 작성 시점에 잠깐 멈춰 점검하게 만드는 것입니다. 매번 깊은 분석을 강요하지 않습니다 — 트리거가 맞을 때만 짧게 점검하고, 위험이 있으면 사용자에게 명시적으로 alert.

---

## 1. 점검 트리거 (해당하면 반드시 점검)

다음 작업이 코드에 등장하면 트리거 조건 충족:

### 암호화·해시
- 비밀번호 해시·검증 (`pbkdf2`, `bcrypt`, `scrypt`, `argon2`)
- 대칭/비대칭 암호화 (`crypto.createCipheriv`, `crypto.publicEncrypt` 등)
- HMAC, 디지털 서명 (`crypto.sign`)
- 키 도출·키페어 생성

### 데이터 직렬화·파싱
- 큰 페이로드 `JSON.parse` / `JSON.stringify` (>100KB 또는 사용자 입력 기반)
- XML, YAML 파서
- CSV/Excel 파싱

### 파일·압축
- 동기 파일 I/O (`fs.readFileSync`, `fs.writeFileSync`, `fs.statSync` 등)
- 동기 압축·해제 (`zlib.gzipSync`, `zlib.deflateSync`, `zlib.brotliCompressSync`)

### 정규식·문자열
- 사용자 입력에 정규식 적용 + 백트래킹 발생 가능 패턴 (ReDoS 위험)
- 큰 문자열의 `replace`/`split`/`match` 반복 호출

### 이미지·PDF·미디어
- 이미지 디코딩·리사이즈 (`sharp` 동기 호출, `jimp`)
- PDF 파싱·렌더링
- 비디오 메타데이터 추출

### 외부 명령
- 동기 child_process (`execSync`, `spawnSync`)

---

## 2. 점검 체크리스트

트리거 조건에 해당하면 다음을 순서대로 확인:

- [ ] **단건 비용 측정**: 1회 호출에 몇 ms 걸리는가? (PBKDF2 310k ≈ 5–15ms, bcrypt cost 12 ≈ 50ms, JSON.parse 1MB ≈ 10–30ms 등)
- [ ] **동시 호출 시나리오 추정**: 동시 N건 시나리오로 곱해서 누적 비용 추정
  - 로그인 폭주 (월요일 아침 100~1000명 동시)
  - bulk import (사용자 일괄 생성, 100~1000건)
  - 외부 webhook 폭주
- [ ] **누적 비용이 50ms를 넘는가?**: hot-path. main thread에 두면 안 됨
- [ ] **동기 API → 비동기 API 대체 가능한가?**: Node `crypto`·`fs`·`zlib`은 대부분 비동기 버전 제공 (`pbkdf2Sync` → `pbkdf2`, `readFileSync` → `readFile`/`promises.readFile`)
- [ ] **비동기로도 부족하면 worker_threads로 분리**: CPU 자체가 무거우면 libuv worker pool(기본 4 threads) 또는 `worker_threads` 모듈
- [ ] **`UV_THREADPOOL_SIZE` 가이드 필요한가?**: 비동기 crypto/fs는 libuv worker pool에서 실행. 8코어 서버라면 환경변수로 8 또는 16으로 상향 권장

---

## 3. 위반 시 운영 영향

체감 가능한 실제 사례:

- `pbkdf2Sync` 1회 ≈ 10ms — 단건은 무해
- 동시 100명 로그인 → 약 1초 동안 **헬스체크·다른 모든 요청 정지** (Express는 single-threaded)
- 동시 1000명 → 5–15초 정지 → K8s probe 실패·SLA 위반·다른 사용자 timeout

이는 sync API가 main thread를 점유하기 때문이며, **알고리즘 비용 자체와는 별개**입니다. 동일 알고리즘을 async로 호출하면 libuv worker pool에서 병렬 실행되어 main thread는 다른 요청을 계속 처리할 수 있음.

---

## 4. 점검 결과 보고 방식

트리거 조건에 해당하는 코드를 작성·수정·리뷰할 때:

1. 점검을 **명시적으로 한 줄 보고**합니다 — "Node hot-path 점검: PBKDF2 호출 발견. 동시 100명+ 시 이벤트 루프 정지 가능성. async 권장."
2. 위험이 있으면 **그 자리에서 사용자에게 결정 요청**: 그대로 sync로 갈지, async로 전환할지, 별도 패치 노트로 분리할지.
3. 위험이 없으면(예: 단건 호출 + 호출 빈도 매우 낮음) **간단히 "점검 완료, 위험 없음" 한 줄**로 끝냅니다.
4. 보고 없이 침묵하고 넘어가면 안 됩니다 — 점검 자체가 본 룰의 목적.

---

## 5. 적용 범위

- 본 룰은 **백엔드(`packages/server/`)**에 우선 적용. 프론트엔드는 브라우저 환경이라 다른 트레이드오프(메인 스레드 freeze)가 있지만 본 룰의 직접 대상은 아님.
- 외부 라이브러리 사용도 본 룰의 트리거 — 라이브러리가 내부적으로 동기 호출을 하는지 확인 (예: `bcryptjs`는 pure-JS sync, `bcrypt`는 native async).

---

## 6. 참고 사례 (본 프로젝트 내)

- `.claude/.dev_note/1.0.10_ver.md` — `pbkdf2Sync` → `pbkdf2`(async) 전환 패치 (이벤트 루프 블로킹 해소)
- `.claude/.dev_note/1.0.6_ver.md` — shared bulk import에서 PBKDF2 호출 횟수 N → 1 단축 (호출 횟수 최적화)
- `.claude/.dev_note/1.0.4_review-bulk-insert.md` — 사용자 import 병목 분석에서 PBKDF2가 진짜 병목임을 식별
