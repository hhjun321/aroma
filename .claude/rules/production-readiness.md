# 운영 환경 대비 구현 규칙 (Production-Readiness)

이 규칙은 상용 환경(Production)에서 시스템이 다운되거나 성능 병목이 발생하는 것을 방지하기 위한 백엔드 필수 준수 사항입니다. 모든 구현 및 리뷰 단계에서 이 규칙을 적용합니다.

## 1. 대용량 데이터 및 메모리 방어 (OOM 방지)
- 수천 건 이상의 JSON 데이터를 파싱하거나 DB에 Insert/Export 할 때는 한 번에 메모리에 올리지 말고 반드시 **Chunk/Batch 처리** 또는 **Stream API**를 사용한다.
- 배열(Array)에 끝없이 데이터를 `push`하는 로직은 지양하고, GC(Garbage Collector)가 작동할 수 있는 구조로 작성한다.

## 2. 에러 핸들링과 로깅 (Observability)
- `try-catch`에서 에러를 잡았을 때, 단순히 에러 메시지만 던지지 말고 **'어떤 유저가/어떤 파라미터로'** 호출하다 터졌는지 추적 가능한 컨텍스트를 로거(`logger.error`)에 남긴다.
- 서버 프로세스를 죽일 수 있는 `uncaughtException`이나 `unhandledRejection`을 방지하기 위해 비동기 로직의 에러는 최상단 또는 Global Exception Handler로 안전하게 위임한다.

## 3. 타임아웃과 서킷 브레이커 (Resilience)
- 외부 API나 서드파티 서비스를 호출할 때는 무한정 기다리며 커넥션을 물고 있지 않도록 반드시 **타임아웃(Timeout)** 을 명시적으로 설정한다.

## 4. 기존 규칙과의 연계
- Node.js 이벤트 루프 블로킹 방지는 `nodejs-hotpath.md`를 우선 따른다.
- 보안 및 입력값 검증은 `validation.md`를 우선 따른다.
- API 레벨의 속도 제한 및 SQL 인젝션 방어는 `rest-api.md`를 우선 따른다.