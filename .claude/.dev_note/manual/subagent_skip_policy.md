# 서브에이전트 훅 스킵 정책

## 배경

`~/.claude/settings.json`에 등록된 훅은 **메인 세션과 서브에이전트 세션 모두**에서 실행된다.
code-reviewer 같은 서브에이전트가 `git diff`, `git log` 등 Bash 호출을 할 때마다
`tmux-reminder`, `git-push-reminder`, `commit-quality`, `auto-tmux-dev` 같은
**개발자 워크플로우 전용 훅**이 불필요하게 실행되어 응답 속도가 저하된다.

Bash 호출 1회당 추가 오버헤드 예시:
- `auto-tmux-dev.js` — legacy spawn (~50–100ms)
- `pre-bash-git-push-reminder.js` — legacy spawn (~50–100ms)
- `pre-bash-commit-quality.js` — in-process이나 ESLint 실행 포함

code-reviewer가 10회 Bash 호출 시 → 약 1–2초 순수 훅 오버헤드 발생.

---

## 핵심 원리

Claude Code는 훅 실행 시 **`CLAUDE_AGENT_NAME` 환경변수**를 설정한다.

- 메인 세션: `CLAUDE_AGENT_NAME` 미설정 (빈 값)
- 서브에이전트: `CLAUDE_AGENT_NAME=<에이전트명>` (예: `code-reviewer`, `planner`)

이 변수를 기준으로 서브에이전트를 감지하고 불필요한 훅을 스킵한다.

---

## 적용 방법

### 방법 A: `hook-flags.js` 수정 (run-with-flags 경유 훅 전체 적용)

`~/.claude/scripts/lib/hook-flags.js`의 `isHookEnabled()` 함수에 서브에이전트 감지를 추가한다.

```js
// 추가할 함수
function isSubAgent() {
  return !!process.env.CLAUDE_AGENT_NAME;
}

// isHookEnabled() 내부 수정
function isHookEnabled(hookId, options = {}) {
  const id = normalizeId(hookId);
  if (!id) return true;

  const disabled = getDisabledHookIds();
  if (disabled.has(id)) return false;

  // 서브에이전트는 minimal 프로필로 강제 다운그레이드
  // → standard/strict 훅 전체 스킵
  const profile = isSubAgent() ? 'minimal' : getHookProfile();
  const allowedProfiles = parseProfiles(options.profiles);
  return allowedProfiles.includes(profile);
}

// module.exports에 isSubAgent 추가
module.exports = { ..., isSubAgent };
```

**효과**: `run-with-flags.js`를 통해 등록된 `standard` 또는 `strict` 프로필 훅이
서브에이전트에서 모두 스킵된다.

스킵 대상 예시:
- `pre:bash:git-push-reminder` (standard,strict)
- `pre:bash:commit-quality` (standard,strict)
- `pre:governance-capture` (standard,strict)
- `post:governance-capture` (standard,strict)
- `post:bash:pr-created` (standard,strict)
- `post:bash:build-complete` (standard,strict)
- `post:quality-gate` (standard,strict)
- `pre:mcp-health-check` (standard,strict)
- `pre:config-protection` (standard,strict)

`minimal,standard,strict` 프로필 훅(세션 종료 훅 등)은 계속 실행된다.

---

### 방법 B: 개별 스크립트에 직접 가드 추가 (run-with-flags 미경유 훅)

`auto-tmux-dev.js`처럼 `settings.json`에서 직접 호출되는 훅은
`hook-flags.js` 수정만으로는 스킵되지 않는다.
해당 스크립트의 stdin `end` 핸들러 최상단에 아래 코드를 추가한다.

```js
process.stdin.on('end', () => {
  // 서브에이전트에서는 즉시 pass-through
  if (process.env.CLAUDE_AGENT_NAME) {
    process.stdout.write(raw); // 변수명은 스크립트마다 다를 수 있음 (raw, data 등)
    return;                     // process.exit(0) 필요한 경우도 있음
  }

  // 기존 로직...
});
```

적용 대상 파일:
- `auto-tmux-dev.js`
- `pre-bash-git-push-reminder.js`
- `pre-bash-tmux-reminder.js`
- 기타 `run-with-flags` 미경유 직접 훅 스크립트

---

## 스킵하지 않는 훅

서브에이전트에서도 유의미한 훅은 `minimal,standard,strict` 프로필로 등록하거나
가드를 추가하지 않아 계속 실행되게 한다.

| 훅 | 이유 |
|---|---|
| `stop:session-end` 등 Stop 훅 | 세션 종료 시 한 번만 실행, 오버헤드 무관 |
| `block-no-verify` | git 무결성 보호, 서브에이전트도 커밋 가능 |

---

## 프로필 체계 참고

`~/.claude/scripts/lib/hook-flags.js` 기준:

| 프로필 | 의미 |
|--------|------|
| `minimal` | 세션 관리 훅만 실행 (서브에이전트 기본값) |
| `standard` | 일반 개발 워크플로우 훅 포함 (메인 세션 기본값) |
| `strict` | 모든 훅 실행 |

환경변수 `ECC_HOOK_PROFILE=minimal|standard|strict`로 수동 오버라이드 가능.

---

## 다른 프로젝트/머신에서 적용 시 체크리스트

- [ ] `~/.claude/scripts/lib/hook-flags.js` — `isSubAgent()` 추가 및 `isHookEnabled()` 수정
- [ ] `~/.claude/scripts/hooks/auto-tmux-dev.js` — `CLAUDE_AGENT_NAME` early return 추가
- [ ] `~/.claude/scripts/hooks/pre-bash-git-push-reminder.js` — 동일
- [ ] `~/.claude/scripts/hooks/pre-bash-tmux-reminder.js` — 동일
- [ ] 기타 직접 호출 훅 스크립트 확인 후 필요 시 동일 패턴 적용
