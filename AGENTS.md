# AGENTS.md

> AI 코딩 에이전트를 위한 repo 컨텍스트. `CLAUDE.md`는 이 파일의 심볼릭 링크다.

## llm-wiki 핸드오프 (세션 시작 시)

이 repo는 llm-wiki 허브에서 관리된다. 이 repo를 다루기 시작할 때(코딩이든 논의·계획이든):

1. **항상** `/home/user/workspace/llm-wiki/wiki/projects/training-framework-comparison-tutorial.md`를 읽는다.
   `## 현재 상태 / 다음`이 이 repo의 핸드오프(현재 state + 미래 plan)다.
2. **항상** `/home/user/workspace/llm-wiki/raw/devlog/training-framework-comparison-tutorial-*.md`를
   최신순 **최대 3개**(직전 세션들)를 읽어 결정 이유·디테일을 보강한다.

작업 후엔 `/devlog`로 세션을 기록 → 위키 "다음"이 갱신돼 다음 세션이 이어받는다.

## 도구 노트 (AI 에이전트 공용)

- 이 repo의 코딩은 어떤 AI 에이전트로 해도 됨 (Claude Code·Codex·opencode 등). 품질 중요한 부분은 Claude, 양 많은 구현은 Codex 식으로 자유.
- **커밋 트레일러(`Co-Authored-By`)는 지우지 말고, 실제 작업한 도구로 정확히 남긴다** — Claude Code가 나중에 `git log`를 읽어 맥락·`/devlog`를 재구성하므로 (부정확하면 잘못 재구성):
  - Claude 작성 → `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  - Codex 작성 → `Co-Authored-By: Codex <...>` (Codex가 실제 남기는 문자열로 통일)
  - 한 도구가 남의 코드를 대신 커밋해도 **코드를 실제 짠 도구**를 트레일러에 명시.
- 커밋 메시지 본문엔 **무엇 + 왜**를 남긴다 — WHY가 없으면 `/devlog`가 재구성 못 해 지어내지 않고 물어야 한다.
- author/committer는 항상 `jaegookyou` (잔디·크레딧은 사람에게 귀속). 트레일러는 AI 관여 표식일 뿐.
- llm-wiki 반영(`/devlog`)·위키 관리는 **Claude Code 전용** (Codex·opencode는 실행 안 함).
