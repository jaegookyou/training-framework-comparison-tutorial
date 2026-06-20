"""1층: 방법별 정규 스키마.

방법(method)마다 정규형이 다르다 — 이게 통제비교에서 "통제 대상이 아닌" 본질적 차이다:
  - SFT      : 멀티턴 메시지 리스트 [{role, content}, ...]
  - DPO      : 선호쌍 {chosen, rejected} (각각 멀티턴 대화)
  - GRPO(RL) : 프롬프트 + reward 채점용 정답 {prompt, answer}
모든 데이터 소스를 일단 해당 정규형으로 맞춘 뒤, 2층(formats)에서 프레임워크 포맷으로 변환한다.
"""

from __future__ import annotations

from typing import Any

# 정규화된 SFT 한 건.
Message = dict[str, str]
SFTExample = list[Message]
# DPO 한 건 = 선호쌍(둘 다 대화). TRL 은 chosen/rejected 의 공통 prefix 를 prompt 로 자동 추출.
PreferenceExample = dict[str, SFTExample]
# GRPO 한 건 = 생성 입력(prompt) + reward 채점에 쓸 정답(answer). on-policy 라 응답은 학습 중 생성.
RLPromptExample = dict[str, Any]

_VALID_ROLES = {"system", "user", "assistant"}


def normalize_messages(messages: list[dict[str, Any]]) -> SFTExample:
    """임의의 메시지 리스트를 검증·정규화한다."""
    normalized: SFTExample = []
    for m in messages:
        role = m["role"]
        if role not in _VALID_ROLES:
            raise ValueError(f"unknown role: {role!r}")
        normalized.append({"role": role, "content": str(m["content"])})
    if not normalized:
        raise ValueError("empty conversation")
    return normalized


def normalize_preference(
    chosen: list[dict[str, Any]], rejected: list[dict[str, Any]]
) -> PreferenceExample:
    """선호쌍을 정규화한다 (DPO 정규형)."""
    return {
        "chosen": normalize_messages(chosen),
        "rejected": normalize_messages(rejected),
    }


def normalize_rl_prompt(
    prompt: list[dict[str, Any]], answer: str
) -> RLPromptExample:
    """프롬프트 + 정답을 정규화한다 (GRPO 정규형)."""
    return {"prompt": normalize_messages(prompt), "answer": str(answer)}
