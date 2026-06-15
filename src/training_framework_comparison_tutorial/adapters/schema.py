"""1층: 방법별 정규 스키마.

SFT 의 정규형 = 멀티턴 메시지 리스트 [{role, content}, ...].
모든 데이터 소스를 일단 이 형태로 정규화한 뒤, 2층(formats)에서 프레임워크 포맷으로 변환한다.
"""

from __future__ import annotations

from typing import Any

# 정규화된 SFT 한 건.
Message = dict[str, str]
SFTExample = list[Message]

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
