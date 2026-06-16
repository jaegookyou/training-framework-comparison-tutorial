"""2층: 정규 SFT 스키마 → 프레임워크별 학습 포맷.

같은 정규형에서 출발하므로 "데이터는 동일, 포맷만 프레임워크별"이 보장된다(통제비교).
프레임워크를 추가할 때 여기에 변환 함수 하나만 등록하면 된다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import SFTExample


def to_trl(example: SFTExample) -> dict[str, Any]:
    """TRL SFTTrainer 의 conversational 포맷(messages 컬럼).

    Unsloth 도 내부적으로 trl SFTTrainer 를 감싸므로 같은 포맷을 재사용한다.
    """
    return {"messages": example}


# 프레임워크 추가 시 등록 (예정): verl, megatron, torchtitan
FORMATS: dict[str, Callable[[SFTExample], dict[str, Any]]] = {
    "trl": to_trl,
    "unsloth": to_trl,  # trl SFTTrainer 래핑 → conversational 포맷 동일
}


def get_format(framework: str) -> Callable[[SFTExample], dict[str, Any]]:
    try:
        return FORMATS[framework]
    except KeyError:
        raise ValueError(f"no data format adapter for framework: {framework!r}") from None
