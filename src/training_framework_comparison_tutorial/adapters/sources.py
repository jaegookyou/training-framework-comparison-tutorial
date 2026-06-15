"""1층 입력: 데이터 소스(HF dataset row) → 정규 SFT 스키마."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import SFTExample, normalize_messages


def from_smoltalk(row: dict[str, Any]) -> SFTExample:
    """HuggingFaceTB/smoltalk: 이미 messages 컬럼(멀티턴)을 가짐."""
    return normalize_messages(row["messages"])


SOURCES: dict[str, Callable[[dict[str, Any]], SFTExample]] = {
    "smoltalk": from_smoltalk,
}


def get_source(name: str) -> Callable[[dict[str, Any]], SFTExample]:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown dataset source: {name!r}") from None
