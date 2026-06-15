"""데이터 어댑터 2층: (소스 → 방법 스키마) → (방법 스키마 → 프레임워크 포맷)."""

from __future__ import annotations

from .formats import FORMATS, get_format, to_trl
from .schema import Message, SFTExample, normalize_messages
from .sources import SOURCES, from_smoltalk, get_source

__all__ = [
    "FORMATS",
    "SOURCES",
    "Message",
    "SFTExample",
    "from_smoltalk",
    "get_format",
    "get_source",
    "normalize_messages",
    "to_trl",
]
