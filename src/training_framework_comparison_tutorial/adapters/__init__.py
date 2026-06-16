"""데이터 어댑터 2층: (소스 → 방법 스키마) → (방법 스키마 → 프레임워크 포맷)."""

from __future__ import annotations

from .chat_template import CHAT_TEMPLATES, REASONING_CHATML, resolve_chat_template
from .formats import FORMATS, get_format, to_trl
from .schema import Message, SFTExample, normalize_messages
from .sources import SOURCES, from_smoltalk, from_traceinversion, get_source

__all__ = [
    "CHAT_TEMPLATES",
    "FORMATS",
    "REASONING_CHATML",
    "SOURCES",
    "Message",
    "SFTExample",
    "from_smoltalk",
    "from_traceinversion",
    "get_format",
    "get_source",
    "normalize_messages",
    "resolve_chat_template",
    "to_trl",
]
