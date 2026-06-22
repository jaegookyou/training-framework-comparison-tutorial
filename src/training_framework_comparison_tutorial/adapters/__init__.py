"""데이터 어댑터 2층: (소스 → 방법 스키마) → (방법 스키마 → 프레임워크 포맷).

+ rewards: GRPO 채점기(태스크 1:1, 통제 변수라 프레임워크 공유).
"""

from __future__ import annotations

from .chat_template import CHAT_TEMPLATES, REASONING_CHATML, resolve_chat_template
from .formats import (
    FORMATS,
    get_format,
    to_slime_grpo,
    to_trl,
    to_trl_dpo,
    to_trl_grpo,
    to_trl_online_dpo,
    to_verl_grpo,
)
from .rewards import REWARDS, compute_score, get_reward_funcs, slime_rm
from .schema import (
    Message,
    PreferenceExample,
    PromptOnlyExample,
    RLPromptExample,
    SFTExample,
    normalize_messages,
    normalize_preference,
    normalize_prompt_only,
    normalize_rl_prompt,
)
from .sources import (
    SOURCES,
    from_gsm8k,
    from_traceinversion,
    from_ultrafeedback,
    from_ultrafeedback_prompt,
    get_source,
)

__all__ = [
    "CHAT_TEMPLATES",
    "FORMATS",
    "REASONING_CHATML",
    "REWARDS",
    "SOURCES",
    "Message",
    "PreferenceExample",
    "PromptOnlyExample",
    "RLPromptExample",
    "SFTExample",
    "compute_score",
    "from_gsm8k",
    "from_traceinversion",
    "from_ultrafeedback",
    "from_ultrafeedback_prompt",
    "get_format",
    "get_reward_funcs",
    "get_source",
    "normalize_messages",
    "normalize_preference",
    "normalize_prompt_only",
    "normalize_rl_prompt",
    "resolve_chat_template",
    "slime_rm",
    "to_slime_grpo",
    "to_trl",
    "to_trl_dpo",
    "to_trl_grpo",
    "to_trl_online_dpo",
    "to_verl_grpo",
]
