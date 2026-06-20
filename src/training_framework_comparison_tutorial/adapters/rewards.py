"""GRPO reward 함수 — on-policy RL 의 채점기.

통제비교에서 reward 는 **고정 변수**다(데이터·모델·HP 와 함께). 그래서 프레임워크가 아니라
태스크(gsm8k)에 1:1 로 묶어 한 곳에 정의하고, TRL·verl·megatron-lm 이 같은 함수를 재사용한다.

TRL GRPOTrainer 규약: reward 함수는 `f(completions, **kwargs) -> list[float]`.
- completions: 그룹 rollout. conversational 이면 각 원소가
  [{role:assistant, content}] 메시지 리스트.
- kwargs: 데이터셋의 여분 컬럼(여기선 `answer`)이 그대로 전달된다
  (adapters.formats.to_trl_grpo 가 통과시킴).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# 최종답 추출 우선순위: \boxed{...} > '#### 뒤' > 텍스트의 마지막 숫자.
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASHED = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_number(text: str) -> str | None:
    """문자열에서 숫자 하나를 정규화해 뽑는다(쉼표 제거, 끝 .0 제거)."""
    m = _NUMBER.search(text)
    if not m:
        return None
    num = m.group(0).replace(",", "").rstrip(".")
    if num.endswith(".0"):
        num = num[:-2]
    return num


def _completion_text(completion: Any) -> str:
    """conversational(list[message]) / plain(str) 양쪽에서 텍스트를 뽑는다."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return str(completion)


def _predicted_answer(text: str) -> str | None:
    for pat in (_BOXED, _HASHED):
        m = pat.search(text)
        if m:
            got = _norm_number(m.group(1))
            if got is not None:
                return got
    return _norm_number(text.split("\n")[-1]) or _norm_number(text)


def gsm8k_correctness_reward(
    completions: list[Any], answer: list[str], **kwargs: Any
) -> list[float]:
    """정답 일치 = 1.0, 불일치/추출실패 = 0.0 (주 신호)."""
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = _predicted_answer(_completion_text(completion))
        rewards.append(1.0 if pred is not None and pred == _norm_number(gold) else 0.0)
    return rewards


def gsm8k_format_reward(
    completions: list[Any], **kwargs: Any
) -> list[float]:
    """최종답을 채점 가능한 형식(\\boxed{} 또는 '#### ')으로 냈으면 0.1 (형식 유도)."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        has_format = bool(_BOXED.search(text) or _HASHED.search(text))
        rewards.append(0.1 if has_format else 0.0)
    return rewards


# reward 세트 = 태스크 1개당 함수 리스트. GRPO 는 여러 reward 의 합을 advantage 계산에 쓴다.
REWARDS: dict[str, list[Callable[..., list[float]]]] = {
    "gsm8k": [gsm8k_correctness_reward, gsm8k_format_reward],
}


def get_reward_funcs(name: str) -> list[Callable[..., list[float]]]:
    try:
        return REWARDS[name]
    except KeyError:
        raise ValueError(f"unknown reward set: {name!r}") from None
