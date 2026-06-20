"""1층 입력: 데이터 소스(HF dataset row) → 방법별 정규 스키마.

소스는 방법과 1:1 이 아니라 데이터셋과 1:1 이다(SFT=traceinversion / DPO=ultrafeedback /
GRPO=gsm8k). 어떤 정규형을 내놓는지는 소스마다 다르다 — trainer 가 자기 method 에 맞는
소스를 config 로 고른다. 그래서 한 레지스트리(SOURCES)에 모아두고 이름으로만 찾는다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import (
    PreferenceExample,
    RLPromptExample,
    SFTExample,
    normalize_messages,
    normalize_preference,
    normalize_rl_prompt,
)


def from_traceinversion(row: dict[str, Any]) -> SFTExample:
    """Jackrong/Claude-opus-4.7-TraceInversion-5000x: reasoning distill (SFT).

    messages 컬럼이 user 질문 → assistant 응답(<think> 재구성 CoT + 최종 답)의
    2-turn role/content 라 그대로 정규화된다.
    """
    return normalize_messages(row["messages"])


def from_ultrafeedback(row: dict[str, Any]) -> PreferenceExample:
    """trl-lib/ultrafeedback_binarized: 선호쌍 (DPO).

    chosen/rejected 가 이미 대화 리스트([{role, content}, ...])로 들어있는 implicit-prompt
    포맷이다(둘이 user turn 을 공유, assistant turn 만 다름). TRL DPOTrainer 가 공통 prefix 를
    prompt 로 자동 추출하므로 우리는 두 대화를 정규화만 해서 넘긴다.
    """
    return normalize_preference(row["chosen"], row["rejected"])


# GRPO 는 base 모델이 채점 가능한 형식으로 답하도록 프롬프트로 유도해야 한다(reasoning + 최종답).
# DeepSeekMath/TRL GRPO 레시피의 표준 system 지시. reward(adapters.rewards)의 파서와 한 쌍.
GSM8K_SYSTEM = (
    "Solve the math problem. Reason step by step, then give the final numeric "
    "answer on the last line after '#### '."
)


def _extract_gsm8k_gold(answer: str) -> str:
    """gsm8k answer 필드("...풀이...\\n#### 18")에서 최종 정답 숫자만 뽑는다."""
    return answer.split("####")[-1].strip().replace(",", "")


def from_gsm8k(row: dict[str, Any]) -> RLPromptExample:
    """openai/gsm8k(main): 수학 문제 (GRPO).

    question → 프롬프트(system 지시 + user 질문), answer → reward 채점용 정답(#### 뒤 숫자).
    on-policy 라 응답 자체는 데이터에 없다 — 학습 중 생성하고 정답으로 채점한다.
    """
    prompt = [
        {"role": "system", "content": GSM8K_SYSTEM},
        {"role": "user", "content": row["question"]},
    ]
    return normalize_rl_prompt(prompt, _extract_gsm8k_gold(row["answer"]))


SOURCES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "traceinversion": from_traceinversion,  # SFT
    "ultrafeedback": from_ultrafeedback,    # DPO
    "gsm8k": from_gsm8k,                     # GRPO
}


def get_source(name: str) -> Callable[[dict[str, Any]], Any]:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown dataset source: {name!r}") from None
