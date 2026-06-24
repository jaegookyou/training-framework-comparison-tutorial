"""gsm8k 환경 (torchtitan RL) — 단일턴: init=질문 1개, step=즉시 종료.

gsm8k 는 본디 단일턴(질문→답) 태스크라 init 이 프롬프트를 내고 step 은 done=True 로 끝낸다
(alphabet_sort 의 multi-turn 과 달리 — 남은 batch 없으면 첫 step 에서 종료하는 구조를 단일턴으로
단순화). 프롬프트 messages = 공유 from_gsm8k(system 지시 + 질문)이 데이터에서 온다.
"""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message
from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

from training_framework_comparison_tutorial.torchtitan_rl.gsm8k.data import Gsm8kSample


class Gsm8kEnv(MessageEnv):
    """한 gsm8k 문제를 단일턴 chat 으로: init 이 질문 프롬프트, step 은 첫 응답 후 즉시 종료."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: Gsm8kSample) -> None:
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        """질문 프롬프트(공유 from_gsm8k 의 system 지시 + user 질문)를 낸다."""
        return MessageEnvInitOutput(
            init_prompt_messages=list(self._env_input.prompt_messages)
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """단일턴 → 첫 응답 후 종료(채점은 rubric 이 마지막 응답으로)."""
        del completion_message
        return MessageEnvStepOutput(done=True)


__all__ = ["Gsm8kEnv"]
