"""gsm8k reward (torchtitan RL) — 공유 gsm8k_score 채점 코어.

다른 GRPO 경로(trl/verl/slime/megatron/nemo)와 같은 adapters.rewards.gsm8k_score(정답 1.0 + 형식
0.1)를 쓴다 = reward 통제 변수(가로비교 성립 조건). 내장 채점이 아니라 우리 코어를 RewardFn 으로
노출(megatron TfctGSM8KAgent·nemo TfctGsm8kEnvironment 의 torchtitan 판).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

from training_framework_comparison_tutorial.adapters.rewards import gsm8k_score
from training_framework_comparison_tutorial.torchtitan_rl.gsm8k.data import Gsm8kSample


class RewardGsm8k(RewardFn):
    """단일턴 gsm8k: 마지막(유일) turn 의 응답을 gsm8k_score 로 채점(정답 1.0 + 형식 0.1)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    async def __call__(self, rollout: Rollout, env_input: Gsm8kSample) -> float:
        # 단일턴 → 마지막 turn 의 응답 텍스트(alphabet_sort rubric 의 completion_message 패턴).
        text = ""
        if rollout.turns:
            message = rollout.turns[-1].completion_message
            text = (message.get("content") or "") if message else ""
        return gsm8k_score(text, env_input.answer)


__all__ = ["RewardGsm8k"]
