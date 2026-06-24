"""gsm8k Rollouter (torchtitan RL) — dataset·env·reward 배선(pure config).

alphabet_sort 의 AlphabetSortRollouter 미러: train/val 데이터셋·단일턴 env·공유 gsm8k_score reward
를 Config 필드로 묶기만 한다. make_env_group/get_*_sample/score_group 은 Rollouter base 가 제공.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rubrics import Rubric

from training_framework_comparison_tutorial.torchtitan_rl.gsm8k.data import Gsm8kDataset
from training_framework_comparison_tutorial.torchtitan_rl.gsm8k.env import Gsm8kEnv
from training_framework_comparison_tutorial.torchtitan_rl.gsm8k.rubric import RewardGsm8k


class Gsm8kRollouter(Rollouter):
    """gsm8k 태스크: train/val 데이터셋 + 단일턴 env + 공유 gsm8k_score reward(weight 1.0)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: Gsm8kDataset.Config = field(
            default_factory=lambda: Gsm8kDataset.Config(seed=42)
        )
        validation_dataset: Gsm8kDataset.Config = field(
            default_factory=lambda: Gsm8kDataset.Config(seed=99)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardGsm8k.Config(weight=1.0)]
            )
        )
        message_env: Gsm8kEnv.Config = field(default_factory=Gsm8kEnv.Config)


__all__ = ["Gsm8kRollouter"]
