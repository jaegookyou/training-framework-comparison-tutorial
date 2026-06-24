"""gsm8k 데이터셋 (torchtitan RL) — 공유 from_gsm8k 로 prompt+정답 생성.

다른 GRPO 경로(trl/verl/slime/megatron/nemo)와 같은 openai/gsm8k + 같은 프롬프트·정답 추출
(adapters.sources.from_gsm8k: system 지시 + 질문 / `####` 뒤 숫자)을 써서 통제비교 데이터 정합을
유지한다. on-policy 라 응답은 데이터에 없고(학습 중 생성) reward(rubric)가 정답으로 채점한다.

⚠️ 컨테이너 전용: torchtitan.config(Configurable)를 import 하므로 cu130 torchtitan-rl 이미지
안에서만 로드된다(호스트/CPU 임포트 안 됨 — 패키지 __init__·테스트가 안 건드림).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset
from torchtitan.config import Configurable

from training_framework_comparison_tutorial.adapters.sources import from_gsm8k


@dataclass(frozen=True, kw_only=True, slots=True)
class Gsm8kSample:
    """한 gsm8k 문제: 모델에 보일 prompt(messages)와 채점용 정답(`####` 뒤 숫자)."""

    prompt_messages: tuple[dict[str, str], ...]  # [{"role": "system"...}, {"role": "user"...}]
    answer: str  # 정답 숫자(rubric 의 gsm8k_score ground_truth)


class Gsm8kDataset(Configurable):
    """openai/gsm8k(main)을 순회하며 Gsm8kSample 을 낸다. prompt·정답 = 공유 from_gsm8k.

    alphabet_sort 의 AlphabetSortDataset 미러 — __iter__/__next__ 무한 스트림 + seed 셔플,
    state_dict/load_state_dict 로 재개 지점 스냅샷.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42
        hf_path: str = "openai/gsm8k"
        hf_name: str = "main"
        hf_split: str = "train"

    def __init__(self, config: Config) -> None:
        self._config = config
        rows = load_dataset(config.hf_path, config.hf_name, split=config.hf_split)
        self._samples = [_to_sample(row) for row in rows]
        self._rng = random.Random(config.seed)
        self._order = list(range(len(self._samples)))
        self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[Gsm8kSample]:
        return self

    def __next__(self) -> Gsm8kSample:
        if self._pos >= len(self._order):  # epoch 끝 → 다시 셔플(무한 스트림)
            self._rng.shuffle(self._order)
            self._pos = 0
        sample = self._samples[self._order[self._pos]]
        self._pos += 1
        return sample

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.getstate(), "pos": self._pos, "order": self._order}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._pos = state_dict["pos"]
        self._order = state_dict["order"]


def _to_sample(row: dict) -> Gsm8kSample:
    """openai/gsm8k row → Gsm8kSample. 공유 from_gsm8k({"prompt": messages, "answer": gold})."""
    example = from_gsm8k(row)
    return Gsm8kSample(prompt_messages=tuple(example["prompt"]), answer=example["answer"])


__all__ = ["Gsm8kDataset", "Gsm8kSample"]
