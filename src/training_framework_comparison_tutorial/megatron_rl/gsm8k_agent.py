"""통제비교용 gsm8k 환경 에이전트 — Megatron-LM `examples/rl` 네이티브 GRPO 경로.

Megatron-LM 의 GRPO 는 verl/slime 처럼 reward 함수를 주입하는 게 아니라 **환경(agent)** 으로
프롬프트·정답·reward 를 캡슐화한다(`megatron.rl.agent.reward_only_agent.RewardOnlyAgent`).
repo 가 제공하는 `examples/rl/environments/math/gsm8k_agent.py:GSM8KAgent` 가 이미 openai/gsm8k
(train 7473)를 로드하고 `#### ` 뒤 정답을 뽑지만, 두 곳이 다른 프레임워크와 어긋난다:

  1. reward: 기본 GSM8KAgent 는 `math_verify`(parse/verify) 기반 채점을 쓴다. TRL/verl/slime 은
     우리 `adapters.rewards.gsm8k_score`(정답 일치 1.0 + 형식 0.1)를 공유한다 → **채점 코어가
     다르면 GRPO 가로비교가 성립하지 않는다**(reward = 통제 변수).
  2. 프롬프트: 기본 `make_prefix` 는 chat template 없는 평문 문자열을 낸다. 다른 프레임워크는
     캐논 REASONING_CHATML 을 적용한다(통제 변수).

그래서 verl(토크나이저에 template 굽기)·slime(JSONL 에 렌더된 프롬프트 박기)이 한 것과 동일하게
여기서도 reward·프롬프트를 우리 캐논으로 맞춘 얇은 서브클래스를 둔다. dataset 로드·`#### ` 정답
추출(reformat_datum)·rollout 루프는 부모 것을 그대로 재사용한다(중복 회피).

env config(트레이너가 런타임에 생성)가 이 클래스를 점경로로 가리킨다:
    agent_type: training_framework_comparison_tutorial.megatron_rl.gsm8k_agent.TfctGSM8KAgent
    agent_args: {answer_format: boxed, hf_model: ..., chat_template: reasoning_chatml}

⚠️ GPU 검증 대기: ① make_prefix 가 낸 렌더 프롬프트를 train_rl 추론 경로가 다시 chat template
로 감싸지 않는지(이중 적용 위험 — slime 과 같은 가정: base 토크나이저엔 자동 template 없음)
② RewardOnlyAgent get_reward 시그니처 정합 ③ examples.rl 점경로 임포트 가능 여부.
"""

from __future__ import annotations

from typing import Any

# examples.rl 은 컨테이너의 Megatron-LM repo(cwd/PYTHONPATH)에서만 임포트된다(이미지 전용).
from examples.rl.environments.math.gsm8k_agent import GSM8KAgent

from ..adapters import resolve_chat_template
from ..adapters.rewards import gsm8k_score


class TfctGSM8KAgent(GSM8KAgent):
    """reward·프롬프트를 우리 캐논으로 고정한 gsm8k 에이전트(통제 변수)."""

    def __init__(
        self,
        answer_format: str = "boxed",
        hf_model: str = "Qwen/Qwen3-8B-Base",
        chat_template: str | None = "reasoning_chatml",
        **kwargs: Any,
    ):
        super().__init__(answer_format=answer_format, **kwargs)
        # 캐논 chat template 을 구운 토크나이저(프롬프트 렌더용). 다른 프레임워크와 동일 입력.
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(hf_model)
        template = resolve_chat_template(chat_template)
        if template:
            self._tokenizer.chat_template = template

    def make_prefix(self, problem_key: str = "problem", **kwargs: Any) -> str:
        """부모의 평문 프롬프트(문제 + boxed 지시)를 캐논 REASONING_CHATML 로 렌더한다.

        slime 과 동일한 방식 — 토크나이저에 template 을 굽지 않고 렌더 결과 문자열을 그대로
        rollout 입력으로 준다(add_generation_prompt 로 assistant 큐까지). reward 추출은
        \\boxed{}/#### 를 보므로 boxed 지시는 유지한다(gsm8k_score 와 정합).
        """
        base_prompt = super().make_prefix(problem_key=problem_key, **kwargs)
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": base_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    async def get_reward(self, response: str, golden: dict) -> float:
        """TRL/verl/slime 과 같은 채점 코어(gsm8k_score) 공유 = 통제 변수.

        golden["numeric_answer"] = 부모 reformat_datum 이 `#### ` 뒤에서 뽑은 정답.
        """
        return gsm8k_score(response, golden["numeric_answer"])
