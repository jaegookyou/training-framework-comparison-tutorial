"""NeMo-RL gsm8k 커스텀 환경 — 통제 변수 reward (컨테이너 전용).

NeMo-RL 의 GRPO/PPO 는 reward 를 **environment**(Ray actor, EnvironmentInterface)로 받는다. 내장
MathEnvironment 는 math_verify 로 채점하지만, 그러면 verl/slime/megatron-lm(우리 gsm8k_score 공유)과
채점 신호가 달라져 가로비교 confound 가 된다. 그래서 megatron-lm 의 TfctGSM8KAgent 와 같은 정신으로,
**같은 채점 코어(adapters.rewards.gsm8k_score)** 를 부르는 얇은 NeMo 환경을 둔다 = reward 통제 변수.

등록: nemo_rl_env.launch 가 register_env("tfct_gsm8k", "<이 클래스 FQN>") 로 ENV_REGISTRY 에 넣고,
config 의 data.default.env_name=tfct_gsm8k 가 이 환경을 고른다(NeMo create_env → get_object(FQN)).

⚠️ GPU 검증 대기: NeMo 의 step() metadata 에 ground truth 가 담기는 정확한 키, message_log 의 마지막
assistant content 추출, EnvironmentReturn 의 observations/terminateds 모양은 NeMo-RL 이미지 안에서
최종 확인(다른 프레임워크 경로와 동일한 단서). gsm8k 는 단일턴이라 terminateds 는 전부 1.
"""

from __future__ import annotations

from typing import Any

import ray
import torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from ..adapters.rewards import gsm8k_score  # verl/slime/megatron-lm 과 공유하는 채점 코어


def _last_assistant_text(message_log: list[dict[str, Any]]) -> str:
    """message_log 에서 마지막 assistant 발화 content 를 뽑는다(생성된 응답)."""
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _ground_truth(meta: Any) -> str:
    """metadata 에서 정답을 뽑는다. NeMo 데이터 포맷에 따라 키가 다를 수 있어 방어적으로."""
    if isinstance(meta, dict):
        for key in ("ground_truth", "answer", "label"):
            if key in meta and meta[key] is not None:
                return str(meta[key])
    return str(meta)


@ray.remote  # pragma: no cover  (NeMo-RL 이미지·Ray 런타임에서만 실행)
class TfctGsm8kEnvironment(EnvironmentInterface):
    """gsm8k 단일턴 환경 — 응답을 gsm8k_score 로 채점해 scalar reward 를 돌려준다."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[Any],
    ) -> EnvironmentReturn:
        rewards = [
            gsm8k_score(_last_assistant_text(log), _ground_truth(meta))
            for log, meta in zip(message_log_batch, metadata)
        ]
        batch = len(message_log_batch)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        terminateds = torch.ones(batch, dtype=torch.float32)  # 단일턴 → 매 step 종료
        # 채점 후 추가 관찰/행동 없음(단일턴): 빈 environment 메시지.
        observations = [{"role": "environment", "content": ""} for _ in range(batch)]
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * batch,
            rewards=rewards_t,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(self, batch: Any) -> tuple[Any, dict[str, Any]]:
        """배치 롤아웃 후 후처리·메트릭. rule reward 라 추가 후처리 없음."""
        mean_reward = None
        try:
            mean_reward = float(batch["rewards"].mean()) if "rewards" in batch else None
        except Exception:  # noqa: BLE001 (메트릭 산출 실패는 학습을 막지 않는다)
            mean_reward = None
        metrics = {} if mean_reward is None else {"gsm8k_mean_reward": mean_reward}
        return batch, metrics
