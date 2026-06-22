"""컨테이너 안 entrypoint: config 를 읽어 framework 에 맞는 trainer 로 dispatch.

`tfct-run --config configs/sft/....yaml` 로 호출된다(sky yaml 의 run 블록이 이걸 실행).
"""

from __future__ import annotations

import argparse
import importlib

from .config import RunConfig

# (method, framework) -> trainer 모듈. method 축(pretrain/sft/rl)으로 네임스페이스를 나눠
# 단일 모델 PT→SFT→RL 수직 파이프라인과 통제비교(가로)가 같은 dispatch 를 공유한다.
# 새 경로 추가 = 해당 method 아래 한 줄.
_PKG = "training_framework_comparison_tutorial.trainers"
TRAINERS: dict[str, dict[str, str]] = {
    "pretrain": {
        "torchtitan": f"{_PKG}.torchtitan_pretrain",
    },
    "sft": {
        "trl": f"{_PKG}.trl_sft",
        "unsloth": f"{_PKG}.unsloth_sft",
        "verl": f"{_PKG}.verl_sft",
        "megatron-lm": f"{_PKG}.megatron_lm_sft",
        "megatron-bridge": f"{_PKG}.megatron_bridge_sft",
        "torchtitan": f"{_PKG}.torchtitan_sft",
        "nemo-rl": f"{_PKG}.nemo_rl_sft",
    },
    # 사후학습 RL 트랙. DPO(offline preference)와 GRPO(online RL)는 패러다임이 달라
    # 별 method 로 둔다(통제비교 = 프레임워크 고정, 방법만 비교). 기준점 = TRL.
    "dpo": {
        "trl": f"{_PKG}.trl_dpo",
        "unsloth": f"{_PKG}.unsloth_dpo",
        "nemo-rl": f"{_PKG}.nemo_rl_dpo",  # 헤비/DTensor offline DPO 가로(위키 계획)
    },
    # online DPO = 같은 DPO loss 의 on-policy 판(생성+RM 채점). offline DPO 와 별 method 로
    # 둬 "같은 method 의 offline↔online" 비교를 명시한다. Unsloth 는 네이티브 경로 부재 → TRL 단독.
    "online_dpo": {
        "trl": f"{_PKG}.trl_online_dpo",
    },
    "grpo": {
        "trl": f"{_PKG}.trl_grpo",
        "unsloth": f"{_PKG}.unsloth_grpo",
        "verl": f"{_PKG}.verl_grpo",
        "slime": f"{_PKG}.slime_grpo",
        "megatron-lm": f"{_PKG}.megatron_lm_grpo",
        "nemo-rl": f"{_PKG}.nemo_rl_grpo",
    },
    # PPO = critic(value model)으로 GAE advantage 추정(GRPO 의 그룹 정규화와 다름). 대규모 RL 인프라
    # 셋(verl=ray main_ppo / slime=SGLang+Megatron / nemo-rl=NeMo, 전부 rule reward 네이티브)으로
    # 가로비교. PPO 는 무거운(critic) 알고리즘이라 대규모 RL 인프라에만 1급으로 있다 — Unsloth·
    # megatron-lm·torchtitan·bridge 는 네이티브 PPO 없음. TRL 은 PPO 가 있어도 neural RM 강제(rule
    # reward 못 씀 = 선호 패러다임)라 gsm8k rule 축엔 부적합 → 제외.
    "ppo": {
        "verl": f"{_PKG}.verl_ppo",
        "slime": f"{_PKG}.slime_ppo",
        "nemo-rl": f"{_PKG}.nemo_rl_ppo",  # 3번째 PPO(rule reward 네이티브, full 전용)
    },
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tfct-run")
    parser.add_argument("--config", required=True, help="run config YAML 경로")
    args = parser.parse_args(argv)

    cfg = RunConfig.from_file(args.config)
    by_method = TRAINERS.get(cfg.method)
    if by_method is None:
        raise SystemExit(f"no trainers registered for method: {cfg.method!r}")
    module_name = by_method.get(cfg.framework)
    if module_name is None:
        raise SystemExit(
            f"no trainer registered for {cfg.method}/{cfg.framework!r}"
        )

    trainer = importlib.import_module(module_name)
    trainer.train(cfg)


if __name__ == "__main__":
    main()
