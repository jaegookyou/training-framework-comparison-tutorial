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
        # 순수 Megatron-LM pretrain_gpt.py — continued-pretrain 전용(8b, init_from 필수):
        # AutoBridge.import_ckpt 로 8B-Base→mcore 시드(convert.py qwen3 블로커 우회) 후
        # --pretrained-checkpoint+--finetune 로 이어학습, export 도 AutoBridge(bridge 이미지).
        "megatron-lm": f"{_PKG}.megatron_lm_pretrain",
    },
    "sft": {
        "trl": f"{_PKG}.trl_sft",
        "unsloth": f"{_PKG}.unsloth_sft",
        "verl": f"{_PKG}.verl_sft",
        "megatron-lm": f"{_PKG}.megatron_lm_sft",
        "megatron-bridge": f"{_PKG}.megatron_bridge_sft",
        "torchtitan": f"{_PKG}.torchtitan_sft",
        "nemo-rl": f"{_PKG}.nemo_rl_sft",
        "slime": f"{_PKG}.slime_sft",  # rollout 추상 재활용(sft_rollout): RL 프레임워크 SFT
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
        # torchtitan experiments/rl(Monarch+vLLM, GRPO-based). full 전용·experimental. reward=공유
        # gsm8k_score. 별도 cu130 이미지(torchtitan-rl) — SFT/사전학습 cu124 와 코어 스택 다름.
        "torchtitan": f"{_PKG}.torchtitan_grpo",
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


def dispatch(cfg: RunConfig) -> None:
    """RunConfig → (method, framework) 에 맞는 trainer 모듈로 dispatch 해 train() 호출.

    단독 실행(main)과 파이프라인 러너(pipeline)가 공유하는 단일 진입 — 단계 하나를 돌리는 의미는
    한 곳에만 둔다(파이프라인은 이걸 단계마다 부른다).
    """
    by_method = TRAINERS.get(cfg.method)
    if by_method is None:
        raise SystemExit(f"no trainers registered for method: {cfg.method!r}")
    module_name = by_method.get(cfg.framework)
    if module_name is None:
        raise SystemExit(
            f"no trainer registered for {cfg.method}/{cfg.framework!r}"
        )
    # 멀티노드 미배선 조합에 nodes>1 을 주면 여기서 정직하게 죽인다(거짓말 knob 방지).
    # 배선된 trainer 는 내부에서 _dist.resolve 로 실제 프로비저닝까지 재확인한다.
    from .trainers import _dist
    _dist.guard_wired(cfg.method, cfg.framework, cfg.section("scale"))
    importlib.import_module(module_name).train(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tfct-run")
    parser.add_argument("--config", required=True, help="run config YAML 경로")
    args = parser.parse_args(argv)

    dispatch(RunConfig.from_file(args.config))


if __name__ == "__main__":
    main()
