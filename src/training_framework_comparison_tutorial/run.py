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
    },
    # 사후학습 RL 트랙. DPO(offline preference)와 GRPO(online RL)는 패러다임이 달라
    # 별 method 로 둔다(통제비교 = 프레임워크 고정, 방법만 비교). 기준점 = TRL.
    "dpo": {
        "trl": f"{_PKG}.trl_dpo",
        "unsloth": f"{_PKG}.unsloth_dpo",
    },
    "grpo": {
        "trl": f"{_PKG}.trl_grpo",
        "unsloth": f"{_PKG}.unsloth_grpo",
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
