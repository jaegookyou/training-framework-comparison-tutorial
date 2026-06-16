"""컨테이너 안 entrypoint: config 를 읽어 framework 에 맞는 trainer 로 dispatch.

`tfct-run --config configs/sft/....yaml` 로 호출된다(sky yaml 의 run 블록이 이걸 실행).
"""

from __future__ import annotations

import argparse
import importlib

from .config import RunConfig

# framework -> trainer 모듈. 프레임워크 추가 시 여기에 한 줄.
TRAINERS: dict[str, str] = {
    "trl": "training_framework_comparison_tutorial.trainers.trl_sft",
    "unsloth": "training_framework_comparison_tutorial.trainers.unsloth_sft",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tfct-run")
    parser.add_argument("--config", required=True, help="run config YAML 경로")
    args = parser.parse_args(argv)

    cfg = RunConfig.from_file(args.config)
    module_name = TRAINERS.get(cfg.framework)
    if module_name is None:
        raise SystemExit(f"no trainer registered for framework: {cfg.framework!r}")

    trainer = importlib.import_module(module_name)
    trainer.train(cfg)


if __name__ == "__main__":
    main()
