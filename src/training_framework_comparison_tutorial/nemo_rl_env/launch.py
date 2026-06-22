"""NeMo-RL GRPO/PPO 런처 — 커스텀 환경 등록 후 NeMo 진입 스크립트를 그대로 실행 (컨테이너 전용).

NeMo 의 examples/run_{grpo,ppo}.py 는 env_name → 환경 클래스 매핑을 모듈 전역 ENV_REGISTRY 에서
찾는다. 우리 통제 reward 환경(TfctGsm8kEnvironment)을 쓰려면 그 스크립트가 레지스트리를 읽기 전에
register_env() 가 호출돼 있어야 한다. NeMo 스크립트를 직접 고치지 않으려고, 이 런처가:
  1. register_env(env_name, TfctGsm8kEnvironment FQN) 로 같은 프로세스의 ENV_REGISTRY 에 등록하고
  2. runpy 로 NeMo 진입 스크립트를 __main__ 으로 실행한다(같은 프로세스라 등록이 보인다).

사용: python -m ...nemo_rl_env.launch <run_script> <env_name> --config <yaml> [hydra overrides...]
  예) ... launch run_grpo.py tfct_gsm8k --config .../grpo_math_8B.yaml policy.model_name=...
"""

from __future__ import annotations

import os
import runpy
import sys

_ENV_FQN = "training_framework_comparison_tutorial.nemo_rl_env.gsm8k_env.TfctGsm8kEnvironment"


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: launch <run_script> <env_name> --config <yaml> [overrides...]")
    run_script = sys.argv[1]
    env_name = sys.argv[2]
    rest = sys.argv[3:]  # --config <yaml> + hydra overrides

    from nemo_rl.environments.utils import ENV_REGISTRY, register_env

    if env_name not in ENV_REGISTRY:  # register_env 는 중복 등록 시 raise
        register_env(env_name, _ENV_FQN)

    nemo_dir = os.environ.get("NEMO_RL_DIR", "/opt/nemo-rl")
    script_path = os.path.join(nemo_dir, "examples", run_script)
    # NeMo 스크립트가 argparse 로 읽도록 argv 를 그 스크립트 기준으로 재구성.
    sys.argv = [script_path, *rest]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
