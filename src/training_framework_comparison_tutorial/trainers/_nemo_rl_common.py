"""NeMo-RL 트레이너 공통 헬퍼.

설계 = **NeMo 의 example config 에 우리 knob 만 Hydra override 로 얹기**(verl 패턴). NeMo-RL 은
거대한 config 트리(policy/data/env/loss/cluster/logger)를 유지하므로, 우리가 전체 config 를
재작성하지 않고 NeMo 가 유지하는 example config(`examples/configs/<base>.yaml`)를 기준으로
`run_X.py --config <base> <dotted overrides>` 로 우리 통제 변수(model/data/배치/lora/출력)만 덮는다.
추측을 최소화(NeMo example 이 진실의 원천) + NeMo 업데이트를 자동 추종.

torch/transformers 는 NeMo-RL 이미지 안에만 → 함수 안에서 지연 임포트. nemo_rl 자체는 import 안 한다
(서브프로세스가 run_X.py 로 임포트).

⚠️ GPU 검증 대기: override 키 정확명(특히 policy.optimizer.* lr 경로·policy.tokenizer.name·lora_cfg
하위 키)·NeMo Hydra override 문법·NGC 컨테이너 venv 에 우리 패키지 설치 경로는 NeMo-RL 이미지
end-to-end 에서 최종 확인(다른 프레임워크 경로와 동일한 단서).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..adapters import resolve_chat_template
from ..config import RunConfig


def bake_tokenizer(cfg: RunConfig, out_dir: Path) -> str | None:
    """캐논 chat template 을 구운 토크나이저를 저장하고 경로를 돌려준다(None 이면 모델 기본).

    base 모델엔 {% generation %}·instruct 동작이 없으니 다른 프레임워크와 동일 REASONING_CHATML 을
    구워 NeMo policy.tokenizer.name 으로 가리킨다(통제 변수).
    """
    chat_template = resolve_chat_template(cfg.section("model").get("chat_template"))
    if not chat_template:
        return None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.section("model")["name"])
    tok.chat_template = chat_template
    tdir = out_dir / "tokenizer"
    tok.save_pretrained(str(tdir))
    return str(tdir)


def common_overrides(cfg: RunConfig, out_dir: Path, tok_dir: str | None) -> list[str]:
    """전 메서드 공통 Hydra override(model/배치/cluster/출력/lora/lr)."""
    model = cfg.section("model")
    hp = cfg.section("hp")
    scale = cfg.section("scale")
    gpus = scale.get("gpus", 1)
    nodes = scale.get("nodes", 1)
    micro = hp["per_device_batch_size"]
    gbs = micro * hp.get("gradient_accumulation", 1) * gpus  # 다른 프레임워크와 같은 눈금

    ov = [
        f"policy.model_name={model['name']}",
        f"cluster.gpus_per_node={gpus}",
        f"cluster.num_nodes={nodes}",
        f"policy.train_micro_batch_size={micro}",
        f"policy.train_global_batch_size={gbs}",
        f"logger.log_dir={out_dir / 'logs'}",
        f"checkpointing.checkpoint_dir={out_dir / 'ckpt'}",
        # lr — NeMo optimizer config 경로(⚠️ 정확 키 GPU 검증). 통제 변수라 명시 override.
        f"policy.optimizer.kwargs.lr={float(hp['learning_rate'])}",
    ]
    if tok_dir:
        ov.append(f"policy.tokenizer.name={tok_dir}")

    # tuning=lora → DTensor v2 의 lora_cfg 활성(NeMo lora.md: SFT/GRPO/DPO 지원). full 이면 끔.
    if cfg.tuning == "lora":
        lora = cfg.section("lora")
        ov += [
            "policy.dtensor_cfg._v2=true",
            "policy.dtensor_cfg.lora_cfg.enabled=true",
            f"policy.dtensor_cfg.lora_cfg.rank={lora.get('r', 16)}",
            f"policy.dtensor_cfg.lora_cfg.alpha={lora.get('alpha', 32)}",
        ]
    return ov


def run(
    run_script: str, base_config: str, overrides: list[str], env_name: str | None = None
) -> None:
    """NeMo 진입 스크립트를 실행한다. env_name 이 있으면 런처(커스텀 환경 등록) 경유.

    env_name 있음(GRPO/PPO): python -m ...nemo_rl_env.launch <run_script> <env_name> --config ...
    env_name 없음(SFT/DPO): python <nemo>/examples/<run_script> --config ...
    """
    nemo_dir = os.environ.get("NEMO_RL_DIR", "/opt/nemo-rl")
    base = f"{nemo_dir}/examples/configs/{base_config}"
    if env_name:
        cmd = [
            sys.executable, "-m", "training_framework_comparison_tutorial.nemo_rl_env.launch",
            run_script, env_name, "--config", base, *overrides,
        ]
    else:
        script = f"{nemo_dir}/examples/{run_script}"
        cmd = [sys.executable, script, "--config", base, *overrides]
    subprocess.run(cmd, check=True, cwd=nemo_dir)
