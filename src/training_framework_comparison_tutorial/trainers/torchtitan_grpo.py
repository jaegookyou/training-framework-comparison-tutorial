"""torchtitan GRPO 학습 경로 (full·experimental) — experiments/rl(Monarch + vLLM).

torchtitan 의 RL(experiments/rl)은 SFT/사전학습과 **다른 코어 스택**(Monarch actors + TorchStore +
vLLM + flash-attn-3 / cu130)이라 별도 torchtitan-rl 이미지를 쓴다 — 한 프레임워크인데 스택이 갈려
2 이미지(megatron 식 '한 이미지 여러 진입점'과 다른 이유: 코어 CUDA/torch 버전 자체가 다름).
⚠️ upstream 자체가 "actively developing, APIs may change, single-turn only for now".

reward 통제: `--module` 로 우리 gsm8k 모듈(torchtitan_rl.gsm8k)을 넘긴다 — rubric=공유 gsm8k_score,
data=공유 from_gsm8k(다른 GRPO 경로와 같은 채점/데이터 = 가로비교 성립). manager.py 가 FQN
module 을 `<FQN>.config_registry` 로 import 하므로(소스 확인) torchtitan 소스트리에 bake 불필요 —
megatron_rl·nemo_rl_env 와 같은 컨테이너 전용 reward 모듈 패턴이다.

런치 = `python -m torchtitan.experiments.rl.train --module <FQN> --config rl_grpo_qwen3_8b_gsm8k
--hf_assets_path=<assets>`. torchrun 이 아니라 Monarch 가 trainer/generator mesh 를 띄운다(README).
HP(lr·group_size 등)는 baked config 함수가 _base grpo 눈금으로 박는다 — nested RLTrainer.Config 의
CLI override 경로 미확인이라(추정 금지) host 는 README 확인된 --hf_assets_path 만 넘긴다.

무거운 deps(Monarch/TorchStore/renderers/FA3/vLLM/torchtitan)는 이미지 안에만. 이 호스트 모듈은
hf 다운로드용 huggingface_hub/transformers 만 지연 import.

⚠️ GPU 검증 대기: cu130 이미지 빌드(Monarch/TorchStore/renderers/FA3/vLLM 조합) · RL train 진입·
mesh 분할 · canonical template vs renderers qwen3 렌더 정합(_prepare_hf_assets 주석 참고).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..adapters import resolve_chat_template
from ..config import RunConfig

# manager.py fully-qualified 분기로 넘길 우리 task 모듈 + config 함수.
_MODULE = "training_framework_comparison_tutorial.torchtitan_rl.gsm8k"
_CONFIG = "rl_grpo_qwen3_8b_gsm8k"


def _prepare_hf_assets(cfg: RunConfig, work: Path) -> str:
    """모델 전체(가중치+토크나이저) 스냅샷 + 캐논 chat template 주입(SFT 경로와 동일).

    initial_load_in_hf=True 가 이 디렉토리 가중치를 정책 시드로 로드한다(베이스 = 사후학습 base 와
    동일, 수직 파이프라인이면 앞 단계 산출 HF). ⚠️ torchtitan RL 의 프롬프트 렌더는 renderers
    라이브러리(RendererConfig name="qwen3")라 우리 REASONING_CHATML 과 다를 수 있다 — 이 실험
    경로의 알려진 한계(reward=gsm8k_score 공유라 주 통제 변수는 유지, 프롬프트 렌더만 native).
    """
    from huggingface_hub import snapshot_download

    model_cfg = cfg.section("model")
    assets = work / "hf_assets"
    snapshot_download(repo_id=model_cfg["name"], local_dir=str(assets))

    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(assets))
        tok.chat_template = chat_template
        tok.save_pretrained(str(assets))
    return str(assets)


def train(cfg: RunConfig) -> None:
    if cfg.tuning != "full":
        raise SystemExit(
            f"torchtitan GRPO 는 full 전용이다(experiments/rl 에 LoRA RL 경로 없음). "
            f"tuning={cfg.tuning!r} 미지원 — LoRA GRPO 는 trl/unsloth 경로를 써라."
        )

    out = cfg.section("output")
    out_dir = Path(out.get("local_dir", "out"))
    work = out_dir / "torchtitan_rl_workspace"
    work.mkdir(parents=True, exist_ok=True)

    assets = _prepare_hf_assets(cfg, work)

    cmd = [
        "python",
        "-m",
        "torchtitan.experiments.rl.train",
        "--module",
        _MODULE,
        "--config",
        _CONFIG,
        f"--hf_assets_path={assets}",
    ]
    subprocess.run(cmd, check=True, env={**os.environ})
