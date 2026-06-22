"""NeMo-RL GRPO 학습 경로 (online on-policy RL, full|lora).

NeMo-RL = NVIDIA 의 종합 RL 사후학습 툴킷(NeMo-Aligner 후속). GRPO 가로비교에 verl·slime·megatron-lm
에 더해 NeMo-RL 추가(6번째). 진입 = `examples/run_grpo.py --config <base> <hydra overrides>`.

reward = 통제 변수: 내장 MathEnvironment(math_verify) 대신 우리 gsm8k_score 를 부르는 커스텀 환경
(nemo_rl_env.gsm8k_env.TfctGsm8kEnvironment)을 쓴다. nemo_rl_env.launch 가 register_env 로 등록하고
config 의 data.default.env_name=tfct_gsm8k 가 그걸 고른다 → verl/slime/megatron 과 같은 채점 코어.

데이터 = NeMo 의 native gsm8k 로더(openai/gsm8k + '####' 추출 = 우리 from_gsm8k 와 같은 소스/추출).
LoRA = DTensor v2 의 lora_cfg(NeMo lora.md: GRPO 지원). full 은 lora_cfg.enabled=false.

무거운 deps(nemo_rl/torch/vllm)는 NeMo-RL 이미지 안에만 → run_grpo.py 서브프로세스가 임포트.

⚠️ GPU 검증 대기: NeMo override 키 정확명(grpo.num_generations_per_prompt 등)·커스텀 env 의 metadata
ground truth 키·DTensor lora·NGC venv 설치는 이미지 end-to-end 에서 최종 확인.
"""

from __future__ import annotations

from pathlib import Path

from ..config import RunConfig
from . import _nemo_rl_common as nemo

_ENV_NAME = "tfct_gsm8k"  # 커스텀 환경 등록 키(통제 reward)


def train(cfg: RunConfig) -> None:
    hp = cfg.section("hp")
    nm = cfg.section("nemo")
    out_dir = Path(cfg.section("output").get("local_dir", "out"))

    tok_dir = nemo.bake_tokenizer(cfg, out_dir)
    overrides = nemo.common_overrides(cfg, out_dir, tok_dir)
    overrides += [
        # 데이터 = native gsm8k(우리와 같은 소스), reward = 커스텀 환경(통제 변수).
        "data.train.dataset_name=gsm8k",
        f"data.default.env_name={_ENV_NAME}",
        # GRPO 그룹 크기 G(advantage 정규화 단위) — 다른 GRPO 와 같은 눈금.
        f"grpo.num_generations_per_prompt={hp.get('num_generations', 8)}",
    ]
    base = nm.get("base_config", "grpo_math_8B.yaml")
    nemo.run("run_grpo.py", base, overrides, env_name=_ENV_NAME)
