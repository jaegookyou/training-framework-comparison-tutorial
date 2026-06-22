"""NeMo-RL PPO 학습 경로 (online on-policy RL, full 전용).

NeMo-RL 은 PPO 가 1급(critic/value model + GAE). PPO 가로비교에 verl·slime 에 더해 NeMo-RL 추가
(3번째 — 전부 대규모 RL 인프라). 진입 = `examples/run_ppo.py --config <base> <hydra overrides>`.

reward = 통제 변수: GRPO 와 같은 커스텀 환경(TfctGsm8kEnvironment, gsm8k_score 공유)을 env_name=
tfct_gsm8k 로 꽂는다. 데이터 = native gsm8k(우리와 같은 소스). KL/critic warmup 등 PPO 세부는 NeMo
base config(ppo_*)가 들고, 우리는 통제 변수(model/data/배치/출력)만 override.

LoRA: NeMo lora.md 가 SFT/GRPO/DPO 만 LoRA 지원이라 명시 → **PPO 는 full 전용**(verl·slime PPO 와
같은 제약). base config 는 megatron 백엔드(ppo_math_1B_megatron) — 우리는 model 을 8B 로 override.

⚠️ GPU 검증 대기: PPO base config 의 1B→8B override(병렬·critic 메모리)·env metadata ground truth
키·NeMo override 키는 NeMo-RL 이미지 end-to-end 에서 최종 확인(verl/slime PPO 와 동일한 단서).
"""

from __future__ import annotations

from pathlib import Path

from ..config import RunConfig
from . import _nemo_rl_common as nemo

_ENV_NAME = "tfct_gsm8k"  # GRPO 와 같은 커스텀 환경(통제 reward)


def train(cfg: RunConfig) -> None:
    if cfg.tuning != "full":
        raise SystemExit(
            f"NeMo-RL PPO 는 full 전용이다(NeMo lora.md: LoRA 는 SFT/GRPO/DPO 만 지원, PPO 제외). "
            f"tuning={cfg.tuning!r} 미지원 — LoRA RL 은 grpo/dpo 또는 trl/unsloth 경로를 써라."
        )

    hp = cfg.section("hp")
    nm = cfg.section("nemo")
    out_dir = Path(cfg.section("output").get("local_dir", "out"))

    tok_dir = nemo.bake_tokenizer(cfg, out_dir)
    overrides = nemo.common_overrides(cfg, out_dir, tok_dir)
    overrides += [
        "data.train.dataset_name=gsm8k",
        f"data.default.env_name={_ENV_NAME}",
        # PPO 는 critic 으로 GAE → 프롬프트당 1개 응답이면 충분(그룹 불필요).
        f"ppo.num_generations_per_prompt={hp.get('num_generations', 1)}",
    ]
    base = nm.get("base_config", "ppo_math_1B_megatron.yaml")
    nemo.run("run_ppo.py", base, overrides, env_name=_ENV_NAME)
