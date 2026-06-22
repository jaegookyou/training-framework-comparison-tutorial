"""verl GRPO 학습 경로 (online on-policy RL, full|lora).

verl 의 본진 — GRPO/PPO 가 1급 시민이다. SFT(verl.trainer.sft_trainer, torchrun)와 달리 GRPO 는
`verl.trainer.main_ppo`(ray 기반, hydra)로 구동된다. 이 모듈의 train() 은:
  1. gsm8k → {prompt 체인, reward_model.ground_truth} parquet 을 떨군다(verl RLHFDataset 포맷).
     data_source 컬럼(=reward.name)을 주입해 reward 라우팅 키로 쓴다.
  2. base 모델용 캐논 chat template 을 토크나이저에 구워 로컬에 저장한다(verl_sft 와 동일 우회).
  3. RunConfig → verl hydra override 로 번역해 `python -m verl.trainer.main_ppo` 를 띄운다.
     reward 는 custom_reward_function(adapters.rewards.compute_score)로 rule-based 채점.

reward = 통제 변수: TRL/Unsloth GRPO 와 **같은 gsm8k 채점 코어**(adapters.rewards)를 공유하되,
verl 규약(compute_score 시그니처)으로 노출한 어댑터를 쓴다 → GRPO 가로비교가 성립한다.

rollout: verl 은 vllm rollout 이 기본(actor_rollout_ref.rollout.name=vllm). docker/verl.Dockerfile
에 vllm 을 추가했다(핀은 GPU 빌드 때 확정 — TODO). TRL GRPO 의 vllm 공백이 verl 경로엔 네이티브.

무거운 deps(verl/torch/vllm)는 이미지 안에만 있고 main_ppo 서브프로세스가 임포트한다. 이 모듈은
datasets/transformers 만 함수 안에서 지연 임포트한다(verl 자체는 import 하지 않는다).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..adapters import get_format, get_source, resolve_chat_template
from ..adapters import rewards as rewards_mod
from ..config import RunConfig


def _prepare_parquet(cfg: RunConfig, out_dir: Path) -> str:
    """gsm8k → {prompt, reward_model, data_source} parquet. 경로를 돌려준다."""
    from datasets import load_dataset

    ds_cfg = cfg.section("dataset")
    reward_name = cfg.section("reward")["name"]  # data_source = reward 라우팅 키
    to_prompt = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: {**to_format(to_prompt(row)), "data_source": reward_name},
        remove_columns=raw.column_names,
    )

    path = out_dir / "data" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    return str(path)


def _prepare_tokenizer_dir(cfg: RunConfig, out_dir: Path) -> str | None:
    """캐논 chat template 을 토크나이저에 구워 저장하고 그 디렉토리를 돌려준다.

    REASONING_CHATML 은 jinja 라 hydra CLI override 로 직접 넘기면 OmegaConf 파서가 깨진다
    (verl_sft 와 동일). 대신 구운 토크나이저를 저장해 경로로 가리킨다. None 이면 모델 토크나이저.
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


def train(cfg: RunConfig) -> None:
    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    lora = cfg.section("lora")
    verl = cfg.section("verl")
    debug = cfg.section("debug")

    out_dir = Path(out.get("local_dir", "out"))
    train_parquet = _prepare_parquet(cfg, out_dir)
    tokenizer_dir = _prepare_tokenizer_dir(cfg, out_dir)

    nodes = scale.get("nodes", 1)
    gpus = scale.get("gpus", 1)

    # 프롬프트 배치(=rollout 단위). trl/unsloth 와 눈금을 맞추려고
    # train_batch_size = per_device × grad_accum × gpus 로 둔다(통제비교).
    micro = hp["per_device_batch_size"]
    train_bs = micro * hp.get("gradient_accumulation", 1) * gpus
    # ppo_mini_batch_size ≤ train_batch_size. 1 epoch-of-rollout 당 정책 갱신 횟수를 정한다.
    mini_bs = min(verl.get("ppo_mini_batch_size", train_bs), train_bs)

    # custom reward = adapters.rewards.compute_score (이 모듈 파일 경로 + 함수명).
    reward_path = Path(rewards_mod.__file__).resolve()

    gpu_mem_util = verl.get("gpu_memory_utilization", 0.6)  # vllm KV 캐시 점유 비율

    # tuning=lora 면 lora_rank>0. full 이면 0(전체 파라미터). verl 은 model.lora_rank 로 분기.
    lora_rank = lora.get("r", 16) if cfg.tuning == "lora" else 0

    overrides = [
        "algorithm.adv_estimator=grpo",            # GRPO: critic 없이 그룹 정규화 advantage
        f"data.train_files={train_parquet}",
        # verl 은 val_files 를 요구한다 → train 재사용 + test_freq=-1 로 실제 평가는 생략(스모크).
        f"data.val_files={train_parquet}",
        "data.prompt_key=prompt",
        f"data.train_batch_size={train_bs}",
        f"data.max_prompt_length={hp.get('max_prompt_length', 512)}",
        f"data.max_response_length={hp.get('max_completion_length', 1024)}",
        f"actor_rollout_ref.model.path={model_cfg['name']}",
        f"actor_rollout_ref.actor.optim.lr={float(hp['learning_rate'])}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro}",
        "actor_rollout_ref.actor.use_kl_loss=true",   # GRPO 는 reward 대신 loss 에 KL
        f"actor_rollout_ref.actor.kl_loss_coef={hp.get('beta', 0.04)}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        # rollout = vllm(verl 기본, 이미지에 vllm 추가됨). n = 그룹 크기 G(advantage 정규화 단위).
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={hp.get('num_generations', 8)}",
        f"actor_rollout_ref.rollout.temperature={hp.get('temperature', 1.0)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={verl.get('rollout_tp', 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro}",
        # rule-based reward → 신경망 RM 끔. 채점은 custom_reward_function 으로.
        "reward_model.enable=false",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        f"trainer.default_local_dir={out_dir / 'ckpt'}",
        f"trainer.project_name={cfg.section('wandb').get('project', 'tfct-grpo')}",
        f"trainer.experiment_name={cfg.run_name()}",
        f"trainer.total_epochs={hp.get('epochs', 1)}",
        "trainer.logger=[console,wandb]",
        f"trainer.nnodes={nodes}",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",   # 주기적 평가 생략(val=train 재사용이라 의미 없음)
    ]
    if lora_rank > 0:
        overrides += [
            f"actor_rollout_ref.model.lora_rank={lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={lora.get('alpha', 32)}",
            f"actor_rollout_ref.model.target_modules={lora.get('target_modules', 'all-linear')}",
        ]
    if tokenizer_dir:
        overrides.append(f"actor_rollout_ref.model.tokenizer_path={tokenizer_dir}")

    # 로컬/스모크: max_steps>0 이면 그 step 만 돌고 끝.
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        overrides.append(f"trainer.total_training_steps={max_steps}")

    # main_ppo = ray 기반(단일 노드는 로컬 ray). torchrun 이 아니다(SFT 경로와 다름).
    # NOTE: verl 은 API churn 이 잦다 — 정확한 hydra 키/인자명은 docker/verl.Dockerfile 핀 기준
    # GPU end-to-end 에서 최종 검증(다른 프레임워크 경로와 동일한 단서).
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    subprocess.run(cmd, check=True)
