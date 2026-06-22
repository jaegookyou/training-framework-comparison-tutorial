"""verl PPO 학습 경로 (online on-policy RL, full|lora).

verl 의 본진 — PPO 가 1급 시민이다(GRPO 와 같은 `verl.trainer.main_ppo` 진입점, ray 기반). GRPO 와
PPO 의 차이는 **advantage 추정**에 있다:
  - GRPO: critic 없이 그룹(num_generations) 내 정규화로 advantage 산출(adv_estimator=grpo).
  - PPO:  **critic(value model)** 으로 GAE 를 추정(adv_estimator=gae). 프롬프트당 1개 응답이면
          충분(그룹 불필요 → rollout.n=1). KL 은 reward 페널티로(PPO 정석, GRPO 는 loss 의 KL).

그래서 GRPO 대비 추가되는 건 (a) critic.* 블록(별도 optim·배치), (b) algorithm.adv_estimator=gae +
gamma/lam, (c) use_kl_in_reward=true(+kl_ctrl.kl_coef). 데이터·reward·rollout 인프라(vllm)는 GRPO 와
**완전히 동일** — 그래서 parquet/tokenizer 준비 헬퍼를 verl_grpo 에서 그대로 재사용한다(통제비교:
PPO vs GRPO 가 같은 데이터·reward·모델, advantage 추정만 다르다).

reward = 통제 변수: GRPO 와 **같은 gsm8k 채점 코어**(adapters.rewards.compute_score)를 rule-based
custom_reward_function 으로 꽂는다(신경망 RM 아님). gsm8k 는 검증 가능 태스크라 규칙 채점이 정확.

critic: actor 와 같은 base 에서 시작한다(critic.model.path=정책 모델). lora 면 actor·critic 둘 다
어댑터만 학습(critic.model.lora_rank>0 — actor 와 같은 hf_model 스키마).

무거운 deps(verl/torch/vllm)는 이미지 안에만 있고 main_ppo 서브프로세스가 임포트한다. 이 모듈은
verl_grpo 의 헬퍼(datasets/transformers 만 지연 임포트)를 빌려 쓴다 — verl 자체는 import 안 한다.

⚠️ GPU 검증 대기: critic 추가로 메모리(정책+ref+critic+vllm)가 GRPO 보다 크다 → full 은 멀티GPU 필수.
verl API churn 잦음 — 정확한 hydra 키/critic 메모리 배치는 docker/verl.Dockerfile 핀 기준 GPU
end-to-end 에서 최종 검증(GRPO 경로와 동일한 단서).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..adapters import rewards as rewards_mod
from ..config import RunConfig

# 데이터·reward·rollout 이 GRPO 와 동일 = 준비 헬퍼 재사용.
from .verl_grpo import _prepare_parquet, _prepare_tokenizer_dir


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

    # 프롬프트 배치(=rollout 단위). GRPO 와 같은 눈금: train_bs = per_device × grad_accum × gpus.
    micro = hp["per_device_batch_size"]
    train_bs = micro * hp.get("gradient_accumulation", 1) * gpus
    # ppo_mini_batch_size ≤ train_batch_size. 1 rollout 당 정책/critic 갱신 단위. actor·critic 공유.
    mini_bs = min(verl.get("ppo_mini_batch_size", train_bs), train_bs)

    # custom reward = adapters.rewards.compute_score (GRPO 와 같은 채점 코어 = 통제 변수).
    reward_path = Path(rewards_mod.__file__).resolve()

    gpu_mem_util = verl.get("gpu_memory_utilization", 0.6)  # vllm KV 캐시 점유 비율

    # tuning=lora 면 lora_rank>0(actor·critic 둘 다). full 이면 0(전체 파라미터).
    lora_rank = lora.get("r", 16) if cfg.tuning == "lora" else 0

    # critic lr: actor 와 별개(critic 은 보통 더 큰 lr). 미지정 시 verl 기본(1e-5).
    critic_lr = float(hp.get("critic_learning_rate", 1.0e-5))

    overrides = [
        # PPO: critic 으로 GAE advantage 추정(GRPO 의 그룹 정규화와 다름). gamma/lam = GAE 파라미터.
        "algorithm.adv_estimator=gae",
        f"algorithm.gamma={hp.get('gamma', 1.0)}",
        f"algorithm.lam={hp.get('lam', 1.0)}",
        # PPO 정석: KL 을 reward 에 페널티로(GRPO 는 loss 의 KL). beta = kl_coef.
        "algorithm.use_kl_in_reward=true",
        "algorithm.kl_ctrl.type=fixed",
        f"algorithm.kl_ctrl.kl_coef={hp.get('beta', 0.04)}",
        f"data.train_files={train_parquet}",
        # verl 은 val_files 를 요구 → train 재사용 + test_freq=-1 로 실제 평가 생략(스모크).
        f"data.val_files={train_parquet}",
        "data.prompt_key=prompt",
        f"data.train_batch_size={train_bs}",
        f"data.max_prompt_length={hp.get('max_prompt_length', 512)}",
        f"data.max_response_length={hp.get('max_completion_length', 1024)}",
        # actor (정책). PPO 는 KL 을 reward 로 넣으므로 actor.use_kl_loss=false.
        f"actor_rollout_ref.model.path={model_cfg['name']}",
        f"actor_rollout_ref.actor.optim.lr={float(hp['learning_rate'])}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro}",
        "actor_rollout_ref.actor.use_kl_loss=false",
        # rollout = vllm(verl 기본). PPO 는 프롬프트당 1개 응답이면 충분(그룹 불필요) → n=1.
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.temperature={hp.get('temperature', 1.0)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={verl.get('rollout_tp', 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro}",
        # critic (value model). adv_estimator=gae 면 verl 이 critic 을 켠다. base = 정책 모델.
        "critic.enable=true",
        f"critic.model.path={model_cfg['name']}",
        f"critic.optim.lr={critic_lr}",
        f"critic.ppo_mini_batch_size={mini_bs}",
        f"critic.ppo_micro_batch_size_per_gpu={micro}",
        # rule-based reward → 신경망 RM 끔. 채점은 GRPO 와 같은 custom_reward_function.
        "reward_model.enable=false",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        f"trainer.default_local_dir={out_dir / 'ckpt'}",
        f"trainer.project_name={cfg.section('wandb').get('project', 'tfct-ppo')}",
        f"trainer.experiment_name={cfg.run_name()}",
        f"trainer.total_epochs={hp.get('epochs', 1)}",
        "trainer.logger=[console,wandb]",
        f"trainer.nnodes={nodes}",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",   # 주기적 평가 생략(val=train 재사용이라 의미 없음)
    ]
    if lora_rank > 0:
        # actor·critic 둘 다 어댑터만 학습(같은 hf_model 스키마의 lora_rank/alpha/target_modules).
        for role in ("actor_rollout_ref.model", "critic.model"):
            overrides += [
                f"{role}.lora_rank={lora_rank}",
                f"{role}.lora_alpha={lora.get('alpha', 32)}",
                f"{role}.target_modules={lora.get('target_modules', 'all-linear')}",
            ]
    if tokenizer_dir:
        overrides.append(f"actor_rollout_ref.model.tokenizer_path={tokenizer_dir}")

    # 로컬/스모크: max_steps>0 이면 그 step 만 돌고 끝.
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        overrides.append(f"trainer.total_training_steps={max_steps}")

    # main_ppo = ray 기반(단일 노드는 로컬 ray). GRPO 와 같은 진입점, override 만 PPO.
    # NOTE: verl 은 API churn 이 잦다 — 정확한 hydra 키/critic 배치는 docker/verl.Dockerfile 핀 기준
    # GPU end-to-end 에서 최종 검증(GRPO 경로와 동일한 단서).
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    subprocess.run(cmd, check=True)
