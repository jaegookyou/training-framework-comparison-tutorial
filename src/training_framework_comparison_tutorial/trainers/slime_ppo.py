"""slime PPO 학습 경로 (online on-policy RL, full 전용).

slime = SGLang 롤아웃 + Megatron 학습 RL 오케스트레이터. GRPO(slime_grpo)와 같은 칸이지만 advantage
추정이 다르다:
  - GRPO: critic 없이 그룹(n-samples-per-prompt) 내 정규화(advantage-estimator=grpo).
  - PPO:  **critic(value model)** 으로 GAE 를 추정(advantage-estimator=ppo). 프롬프트당 1개 응답이면
          충분(그룹 불필요 → n=1). KL 은 reward 페널티(--kl-coef; GRPO 는 loss 의 KL).

slime PPO 의 critic 설정 = **role-tagged megatron config**(--megatron-config-path): 한 YAML 의
top-level `megatron` 리스트에 role=critic / role=actor 엔트리를 두고 per-role override(lr 등)를 준다
(slime PPO 테스트 recipe 미러). critic 은 별도 체크포인트 없이 같은 base(ref-load)에서 시작하고,
override 로 critic lr 만 actor 와 다르게 둔다. --num-critic-only-steps 로 초기 critic 워밍업.

데이터·reward·롤아웃 렌더는 GRPO 와 **완전히 동일**(통제비교: PPO vs GRPO = advantage 추정만 다름)
→ JSONL 준비 헬퍼·reward 진입점을 slime_grpo 에서 그대로 재사용한다. reward = 통제 변수:
GRPO/verl 과 같은 채점 코어(adapters.rewards.slime_rm)를 --custom-rm-path 로 꽂는다(내장 rm 대신).

무거운 deps(slime/sglang/megatron)는 공식 이미지(docker/slime.Dockerfile) 안에만 있고 train.py
서브프로세스가 임포트한다. 이 모듈은 slime_grpo 헬퍼(datasets/transformers 지연 임포트)를 빌려 쓴다.

⚠️ GPU 검증 대기: critic 추가로 메모리·병렬 곱이 GRPO 보다 크다(정책+critic+SGLang). role-config
키 정합·num-critic-only-steps·value-clip·batch 제약(rollout_bs×n = global_bs×steps)은 GPU
end-to-end 에서 최종 확인(GRPO 경로와 동일한 단서).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from ..config import RunConfig

# 데이터·reward 가 GRPO 와 동일 = JSONL 준비 헬퍼·reward 진입점 재사용.
from .slime_grpo import _RM_PATH, _prepare_jsonl


def _write_megatron_config(out_dir: Path, actor_lr: float, critic_lr: float) -> str:
    """role-tagged megatron config(critic/actor lr override) YAML 을 떨구고 경로를 돌려준다.

    slime PPO 의 critic 설정 메커니즘 — top-level `megatron` 리스트에서 critic 런타임이 role=critic
    엔트리 하나를 골라 override 적용(slime PPO 테스트와 동일 구조). yaml 의존 없이 손으로 쓴다.
    """
    lines = [
        "megatron:",
        "  - name: default",
        "    role: critic",
        "    overrides:",
        f"      lr: {critic_lr}",
        "  - name: default",
        "    role: actor",
        "    overrides:",
        f"      lr: {actor_lr}",
        "",
    ]
    path = out_dir / "megatron_config.yaml"
    path.write_text("\n".join(lines))
    return str(path)


def train(cfg: RunConfig) -> None:
    if cfg.tuning != "full":
        raise SystemExit(
            f"slime 은 full RL 전용이다(base slime 에 LoRA 없음 — Miles 확장 필요). "
            f"tuning={cfg.tuning!r} 미지원 — LoRA RL 은 trl/unsloth 경로를 써라."
        )

    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    sl = cfg.section("slime")
    wandb_cfg = cfg.section("wandb")
    debug = cfg.section("debug")

    slime_dir = os.environ.get("SLIME_DIR", "/root/slime")
    megatron_dir = os.environ.get("MEGATRON_LM_DIR", "/root/Megatron-LM")
    model_script = Path(slime_dir) / "scripts" / "models" / f"{sl['model_script']}.sh"

    out_dir = Path(out.get("local_dir", "out"))
    train_jsonl = _prepare_jsonl(cfg, out_dir)
    torch_dist = out_dir / "ref_torch_dist"   # HF→Megatron 변환(=ref-load, actor·critic 공통 base)
    slime_ckpt = out_dir / "slime_ckpt"        # 학습 체크포인트(load/save)

    # actor lr 은 base --lr, critic lr 은 role override(critic 은 보통 더 큰 lr).
    actor_lr = float(hp["learning_rate"])
    critic_lr = float(hp.get("critic_learning_rate", 1.0e-5))
    megatron_config = _write_megatron_config(out_dir, actor_lr, critic_lr)

    gpus = scale.get("gpus", 1)
    nodes = scale.get("nodes", 1)
    tp = sl.get("tensor_model_parallel_size", 1)
    engine_gpus = sl.get("rollout_num_gpus_per_engine") or tp

    # batch 제약(slime): rollout_batch_size × n_samples = global_batch_size × num_steps_per_rollout.
    # PPO 는 그룹 불필요 → n_samples=1(GRPO 의 그룹 G 와 차이). num_steps=1 → global = rollout_bs.
    group = hp.get("num_generations", 1)          # PPO = 프롬프트당 1개 응답
    rollout_bs = sl.get("rollout_batch_size", 32)  # 라운드당 프롬프트 수
    global_bs = rollout_bs * group

    # 스모크: max_steps>0 이면 그만큼만 rollout 루프. 아니면 slime.num_rollout(태스크 기본).
    max_steps = debug.get("max_steps", -1)
    num_rollout = max_steps if (max_steps and max_steps > 0) else sl.get("num_rollout", 100)

    # 초기 critic 워밍업(critic-only) step. critic 이 무작위 init 이라 정책 갱신 전 value 정렬.
    critic_only_steps = sl.get("num_critic_only_steps", 1)

    # train.py args (PPO 테스트 recipe 미러). MODEL_ARGS 는 bash source 가 채운다.
    args = [
        "--actor-num-nodes", str(nodes),
        "--actor-num-gpus-per-node", str(gpus),
        "--colocate",                                  # 학습/롤아웃이 같은 GPU 공유(단노드)
        # CKPT (critic 도 같은 base 에서 시작 — role override 는 lr 만)
        "--hf-checkpoint", model_cfg["name"],
        "--ref-load", str(torch_dist),
        "--load", str(slime_ckpt),
        "--save", str(slime_ckpt),
        # ROLLOUT (prompt 는 이미 렌더됨 → --apply-chat-template 안 씀)
        "--prompt-data", train_jsonl,
        "--input-key", "prompt",
        "--label-key", "label",
        "--metadata-key", "metadata",
        "--rollout-shuffle",
        "--custom-rm-path", _RM_PATH,                  # 우리 reward(통제 변수), 내장 rm-type 대신
        "--num-rollout", str(num_rollout),
        "--rollout-batch-size", str(rollout_bs),
        "--n-samples-per-prompt", str(group),
        "--rollout-max-response-len", str(hp.get("max_completion_length", 1024)),
        "--rollout-temperature", str(hp.get("temperature", 1.0)),
        "--global-batch-size", str(global_bs),
        "--balance-data",
        # PPO (critic 으로 GAE). role-config 로 critic lr override, KL 은 reward 페널티(--kl-coef).
        "--advantage-estimator", "ppo",
        "--megatron-config-path", megatron_config,
        "--num-critic-only-steps", str(critic_only_steps),
        "--eps-clip", str(sl.get("eps_clip", 0.2)),    # PPO 정책 clip 범위
        "--value-clip", str(sl.get("value_clip", 0.2)),  # value(critic) loss clip
        "--normalize-advantages",
        "--kl-coef", str(hp.get("beta", 0.04)),        # reward shaping KL(PPO 정석; GRPO=loss KL)
        # OPTIMIZER (actor; critic lr 은 role override)
        "--optimizer", "adam",
        "--lr", str(actor_lr),
        "--lr-decay-style", hp.get("lr_scheduler", "constant"),
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        # PERF / 병렬
        "--tensor-model-parallel-size", str(tp),
        "--sequence-parallel",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(sl.get("max_tokens_per_gpu", 9216)),
        # SGLANG 롤아웃 엔진
        "--rollout-num-gpus-per-engine", str(engine_gpus),
        "--sglang-mem-fraction-static", str(sl.get("sglang_mem_fraction", 0.7)),
        # MISC (run 스크립트 권장값)
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        # WANDB
        "--use-wandb",
        "--wandb-project", wandb_cfg.get("project", "tfct-ppo"),
        "--wandb-group", cfg.run_name(),
    ]

    # bash 래퍼: 모델 arch args 를 source 로 채우고(추정 금지 — slime repo 가 제공), 필요시 변환,
    # ray 기동 후 job submit. slime_grpo 와 동일 패턴(다른 건 args 가 PPO 라는 점뿐).
    extra = shlex.join(args)
    rt_env = json.dumps({"env_vars": {
        "PYTHONPATH": megatron_dir, "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }})
    convert_py = shlex.quote(f"{slime_dir}/tools/convert_hf_to_torch_dist.py")
    train_py = shlex.quote(f"{slime_dir}/train.py")
    hf = shlex.quote(model_cfg["name"])
    ref = shlex.quote(str(torch_dist))
    mega = shlex.quote(megatron_dir)
    script = f"""set -ex
source {shlex.quote(str(model_script))}
if [ ! -d {ref} ]; then
  PYTHONPATH={mega} python3 {convert_py} \
    "${{MODEL_ARGS[@]}}" --hf-checkpoint {hf} --save {ref}
fi
ray start --head --node-ip-address 127.0.0.1 --num-gpus {gpus} \
  --disable-usage-stats --dashboard-port=8265
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json={shlex.quote(rt_env)} \
  -- python3 {train_py} "${{MODEL_ARGS[@]}}" {extra}
"""
    subprocess.run(["bash", "-c", script], check=True, cwd=slime_dir)
