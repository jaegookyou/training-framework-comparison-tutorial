"""순수 Megatron-LM GRPO 학습 경로 (online on-policy RL, full 전용).

verl 백엔드/래퍼가 아니라 **Megatron-LM repo 의 examples/rl** (train_rl.py + megatron/rl 모듈)을
그대로 구동한다(= 기업이 콕 집어 요구하는 'Megatron-LM 경험'의 RL 판; SFT 의 post_training/modelopt
경로와 같은 정신, 다른 서브트리). GRPO 가로비교 3번째 축: verl(ray+vllm)·slime(SGLang+Megatron)
에 더해 megatron-lm(네이티브 train_rl).

구동(examples/rl README 미러): bash 오케스트레이션.
  1. env config(YAML)을 떨군다 — 우리 TfctGSM8KAgent(gsm8k + 캐논 template + 공유 reward)를
     점경로로 가리킨다. reward·프롬프트가 다른 프레임워크와 동일 = 통제 변수.
  2. HF Qwen3-8B-Base → Megatron mcore(torch_dist) 체크포인트로 변환(없으면).
  3. `examples/rl/model_configs/<model_script>.sh` 를 source 해 arch args(MODEL_OPTIONS/
     COMMON_OPTIONS/ENV_DEPENDENT)를 채우고, env(TP/PP/CHECKPOINT/ENV_CONFIG/GRPO_*)로 조정한 뒤
     `torchrun train_rl.py <static flags> $COMMON_OPTIONS $MODEL_OPTIONS $ENV_DEPENDENT` 로 띄운다.

reward = 통제 변수: TfctGSM8KAgent.get_reward 가 adapters.rewards.gsm8k_score 를 쓴다(TRL/verl/
slime 과 같은 채점 코어). 기본 GSM8KAgent 의 math_verify 채점이 아니라.

무거운 deps(megatron/te/RL deps)는 이미지(docker/megatron-lm.Dockerfile, RL 추가분 포함) 안에만
있고 torchrun 서브프로세스가 임포트한다. 이 모듈은 transformers 도 안 쓴다(템플릿=에이전트 몫).

⚠️ GPU 검증 대기:
  - **HF→mcore 변환**(최대 블로커): core_v0.17.1 의 tools/checkpoint/convert.py(llama_mistral
    로더)는 --model-size 가 qwen2.5 까지만(qwen3 없음). qwen3_8b.sh 헤더가 "MegatronBridge
    run_config 기반"이라 적혀 있듯 Qwen3 mcore 는 **Megatron-Bridge import** 로 만드는 게 정석.
    → megatron.mcore_checkpoint 로 미리 변환한 ckpt 경로를 넘기면 변환을 건너뛴다(권장 탈출구).
    경로 미지정 시 문서화된 convert.py 를 그대로 호출하되 model_size 는 GPU 에서 확정.
  - train_rl 정확한 플래그/exit 의미(exit-interval vs train-samples)·병렬 곱·이중 template 위험.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from ..config import RunConfig

# env config 가 가리킬 우리 커스텀 에이전트(통제 변수 reward·프롬프트).
_AGENT_TYPE = "training_framework_comparison_tutorial.megatron_rl.gsm8k_agent.TfctGSM8KAgent"


def _write_env_config(cfg: RunConfig, out_dir: Path) -> str:
    """examples/rl 포맷의 env config(에이전트 리스트) YAML 을 떨구고 경로를 돌려준다.

    네이티브 gsm8k.yaml 과 같은 모양이되 agent_type 을 우리 TfctGSM8KAgent 로 바꿔 reward·
    프롬프트를 통제 변수로 고정한다. yaml 의존 없이 손으로 쓴다(들여쓰기 단순).
    """
    model_cfg = cfg.section("model")
    agent_args = {
        "answer_format": "boxed",
        "hf_model": model_cfg["name"],
        "chat_template": model_cfg.get("chat_template"),
    }
    lines = [
        f"- agent_type: {_AGENT_TYPE}",
        "  agent_args:",
        *[f"    {k}: {json.dumps(v)}" for k, v in agent_args.items()],
        "  weight: 1.0",
        "  evaluation_only: false",
        "",
    ]
    path = out_dir / "env_config" / "gsm8k.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return str(path)


def train(cfg: RunConfig) -> None:
    if cfg.tuning != "full":
        raise SystemExit(
            f"megatron-lm GRPO 는 full RL 전용이다(examples/rl 에 LoRA 경로 없음). "
            f"tuning={cfg.tuning!r} 미지원 — LoRA RL 은 trl/unsloth 경로를 써라."
        )

    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    mg = cfg.section("megatron")
    wandb_cfg = cfg.section("wandb")
    debug = cfg.section("debug")

    repo = os.environ.get("MEGATRON_LM_DIR", "/opt/Megatron-LM")
    rl_dir = Path(repo) / "examples" / "rl"
    model_script = rl_dir / "model_configs" / f"{mg['model_script']}.sh"

    out_dir = Path(out.get("local_dir", "out"))
    env_config = _write_env_config(cfg, out_dir)
    run_dir = out_dir / "rl_run"
    ckpt_dir = run_dir / "checkpoints"
    data_cache = run_dir / "data_cache"
    for d in (ckpt_dir, data_cache):
        d.mkdir(parents=True, exist_ok=True)

    # HF→mcore: 미리 변환한 경로가 있으면 그걸 쓰고(권장), 없으면 문서화된 convert.py 로 만든다.
    mcore_ckpt = mg.get("mcore_checkpoint") or str(run_dir / "mcore_init")

    # 병렬 곱 = nproc_per_node × nnodes. TP×PP×DP = gpus(노드 내).
    gpus = scale.get("gpus", 1)
    nodes = scale.get("nodes", 1)
    tp = mg.get("tensor_model_parallel_size", 1)
    pp = mg.get("pipeline_model_parallel_size", 1)

    # GRPO 눈금(통제비교): num_generations=그룹 G, beta=KL. prompts_per_step·global_bs 도 knob.
    group = hp.get("num_generations", 8)
    kl_beta = hp.get("beta", 0.04)
    prompts_per_step = mg.get("grpo_prompts_per_step", 64)
    global_bs = mg.get("global_batch_size", 256)
    micro_bs = mg.get("micro_batch_size", 1)
    # gsm8k 짧은 프롬프트 + reasoning 응답 상한(prompt+completion). model.max_seq_len 가 가늠.
    max_seq_len = model_cfg.get(
        "max_seq_len",
        hp.get("max_prompt_length", 512) + hp.get("max_completion_length", 1024),
    )

    # 스모크: max_steps>0 이면 그만큼 iteration 후 종료(exit-interval). 아니면 megatron knob.
    max_steps = debug.get("max_steps", -1)
    exit_interval = max_steps if (max_steps and max_steps > 0) else mg.get("exit_interval", 200)
    save_interval = mg.get("save_interval", exit_interval)
    # RL 은 train-samples 를 상한(ceiling)으로만 쓰고 실제 종료는 exit-interval 이 건다(README).
    train_samples = mg.get("train_samples", 48828125)

    # qwen3_8b.sh 가 읽는 env(arch args 를 source 가 채움 — 추정 금지, repo 제공).
    model_env = {
        **os.environ,
        "TP": str(tp),
        "PP": str(pp),
        "CHECKPOINT": mcore_ckpt,
        "ENV_CONFIG": env_config,
        "GRPO_GROUP_SIZE": str(group),
        "GRPO_KL_BETA": str(kl_beta),
        "GRPO_PROMPTS_PER_STEP": str(prompts_per_step),
        "TRAINING_BATCH_SIZE": str(global_bs),
        "MICRO_BATCH_SIZE": str(micro_bs),
        "MAX_SEQ_LENGTH": str(max_seq_len),
        "EXIT_INTERVAL": str(exit_interval),
        "CHKPT_SAVE_INTERVAL": str(save_interval),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }

    # train_rl static flags (README 실험 커맨드 미러; model_script 가 안 채우는 것만). lr 은
    # 끝에 붙여 MODEL_OPTIONS 의 기본 --lr 을 덮는다(통제 변수, argparse 는 마지막 값 채택).
    static = [
        "--mock-data",                       # RL 은 자체 rollout 으로 생성 → 표준 dataloader 더미
        "--perform-rl-step",
        "--finetune",
        "--sequence-parallel",
        "--use-distributed-optimizer",
        "--no-create-attention-mask-in-dataloader",
        "--accumulate-allreduce-grads-in-fp32",
        "--calculate-per-token-loss",
        "--train-samples", str(train_samples),
        "--exit-interval", str(exit_interval),
        "--save-interval", str(save_interval),
        "--eval-interval", str(mg.get("eval_interval", 20)),
        "--rl-prompts-per-eval", str(mg.get("rl_prompts_per_eval", 32)),
        "--log-interval", "10",
        "--distributed-timeout-minutes", "60",
        "--seed", str(mg.get("seed", 42)),
        "--data-cache-path", str(data_cache),
        "--save", str(ckpt_dir),
        "--load", str(ckpt_dir),
        "--wandb-project", wandb_cfg.get("project", "tfct-grpo"),
        "--wandb-exp-name", cfg.run_name(),
        "--lr", str(float(hp["learning_rate"])),
    ]

    # 선택적 변환: mcore_checkpoint 미지정 + 미존재 시 문서화된 convert.py 호출(GPU 검증 대상).
    do_convert = not mg.get("mcore_checkpoint") and not Path(mcore_ckpt).exists()
    convert_block = ""
    if do_convert:
        model_size = mg.get("convert_model_size", "qwen2.5")  # ⚠️ qwen3 미지원 — GPU 에서 확정
        convert_args = shlex.join([
            "--bf16", "--model-type", "GPT", "--loader", "llama_mistral", "--saver", "core",
            "--target-tensor-parallel-size", str(tp), "--checkpoint-type", "hf",
            "--load-dir", model_cfg["name"], "--save-dir", mcore_ckpt,
            "--tokenizer-model", model_cfg["name"], "--model-size", model_size,
            "--loader-transformer-impl", "transformer_engine",
            "--make-vocab-size-divisible-by", "128",
        ])
        convert_block = (
            f"if [ ! -d {shlex.quote(mcore_ckpt)} ]; then\n"
            f"  python3 tools/checkpoint/convert.py {convert_args}\n"
            f"fi\n"
        )

    extra = shlex.join(static)
    nproc = gpus
    # bash 래퍼: arch args 는 model_script source 가 채운다(MODEL_OPTIONS/COMMON_OPTIONS/
    # ENV_DEPENDENT 는 bash 배열/문자열이라 Python 에서 못 만든다 — source 가 정석, slime 과 같은
    # 패턴). cwd=repo(examples.rl 점경로 임포트·상대경로 정합).
    script = f"""set -ex
{convert_block}source {shlex.quote(str(model_script))}
torchrun --nproc-per-node={nproc} --nnodes={nodes} examples/rl/train_rl.py \
  {extra} $COMMON_OPTIONS $MODEL_OPTIONS $ENV_DEPENDENT
"""
    subprocess.run(["bash", "-c", script], check=True, env=model_env, cwd=repo)
