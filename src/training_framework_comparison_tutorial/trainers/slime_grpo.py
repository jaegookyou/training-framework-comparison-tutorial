"""slime GRPO 학습 경로 (online on-policy RL, full 전용).

slime = THUDM 의 RL 사후학습 프레임워크. 정체성 = **SGLang 롤아웃 + Megatron 학습**을 잇는 RL
오케스트레이터. verl 과 같은 칸(on-policy GRPO)이지만 롤아웃 엔진이 SGLang(verl=vllm)이고 학습
백엔드가 Megatron 으로 고정이다. 한때 "verl 과 겹쳐" 후보에서 빠졌다가 RL 가로비교 축으로 재추가.

구동(run-qwen3-*.sh 미러): slime 은 bash 배열 args + ray 로 돈다 — TRL/Unsloth 의 인프로세스
trainer.train() 와 근본적으로 다르다(verl 보다도 멀다). 이 모듈의 train() 은:
  1. gsm8k → JSONL 을 떨군다. prompt 는 **우리 캐논 REASONING_CHATML 로 미리 렌더한 문자열**
     (--apply-chat-template 대신 → base 모델 토크나이저에 template 이 없어도 다른 프레임워크와
     동일한 프롬프트 = 통제 변수). label=정답, metadata={data_source} 로 reward 라우팅.
  2. HF 모델 → Megatron torch_dist 체크포인트로 변환(slime tools/convert_hf_to_torch_dist.py).
     ref-load 가 torch_dist 를 요구. 이미 있으면 건너뛴다.
  3. ray 를 띄우고 `ray job submit -- python train.py <args>` 로 GRPO 를 돌린다.
     reward = --custom-rm-path 로 우리 slime_rm(adapters.rewards) 주입(내장 --rm-type math 대신
     → TRL/verl 과 같은 채점 코어 공유 = 통제 변수).

무거운 deps(slime/sglang/megatron)는 공식 이미지(docker/slime.Dockerfile = slimerl/slime 기반)
안에만 있고 train.py 서브프로세스가 임포트한다. 이 모듈은 datasets/transformers 만 지연 임포트한다.

⚠️ GPU 검증 대기: slime arg 정합(특히 batch 제약 rollout_bs×n_samples = global_bs×steps)·
torch_dist 변환·SGLang/Megatron 병렬 곱·캐논 template 렌더는 GPU end-to-end 에서 최종 확인 대상.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig

# 우리 reward 를 slime --custom-rm-path 로 넘길 모듈 경로(함수 slime_rm).
_RM_PATH = "training_framework_comparison_tutorial.adapters.rewards.slime_rm"


def _prepare_jsonl(cfg: RunConfig, out_dir: Path) -> str:
    """gsm8k → {prompt(렌더된 문자열), label, metadata} JSONL. 경로를 돌려준다."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds_cfg = cfg.section("dataset")
    model_cfg = cfg.section("model")
    reward_name = cfg.section("reward")["name"]  # = data_source(라우팅 키)
    to_prompt = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)

    # 캐논 chat template 으로 프롬프트를 미리 렌더(add_generation_prompt → rollout 큐 정합).
    # 다른 프레임워크는 토크나이저에 template 을 구워 rollout 때 적용 — slime 은 렌더 결과를
    # JSONL 에 박아 동일한 입력을 보장한다(통제 변수).
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    path = out_dir / "data" / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in raw:
            ex = to_format(to_prompt(row))  # {prompt: messages, label: 정답}
            rendered = tokenizer.apply_chat_template(
                ex["prompt"], tokenize=False, add_generation_prompt=True
            )
            line = {
                "prompt": rendered,
                "label": ex["label"],
                "metadata": {"data_source": reward_name},
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
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
    torch_dist = out_dir / "ref_torch_dist"   # HF→Megatron 변환 산출(=ref-load)
    slime_ckpt = out_dir / "slime_ckpt"        # 학습 체크포인트(load/save)

    gpus = scale.get("gpus", 1)
    nodes = scale.get("nodes", 1)
    tp = sl.get("tensor_model_parallel_size", 1)
    engine_gpus = sl.get("rollout_num_gpus_per_engine") or tp

    # batch 제약(slime): rollout_batch_size × n_samples_per_prompt = global_batch_size ×
    # num_steps_per_rollout. num_steps=1 로 고정 → global = rollout_bs × G.
    group = hp.get("num_generations", 8)          # = n-samples-per-prompt(그룹 크기 G)
    rollout_bs = sl.get("rollout_batch_size", 32)  # 라운드당 프롬프트 수
    global_bs = rollout_bs * group

    # 스모크: max_steps>0 이면 그만큼만 rollout 루프. 아니면 slime.num_rollout(태스크 기본).
    # NOTE: slime 은 epochs 가 아니라 rollout 루프 수로 학습량을 센다(프레임워크 본질 차이).
    max_steps = debug.get("max_steps", -1)
    num_rollout = max_steps if (max_steps and max_steps > 0) else sl.get("num_rollout", 100)

    # train.py args (run-qwen3-*.sh 의 배열들을 한 리스트로). MODEL_ARGS 는 bash source 가 채운다.
    args = [
        "--actor-num-nodes", str(nodes),
        "--actor-num-gpus-per-node", str(gpus),
        "--colocate",                                  # 학습/롤아웃이 같은 GPU 공유(단노드)
        # CKPT
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
        # GRPO
        "--advantage-estimator", "grpo",
        "--use-kl-loss",
        "--kl-loss-coef", str(hp.get("beta", 0.04)),
        "--kl-loss-type", "low_var_kl",
        "--eps-clip", "0.2",
        "--eps-clip-high", "0.28",
        # OPTIMIZER
        "--optimizer", "adam",
        "--lr", str(float(hp["learning_rate"])),
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
        "--wandb-project", wandb_cfg.get("project", "tfct-grpo"),
        "--wandb-group", cfg.run_name(),
    ]

    # bash 래퍼: 모델 arch args 를 source 로 채우고(추정 금지 — slime repo 가 제공), 필요시 변환,
    # ray 기동 후 job submit. MODEL_ARGS 는 bash 배열이라 Python 에서 못 만든다 → source 가 정석.
    # shlex-quote 된 조각들을 bash 변수로 먼저 묶어 각 호출 줄을 짧게 유지한다.
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
