"""slime SFT 학습 경로 (offline 지도학습, full 전용).

slime 의 정체성은 RL(SGLang 롤아웃 + Megatron 학습)이지만, SFT 도 **네이티브로** 지원한다 —
"rollout" 추상을 재활용해서다. RL 에선 rollout = SGLang 생성이지만, SFT 에선 rollout =
고정 데이터(messages)를 토크나이즈하고 loss mask 만 만드는 것(slime.rollout.sft_rollout).
즉 학습 백엔드(Megatron)는 그대로 두고, 데이터 생산 단계만 생성→고정데이터로 바꾼다.

구동(scripts/run-qwen3-4B-base-sft.sh 미러): slime 은 bash 배열 args + ray 로 돈다. 이 모듈의
train():
  1. traceinversion → {messages} JSONL 을 떨군다(--input-key messages). prompt 가 아니라 messages
     리스트 그대로다 — sft_rollout 이 messages 를 받아 loss mask 를 직접 계산하기 때문.
  2. base 모델을 로컬로 받아 **캐논 REASONING_CHATML 을 토크나이저에 구워** hf-checkpoint 로 쓴다
     (아래 _prepare_hf_dir 주석 — slime 의 loss mask 가 그 토크나이저의 chat template 으로
     계산되므로 이게 통제 변수다). 이어서 HF→Megatron torch_dist 변환(ref-load).
  3. ray 를 띄우고 `ray job submit -- python train_async.py <args>` 로 SFT 를 돌린다.
     핵심 SFT 플래그: --rollout-function-path slime.rollout.sft_rollout.generate_rollout ·
     --loss-type sft_loss · --loss-mask-type qwen3 · --calculate-per-token-loss ·
     --disable-compute-advantages-and-returns · --debug-train-only(SGLang 엔진 스킵, reward 없음).

통제비교 정합 = **loss mask 가 우리 캐논 template 으로 계산된다**. slime 의
MultiTurnLossMaskGenerator 는 tokenizer.apply_chat_template(...) 로 assistant span 을 찾는다
(tokenizer_type=qwen3 는 *알고리즘* 선택일 뿐, template 은 토크나이저가 들고 있는 걸 쓴다). 우리
REASONING_CHATML 을 구워 넘기면 다른 프레임워크(assistant_only_loss + {% generation %})와 같은
응답 마스킹 = 통제 변수. RL 경로(slime_grpo)가 prompt 를 미리 렌더해 우회한 것과 달리, SFT 는
slime 이 직접 마스킹하므로 template 주입이 불가피하다.

무거운 deps(slime/megatron)는 공식 이미지(docker/slime.Dockerfile) 안에만 있고 train_async.py
서브프로세스가 임포트한다. 이 모듈은 datasets/transformers/huggingface_hub 만 지연 임포트한다.

⚠️ GPU 검증 대기: --hf-checkpoint 로 박은 캐논 template 이 qwen3 mask 와 맞물리는지(응답만 1)·
torch_dist 변환·num-epoch→num_rollout 환산·batch 제약(SFT 는 rollout_bs=global_bs, 그룹 없음)·
train_async --debug-train-only(콜로케이트 없이)는 GPU end-to-end 에서 최종 확인 대상.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig


def _prepare_sft_jsonl(cfg: RunConfig, out_dir: Path) -> str:
    """traceinversion → {messages} JSONL. 경로를 돌려준다.

    RL 경로(slime_grpo)는 prompt 를 렌더한 문자열을 박지만, SFT 는 messages 리스트 그대로 둔다 —
    sft_rollout 이 messages 를 받아 토크나이즈 + loss mask 를 만들기 때문(--input-key messages).
    """
    from datasets import load_dataset

    ds_cfg = cfg.section("dataset")
    to_messages = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)  # sft/slime → to_trl = {messages: [...]}

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    path = out_dir / "data" / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in raw:
            line = to_format(to_messages(row))  # {"messages": [{role, content}, ...]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return str(path)


def _prepare_hf_dir(cfg: RunConfig, out_dir: Path) -> str:
    """base 모델을 로컬로 받아 캐논 chat template 을 토크나이저에 굽고 그 디렉토리를 돌려준다.

    왜 머터리얼라이즈인가: slime 은 토크나이저를 `--hf-checkpoint` 에서 로드하고(별도
    --tokenizer-path 없음), 그 토크나이저의 chat template 으로 SFT loss mask 를 계산한다. base
    모델엔 template 이 없으므로 우리 REASONING_CHATML 을 구워 넣어야 한다 = 통제 변수. hf-checkpoint
    는 HF→torch_dist 변환에도 쓰이므로(가중치 필요) 토크나이저만이 아니라 모델 전체를 받는다.

    chat_template 이 없으면(None) base 모델 hub id 를 그대로 돌려준다 — 단 slime qwen3 mask 는
    template 을 요구하므로 SFT 엔 사실상 항상 template 이 있어야 한다(_base 가 reasoning_chatml).
    """
    model_cfg = cfg.section("model")
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if not chat_template:
        return model_cfg["name"]

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    local = out_dir / "hf_model"
    snapshot_download(repo_id=model_cfg["name"], local_dir=str(local))
    tok = AutoTokenizer.from_pretrained(str(local))
    tok.chat_template = chat_template
    tok.save_pretrained(str(local))  # 같은 디렉토리에 template 덮어쓰기(가중치는 그대로)
    return str(local)


def train(cfg: RunConfig) -> None:
    if cfg.tuning != "full":
        raise SystemExit(
            f"slime 은 full 전용이다(base slime 에 LoRA 없음 — Miles 확장 필요). "
            f"tuning={cfg.tuning!r} 미지원 — LoRA SFT 는 trl/unsloth/verl 등을 써라."
        )

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
    train_jsonl = _prepare_sft_jsonl(cfg, out_dir)
    hf_checkpoint = _prepare_hf_dir(cfg, out_dir)   # 캐논 template 구운 로컬 모델 dir
    torch_dist = out_dir / "ref_torch_dist"          # HF→Megatron 변환(=init/ref-load)
    slime_ckpt = out_dir / "slime_ckpt"              # 학습 체크포인트(load/save)

    gpus = scale.get("gpus", 1)
    nodes = scale.get("nodes", 1)
    tp = sl.get("tensor_model_parallel_size", 1)

    # SFT batch: 그룹(n-samples-per-prompt) 없음 → rollout_batch_size = global_batch_size
    # (공식 SFT 레시피가 128/128). RL 의 global = rollout_bs × G 와 다른 SFT 의 본질.
    batch = sl.get("rollout_batch_size", 128)

    # 학습량: 기본은 num-epoch(slime 이 dataset 크기로 num_rollout 환산). 스모크(max_steps>0)는
    # rollout 루프 수를 직접 박는다(--num-rollout, epoch 환산 우회).
    max_steps = debug.get("max_steps", -1)
    epoch_or_rollout = (
        ["--num-rollout", str(max_steps)]
        if (max_steps and max_steps > 0)
        else ["--num-epoch", str(hp.get("epochs", 3))]
    )

    # train_async.py args (run-qwen3-4B-base-sft.sh 미러). MODEL_ARGS 는 bash source 가 채운다.
    args = [
        "--actor-num-nodes", str(nodes),
        "--actor-num-gpus-per-node", str(gpus),
        # CKPT (init/ref = 변환한 base, load/save = 학습 ckpt)
        "--hf-checkpoint", hf_checkpoint,
        "--ref-load", str(torch_dist),
        "--load", str(slime_ckpt),
        "--save", str(slime_ckpt),
        "--save-interval", str(sl.get("save_interval", 1000)),
        # ROLLOUT = 고정데이터 인입(생성 아님). messages 리스트를 sft_rollout 이 마스킹.
        "--rollout-function-path", "slime.rollout.sft_rollout.generate_rollout",
        "--prompt-data", train_jsonl,
        "--input-key", "messages",
        "--rollout-shuffle",
        *epoch_or_rollout,
        "--rollout-batch-size", str(batch),
        "--global-batch-size", str(batch),
        # SFT loss: 응답만 마스킹(loss-mask-type=qwen3, 토크나이저 = 우리 캐논 template),
        # advantage/returns 끔, SGLang 엔진 스킵(reward 없음).
        "--loss-type", "sft_loss",
        "--loss-mask-type", sl.get("loss_mask_type", "qwen3"),
        "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--debug-train-only",
        # OPTIMIZER (SFT 레시피값: betas 0.9/0.95, wd 0.1)
        "--optimizer", "adam",
        "--lr", str(float(hp["learning_rate"])),
        "--lr-decay-style", hp.get("lr_scheduler", "cosine"),
        "--min-lr", str(float(hp.get("min_learning_rate", 1.0e-6))),
        "--lr-warmup-fraction", str(hp.get("warmup_ratio", 0.1)),
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        # PERF / 병렬
        "--tensor-model-parallel-size", str(tp),
        "--sequence-parallel",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(sl.get("max_tokens_per_gpu", 9216)),
        # MISC (run 스크립트 권장값)
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        # WANDB
        "--use-wandb",
        "--wandb-project", wandb_cfg.get("project", "tfct-sft"),
        "--wandb-group", cfg.run_name(),
    ]

    # bash 래퍼: 모델 arch args 를 source 로 채우고(추정 금지 — slime repo 가 제공), 필요시 변환,
    # ray 기동 후 job submit. slime_grpo 와 같은 패턴이되 train_async.py + 콜로케이트 없음
    # (async 는 콜로케이트 불가, SFT 는 SGLang 스킵이라 콜로케이트 무의미 — 공식 SFT 레시피 그대로).
    extra = shlex.join(args)
    rt_env = json.dumps({"env_vars": {
        "PYTHONPATH": megatron_dir, "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }})
    convert_py = shlex.quote(f"{slime_dir}/tools/convert_hf_to_torch_dist.py")
    train_py = shlex.quote(f"{slime_dir}/train_async.py")
    hf = shlex.quote(hf_checkpoint)
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
