"""순수 Megatron-LM 사전학습 경로 (from-scratch·full) — 수직 파이프라인 1단계 + 가로비교.

torchtitan 사전학습과 같은 칸(초소형 Qwen3 from-scratch on wikitext)이지만 프레임워크가 순수
Megatron-LM(`pretrain_gpt.py`)이다 — 기업이 콕 집어 요구하는 'Megatron-LM 경험'. SFT(post_training/
modelopt)·GRPO(examples/rl)에 이은 megatron-lm 세 번째 진입점(같은 이미지, 다른 entry).

torchtitan 과 동일 arch·데이터로 from-scratch 학습해 가로비교가 성립한다(통제: 모델·코퍼스 고정,
프레임워크만 변수). arch 는 model_sizes.megatron_arch_args(tfct_tiny 와 동일 치수 — dim512/6층/
heads8/kv4/ffn1536/vocab151936/RMSNorm/SwiGLU/rope/qk-layernorm/tied). 플래그명·값은 upstream
core_v0.17.1 의 examples/rl/model_configs/qwen3_8b.sh(arch)·examples/gpt3/train_gpt3_175b_
distributed.sh(training/data/logging)를 미러 = 같은 태그라 버전 정합(추정 아님).

이 모듈의 train() 은 3단계 subprocess:
  1. wikitext → {"text"} JSONL 을 떨군다(빈 줄 제외).
  2. Megatron 은 인덱스(.bin/.idx) 데이터를 요구 → tools/preprocess_data.py 로 토크나이즈·인덱싱
     (--tokenizer-type HuggingFaceTokenizer = Qwen3, vocab 151936). 산출 prefix `_text_document`.
  3. `torchrun pretrain_gpt.py <arch> <training> <data> <logging>` 로 from-scratch 학습(DCP 저장).

무거운 deps(megatron-core/TE)는 이미지(docker/megatron-lm.Dockerfile, SFT/GRPO 와 동일)에만 있고
torchrun/preprocess 서브프로세스가 임포트한다. 호스트 모듈은 datasets/transformers 만 지연 임포트.

⚠️ GPU 검증 대기: Megatron 은 CPU 스모크 불가(TE/CUDA) → 학습 루프·preprocess_data 인자 정합·
arch 플래그(특히 core_v0.17.1 이 TransformerConfig 에서 자동생성하는 --qk-layernorm/--disable-bias-
linear 의 boolean 관례)·wandb 플래그명·tied embedding 정합은 GPU end-to-end 에서 최종 확인.
파이프라인 다음 단계(SFT) 입력용 mcore→HF export 는 Megatron-Bridge 가 담당(별도 경로) — 이
트레이너 범위 밖(convert.py qwen3 로더 블로커와 같은 사정).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..config import RunConfig
from ..model_sizes import megatron_arch_args


def _stage_tokenizer(cfg: RunConfig, work: Path) -> str:
    """Qwen3 토크나이저를 디렉토리로 저장하고 경로를 돌려준다.

    사전학습은 raw text 라 chat template 불필요(SFT/RL 이 같은 vocab 151936 토크나이저를 이어 써
    파이프라인 정합). preprocess 와 학습이 같은 토크나이저를 쓰도록 한 디렉토리를 공유한다.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.section("model")["tokenizer"])
    tdir = work / "hf_tokenizer"
    tok.save_pretrained(str(tdir))
    return str(tdir)


def _prepare_indexed_data(cfg: RunConfig, work: Path, tokenizer_dir: str, repo: str) -> str:
    """wikitext → JSONL → Megatron 인덱스(.bin/.idx). data-path prefix 를 돌려준다.

    Megatron GPTDataset 은 preprocess_data.py 로 만든 인덱스를 요구한다(HF 스트리밍 아님). json-keys
    text → 산출 `<prefix>_text_document`. 빈 줄(wikitext 헤더/공백)은 제외.
    """
    from datasets import load_dataset

    ds_cfg = cfg.section("dataset")
    mg = cfg.section("megatron")
    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])

    jsonl = work / "wikitext.jsonl"
    with jsonl.open("w") as f:
        for row in raw:
            text = (row.get("text") or "").strip()
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    prefix = work / "wikitext"
    preprocess = Path(repo) / "tools" / "preprocess_data.py"
    cmd = [
        "python",
        str(preprocess),
        "--input", str(jsonl),
        "--json-keys", "text",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_dir,
        "--output-prefix", str(prefix),
        "--append-eod",
        "--workers", str(mg.get("preprocess_workers", 8)),
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    return f"{prefix}_text_document"


def train(cfg: RunConfig) -> None:
    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    mg = cfg.section("megatron")
    wandb_cfg = cfg.section("wandb")

    repo = os.environ.get("MEGATRON_LM_DIR", "/opt/Megatron-LM")

    out_dir = Path(out.get("local_dir", "out"))
    work = out_dir / "megatron_pretrain_workspace"
    dcp_dir = work / "checkpoint"   # DCP 체크포인트(torch_dist)
    work.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = _stage_tokenizer(cfg, work)
    data_path = _prepare_indexed_data(cfg, work, tokenizer_dir, repo)

    gpus = scale.get("gpus", 1)
    nodes = scale.get("nodes", 1)
    tp = mg.get("tensor_model_parallel_size", 1)
    pp = mg.get("pipeline_model_parallel_size", 1)
    dp = max(1, gpus // (tp * pp))

    micro = hp["per_device_batch_size"]
    global_bs = micro * dp  # 사전학습엔 grad_accum knob 없음 → global = micro × dp

    steps = hp.get("steps", 2000)
    debug_steps = cfg.section("debug").get("max_steps", -1)
    if debug_steps and debug_steps > 0:
        steps = debug_steps

    seq_len = model_cfg.get("max_seq_len", 2048)
    lr = float(hp["learning_rate"])
    min_lr = float(hp.get("min_learning_rate", lr / 10))

    args = [
        # ARCH (model_sizes = tfct_tiny 정합, qwen3_8b.sh 미러). tied → --untie 안 붙임.
        *megatron_arch_args(model_cfg["size"], seq_len),
        # 병렬 / 백엔드
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        "--attention-backend", "auto",
        "--bf16",
        # TRAINING (gpt3 예제 미러)
        "--micro-batch-size", str(micro),
        "--global-batch-size", str(global_bs),
        "--train-iters", str(steps),
        "--lr", str(lr),
        "--min-lr", str(min_lr),
        "--lr-decay-style", hp.get("lr_scheduler", "cosine"),
        "--lr-decay-iters", str(steps),
        "--lr-warmup-fraction", str(hp.get("warmup_ratio", 0.02)),
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--clip-grad", "1.0",
        "--init-method-std", "0.02",
        # TOKENIZER (Qwen3, vocab 151936 = 파이프라인 정합)
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_dir,
        # DATA (val split 없음 → split 100,0,0 + eval-iters 0)
        "--data-path", data_path,
        "--split", "100,0,0",
        # CKPT / LOGGING
        "--save", str(dcp_dir),
        "--load", str(dcp_dir),
        "--ckpt-format", "torch_dist",
        "--save-interval", str(mg.get("save_interval", 1000)),
        "--log-interval", str(mg.get("log_interval", 10)),
        "--eval-interval", str(mg.get("save_interval", 1000)),
        "--eval-iters", "0",
        "--wandb-project", wandb_cfg.get("project", "tfct-pretrain"),
        "--wandb-exp-name", cfg.run_name(),
    ]

    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        f"--nnodes={nodes}",
        str(Path(repo) / "pretrain_gpt.py"),
        *args,
    ]
    env = {**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"}
    subprocess.run(cmd, check=True, cwd=repo, env=env)
