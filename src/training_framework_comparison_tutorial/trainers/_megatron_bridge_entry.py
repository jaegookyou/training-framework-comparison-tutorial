"""torchrun 안에서만 도는 Megatron-Bridge 진입점. `torchrun -m <이 모듈>` 로 실행된다.

호스트 프로세스(megatron_bridge_sft.py)는 megatron 을 import 하지 않는다 — 무거운 deps 는
이 모듈이 torchrun rank 별로 import 한다. 두 stage:
  --stage convert  : AutoBridge.import_ckpt 로 HF Qwen3-8B-Base → Megatron-core 체크포인트.
  --stage finetune : qwen3 SFT/PEFT 레시피를 우리 RunConfig 로 override 한 뒤 finetune().

megatron import 는 전부 함수 안(지연)이라 megatron 없는 dev/CI 에서도 모듈 import 는 된다.
"""

from __future__ import annotations

import argparse

from ..adapters import get_source
from ..config import RunConfig


def _convert(cfg: RunConfig, megatron_path: str) -> None:
    """HF base 가중치 → Megatron-core 체크포인트(finetune 의 pretrained_checkpoint)."""
    import torch
    from megatron.bridge import AutoBridge

    AutoBridge.import_ckpt(
        hf_model_id=cfg.section("model")["name"],
        megatron_path=megatron_path,
        torch_dtype=torch.bfloat16 if cfg.section("hp").get("bf16", True) else torch.float32,
    )


def _build_finetune_config(cfg: RunConfig, megatron_path: str, tokenizer_dir: str | None):
    """RunConfig → Megatron-Bridge ConfigContainer (full=SFT 레시피 / lora=PEFT 레시피)."""
    from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
    from megatron.bridge.recipes.qwen.qwen3 import (
        qwen3_8b_peft_config,
        qwen3_8b_sft_config,
    )

    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    scale = cfg.section("scale")
    lora = cfg.section("lora")
    mb = cfg.section("megatron")
    ds_cfg = cfg.section("dataset")
    debug = cfg.section("debug")

    if cfg.tuning == "lora":
        config = qwen3_8b_peft_config("lora")
        config.peft.dim = lora.get("r", 16)
        config.peft.alpha = lora.get("alpha", 32)
        config.peft.dropout = lora.get("dropout", 0.0)
    else:
        config = qwen3_8b_sft_config()

    # base 가중치 = convert 산출 mcore 체크포인트. 레시피는 arch(Qwen3-8B=Base 동형)만 제공.
    config.checkpoint.pretrained_checkpoint = megatron_path

    # 토크나이저 = 캐논 chat template 을 구운 디렉토리(없으면 모델 자체). chat SFT 의 마스킹이
    # 이 template 의 assistant 구간(REASONING_CHATML 의 {% generation %})을 쓴다 → 통제비교 정합.
    config.tokenizer.tokenizer_model = tokenizer_dir or model_cfg["name"]

    # 병렬 = megatron 고유 knob. 곱(TP×PP×DP)=gpus 로 DP 를 도출(다른 경로와 같은 눈금).
    tp = mb.get("tensor_model_parallel_size", config.model.tensor_model_parallel_size)
    pp = mb.get("pipeline_model_parallel_size", 1)
    config.model.tensor_model_parallel_size = tp
    config.model.pipeline_model_parallel_size = pp

    config.model.seq_length = model_cfg.get("max_seq_len", 2048)
    config.optimizer.lr = float(hp["learning_rate"])

    micro = hp.get("per_device_batch_size", 1)
    config.train.micro_batch_size = micro
    gpus = scale.get("gpus", 1)
    dp = max(1, gpus // (tp * pp))
    global_bs = micro * hp.get("gradient_accumulation", 1) * dp
    config.train.global_batch_size = global_bs

    # Megatron 은 epoch 이 아니라 iter 단위 → train_samples(≈epochs×N) / global_batch.
    train_samples = mb.get("train_samples", 15000)
    config.train.train_iters = max(1, train_samples // global_bs)

    # 데이터 = 우리 reasoning HF 데이터셋을 chat-template SFT 로. 다른 프레임워크와 동일하게
    # get_source 로 정규화한 messages 를 넘긴다(megatron-lm 처럼 row-level FORMATS 는 거치지 않음).
    to_messages = get_source(ds_cfg["source"])

    def keep_messages(example, tokenizer=None):  # noqa: ANN001 - bridge 콜백 시그니처
        del tokenizer
        return {"messages": to_messages(example)}

    config.dataset = HFDatasetConfig(
        dataset_name=ds_cfg["hf_path"],
        dataset_subset=ds_cfg.get("hf_name"),
        process_example_fn=keep_messages,
        split=ds_cfg.get("split", "train"),
        seq_length=config.model.seq_length,
        dataloader_type="batch",
        do_validation=False,
        do_test=False,
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "pad_to_max_length": True,
        },
        packed_sequence_specs=None,
        rewrite=False,
    )

    # 로컬/스모크: max_steps>0 이면 그만큼만.
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        config.train.train_iters = max_steps

    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="megatron-bridge-entry")
    parser.add_argument("--stage", choices=["convert", "finetune"], required=True)
    parser.add_argument("--config", required=True, help="병합된 RunConfig YAML")
    parser.add_argument("--megatron-path", required=True, help="mcore 체크포인트 경로")
    parser.add_argument("--tokenizer-dir", default=None, help="캐논 template 을 구운 토크나이저")
    args = parser.parse_args(argv)

    cfg = RunConfig.from_file(args.config)

    if args.stage == "convert":
        _convert(cfg, args.megatron_path)
        return

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    config = _build_finetune_config(cfg, args.megatron_path, args.tokenizer_dir)
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
