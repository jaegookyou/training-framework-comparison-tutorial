"""TRL SFT 학습 경로.

torch/trl/transformers/datasets 는 docker/trl.Dockerfile 안에만 있다 → 함수 안에서 지연 임포트.
패키지 임포트만으로 무거운 deps 가 끌려오지 않게 한다(CI 는 .[dev] 만 설치).
"""

from __future__ import annotations

from ..adapters import get_format, get_source
from ..config import RunConfig


def _lora_config(cfg: RunConfig):
    """config 의 lora 블록 → peft LoraConfig. tuning=lora 일 때만 호출."""
    from peft import LoraConfig

    lora = cfg.section("lora")
    return LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def train(cfg: RunConfig) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")

    to_messages = get_source(ds_cfg["source"])
    to_format = get_format(cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: to_format(to_messages(row)),
        remove_columns=raw.column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    # tuning=lora 면 peft LoraConfig 를 SFTTrainer 에 넘긴다. full 이면 None(전체 파라미터).
    peft_config = _lora_config(cfg) if cfg.tuning == "lora" else None

    # NOTE: TRL 은 API churn 이 잦다. 정확한 인자 호환은 docker/trl.Dockerfile 의 핀 기준.
    args = SFTConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        max_length=model_cfg.get("max_seq_len", 2048),
        assistant_only_loss=hp.get("assistant_only_loss", False),
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    trainer = SFTTrainer(
        model=model_cfg["name"],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if out.get("hf_repo"):
        trainer.push_to_hub()
