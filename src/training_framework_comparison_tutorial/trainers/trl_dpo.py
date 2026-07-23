"""TRL DPO 학습 경로 (offline preference optimization).

DPO 는 SFT 와 가깝다 — generation 없이 chosen/rejected 선호쌍의 logprob 만 비교한다.
그래서 trl_sft 의 패턴(지연 임포트·캐논 template·peft)을 거의 그대로 따르고, 데이터 정규형
(선호쌍)과 trainer(DPOTrainer/DPOConfig)만 다르다. ref 모델은 TRL 이 자동 처리
(full=내부 복제 / lora=어댑터 끄고 base 재사용).

torch/trl/transformers/datasets 는 docker/trl.Dockerfile 안에만 있다 → 함수 안에서 지연 임포트.
"""

from __future__ import annotations

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig
from .trl_sft import _lora_config, apply_multigpu_fsdp  # TRL-family 공유 헬퍼


def train(cfg: RunConfig) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")

    to_pref = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: to_format(to_pref(row)),
        remove_columns=raw.column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    # base 모델 → 기본 template 에 {% generation %} 없음. DPO 는 그 마스크를 안 쓰지만,
    # 통제비교를 위해 SFT 와 동일한 캐논 ChatML template 으로 chosen/rejected 를 렌더한다.
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    peft_config = _lora_config(cfg) if cfg.tuning == "lora" else None

    # NOTE: TRL 은 API churn 이 잦다. 정확한 인자 호환은 docker/trl.Dockerfile 의 핀 기준.
    args = DPOConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        beta=hp.get("beta", 0.1),  # DPO KL 정규화 강도
        max_length=model_cfg.get("max_seq_len", 2048),
        max_prompt_length=hp.get("max_prompt_length", 1024),
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    # 멀티노드/멀티GPU(torchrun) 런치면 full FT 에 FSDP 샤딩(단일 프로세스면 no-op).
    apply_multigpu_fsdp(args, cfg.tuning)

    trainer = DPOTrainer(
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
