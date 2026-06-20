"""TRL GRPO 학습 경로 (online on-policy RL).

DPO 와 패러다임이 다르다 — 데이터엔 정답만 있고, 학습 중 모델이 그룹 단위로 응답을 **생성**한 뒤
reward(adapters.rewards)로 채점해 group-normalized advantage 로 정책을 민다. 그래서 추가로
reward_funcs 가 필요하고, rollout 생성 비용이 크다.

generation 가속: GRPOConfig.use_vllm=True 면 vllm 로 빠르게 rollout 한다(현실적 8B GRPO 엔 사실상
필수). 다만 현재 trl 이미지엔 vllm 핀이 없다 → 실사용 전 docker/trl.Dockerfile 에 vllm 추가 필요.
배선/스모크는 use_vllm=false(transformers generate)로 돈다.

torch/trl/transformers/datasets 는 docker/trl.Dockerfile 안에만 있다 → 함수 안에서 지연 임포트.
"""

from __future__ import annotations

from ..adapters import get_format, get_reward_funcs, get_source, resolve_chat_template
from ..config import RunConfig
from .trl_sft import _lora_config  # TRL-family 공유 헬퍼(config.lora → peft LoraConfig)


def train(cfg: RunConfig) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")

    to_prompt = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)
    reward_funcs = get_reward_funcs(cfg.section("reward")["name"])

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: to_format(to_prompt(row)),
        remove_columns=raw.column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    # base 모델 → 캐논 ChatML template (add_generation_prompt 로 rollout 프롬프트 큐 정합).
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    peft_config = _lora_config(cfg) if cfg.tuning == "lora" else None

    # NOTE: TRL 은 API churn 이 잦다. 정확한 인자 호환은 docker/trl.Dockerfile 의 핀 기준.
    args = GRPOConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        beta=hp.get("beta", 0.04),                # KL 계수 (DeepSeekMath 0.04)
        num_generations=hp.get("num_generations", 8),  # 그룹 크기 G (advantage 정규화 단위)
        max_prompt_length=hp.get("max_prompt_length", 512),
        max_completion_length=hp.get("max_completion_length", 1024),
        temperature=hp.get("temperature", 1.0),
        use_vllm=hp.get("use_vllm", False),       # true = vllm rollout (이미지에 vllm 필요)
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    trainer = GRPOTrainer(
        model=model_cfg["name"],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if out.get("hf_repo"):
        trainer.push_to_hub()
