"""Unsloth GRPO 학습 경로 (full|lora·단일 GPU).

Unsloth GRPO 의 본령 = **fast_inference(vllm 내장)** 로 단일 GPU 에서 rollout 을 빠르게 뽑는 것.
정책 모델과 vllm 엔진이 같은 카드에서 가중치를 공유(gpu_memory_utilization 로 분할)하므로 TRL
GRPO 처럼 별도 vllm 셋업이 없다. lora 면 max_lora_rank 로 어댑터 rank 를 vllm 에 알린다.

unsloth 는 transformers/trl 보다 먼저 임포트. deps 는 docker/unsloth.Dockerfile(trl<=0.24 + vllm)
에만 → 지연 임포트. trl 0.24 API 라 trl_grpo.py(trl 1.x)와 인자가 다르다.
"""

from __future__ import annotations

from ..adapters import get_format, get_reward_funcs, get_source, resolve_chat_template
from ..config import RunConfig

_ALL_LINEAR_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _target_modules(lora: dict) -> list[str]:
    tm = lora.get("target_modules", "all-linear")
    return list(_ALL_LINEAR_MODULES) if tm == "all-linear" else list(tm)


def train(cfg: RunConfig) -> None:
    if cfg.tuning not in ("full", "lora"):
        raise SystemExit(f"unsloth: tuning 은 full|lora 만 지원(받음: {cfg.tuning!r}).")

    from unsloth import FastLanguageModel  # noqa: I001  # transformers/trl 보다 먼저
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")
    lora = cfg.section("lora")

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

    full = cfg.tuning == "full"
    use_vllm = hp.get("use_vllm", True)  # unsloth 본령 = vllm rollout (기본 켬)

    # lora 면 vllm 이 어댑터 rank 를 알아야 한다 → max_lora_rank. full 은 어댑터 없음.
    load_kwargs = {} if full else {"max_lora_rank": lora.get("r", 16)}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_len", 2048),
        dtype=None,
        load_in_4bit=False,
        full_finetuning=full,
        fast_inference=use_vllm,                       # vllm 엔진 내장
        gpu_memory_utilization=hp.get("gpu_memory_utilization", 0.6),  # 정책/vllm 가 카드 공유
        **load_kwargs,
    )
    if not full:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.0),
            target_modules=_target_modules(lora),
            use_gradient_checkpointing="unsloth",
            random_state=ds_cfg.get("seed", 42),
        )

    # base 모델 → 캐논 ChatML template (add_generation_prompt 로 rollout 프롬프트 큐 정합).
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    # NOTE: trl 0.24 API. 정확한 인자 호환은 docker/unsloth.Dockerfile 의 핀 기준.
    args = GRPOConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        beta=hp.get("beta", 0.04),
        num_generations=hp.get("num_generations", 8),
        max_prompt_length=hp.get("max_prompt_length", 512),
        max_completion_length=hp.get("max_completion_length", 1024),
        temperature=hp.get("temperature", 1.0),
        use_vllm=use_vllm,
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if out.get("hf_repo"):
        trainer.push_to_hub()
