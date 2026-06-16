"""Unsloth SFT 학습 경로 (LoRA 전용·단일 GPU).

Unsloth 는 단일 GPU LoRA/QLoRA 에 특화 → 통제비교에서 tuning=lora·1 GPU 슬롯만
담당한다(full FT·멀티GPU 는 안 함, 매트릭스 결정).

unsloth 는 transformers/trl 보다 먼저 임포트돼야 자기 최적화 패치가 걸린다 →
함수 안에서, 다른 무거운 deps 보다 위에서 임포트한다. deps 는 docker/unsloth.Dockerfile
에만 있고(torch<2.11·trl<=0.24 라 trl 이미지와 핀 충돌 → 별도 이미지), 패키지 임포트만으로
끌려오지 않게 지연 임포트한다.
"""

from __future__ import annotations

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig

# Unsloth get_peft_model 은 명시 모듈 리스트를 원한다("all-linear" 미지원).
# config 가 "all-linear" 면 표준 트랜스포머 proj 집합으로 번역한다.
_ALL_LINEAR_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _target_modules(lora: dict) -> list[str]:
    tm = lora.get("target_modules", "all-linear")
    return list(_ALL_LINEAR_MODULES) if tm == "all-linear" else list(tm)


def train(cfg: RunConfig) -> None:
    if cfg.tuning != "lora":
        raise SystemExit(
            f"unsloth 는 LoRA 전용이다(단일 GPU). tuning={cfg.tuning!r} 미지원 "
            "— full FT 는 trl/megatron 경로를 써라."
        )

    from unsloth import FastLanguageModel  # noqa: I001  # transformers/trl 보다 먼저
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")
    lora = cfg.section("lora")

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

    # bf16 LoRA (QLoRA 아님 — Qwen3.5 는 4bit 양자화 오차 커서 Unsloth 도 비권장).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_len", 2048),
        dtype=None,            # Ampere+ 에서 bf16 자동
        load_in_4bit=False,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=_target_modules(lora),
        use_gradient_checkpointing="unsloth",
        random_state=ds_cfg.get("seed", 42),
    )

    # base 모델엔 {% generation %} 마커가 없다 → trl 경로와 동일한 캐논 학습 template 으로
    # 덮어쓴다(두 프레임워크 동일 포맷 = 통제비교). assistant_only_loss 가 이 마커를 쓴다.
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    # NOTE: unsloth 는 trl<=0.24 핀(docker/unsloth.Dockerfile) → trl 1.x(trl_sft.py)와
    # API 가 다르다. 여기선 trl 0.x 의 SFTConfig 인자(max_seq_length 등)를 쓴다.
    args = SFTConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        max_seq_length=model_cfg.get("max_seq_len", 2048),
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if out.get("hf_repo"):
        trainer.push_to_hub()
