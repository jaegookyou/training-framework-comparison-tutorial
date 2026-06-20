"""Unsloth DPO 학습 경로 (full|lora·단일 GPU).

DPO 는 offline preference (generation 없음) → SFT 와 거의 같은 메모리/실행 모양이라 unsloth_sft
패턴을 그대로 따른다. 차이는 데이터 정규형(선호쌍)·DPOTrainer 뿐. lora 면 ref=어댑터 끈 base
(추가 메모리 0), full 이면 ref 별도 복제(2x VRAM → 단일 GPU 라도 큰 VRAM, sky 참고).

unsloth 는 transformers/trl 보다 먼저 임포트돼야 패치가 걸린다. deps 는 docker/unsloth.Dockerfile
(trl<=0.24 핀)에만 → 지연 임포트. trl 0.24 API 라 trl_dpo.py(trl 1.x)와 인자가 다르다.
"""

from __future__ import annotations

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig

# unsloth get_peft_model 은 "all-linear" 미지원 → 표준 proj 집합으로 번역(unsloth_sft 와 동일).
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

    # FastLanguageModel 먼저(패치). PatchDPOTrainer 는 unsloth DPO 의 문서화된 진입 패턴.
    from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: I001

    PatchDPOTrainer()
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")
    lora = cfg.section("lora")

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

    full = cfg.tuning == "full"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_len", 2048),
        dtype=None,            # Ampere+ 에서 bf16 자동
        load_in_4bit=False,
        full_finetuning=full,
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

    # base 모델 → 캐논 ChatML template (trl 경로와 동일 포맷 = 통제비교).
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    # NOTE: trl 0.24 API. 정확한 인자 호환은 docker/unsloth.Dockerfile 의 핀 기준.
    args = DPOConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        beta=hp.get("beta", 0.1),
        max_length=model_cfg.get("max_seq_len", 2048),
        max_prompt_length=hp.get("max_prompt_length", 1024),
        # full 은 unsloth gradient ckpt(get_peft_model)를 못 받으므로 여기서 켠다(메모리 절약).
        gradient_checkpointing=full,
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if out.get("hf_repo"):
        trainer.push_to_hub()
