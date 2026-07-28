"""TRL SFT 학습 경로.

torch/trl/transformers/datasets 는 docker/trl.Dockerfile 안에만 있다 → 함수 안에서 지연 임포트.
패키지 임포트만으로 무거운 deps 가 끌려오지 않게 한다(CI 는 .[dev] 만 설치).
"""

from __future__ import annotations

import os

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig


def multigpu_fsdp_kwargs(tuning: str) -> dict:
    """torchrun 멀티프로세스 런치일 때 full FT 에 줄 FSDP 인자(TRL-family 공유).

    HF Trainer 는 torchrun env(WORLD_SIZE>1)를 자동 감지해 분산 학습한다. 단 4B full 은 DDP(모델
    복제)로는 GPU 당 OOM → **FSDP 샤딩**이 필수다. LoRA 는 base(4B bf16 ≈ 8GB)가 GPU 당 들어가
    DDP 로 충분하므로 건드리지 않는다. 단일 프로세스(WORLD_SIZE=1)면 FSDP 불필요 → 빈 dict.

    **반드시 생성자 인자로 넘긴다**(예전엔 만들어진 args 에 `args.fsdp=...` 로 꽂았다). 이유는
    2026-07-28 첫 2노드 런에서 잡은 실패: FSDP 정규화는 `TrainingArguments._process_fsdp_args()`
    가 `__post_init__` 에서 하는데, 생성 후 속성 대입은 그걸 **건너뛴다** → `fsdp_config` 가 None 인
    채로 남아 `trainer.py:452` 의 `args.fsdp_config.get("xla", …)` 가 AttributeError 로 죽었다.
    사후 대입은 검증·정규화를 통째로 우회한다는 일반 교훈.

    값의 근거(transformers **5.12.0** 실물 — 이미지 핀과 동일 버전):
      - 문자열 `fsdp="full_shard auto_wrap"` 은 **deprecated**(v5.20 제거 예정, training_args.py:
        2811-2815) → `fsdp=True` + `fsdp_config` 가 현재 API.
      - `version` 기본 2(FSDP2). `reshard_after_forward=True` 가 곧 full_shard(2798-2807).
      - `auto_wrap_policy` 기본 = TRANSFORMER_BASED_WRAP 이고, `transformer_layer_cls_to_wrap` 을
        **주지 않으면** 플러그인이 모델의 `_no_split_modules`(Qwen3 는 `Qwen3DecoderLayer`)를
        쓴다(2749-2755) → 레이어 클래스명을 하드코딩하지 않는다(추정 회피).

    **FSDP1(version=1)을 쓰는 이유 = Qwen3-4B 의 tied embeddings**(2026-07-28 2노드 런에서 실측):
    FSDP2 기본으로 돌리면 `fully_shard` 가
      `ValueError: Parameter 'model.embed_tokens.weight' is shared with a parameter already
       managed by another FSDP group.`
    로 죽는다 — 묶인 `embed_tokens`/`lm_head` 가 서로 다른 FSDP 그룹에 들어가는데, FSDP2 는 공유
    파라미터가 한 그룹 안에 있기를 요구한다. transformers 5.12·accelerate 어느 쪽에도 이걸 자동으로
    풀어주는 처리가 없다(양쪽 소스에 tie 관련 FSDP 분기 0건). FSDP1 은 decoder layer 만 감싸고
    embed/lm_head 를 root 유닛에 함께 남기므로 묶인 파라미터가 같은 유닛에 있어 성립한다.
    **모델을 untie 하는 우회는 하지 않는다** — 4B 는 tied 가 정식 arch 이고, 풀면 파라미터 수와
    초기화가 base 체크포인트와 달라져 통제비교의 모델 축이 오염된다(프레임워크만 변수여야 한다).
    ⚠️ FSDP1 은 transformers v5.20 에서 제거 예정 — 우리는 이미지로 5.12 를 박제해 지금은 안전하고,
    올릴 때 재검토할 지점이다(이미지 핀 전략의 전형적 이월 항목).

    ⚠️ 남은 GPU 검증: 캐논 template + FSDP + assistant_only_loss 정합.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or tuning != "full":
        return {}
    return {
        "fsdp": True,
        "fsdp_config": {"version": 1, "reshard_after_forward": "full_shard"},
    }


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
    to_format = get_format(cfg.method, cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: to_format(to_messages(row)),
        remove_columns=raw.column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    # base 모델은 assistant_only_loss 가 요구하는 {% generation %} 마커가 토크나이저 기본
    # template 에 없다 → config 가 지정한 캐논 학습 template 으로 덮어쓴다(통제비교 = 동일 포맷).
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

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
        # HF 기본 500 이면 5 step 스모크는 W&B 에 한 점도 안 찍힌다(빈 차트=판정 불가).
        logging_steps=cfg.log_every_n_steps(),
        # 멀티노드/멀티GPU(torchrun) 런치면 full FT 에 FSDP 샤딩(단일 프로세스·LoRA 면 빈 dict).
        **multigpu_fsdp_kwargs(cfg.tuning),
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
