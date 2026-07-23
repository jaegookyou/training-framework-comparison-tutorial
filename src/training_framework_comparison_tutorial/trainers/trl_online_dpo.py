"""TRL Online DPO 학습 경로 (online preference, on-policy).

offline DPO(trl_dpo)와 **같은 DPO loss** 지만 선호쌍을 데이터에서 받지 않고 학습 중 생성한다:
정책이 프롬프트당 응답을 생성 → reward model 이 채점 → 높은 쪽=chosen / 낮은 쪽=rejected 로
그 자리에서 쌍을 구성해 DPO. 그래서 데이터 = **prompt-only**(trl-lib/ultrafeedback-prompt) +
**reward model**(offline 의 고정 선호쌍 대신). offline↔online DPO 비교 = 같은 ultrafeedback
도메인, "쌍을 미리 굽냐 / 즉석에 만드냐"만 차이(OAIF 셋업) = 통제비교.

OnlineDPO 는 trl 1.6 에서 trl.experimental.online_dpo 에 있다. reward_funcs 로 RM 을,
reward_processing_classes 로 RM 토크나이저를 넘긴다. ref 모델은 TRL 자동(full=복제/lora=어댑터 끔).
reward 는 통제 변수가 아니라 **도메인 채점기** — gsm8k rule reward 와 달리 ultrafeedback 은 열린
채팅이라 rule 채점 불가 → 커뮤니티 검증 RM(config 의 reward.model)으로 채점.

Unsloth 는 online DPO 네이티브 경로가 없어(PatchOnlineDPO 부재) 이 트랙은 TRL 단독이다
(offline DPO 의 Unsloth 와 비대칭 — repo 원칙: 네이티브 경로만).

torch/trl/transformers/datasets 는 docker/trl.Dockerfile(trl 1.6) 안에만 → 지연 임포트.

⚠️ GPU 검증 대기: 정책+ref+RM 3개 모델 메모리(full 이면 8B×3 → 멀티GPU 필수)·생성 길이/EOS·
RM 토크나이저 정합·OnlineDPOConfig max_length>max_new_tokens 제약.
"""

from __future__ import annotations

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig
from .trl_sft import _lora_config, apply_multigpu_fsdp  # TRL-family 공유 헬퍼


def train(cfg: RunConfig) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

    model_cfg = cfg.section("model")
    ds_cfg = cfg.section("dataset")
    reward_cfg = cfg.section("reward")
    hp = cfg.section("hp")
    out = cfg.section("output")
    debug = cfg.section("debug")

    to_prompt = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: to_format(to_prompt(row)),
        remove_columns=raw.column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # base 모델 → 캐논 ChatML template 으로 prompt 렌더(offline DPO·SFT 와 동일 = 통제 변수).
    # conversational dataset 이면 OnlineDPOTrainer 가 이 template 을 자동 적용한다.
    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        tokenizer.chat_template = chat_template

    # reward model = 도메인 채점기(커뮤니티 검증 RM). num_labels=1 의 SequenceClassification.
    reward_name = reward_cfg["model"]
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name, num_labels=1)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_name, truncation_side="left")
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    peft_config = _lora_config(cfg) if cfg.tuning == "lora" else None

    # max_length = prompt+생성 총길이(제약: > max_new_tokens). max_new_tokens = 생성 길이.
    max_new_tokens = hp.get("max_completion_length", 512)
    max_length = model_cfg.get("max_seq_len", 2048)

    # NOTE: TRL 은 API churn 이 잦다. 정확한 인자 호환은 docker/trl.Dockerfile(trl 1.6) 핀 기준.
    args = OnlineDPOConfig(
        output_dir=out.get("local_dir", "out"),
        per_device_train_batch_size=hp["per_device_batch_size"],
        gradient_accumulation_steps=hp.get("gradient_accumulation", 1),
        learning_rate=float(hp["learning_rate"]),
        num_train_epochs=hp.get("epochs", 1),
        warmup_ratio=hp.get("warmup_ratio", 0.0),
        lr_scheduler_type=hp.get("lr_scheduler", "linear"),
        bf16=hp.get("bf16", False),
        beta=hp.get("beta", 0.1),                       # DPO KL 정규화 강도
        max_new_tokens=max_new_tokens,                  # rollout 생성 길이
        max_length=max_length,                          # prompt+생성 총상한
        temperature=hp.get("temperature", 0.9),         # rollout 샘플링 온도
        missing_eos_penalty=hp.get("missing_eos_penalty", 1.0),  # EOS 누락 페널티
        max_steps=debug.get("max_steps", -1),
        report_to="wandb",
        run_name=cfg.run_name(),
        push_to_hub=bool(out.get("hf_repo")),
        hub_model_id=out.get("hf_repo"),
    )

    # 멀티노드/멀티GPU(torchrun) 런치면 full FT 에 FSDP 샤딩(단일 프로세스면 no-op).
    # ⚠️ online DPO = 정책+ref+RM 3모델 + 루프 내 생성 → full 은 사실상 MN 필수(8B×3). GPU 검증 대기.
    apply_multigpu_fsdp(args, cfg.tuning)

    trainer = OnlineDPOTrainer(
        model=model_cfg["name"],
        reward_funcs=reward_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_processing_classes=reward_tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if out.get("hf_repo"):
        trainer.push_to_hub()
