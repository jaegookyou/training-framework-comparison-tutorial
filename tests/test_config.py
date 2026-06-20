from pathlib import Path

from training_framework_comparison_tutorial.config import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__trl__full.yaml"
LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__trl__lora.yaml"
UNSLOTH = CONFIGS / "sft" / "qwen3-8b_traceinversion__unsloth__lora.yaml"
UNSLOTH_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__unsloth__full.yaml"
VERL_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__verl__full.yaml"
VERL_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__verl__lora.yaml"
MEGATRON_LM = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-lm__full.yaml"
MEGATRON_BRIDGE_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-bridge__full.yaml"
MEGATRON_BRIDGE_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-bridge__lora.yaml"
TORCHTITAN = CONFIGS / "sft" / "qwen3-8b_traceinversion__torchtitan__full.yaml"
PRETRAIN = CONFIGS / "pretrain" / "qwen3-tiny_wikitext__torchtitan.yaml"
DPO_FULL = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__trl__full.yaml"
DPO_LORA = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__trl__lora.yaml"
GRPO_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__trl__full.yaml"
GRPO_LORA = CONFIGS / "grpo" / "qwen3-8b_gsm8k__trl__lora.yaml"
DPO_UNSLOTH_FULL = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__unsloth__full.yaml"
DPO_UNSLOTH_LORA = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__unsloth__lora.yaml"
GRPO_UNSLOTH_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__unsloth__full.yaml"
GRPO_UNSLOTH_LORA = CONFIGS / "grpo" / "qwen3-8b_gsm8k__unsloth__lora.yaml"


def test_extends_inherits_base():
    cfg = RunConfig.from_file(FULL)
    assert cfg.framework == "trl"
    assert cfg.method == "sft"
    # _base.yaml 에서 상속된 공통 축
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"


def test_run_overrides_win():
    cfg = RunConfig.from_file(LORA)
    # run 파일에만 있는 값
    assert cfg.image.endswith("/trl:latest")
    assert cfg.tuning == "lora"
    assert cfg.section("scale")["gpus"] == 1
    # lora run 이 _base 의 lr 을 override
    assert float(cfg.section("hp")["learning_rate"]) == 2.0e-4


def test_tuning_axis_defaults_and_overrides():
    assert RunConfig.from_file(FULL).tuning == "full"
    assert RunConfig.from_file(LORA).tuning == "lora"


def test_run_name_is_descriptive():
    cfg = RunConfig.from_file(LORA)
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-trl-lora"


def test_unsloth_config_is_single_gpu_lora():
    cfg = RunConfig.from_file(UNSLOTH)
    assert cfg.framework == "unsloth"
    assert cfg.tuning == "lora"
    assert cfg.section("scale")["gpus"] == 1
    assert cfg.image.endswith("/unsloth:latest")
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-unsloth-lora"


def test_unsloth_full_config_is_single_gpu():
    cfg = RunConfig.from_file(UNSLOTH_FULL)
    assert cfg.framework == "unsloth"
    assert cfg.tuning == "full"  # Unsloth 단일 GPU full FT (full_finetuning=True)
    assert cfg.section("scale")["gpus"] == 1  # full 도 단일 GPU
    assert cfg.image.endswith("/unsloth:latest")
    # full 은 _base 의 full lr 을 그대로 쓴다(lora 만 2e-4 override)
    assert float(cfg.section("hp")["learning_rate"]) == 2.0e-5
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-unsloth-full"


def test_verl_full_and_lora_configs():
    full = RunConfig.from_file(VERL_FULL)
    assert full.framework == "verl"
    assert full.tuning == "full"
    assert full.image.endswith("/verl:latest")
    assert full.run_name() == "sft-Qwen3-8B-Base-traceinversion-verl-full"

    lora = RunConfig.from_file(VERL_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base 의 lr 을 override (trl/unsloth lora 와 동일 눈금)
    assert float(lora.section("hp")["learning_rate"]) == 2.0e-4


def test_megatron_lm_config_is_full_only_with_megatron_section():
    cfg = RunConfig.from_file(MEGATRON_LM)
    assert cfg.framework == "megatron-lm"
    assert cfg.tuning == "full"  # 순수 Megatron-LM SFT 는 full 전용(LoRA 없음)
    assert cfg.image.endswith("/megatron-lm:latest")
    # _base 공통 축 상속 (모델/데이터/템플릿 = 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"
    # Megatron 고유 knob
    mg = cfg.section("megatron")
    assert mg["model_cfg"] == "Qwen/Qwen3-8B"
    assert mg["tensor_model_parallel_size"] == cfg.section("scale")["gpus"]
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-megatron-lm-full"


def test_megatron_bridge_full_and_lora_configs():
    full = RunConfig.from_file(MEGATRON_BRIDGE_FULL)
    assert full.framework == "megatron-bridge"
    assert full.tuning == "full"
    assert full.image.endswith("/megatron-bridge:latest")
    # _base 공통 축 상속 (모델/데이터/템플릿 = 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "traceinversion"
    # Megatron 고유 knob (NVIDIA qwen3-8b SFT 권장 TP=4)
    assert full.section("megatron")["tensor_model_parallel_size"] == 4
    assert full.run_name() == "sft-Qwen3-8B-Base-traceinversion-megatron-bridge-full"

    lora = RunConfig.from_file(MEGATRON_BRIDGE_LORA)
    assert lora.tuning == "lora"  # 순수 Megatron-LM 과 달리 Bridge 는 네이티브 PEFT 지원
    assert lora.section("scale")["gpus"] == 1
    assert lora.section("megatron")["tensor_model_parallel_size"] == 1
    # lora run 이 _base 의 lr 을 override (trl/unsloth/verl lora 와 동일 눈금)
    assert float(lora.section("hp")["learning_rate"]) == 2.0e-4


def test_torchtitan_config_is_full_only():
    cfg = RunConfig.from_file(TORCHTITAN)
    assert cfg.framework == "torchtitan"
    assert cfg.tuning == "full"  # torchtitan SFT 는 full 전용(네이티브 LoRA 없음)
    assert cfg.image.endswith("/torchtitan:latest")
    # _base 공통 축 상속 (모델/데이터/템플릿 = 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"
    # torchtitan 고유 knob (step 환산용 train_samples)
    assert cfg.section("torchtitan")["train_samples"] == 15000
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-torchtitan-full"


def test_pretrain_config_torchtitan():
    cfg = RunConfig.from_file(PRETRAIN)
    assert cfg.method == "pretrain"
    assert cfg.framework == "torchtitan"
    # from-scratch 라 model.name 이 아니라 model.size (arch preset)
    assert cfg.section("model")["size"] == "tiny"
    assert cfg.section("model")["tokenizer"] == "Qwen/Qwen3-8B-Base"  # SFT/RL 과 동일 토크나이저
    assert cfg.section("dataset")["source"] == "wikitext"
    # 사전학습 run_name = method-size-ds-framework (tuning 축 없음)
    assert cfg.run_name() == "pretrain-tiny-wikitext-torchtitan"


def test_model_size_presets_map_to_torchtitan():
    from training_framework_comparison_tutorial.model_sizes import (
        torchtitan_config_fn,
        torchtitan_flavor,
    )

    assert torchtitan_flavor("tiny") == "tfct_tiny"
    assert torchtitan_config_fn("tiny") == "pretrain_qwen3_tiny_wikitext"


def test_dpo_full_and_lora_configs():
    full = RunConfig.from_file(DPO_FULL)
    assert full.framework == "trl"
    assert full.method == "dpo"
    assert full.tuning == "full"
    assert full.image.endswith("/trl:latest")
    # _base 공통 축 상속 (모델 = SFT 와 동일, 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "ultrafeedback"
    assert full.section("hp")["beta"] == 0.1  # DPO KL 강도
    assert full.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-trl-full"

    lora = RunConfig.from_file(DPO_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base 의 DPO lr 을 override (full 5e-7 → lora 5e-6)
    assert float(lora.section("hp")["learning_rate"]) == 5.0e-6
    assert lora.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-trl-lora"


def test_grpo_full_and_lora_configs():
    full = RunConfig.from_file(GRPO_FULL)
    assert full.framework == "trl"
    assert full.method == "grpo"
    assert full.tuning == "full"
    assert full.image.endswith("/trl:latest")
    # _base 공통 축 상속 (모델 = SFT/DPO 와 동일, 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "gsm8k"
    # GRPO 고유 knob: reward 세트 + 그룹 크기
    assert full.section("reward")["name"] == "gsm8k"
    assert full.section("hp")["num_generations"] == 8
    assert full.run_name() == "grpo-Qwen3-8B-Base-gsm8k-trl-full"

    lora = RunConfig.from_file(GRPO_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base 의 GRPO lr 을 override (full 1e-6 → lora 1e-5)
    assert float(lora.section("hp")["learning_rate"]) == 1.0e-5
    assert lora.run_name() == "grpo-Qwen3-8B-Base-gsm8k-trl-lora"


def test_dpo_unsloth_full_and_lora_configs():
    full = RunConfig.from_file(DPO_UNSLOTH_FULL)
    assert full.framework == "unsloth"
    assert full.method == "dpo"
    assert full.tuning == "full"
    assert full.section("scale")["gpus"] == 1  # Unsloth 단일 GPU 전용
    assert full.image.endswith("/unsloth:latest")
    # full 은 _base 의 DPO full lr(5e-7) 을 그대로
    assert float(full.section("hp")["learning_rate"]) == 5.0e-7
    assert full.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-unsloth-full"

    lora = RunConfig.from_file(DPO_UNSLOTH_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    assert float(lora.section("hp")["learning_rate"]) == 5.0e-6


def test_grpo_unsloth_full_and_lora_configs():
    full = RunConfig.from_file(GRPO_UNSLOTH_FULL)
    assert full.framework == "unsloth"
    assert full.method == "grpo"
    assert full.tuning == "full"
    assert full.section("scale")["gpus"] == 1  # Unsloth 단일 GPU 전용
    assert full.section("reward")["name"] == "gsm8k"
    assert full.run_name() == "grpo-Qwen3-8B-Base-gsm8k-unsloth-full"

    lora = RunConfig.from_file(GRPO_UNSLOTH_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    assert float(lora.section("hp")["learning_rate"]) == 1.0e-5


def test_dispatch_namespaced_by_method():
    from training_framework_comparison_tutorial.run import TRAINERS

    # method 축으로 네임스페이스 — 같은 framework(torchtitan)가 pretrain·sft 양쪽에
    assert "torchtitan" in TRAINERS["pretrain"]
    assert "torchtitan" in TRAINERS["sft"]
    assert TRAINERS["pretrain"]["torchtitan"].endswith("torchtitan_pretrain")
    assert TRAINERS["sft"]["trl"].endswith("trl_sft")
    # RL 트랙: DPO·GRPO 는 별 method (TRL 기준점 + Unsloth 단일 GPU)
    assert TRAINERS["dpo"]["trl"].endswith("trl_dpo")
    assert TRAINERS["grpo"]["trl"].endswith("trl_grpo")
    assert TRAINERS["dpo"]["unsloth"].endswith("unsloth_dpo")
    assert TRAINERS["grpo"]["unsloth"].endswith("unsloth_grpo")
