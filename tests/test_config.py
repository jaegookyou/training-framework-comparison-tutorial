from pathlib import Path

from training_framework_comparison_tutorial.config import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__trl__full.yaml"
LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__trl__lora.yaml"
UNSLOTH = CONFIGS / "sft" / "qwen3-8b_traceinversion__unsloth__lora.yaml"
VERL_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__verl__full.yaml"
VERL_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__verl__lora.yaml"
MEGATRON_LM = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-lm__full.yaml"


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
