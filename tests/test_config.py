from pathlib import Path

from training_framework_comparison_tutorial.config import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FULL = CONFIGS / "sft" / "qwen3.5-9b_traceinversion__trl__full.yaml"
LORA = CONFIGS / "sft" / "qwen3.5-9b_traceinversion__trl__lora.yaml"
UNSLOTH = CONFIGS / "sft" / "qwen3.5-9b_traceinversion__unsloth__lora.yaml"


def test_extends_inherits_base():
    cfg = RunConfig.from_file(FULL)
    assert cfg.framework == "trl"
    assert cfg.method == "sft"
    # _base.yaml 에서 상속된 공통 축
    assert cfg.section("model")["name"] == "Qwen/Qwen3.5-9B-Base"
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
    assert cfg.run_name() == "sft-Qwen3.5-9B-Base-traceinversion-trl-lora"


def test_unsloth_config_is_single_gpu_lora():
    cfg = RunConfig.from_file(UNSLOTH)
    assert cfg.framework == "unsloth"
    assert cfg.tuning == "lora"
    assert cfg.section("scale")["gpus"] == 1
    assert cfg.image.endswith("/unsloth:latest")
    assert cfg.run_name() == "sft-Qwen3.5-9B-Base-traceinversion-unsloth-lora"
