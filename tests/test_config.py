from pathlib import Path

from training_framework_comparison_tutorial.config import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_extends_inherits_base():
    cfg = RunConfig.from_file(CONFIGS / "sft" / "qwen0.5b_smoltalk__trl__1gpu.yaml")
    assert cfg.framework == "trl"
    assert cfg.method == "sft"
    # _base.yaml 에서 상속된 공통 축
    assert cfg.section("model")["name"] == "Qwen/Qwen2.5-0.5B"
    assert cfg.section("hp")["learning_rate"] == 2.0e-5


def test_run_overrides_win():
    cfg = RunConfig.from_file(CONFIGS / "sft" / "qwen0.5b_smoltalk__trl__1gpu.yaml")
    # run 파일에만 있는 값
    assert cfg.image.endswith("tfct-trl:latest")
    assert cfg.section("scale")["gpus"] == 1


def test_run_name_is_descriptive():
    cfg = RunConfig.from_file(CONFIGS / "sft" / "qwen0.5b_smoltalk__trl__1gpu.yaml")
    assert cfg.run_name() == "sft-Qwen2.5-0.5B-smoltalk-trl"
