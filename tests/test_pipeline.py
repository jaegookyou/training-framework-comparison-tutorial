"""수직 파이프라인 러너(Phase B) 테스트 — 경로 threading·산출 경로 resolver·full 가드.

GPU 없이 검증 가능한 범위 = plan_pipeline 의 순수 로직(단계 config 빌드 + model.name 이어주기 +
output.local_dir 분리). 실제 단계 실행(run_pipeline→dispatch→trainer.train)은 GPU 의존이라 제외.
"""

from pathlib import Path

import pytest
import yaml

from training_framework_comparison_tutorial.pipeline import plan_pipeline

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
SPEC = ROOT / "pipelines" / "qwen3-8b.yaml"

PRETRAIN_8B = "configs/pretrain/qwen3-8b_wikitext__torchtitan.yaml"
SFT_TRL_FULL = "configs/sft/qwen3-8b_traceinversion__trl__full.yaml"
SFT_TRL_LORA = "configs/sft/qwen3-8b_traceinversion__trl__lora.yaml"
SFT_VERL_FULL = "configs/sft/qwen3-8b_traceinversion__verl__full.yaml"
GRPO_TRL_FULL = "configs/grpo/qwen3-8b_gsm8k__trl__full.yaml"


def test_example_pipeline_threads_model_and_dirs():
    spec = yaml.safe_load(SPEC.read_text())
    planned = plan_pipeline(spec)
    assert [c.method for c in planned] == ["pretrain", "sft", "grpo"]

    ws = Path(spec["workspace"])
    s0, s1, s2 = planned

    # 단계별 분리 출력 디렉토리
    assert s0.section("output")["local_dir"] == str(ws / "stage0_pretrain_torchtitan")
    assert s1.section("output")["local_dir"] == str(ws / "stage1_sft_trl")
    assert s2.section("output")["local_dir"] == str(ws / "stage2_grpo_trl")

    # 첫 단계는 자기 model(size/init_from) 유지 — 러너가 model.name 안 박음
    assert s0.section("model")["size"] == "8b"
    assert "name" not in s0.section("model")

    # SFT 입력 = pretrain 산출(out/hf). GRPO 입력 = SFT 산출(trl save_model = local_dir 통째)
    assert s1.section("model")["name"] == str(ws / "stage0_pretrain_torchtitan" / "hf")
    assert s2.section("model")["name"] == str(ws / "stage1_sft_trl")

    # 이어받은 단계도 _base 의 다른 model 키(max_seq_len 등)는 보존(부분 override)
    assert "max_seq_len" in s1.section("model")


def test_plan_does_not_mutate_source_configs():
    # _override 가 frozen RunConfig 를 새로 만들어 원본 로드본을 안 건드리는지(독립 단계 실행 보호)
    spec = yaml.safe_load(SPEC.read_text())
    plan_pipeline(spec)
    from training_framework_comparison_tutorial.config import RunConfig

    fresh = RunConfig.from_file(ROOT / SFT_TRL_FULL)
    assert fresh.section("model")["name"] == "Qwen/Qwen3-8B-Base"  # 파이프라인이 안 바꿈
    assert fresh.section("output")["local_dir"] == "/workspace/out"


def test_consumed_lora_stage_rejected():
    # 소비되는(마지막 아닌) 단계가 lora 면 거부 — 어댑터를 model.name 으로 핸드오프 불가
    spec = {"workspace": "/tmp/p", "stages": [PRETRAIN_8B, SFT_TRL_LORA, GRPO_TRL_FULL]}
    with pytest.raises(ValueError, match="full 이어야"):
        plan_pipeline(spec)


def test_unsupported_output_framework_errors():
    # 산출 HF 경로 규칙이 없는 프레임워크(verl SFT)가 소비 단계면 명시적 에러(조용한 오연결 방지)
    spec = {"workspace": "/tmp/p", "stages": [PRETRAIN_8B, SFT_VERL_FULL, GRPO_TRL_FULL]}
    with pytest.raises(ValueError, match="산출 경로 규칙 미정의"):
        plan_pipeline(spec)


def test_empty_stages_rejected():
    with pytest.raises(ValueError, match="stages"):
        plan_pipeline({"stages": []})
