"""W&B 식별자 주입(_wandb) 테스트.

여기서 지키는 것 = **런을 나중에 알아볼 수 있는가**. 배선 검증 런은 결과가 W&B 화면에만 남는데
프로젝트가 새거나(huggingface 로 감)·이름이 겹치거나·스칼라가 0 점이면 태운 돈이 증발한다.
"""

from pathlib import Path

from training_framework_comparison_tutorial.config import RunConfig
from training_framework_comparison_tutorial.trainers import _wandb
from training_framework_comparison_tutorial.trainers._dist import (
    MULTINODE_MECHANISM,
    MULTINODE_WIRED,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SFT_TRL_LORA = CONFIGS / "sft" / "qwen3-4b_traceinversion__trl__lora.yaml"
SMOKE_TRL = CONFIGS / "sft" / "_smoke_trl_gpu.yaml"
SMOKE_TT_MN = CONFIGS / "pretrain" / "_smoke_torchtitan_multinode_gpu.yaml"


def test_real_run_goes_to_method_project():
    """실측 run 은 method 프로젝트로. framework 로 안 쪼개야 겹쳐보기(가로비교)가 성립한다."""
    env = _wandb.env(RunConfig.from_file(SFT_TRL_LORA))
    assert env["WANDB_PROJECT"] == "tfct-sft"
    assert env["WANDB_RUN_GROUP"] == "trl"


def test_smoke_is_isolated_from_measurement_projects():
    """스모크는 진단용 쓰레기 run — 실측 프로젝트에 섞이면 안 된다(삭제가 수동이라 청소도 비쌈)."""
    cfg = RunConfig.from_file(SMOKE_TRL)
    assert cfg.is_smoke()
    env = _wandb.env(cfg)
    assert env["WANDB_PROJECT"] == _wandb.SMOKE_PROJECT
    assert "smoke" in env["WANDB_TAGS"].split(",")


def test_multinode_run_is_distinguishable_from_single_node():
    """같은 config 를 SNSG·MNMG 로 돌리므로 이름·태그로 구분돼야 한다(안 그러면 판정 불가)."""
    cfg = RunConfig.from_file(SMOKE_TT_MN)
    env = _wandb.env(cfg)
    assert env["WANDB_NAME"].endswith("-mn2")
    tags = env["WANDB_TAGS"].split(",")
    assert "mnmg" in tags and "nodes-2" in tags
    # 멀티노드 검증의 정보 단위 = 메커니즘(조합 20 개가 아니라 메커니즘 4 개).
    assert "mech-torchrun" in tags


def test_single_node_run_name_unchanged():
    """단노드 이름은 기존 그대로 — 접미사는 멀티노드에만 붙는다."""
    assert RunConfig.from_file(SFT_TRL_LORA).run_name().endswith("-lora")


def test_explicit_env_beats_derived(monkeypatch):
    """런치 때 `--env WANDB_TAGS=...` 로 준 값을 config 파생값이 덮으면 안 된다(명시 > 파생)."""
    monkeypatch.setenv("WANDB_TAGS", "manual-label")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    _wandb.apply_env(RunConfig.from_file(SFT_TRL_LORA))
    import os

    assert os.environ["WANDB_TAGS"] == "manual-label"
    assert os.environ["WANDB_PROJECT"] == "tfct-sft"


def test_log_interval_is_dense_enough_for_smoke():
    """5 step 스모크에 간격 10(torchtitan/megatron 기본)·500(HF 기본)이면 차트가 빈다."""
    assert RunConfig.from_file(SMOKE_TRL).log_every_n_steps() == 1
    assert RunConfig.from_file(SFT_TRL_LORA).log_every_n_steps() == 10


def test_wired_whitelist_is_derived_from_mechanism_map():
    """가드용 화이트리스트와 메커니즘 맵이 어긋날 수 없어야 한다(파생 관계)."""
    assert MULTINODE_WIRED == frozenset(MULTINODE_MECHANISM)
    assert set(MULTINODE_MECHANISM.values()) == {"torchrun", "ray", "hf-torchrun", "megatron"}
