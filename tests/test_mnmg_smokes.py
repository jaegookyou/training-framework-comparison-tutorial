"""MNMG(멀티노드) 스모크 config 묶음의 계약.

이 묶음의 목적은 "배선된 20조합을 다 태우기"가 아니다 — 정보는 조합이 아니라 **메커니즘 4개**에
있으므로(torchrun / ray / hf-torchrun / megatron), 묶음이 그 4개를 빠짐없이 덮는지와, 각 config 가
실제로 멀티노드로 뜰 수 있는 상태인지(가드 통과·nodes>1·스모크 레버 있음)를 여기서 잠근다.
GPU 를 빌린 뒤에 "이 config 는 못 뜬다"를 발견하면 그때는 돈이 나가고 있다.
"""

from pathlib import Path

import pytest

from training_framework_comparison_tutorial.config import RunConfig
from training_framework_comparison_tutorial.trainers import _wandb
from training_framework_comparison_tutorial.trainers._dist import (
    MULTINODE_MECHANISM,
    guard_wired,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MNMG_SMOKES = sorted(CONFIGS.glob("*/_smoke_*multinode*.yaml"))


def test_mnmg_smoke_set_is_not_empty():
    assert MNMG_SMOKES, "MNMG 스모크가 하나도 없다 — 멀티노드 검증을 태울 수 없다"


@pytest.mark.parametrize("path", MNMG_SMOKES, ids=lambda p: p.stem)
def test_each_mnmg_smoke_is_launchable(path):
    """가드 통과 · nodes>1 · 스모크 레버 — 셋 중 하나라도 빠지면 GPU 위에서야 드러난다."""
    cfg = RunConfig.from_file(path)
    scale = cfg.section("scale")

    assert int(scale.get("nodes", 1)) > 1, "MNMG 인데 nodes 가 1 이다"
    # nodes>1 인데 미배선 조합이면 여기서 SystemExit — 즉 이게 통과해야 실제로 뜬다.
    guard_wired(cfg.method, cfg.framework, scale)
    assert cfg.is_smoke(), "스모크로 인식 안 되면 W&B 격리·로깅 간격 정책이 안 걸린다"
    # 축소 레버가 실제로 있어야 풀런 비용이 안 나간다. megatron-lm 만 레버 이름이 다르다.
    if cfg.framework == "megatron-lm":
        assert cfg.section("megatron").get("train_samples", 10**9) <= 1000
    else:
        assert int(cfg.section("debug").get("max_steps", -1)) > 0


@pytest.mark.parametrize("path", MNMG_SMOKES, ids=lambda p: p.stem)
def test_each_mnmg_smoke_is_identifiable_in_wandb(path):
    """판정은 W&B 화면으로만 한다 — 이름·태그가 SNSG 런과 구분돼야 판정이 성립한다."""
    cfg = RunConfig.from_file(path)
    env = _wandb.env(cfg)
    tags = env["WANDB_TAGS"].split(",")

    assert env["WANDB_NAME"].endswith("-mn2")
    assert "mnmg" in tags
    assert any(t.startswith("mech-") for t in tags)


def test_smoke_set_covers_every_multinode_mechanism():
    """메커니즘 하나라도 안 덮이면 그 배선은 검증 안 된 채로 남는다(=이 묶음의 존재 이유)."""
    covered = {
        MULTINODE_MECHANISM[(cfg.method, cfg.framework)]
        for cfg in (RunConfig.from_file(p) for p in MNMG_SMOKES)
    }
    assert covered == set(MULTINODE_MECHANISM.values())
