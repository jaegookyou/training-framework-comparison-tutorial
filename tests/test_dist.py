"""_dist 토폴로지 해석 — "조용히 틀리지 않는다"가 이 모듈의 계약이라 그걸 테스트한다."""

from __future__ import annotations

import pytest

from training_framework_comparison_tutorial.trainers import _dist


def test_single_node_needs_no_skypilot_env(monkeypatch):
    """로컬/단노드는 SkyPilot env 없이 그대로 — 기존 검증된 경로를 안 건드린다."""
    for var in ("SKYPILOT_NUM_NODES", "SKYPILOT_NODE_IPS", "SKYPILOT_NODE_RANK"):
        monkeypatch.delenv(var, raising=False)

    topo = _dist.resolve({"nodes": 1, "gpus": 8})

    assert topo.world_size == 8
    assert not topo.is_multinode
    assert topo.is_head
    assert _dist.torchrun_args(topo) == ["--standalone", "--nnodes=1", "--nproc_per_node=8"]


def test_multinode_builds_static_rendezvous(monkeypatch):
    """멀티노드면 head IP 를 master_addr 로, 자기 rank 를 node_rank 로 넘긴다."""
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "1")

    topo = _dist.resolve({"nodes": 2, "gpus": 4})

    assert topo.world_size == 8            # dp 눈금 = nodes×gpus
    assert topo.is_multinode
    assert not topo.is_head                # rank 1 = worker → HF export 안 함
    assert _dist.torchrun_args(topo) == [
        "--nnodes=2",
        "--nproc_per_node=4",
        "--node_rank=1",
        "--master_addr=10.0.0.1",          # NODE_IPS 첫 줄 = head
        f"--master_port={_dist.MASTER_PORT}",
    ]


def test_node_count_mismatch_dies(monkeypatch):
    """config 가 2노드를 요구하는데 1노드만 떴으면 학습하지 말고 죽어야 한다.

    조용히 진행하면 dp 눈금이 절반이 돼 step 수·유효 배치가 달라지고, 에러 없이 가로비교가 오염된다.
    """
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")

    with pytest.raises(SystemExit, match="노드 수 불일치"):
        _dist.resolve({"nodes": 2, "gpus": 1})


def test_multinode_without_skypilot_env_dies(monkeypatch):
    """nodes>1 인데 SkyPilot 멀티노드 env 가 없으면(로컬 실행 등) 랑데부가 불가능하다."""
    for var in ("SKYPILOT_NUM_NODES", "SKYPILOT_NODE_IPS", "SKYPILOT_NODE_RANK"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(SystemExit, match="SKYPILOT_NUM_NODES"):
        _dist.resolve({"nodes": 2, "gpus": 1})


def test_ip_count_mismatch_dies(monkeypatch):
    """NODE_IPS 가 선언한 노드 수와 안 맞으면 head 선택이 틀릴 수 있으니 죽인다."""
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")

    with pytest.raises(SystemExit, match="SKYPILOT_NODE_IPS"):
        _dist.resolve({"nodes": 2, "gpus": 1})


# --- guard_wired: 미배선 조합의 거짓말 knob 방지 ---

def test_guard_allows_single_node_for_any_framework():
    """단노드는 어떤 (method, framework)든 통과 — 가드는 멀티노드에만 관여."""
    _dist.guard_wired("grpo", "slime", {"nodes": 1, "gpus": 8})       # 예외 안 남
    _dist.guard_wired("sft", "unsloth", {"nodes": 1, "gpus": 1})      # 예외 안 남


def test_guard_allows_wired_multinode_combos():
    """배선된 조합은 멀티노드 통과 — torchrun 계열 + ray 계열 둘 다."""
    wired = [
        ("sft", "torchtitan"), ("pretrain", "torchtitan"), ("sft", "verl"),   # torchrun
        ("grpo", "verl"), ("ppo", "verl"),                                     # ray
        ("sft", "slime"), ("grpo", "slime"), ("ppo", "slime"),                 # ray
        ("sft", "nemo-rl"), ("dpo", "nemo-rl"), ("grpo", "nemo-rl"), ("ppo", "nemo-rl"),  # ray
        ("sft", "trl"), ("dpo", "trl"),                                       # HF torchrun
    ]
    for method, fw in wired:
        _dist.guard_wired(method, fw, {"nodes": 2, "gpus": 2})        # 예외 안 남


def test_guard_blocks_unwired_multinode():
    """미배선 조합에 nodes>1 이면 죽는다: unsloth(단일GPU 전용)·trl RL(vLLM 분산 GPU 검증 선행)."""
    unwired = [("grpo", "trl"), ("online_dpo", "trl"), ("sft", "unsloth"), ("grpo", "unsloth")]
    for method, fw in unwired:
        with pytest.raises(SystemExit, match="멀티노드 미배선"):
            _dist.guard_wired(method, fw, {"nodes": 2, "gpus": 8})
