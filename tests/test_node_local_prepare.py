"""노드-로컬 준비(`--prepare-only`) 계약.

배경: ray 메커니즘만 드라이버가 head 에서만 돌아, 드라이버가 만든 파일이 워커엔 없다. 그래서
워커가 파일 경로를 여는 순간 죽는다(2026-07-28 verl GRPO 2노드 런에서 실측). 여기서 잠그는 건
**"필요한 곳에 prepare 가 있고, 그게 sky 배선과 어긋나지 않는다"** — 어긋나면 GPU 위에서만 드러난다.
"""

import importlib
import re
from pathlib import Path

import pytest

from training_framework_comparison_tutorial.run import TRAINERS
from training_framework_comparison_tutorial.trainers._dist import MULTINODE_MECHANISM

ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT / "sky"

# 노드-로컬 산출물을 **파일 경로로** 프레임워크에 넘기는 조합 = 워커에도 실물이 필요하다.
# slime GRPO/PPO 가 빠진 건 누락이 아니라 경로 부재다 — 걔들은 `--hf-checkpoint` 로 hub id 를
# 넘겨서 워커가 알아서 받는다(필요 없는 곳에 knob 을 달지 않는다).
NEEDS_PREPARE = {
    ("grpo", "verl"),
    ("ppo", "verl"),
    ("sft", "slime"),
    ("sft", "nemo-rl"),
    ("dpo", "nemo-rl"),
    ("grpo", "nemo-rl"),
    ("ppo", "nemo-rl"),
}


def _module(method: str, framework: str):
    return importlib.import_module(TRAINERS[method][framework])


@pytest.mark.parametrize("cell", sorted(NEEDS_PREPARE), ids=lambda c: f"{c[0]}-{c[1]}")
def test_needed_trainers_expose_prepare(cell):
    """워커가 파일 경로를 여는 조합엔 prepare() 가 있어야 한다."""
    method, framework = cell
    assert hasattr(_module(method, framework), "prepare"), (
        f"{method}/{framework} 는 노드-로컬 산출물을 경로로 넘기는데 prepare() 가 없다"
    )


def test_prepare_only_on_ray_mechanism():
    """prepare 가 필요한 조합은 전부 ray 메커니즘이어야 한다.

    torchrun 계열은 모든 노드가 train() 을 돌아 각자 만든다 → 준비 진입점이 필요 없다.
    여기가 깨지면 둘 중 하나다: 메커니즘 분류가 틀렸거나, 불필요한 prepare 를 단 것.
    """
    for method, framework in NEEDS_PREPARE:
        assert MULTINODE_MECHANISM.get((method, framework)) == "ray", (
            f"{method}/{framework} 는 ray 가 아닌데 노드-로컬 준비가 필요하다고 표시돼 있다"
        )


def test_sky_yaml_passes_prepare_exactly_where_needed():
    """sky 배선과 코드가 어긋나면 GPU 위에서만 드러난다 → 여기서 프로그램으로 대조한다."""
    wired = set()
    for yaml in SKY.glob("*.yaml"):
        text = yaml.read_text()
        if "ray_bootstrap.sh" not in text:
            continue
        # 파일명 규칙: <method>.<framework>.sky.yaml (framework 에 '-' 포함 가능)
        method, framework = yaml.name.split(".")[0], yaml.name.split(".")[1]
        if re.search(r"--prepare-only", text):
            wired.add((method, framework))

    assert wired == NEEDS_PREPARE, (
        f"sky yaml 의 --prepare-only 배선이 코드와 불일치.\n"
        f"  yaml 에만: {sorted(wired - NEEDS_PREPARE)}\n"
        f"  코드에만: {sorted(NEEDS_PREPARE - wired)}"
    )


def test_bootstrap_runs_prepare_before_join():
    """준비 실패 시 조인 전에 죽어야 한다 — 조인만 해두면 진단이 액터 생성 시점까지 밀린다."""
    script = (SKY / "ray_bootstrap.sh").read_text()
    prepare_pos = script.index('bash -c "$PREPARE"')
    join_pos = script.index('ray start --address=')
    assert prepare_pos < join_pos, "워커 준비가 ray 조인보다 뒤에 있다"
