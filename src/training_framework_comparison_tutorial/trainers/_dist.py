"""멀티노드 토폴로지 해석 — config(의도)와 SkyPilot(실제)을 대조해 랑데부 인자를 만든다.

SkyPilot 멀티노드는 `run:` 블록을 **모든 노드에서** 실행한다 → `tfct-run` 이 노드마다 뜬다.
각 프로세스가 자기 node_rank 와 head IP 를 알아야 torchrun 이 여러 노드를 한 job 으로 묶는다.
그 사실을 SkyPilot 이 env 로 흘려주고(SKYPILOT_NODE_RANK/NODE_IPS/NUM_NODES), 이 모듈이 그걸
읽어 프레임워크별 trainer 가 쓰는 형태로 번역한다.

**SoT 를 둘로 쪼갠 이유**:
  - config `scale.nodes` = 실험이 요구하는 노드 수(통제비교 축이자 재현 기록 — YAML 이 단일 출처).
  - SKYPILOT_* env       = 실제 프로비저닝된 사실(rank·IP — 런타임에만 알 수 있음).

둘이 어긋나면 **조용히 틀리게 학습하는 대신 즉시 죽인다**. nodes=2 로 선언했는데 노드가 1개면
dp 눈금(= nodes×gpus)이 어긋나 step 수·유효 배치가 다른 프레임워크와 달라지고, 그건 에러 없이
"그럴듯한 손실 곡선"으로 나와 가로비교를 통째로 오염시킨다. 노드 수는 조용히 틀리면 안 되는 축이다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# 고정 포트. SkyPilot 이 띄우는 노드는 같은 VPC 안이라 충돌 위험이 없고, 값이 고정이면
# 로그·디버깅이 재현된다(랜덤 포트면 실패 재현이 어렵다).
MASTER_PORT = 29500


@dataclass(frozen=True)
class Topology:
    """이 프로세스가 속한 학습 job 의 토폴로지."""

    nodes: int
    gpus: int                       # 노드당 GPU (config scale.gpus)
    rank: int                       # node rank (0 = head)
    master_addr: str | None         # 단노드면 None (standalone 랑데부)

    @property
    def world_size(self) -> int:
        """전체 프로세스 수 = dp 눈금의 분모(step 계산에 쓰인다)."""
        return self.nodes * self.gpus

    @property
    def is_multinode(self) -> bool:
        return self.nodes > 1

    @property
    def is_head(self) -> bool:
        """head 노드에서만 해야 하는 일(HF export 등)의 가드."""
        return self.rank == 0


def resolve(scale: dict[str, Any]) -> Topology:
    """config `scale` 섹션 + SkyPilot env → Topology. 불일치면 SystemExit."""
    nodes = int(scale.get("nodes", 1))
    gpus = int(scale.get("gpus", 1))

    env_nodes = os.environ.get("SKYPILOT_NUM_NODES")
    actual = int(env_nodes) if env_nodes else None

    # 양방향 불일치 검사. 어느 쪽으로 어긋나도 사고다 — 노드가 모자라면 결과가 오염되고,
    # 남으면 놀고 있는 노드에 돈을 낸다.
    if actual is not None and actual != nodes:
        raise SystemExit(
            f"노드 수 불일치: config scale.nodes={nodes} 인데 SkyPilot 이 띄운 노드는 "
            f"{actual} 개다. `sky launch --num-nodes {nodes}` 로 맞추거나 config 의 "
            f"scale.nodes 를 {actual} 로 바꿔라."
        )

    if nodes == 1:
        return Topology(nodes=1, gpus=gpus, rank=0, master_addr=None)

    if actual is None:
        raise SystemExit(
            f"config scale.nodes={nodes} (멀티노드)인데 SkyPilot 멀티노드 env"
            "(SKYPILOT_NUM_NODES)가 없다. 멀티노드 런은 `sky launch --num-nodes N` 으로 띄워야 "
            "한다 (로컬/단노드 실행은 scale.nodes=1)."
        )

    raw_ips = os.environ.get("SKYPILOT_NODE_IPS", "")
    ips = [line.strip() for line in raw_ips.splitlines() if line.strip()]
    if len(ips) != nodes:
        raise SystemExit(
            f"SKYPILOT_NODE_IPS 의 IP 개수({len(ips)})가 scale.nodes({nodes})와 다르다: {ips!r}"
        )

    return Topology(
        nodes=nodes,
        gpus=gpus,
        rank=int(os.environ.get("SKYPILOT_NODE_RANK", "0")),
        master_addr=ips[0],         # SkyPilot 규약: 첫 줄이 head
    )


# 멀티노드가 실제로 배선된 (method, framework) 조합. 여기 없는 조합에 nodes>1 을 주면
# guard_wired 가 즉시 죽인다 — knob 이 조용히 거짓말하는 대신("nodes=2 넣었는데 실은 1노드로
# 학습") 정직하게 "미배선"이라고 말하게 한다. 새 프레임워크를 멀티노드로 배선하면 여기 한 줄 추가.
#
# 두 배선 계열:
#   ① torchrun 랑데부(이 모듈 torchrun_args): torchtitan SFT·pretrain, verl SFT.
#   ② ray 클러스터 부트스트랩(sky/ray_bootstrap.sh, head/worker `ray start`): verl RL·slime·
#      nemo-rl. 트레이너는 안 바뀐다(이미 nnodes 를 프레임워크에 전달) — sky run 블록이 노드 간
#      ray 를 세우고 head 에서만 드라이버를 돌린다. verl 공식 멀티노드 문서 + SkyPilot 예제 기준.
# ③ HF Trainer torchrun 런처(sky/hf_torchrun_launch.sh): trl SFT·DPO·GRPO·online_dpo(full=FSDP).
# 미배선으로 남는 멀티노드: **unsloth**(단일 GPU 전용 설계 = 영구) · **torchtitan GRPO**
# (experiments/rl 은 torchrun/ray 가 아니라 Monarch actor mesh = 우리 메커니즘 밖, experimental).
MULTINODE_WIRED: frozenset[tuple[str, str]] = frozenset({
    # ① torchrun 랑데부
    ("sft", "torchtitan"),
    ("pretrain", "torchtitan"),
    ("sft", "verl"),
    # ② ray 클러스터 부트스트랩(sky/ray_bootstrap.sh)
    ("grpo", "verl"),
    ("ppo", "verl"),
    ("sft", "slime"),
    ("grpo", "slime"),
    ("ppo", "slime"),
    ("sft", "nemo-rl"),
    ("dpo", "nemo-rl"),
    ("grpo", "nemo-rl"),
    ("ppo", "nemo-rl"),
    # ③ HF Trainer torchrun 런처(hf_torchrun_launch.sh, full=FSDP): trl SFT·DPO·GRPO·online_dpo.
    #    넷 다 Trainer(model=문자열) 동일 구조 → 같은 런처. RL(grpo/online_dpo)은 루프 내 생성이
    #    있어 SFT 보다 복잡하나 하드 블로커 아님(TRL FSDP 지원, use_vllm 기본 false=HF generate).
    ("sft", "trl"),
    ("dpo", "trl"),
    ("grpo", "trl"),
    ("online_dpo", "trl"),
    # ④ megatron(다단계): convert=노드별 로컬 / train=torchrun 랑데부 / export=head 전용.
    #    megatron_lm_sft 는 arguments.sh 의 LAUNCH_SCRIPT override 훅으로 랑데부 주입.
    #    ⚠️ export·resume 은 torch_dist 분산 ckpt 라 멀티노드에선 공유 FS 필요(스모크=train 검증).
    ("pretrain", "megatron-lm"),
    ("sft", "megatron-lm"),
    ("sft", "megatron-bridge"),
    ("grpo", "megatron-lm"),
})


def guard_wired(method: str, framework: str, scale: dict[str, Any]) -> None:
    """nodes>1 인데 이 (method, framework)가 멀티노드 미배선이면 SystemExit.

    dispatch 초입에서 부른다 — 배선 안 된 경로에 멀티노드 config 를 주면 학습이 시작되기 전에
    막아, 조용히 단노드로 돌거나(눈금 오염) 프레임워크가 리소스를 기다리며 행 거는 걸 예방한다.
    """
    if int(scale.get("nodes", 1)) <= 1:
        return
    if (method, framework) in MULTINODE_WIRED:
        return
    raise SystemExit(
        f"{method}/{framework} 는 아직 멀티노드 미배선인데 scale.nodes>1 이다. "
        f"멀티노드 배선된 조합: {sorted(MULTINODE_WIRED)}. "
        "이 조합은 scale.nodes=1 로 돌리거나, 멀티노드 배선을 먼저 추가하라 "
        "(ray 계열은 노드 간 ray 부트스트랩, trl/unsloth 는 accelerate 런처가 필요)."
    )


def torchrun_args(topo: Topology) -> list[str]:
    """torchrun 랑데부 인자.

    단노드는 `--standalone`(= c10d + localhost + 랜덤 포트) 로 기존 검증 경로를 그대로 둔다.
    멀티노드는 static 랑데부(node_rank + master_addr/port) — SkyPilot 공식 torchrun 예제와 동형.
    """
    if not topo.is_multinode:
        return ["--standalone", "--nnodes=1", f"--nproc_per_node={topo.gpus}"]
    return [
        f"--nnodes={topo.nodes}",
        f"--nproc_per_node={topo.gpus}",
        f"--node_rank={topo.rank}",
        f"--master_addr={topo.master_addr}",
        f"--master_port={MASTER_PORT}",
    ]
