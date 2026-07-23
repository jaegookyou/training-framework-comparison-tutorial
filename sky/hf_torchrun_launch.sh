#!/usr/bin/env bash
# HF Trainer(trl) 멀티노드/멀티GPU 런처 — trl sky run 블록이 호출한다.
#
# 왜 필요한가: trl 트레이너는 인프로세스 `SFTTrainer().train()` 이다(torchtitan/verl 처럼 자기
# torchrun 을 안 띄운다). HF Trainer 는 **torchrun 이 세팅한 분산 env(RANK/WORLD_SIZE/LOCAL_RANK/
# MASTER_ADDR)** 를 자동 감지해 DDP/FSDP 샤딩한다 → 우리가 tfct-run 을 torchrun 으로 감싸주면 된다.
# ray 계열과 달리 별도 클러스터가 아니라 torchrun 랑데부(torchtitan/verl SFT 와 같은 메커니즘).
#
# 분기:
#   전체 프로세스 1개(1노드 1GPU): torchrun 없이 그대로 실행(기존 LoRA 스모크 경로 보존).
#   그 외(멀티GPU/멀티노드):        torchrun static 랑데부로 tfct-run 을 감싼다.
#
# full FT 의 FSDP 샤딩은 트레이너(trl_sft.apply_multigpu_fsdp)가 WORLD_SIZE>1 을 보고 켠다.
#
# ⚠️ GPU 검증 대기: HF Trainer+FSDP+캐논 template 정합은 GPU end-to-end 확인(다른 경로와 동일).
#
# 사용: run 블록에서 `bash sky/hf_torchrun_launch.sh "$CONFIG"`

set -euo pipefail

CONFIG="$1"
NUM_NODES="${SKYPILOT_NUM_NODES:-1}"
GPUS="${SKYPILOT_NUM_GPUS_PER_NODE:-1}"
DRIVER_MODULE="training_framework_comparison_tutorial.run"

# 단일 프로세스: torchrun 불필요 — 콘솔 스크립트 그대로(기존 검증 경로).
if [ "$NUM_NODES" -le 1 ] && [ "$GPUS" -le 1 ]; then
  exec tfct-run --config "$CONFIG"
fi

if [ "$NUM_NODES" -le 1 ]; then
  # 단노드 멀티GPU: standalone 랑데부.
  exec torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    -m "$DRIVER_MODULE" --config "$CONFIG"
fi

# 멀티노드: static 랑데부(_dist.torchrun_args 와 동형 — SkyPilot NODE_IPS 첫 줄 = head).
HEAD_IP="$(echo "$SKYPILOT_NODE_IPS" | head -n1)"
exec torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$GPUS" \
  --node_rank="${SKYPILOT_NODE_RANK}" \
  --master_addr="$HEAD_IP" \
  --master_port=29500 \
  -m "$DRIVER_MODULE" --config "$CONFIG"
