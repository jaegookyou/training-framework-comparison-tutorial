#!/usr/bin/env bash
# SNSG 스윕 — 단노드·단GPU LoRA 스모크 14개를 순차로 검증한다(repo 최초 GPU 실행용).
#
# 각 런: sky launch(autodown) → 5-step 스모크 → 즉시 down. 하나 실패해도 다음으로 넘어간다.
# autodown(-i --down)이 3중 안전망: 스크립트가 죽어도 idle 후 인스턴스가 자동 파기된다
# (과금 모니터 못 하는 상황 대비). 로그는 프레임워크별 파일로 남긴다.
#
# 사전조건:
#   - sky check vast 통과(또는 --cloud 로 지정한 클라우드) + 크레딧 충전
#   - export WANDB_API_KEY=...  HF_TOKEN=...   (W&B 로깅 · HF Qwen3-8B 다운로드)
#
# 사용:
#   export WANDB_API_KEY=...  HF_TOKEN=...
#   bash sky/snsg_sweep.sh                 # 기본 L40S:1 · vast
#   bash sky/snsg_sweep.sh RTX3090:1 vast  # 더 싸게(단 online_dpo 는 OOM 위험 — 24GB)
#   bash sky/snsg_sweep.sh A100:1 nebius   # Nebius 로
#
# ⚠️ 먼저 1개만: 처음이면 아래 SMOKES 의 trl 한 줄만 남기고 돌려 파이프라인(이미지 pull·HF·wandb)
#    을 확인한 뒤 전체를 도는 게 안전하다(첫 런이 제일 값진 발견).

set -u
GPU="${1:-L40S:1}"       # 48GB — 8B LoRA 전부 커버(online_dpo=정책+RM 2모델 포함)
CLOUD="${2:-vast}"
IDLE=15                  # idle 분 후 자동 파기(안전망)

: "${WANDB_API_KEY:?먼저 export WANDB_API_KEY}"
: "${HF_TOKEN:?먼저 export HF_TOKEN}"

# (method  framework  smoke_config  sky_yaml)
SMOKES=(
  "sft         trl              configs/sft/_smoke_trl_gpu.yaml                 sky/sft.trl.sky.yaml"
  "sft         unsloth          configs/sft/_smoke_unsloth_gpu.yaml             sky/sft.unsloth.sky.yaml"
  "sft         verl             configs/sft/_smoke_verl_gpu.yaml                sky/sft.verl.sky.yaml"
  "sft         torchtitan       configs/sft/_smoke_torchtitan_gpu.yaml          sky/sft.torchtitan.sky.yaml"
  "sft         megatron-bridge  configs/sft/_smoke_megatron-bridge_gpu.yaml     sky/sft.megatron-bridge.sky.yaml"
  "sft         nemo-rl          configs/sft/_smoke_nemo-rl_gpu.yaml             sky/sft.nemo-rl.sky.yaml"
  "dpo         trl              configs/dpo/_smoke_trl_gpu.yaml                 sky/dpo.trl.sky.yaml"
  "dpo         unsloth          configs/dpo/_smoke_unsloth_gpu.yaml             sky/dpo.unsloth.sky.yaml"
  "dpo         nemo-rl          configs/dpo/_smoke_nemo-rl_gpu.yaml             sky/dpo.nemo-rl.sky.yaml"
  "grpo        trl              configs/grpo/_smoke_trl_gpu.yaml                sky/grpo.trl.sky.yaml"
  "grpo        unsloth          configs/grpo/_smoke_unsloth_gpu.yaml            sky/grpo.unsloth.sky.yaml"
  "grpo        verl             configs/grpo/_smoke_verl_gpu.yaml               sky/grpo.verl.sky.yaml"
  "grpo        nemo-rl          configs/grpo/_smoke_nemo-rl_gpu.yaml            sky/grpo.nemo-rl.sky.yaml"
  "online_dpo  trl              configs/online_dpo/_smoke_trl_gpu.yaml          sky/online_dpo.trl.sky.yaml"
)

LOGDIR="snsg_sweep_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "=== SNSG 스윕 시작 | GPU=$GPU CLOUD=$CLOUD | 로그=$LOGDIR ==="
echo "총 ${#SMOKES[@]}개 · 순차 실행 · 각 완료 후 즉시 down"

declare -a RESULTS
i=0
for row in "${SMOKES[@]}"; do
  read -r method fw config sky <<< "$row"
  i=$((i+1))
  name="tfct-${method}-${fw}"
  log="$LOGDIR/${i}_${method}_${fw}.log"
  echo
  echo "===== [$i/${#SMOKES[@]}] $method/$fw launch ($(date +%H:%M:%S)) ====="
  if sky launch -c "$name" "$sky" --gpus "$GPU" --cloud "$CLOUD" -y -i "$IDLE" --down \
      --env CONFIG="$config" --env WANDB_API_KEY --env HF_TOKEN > "$log" 2>&1; then
    echo "  ✓ PASS  $method/$fw"
    RESULTS+=("✓ PASS  $method/$fw")
  else
    echo "  ✗ FAIL  $method/$fw  (로그: $log 마지막 줄:)"
    tail -3 "$log" | sed 's/^/      /'
    RESULTS+=("✗ FAIL  $method/$fw  → $log")
  fi
  sky down "$name" -y >/dev/null 2>&1 || true   # 확실히 파기(launch 실패로 켜져 있어도)
done

echo
echo "========== SNSG 스윕 결과 =========="
printf '%s\n' "${RESULTS[@]}"
echo "로그 전체: $LOGDIR/"
echo "혹시 남은 클러스터 확인: sky status   (있으면: sky down --all -y)"
