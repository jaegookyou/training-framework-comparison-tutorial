#!/usr/bin/env bash
# SNSG 계단식 전체 스윕 — 이미지 preflight 로 게이팅 후 통과한 셀만 pretrain→SFT→DPO→RL 순서로 스모크.
#
# 왜 계단식인가(2026-07-30): 무작정 다 태우면 아는 실패(torchtitan 이미지 NG · verl GRPO
# vLLM-Blackwell 벽)에 돈을 태우고, 미확인 이미지(unsloth 등)를 확인 없이 12분씩 학습을 돈다.
# → ① 미확인 이미지 preflight($0.1) 로 Blackwell 실행 여부 확정 ② 통과 이미지만 학습 스모크 ③ 아는
#    벽은 SKIP 목록으로 건너뛰고 결과에 사유 기록.
#
# 전제: `set -a; source .env; set +a` (WANDB_API_KEY). HF_TOKEN 은 선택(Qwen3-4B-Base=public).
# 사용: bash sky/snsg_full_sweep.sh [GPU] [INFRA]   기본 RTX6000:1 nebius
#   ⚠️ serial 강제(같은 리전 spot 경합). 각 런 autodown 3중 안전망. 다른 클러스터 없을 때 시작.
#
# 커버리지: SNSG 스모크 config 가 있는 셀만(pretrain·SFT·DPO·online_dpo·GRPO). PPO 3셀·grpo{slime,
# megatron-lm}은 SNSG 스모크 config 미작성(추후) — 대부분 RL rollout 복잡/vLLM 벽이라 우선순위 낮음.

set -u
GPU="${1:-RTX6000:1}"
INFRA="${2:-nebius}"
IDLE=15
OWNER="ghcr.io/jaegookyou/training-framework-comparison-tutorial"
: "${WANDB_API_KEY:?먼저 set -a; source .env; set +a}"
HF_ENV=(); [ -n "${HF_TOKEN:-}" ] && HF_ENV=(--env HF_TOKEN)   # 있으면 전달(public 이라 없어도 됨)

LOGDIR="snsg_sweep_logs_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"

# 이미지 사전상태: OK=검증됨(스킵) · NG=죽음(스킵) · ""=preflight 로 확인.
declare -A IMG_STATE=(
  [trl]=OK [verl]=OK              # 2노드 SFT 5/5 실증(07-28)
  [megatron-bridge]=OK           # NeMo 피벗 convert 실증(07-30) — 4B→mcore import 성공
  [torchtitan]=NG                # sm_120 커널 없음(07-29 preflight)
  [unsloth]="" [megatron-lm]="" [slime]=""
)
PREFLIGHT_IMAGES=(megatron-lm slime unsloth)   # SMOKES 에서 게이팅하는 미확인 이미지만

# (method framework config sky_yaml image)
SMOKES=(
  "pretrain    megatron-lm      configs/pretrain/_smoke_megatron-lm_gpu.yaml  sky/pretrain.megatron-lm.sky.yaml  megatron-bridge"
  "sft         trl              configs/sft/_smoke_trl_gpu.yaml               sky/sft.trl.sky.yaml               trl"
  "sft         verl             configs/sft/_smoke_verl_gpu.yaml              sky/sft.verl.sky.yaml              verl"
  "sft         megatron-lm      configs/sft/_smoke_megatron-lm_gpu.yaml       sky/sft.megatron-lm.sky.yaml       megatron-lm"
  "sft         megatron-bridge  configs/sft/_smoke_megatron-bridge_gpu.yaml   sky/sft.megatron-bridge.sky.yaml   megatron-bridge"
  "sft         slime            configs/sft/_smoke_slime_gpu.yaml             sky/sft.slime.sky.yaml             slime"
  "sft         unsloth          configs/sft/_smoke_unsloth_gpu.yaml           sky/sft.unsloth.sky.yaml           unsloth"
  "sft         torchtitan       configs/sft/_smoke_torchtitan_gpu.yaml        sky/sft.torchtitan.sky.yaml        torchtitan"
  "dpo         trl              configs/dpo/_smoke_trl_gpu.yaml               sky/dpo.trl.sky.yaml               trl"
  "dpo         unsloth          configs/dpo/_smoke_unsloth_gpu.yaml           sky/dpo.unsloth.sky.yaml           unsloth"
  "online_dpo  trl              configs/online_dpo/_smoke_trl_gpu.yaml        sky/online_dpo.trl.sky.yaml        trl"
  "grpo        trl              configs/grpo/_smoke_trl_gpu.yaml              sky/grpo.trl.sky.yaml              trl"
  "grpo        unsloth          configs/grpo/_smoke_unsloth_gpu.yaml          sky/grpo.unsloth.sky.yaml          unsloth"
  "grpo        verl             configs/grpo/_smoke_verl_gpu.yaml             sky/grpo.verl.sky.yaml             verl"
)
# 아는 벽 — 이미지가 OK 여도 건너뛴다(사유 기록).
declare -A SKIP=(
  ["grpo/verl"]="vLLM-Blackwell weight-sync 벽(upstream, 별도 sglang 피벗)"
)

run1() { # name sky config log [extra args...]
  local name="$1" sky="$2" cfg="$3" log="$4"; shift 4
  sky launch -c "$name" "$sky" --gpus "$GPU" --infra "$INFRA" -y -i "$IDLE" --down \
    --env CONFIG="$cfg" --env WANDB_API_KEY "${HF_ENV[@]}" "$@" > "$log" 2>&1
  local rc=$?
  sky down "$name" -y >/dev/null 2>&1 || true
  return $rc
}

echo "===== [Stage 1] 이미지 preflight ($(date +%H:%M:%S)) ====="
for img in "${PREFLIGHT_IMAGES[@]}"; do
  log="$LOGDIR/pf_${img}.log"; echo "  preflight $img ..."
  if run1 "tfct-pf" sky/preflight.sky.yaml "-" "$log" --image-id "docker:${OWNER}/${img}:latest"; then
    IMG_STATE[$img]=OK;  echo "    ✓ OK  $img"
  else
    IMG_STATE[$img]=NG;  echo "    ✗ NG  $img (로그 $log)"
  fi
done

echo; echo "===== [Stage 2] 학습 스모크 (이미지 통과 + 아는 벽 제외) ($(date +%H:%M:%S)) ====="
declare -a RESULTS; i=0
for row in "${SMOKES[@]}"; do
  read -r method fw cfg sky img <<< "$row"; i=$((i+1))
  state="${IMG_STATE[$img]:-?}"; skey="${method}/${fw}"
  if [ -n "${SKIP[$skey]:-}" ]; then
    RESULTS+=("⤵ SKIP  $skey  (${SKIP[$skey]})"); echo "  ⤵ SKIP $skey"; continue
  fi
  if [ "$state" != "OK" ]; then
    RESULTS+=("⛔ IMG-$state  $skey  (이미지 $img 미통과)"); echo "  ⛔ IMG-$state $skey"; continue
  fi
  name="tfct-${method}-${fw}"; log="$LOGDIR/${i}_${method}_${fw}.log"
  echo "  ▶ [$i] $skey launch ($(date +%H:%M:%S))"
  if run1 "$name" "$sky" "$cfg" "$log"; then
    RESULTS+=("✓ PASS  $skey"); echo "    ✓ PASS $skey"
  else
    RESULTS+=("✗ FAIL  $skey  → $log"); echo "    ✗ FAIL $skey"; tail -3 "$log" | sed 's/^/        /'
  fi
done

echo; echo "========== SNSG 계단식 스윕 결과 =========="
printf '%s\n' "${RESULTS[@]}"
echo "로그: $LOGDIR/  |  남은 클러스터 확인: sky status (있으면 sky down --all -y)"
