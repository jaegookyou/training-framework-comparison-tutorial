#!/usr/bin/env bash
# Ray 멀티노드 부트스트랩 — ray 계열 프레임워크(verl RL·slime·nemo-rl)의 sky run 블록이 호출한다.
#
# 왜 sky yaml 의 일인가: SkyPilot 은 run 블록을 **모든 노드에서** 실행하고 SKYPILOT_NODE_RANK/
# NODE_IPS/NUM_NODES 를 준다. ray 계열은 torchrun 랑데부가 아니라 **노드 간 ray 클러스터**가
# 있어야 프레임워크(main_ppo/train.py)가 그 위에 액터를 스케줄한다. 그 클러스터를 여기서 세운다.
# 트레이너(Python)는 안 바꾼다 — 이미 nnodes/n_gpus_per_node 를 프레임워크에 넘기고 있고,
# 드라이버는 head 에서만 도니 데이터 로딩·디스패치는 프레임워크가 ray 로 처리한다.
#
# 패턴 출처(추정 아님): verl 공식 멀티노드 문서 + SkyPilot verl 예제.
#   - head(rank0): `ray start --head` → 모든 워커 조인 대기(ray status) → 드라이버 실행.
#   - worker:      `ray start --address` → sleep infinity(노드를 ray 자원으로 유지, 드라이버 안 돎).
#
# ⚠️ GPU 검증 대기(다른 경로와 동일한 단서):
#   - 프레임워크 ray.init 이 우리 클러스터(포트 6385)에 붙는지 — RAY_ADDRESS export 로 명시하지만
#     verl/slime/nemo 각자의 init 인자 확정은 GPU end-to-end 에서.
#   - 데이터 로컬리티: 드라이버(head)가 데이터를 로드해 ray 로 분배한다는 가정(verl 기본 동작).
#   - nemo-rl 은 uv venv → `ray` 실행파일 경로가 다를 수 있음(그 sky yaml 주석 참고).
#
# 사용: run 블록에서 `bash sky/ray_bootstrap.sh "tfct-run --config $CONFIG"`

set -euo pipefail

DRIVER="$1"                                   # head 에서 실행할 드라이버 명령(문자열)
NUM_NODES="${SKYPILOT_NUM_NODES:-1}"

# 단노드: ray 부트스트랩 불필요 — 프레임워크가 로컬 ray 를 자동 기동한다(기존 검증 경로 그대로).
if [ "$NUM_NODES" -le 1 ]; then
  exec bash -c "$DRIVER"
fi

HEAD_IP="$(echo "$SKYPILOT_NODE_IPS" | head -n1)"
RAY_PORT=6385                                 # SkyPilot 내부 ray(6379)와 충돌 회피용 별도 포트
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"       # 프레임워크 ray.init 이 우리 클러스터에 붙게

if [ "${SKYPILOT_NODE_RANK}" == "0" ]; then
  ray start --head --port="$RAY_PORT" --disable-usage-stats
  # 모든 워커가 조인할 때까지 대기(최대 5분). 드라이버가 자원을 못 찾고 실패/행 거는 걸 방지.
  for _ in $(seq 1 30); do
    joined="$(ray status 2>/dev/null | grep -c 'node_' || true)"
    [ "${joined:-0}" -ge "$NUM_NODES" ] && break
    sleep 10
  done
  echo "ray 클러스터 준비 (${joined:-0}/${NUM_NODES} 노드) — 드라이버 실행"
  exec bash -c "$DRIVER"
else
  sleep 15                                    # head 의 --head 기동을 먼저 기다린다
  ray start --address="$HEAD_IP:$RAY_PORT" --disable-usage-stats
  sleep infinity                              # 노드를 ray 자원으로 유지(SkyPilot 이 job 종료 시 파기)
fi
