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
# 사용: run 블록에서
#   bash sky/ray_bootstrap.sh "tfct-run --config $CONFIG" "tfct-run --config $CONFIG --prepare-only"
# 2번째 인자(선택) = **워커 노드에서 돌릴 준비 명령**. 왜 필요한가: ray 계열은 드라이버가 head 에서만
# 돌아서 드라이버가 만든 노드-로컬 산출물(캐논 template 구운 토크나이저 등)이 워커엔 없다 →
# 워커가 그 경로를 열다 죽는다(2026-07-28 실측: HFValidationError). 근거·설계는 run.py:_run_prepare.

set -euo pipefail

DRIVER="$1"                                   # head 에서 실행할 드라이버 명령(문자열)
PREPARE="${2:-}"                              # (선택) 워커에서 돌릴 노드-로컬 준비 명령
NUM_NODES="${SKYPILOT_NUM_NODES:-1}"

# 단노드: ray 부트스트랩 불필요 — 프레임워크가 로컬 ray 를 자동 기동한다(기존 검증 경로 그대로).
if [ "$NUM_NODES" -le 1 ]; then
  exec bash -c "$DRIVER"
fi

HEAD_IP="$(echo "$SKYPILOT_NODE_IPS" | head -n1)"
RAY_PORT=6385                                 # SkyPilot 내부 ray(6379)와 충돌 회피용 별도 포트
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"       # 프레임워크 ray.init 이 우리 클러스터에 붙게

# collective(NCCL/Gloo) 네트워킹 고정 — Gloo 의 127.0.1.1 함정 등. 근거는 _netenv.sh 주석.
# shellcheck source=/dev/null
source "$(dirname "$0")/_netenv.sh"

# 앞선 job 의 ray 가 노드에 살아 있으면 `ray start` 가 죽는다:
#   ConnectionError: Ray is trying to start at <ip>:6385, but is already running at <ip>:6385.
# 클러스터를 유지한 채 job 만 여러 번 돌리는 게 정상 사용 패턴이라(스윕·수정 후 재실행) 반드시 밟는다.
#
# ⛔ **`ray stop` 으로 풀면 안 된다** (2026-07-28 실측으로 확인): `ray stop` 은 **포트 스코프가 없어**
#    그 노드의 ray 프로세스를 전부 죽인다 → 위 주석대로 SkyPilot 도 자기 ray 를 돌리므로 **SkyPilot 의
#    job 제어면까지 같이 죽는다**. 실제로 클러스터가 INIT 로 빠져 sky queue/logs/down 이 전부 막히고
#    (노드는 살아서 과금 중) 드라이버는 `Failed to connect to GCS ... terminated by ray stop` 으로
#    죽었다. 복구에 `sky start` 가 필요했다.
# ⛔ **재사용으로도 풀면 안 된다** (2026-07-28, 재사용을 넣었다가 되돌림): 살아 있는 ray 를 그냥
#    쓰면 액터를 낳는 **raylet 이 이전 job 의 env 를 그대로 들고 있다**. 그래서 이 스크립트가
#    새로 export 한 GLOO_SOCKET_IFNAME/NCCL_SOCKET_IFNAME 이 액터에 **안 먹고**, 워커가 다시
#    127.0.1.1 로 붙으려다 죽었다(부트스트랩 셸의 env 는 드라이버에만 적용된다).
#    같은 이유로 upstream 지식도 "ray 는 fresh 클러스터만" 이다 → [[multi-node-gpu-provisioning]].
# ✅ 그래서 **정직하게 죽고 fresh 를 요구한다**. 조용히 이전 env 로 도는 것보다, 무엇을 해야 하는지
#    말해주고 멈추는 게 싸다(무증상 행/오염된 런이 훨씬 비싸다).
# 판정 기준은 "fresh 냐"가 아니라 **"살아 있는 raylet 이 이번 job 과 같은 collective env 를 들고
# 있느냐"** 다. 그래서 ray 를 띄울 때 env 지문을 남기고, 재사용 전에 대조한다:
#   · 지문 일치  → 재사용(디버그 루프에서 클러스터를 매번 다시 띄우지 않아도 된다)
#   · 지문 불일치/부재 → 정직하게 실패(조용히 이전 env 로 도는 걸 막는다 — 그게 이 가드의 존재 이유)
_ray_env_marker="/tmp/tfct_ray_env.$RAY_PORT"
# 지문엔 **collective 설정만** 넣는다. VLLM_HOST_IP 같은 노드 고유값은 제외 — 노드마다 달라서
# 지문의 의미(같은 설정으로 뜬 ray 인가)를 흐리고, 노드 간 비교를 불가능하게 만든다.
_ray_env_fingerprint="${GLOO_SOCKET_IFNAME:-}|${NCCL_SOCKET_IFNAME:-}|${NCCL_IB_DISABLE:-}"

_reuse_ray=0
if ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
  if [ -f "$_ray_env_marker" ] && [ "$(cat "$_ray_env_marker")" == "$_ray_env_fingerprint" ]; then
    _reuse_ray=1
    echo "기존 ray 재사용 — collective env 지문 일치: $_ray_env_fingerprint"
  else
    echo "ERROR: $RAY_ADDRESS 에 ray 가 살아 있는데 **collective env 지문이 다르다**." >&2
    echo "  기대: $_ray_env_fingerprint" >&2
    echo "  실제: $( [ -f "$_ray_env_marker" ] && cat "$_ray_env_marker" || echo '<지문 없음 — 이 스크립트가 안 띄운 ray>')" >&2
    echo "  살아 있는 raylet 은 이전 env 를 들고 있어 이번 설정이 액터에 안 먹는다" >&2
    echo "  (Gloo/NCCL 이 엉뚱한 인터페이스를 잡아 무증상 행이 될 수 있다)." >&2
    echo "  조치: sky down <cluster> 후 sky launch 로 새로 띄운다." >&2
    exit 1
  fi
fi

if [ "${SKYPILOT_NODE_RANK}" == "0" ]; then
  if [ "$_reuse_ray" == "0" ]; then
    ray start --head --port="$RAY_PORT" --disable-usage-stats
    echo "$_ray_env_fingerprint" > "$_ray_env_marker"
  fi
  # 모든 워커가 조인할 때까지 대기(최대 5분). 드라이버가 자원을 못 찾고 실패/행 거는 걸 방지.
  for _ in $(seq 1 30); do
    joined="$(ray status 2>/dev/null | grep -c 'node_' || true)"
    [ "${joined:-0}" -ge "$NUM_NODES" ] && break
    sleep 10
  done
  echo "ray 클러스터 준비 (${joined:-0}/${NUM_NODES} 노드) — 드라이버 실행"
  exec bash -c "$DRIVER"
else
  # 노드-로컬 준비물을 **먼저** 만든다(head 는 드라이버가 train() 안에서 직접 만든다).
  # ray 조인보다 앞에 두는 이유: 준비가 실패하면 이 노드를 클러스터에 넣지 않고 정직하게 죽는 게 낫다
  # (조인만 해두면 드라이버가 액터를 띄운 뒤에야 파일이 없는 걸 발견해 진단이 멀어진다).
  if [ -n "$PREPARE" ]; then
    echo "워커 노드-로컬 준비 실행: $PREPARE"
    bash -c "$PREPARE"
  fi
  sleep 15                                    # head 의 --head 기동을 먼저 기다린다
  # 재사용(지문 일치)이면 이 노드는 이미 조인돼 있다 → 다시 join 하면 "already running" 으로 죽는다.
  if [ "$_reuse_ray" == "0" ]; then
    ray start --address="$HEAD_IP:$RAY_PORT" --disable-usage-stats
    echo "$_ray_env_fingerprint" > "$_ray_env_marker"
  fi
  sleep infinity                              # 노드를 ray 자원으로 유지(SkyPilot 이 job 종료 시 파기)
fi
