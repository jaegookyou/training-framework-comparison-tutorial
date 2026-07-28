#!/usr/bin/env bash
# 멀티노드 collective 네트워킹 env 고정 — 멀티노드 런처들이 **source** 한다(실행 아님).
#
# 왜 필요한가(2026-07-28 verl GRPO 2노드 런에서 실측):
#   rank=1 → remote=[127.0.1.1]:6570  SO_ERROR: Connection refused
#   RuntimeError: Gloo connectFullMesh failed
# Gloo 가 상대 주소를 **hostname 으로 해석**하는데, Debian 계열 `/etc/hosts` 는 hostname 을
# `127.0.1.1`(루프백)로 매핑한다 → 워커가 head 대신 자기 루프백에 붙으려다 실패한다.
# 랑데부(우리가 IP 로 넘기는 부분)는 멀쩡해도 **collective 는 따로 해석**하므로 별개로 막아야 한다.
# 2026-07-28 이전엔 "우리 랑데부는 IP 기반이라 해당 없음"으로 이식을 보류했는데, 그 판단이 틀렸다.
#
# 출처: [[multi-node-gpu-provisioning]] (llm-serving-gateway-tutorial 이 2026-07-27 에 같은 계열
# 함정 13 개를 실측하며 정리) — 아래 규칙은 그 위키의 처방을 그대로 따른다:
#   - **NCCL 과 Gloo 는 항상 같이 고정한다.** 하나만 고정하면 나머지가 docker0(172.17.x)를 골라
#     **무증상 행**(에러 없이 멈춤)이 된다 — 실패보다 훨씬 비싼 실패 방식.
#   - **NIC 이름을 하드코딩하지 않는다.** 리눅스 인터페이스 이름은 15자 상한이라 Nebius 의
#     `network-interface-0` 이 커널엔 `network-interfa` 로 등록된다(`eth0` 가정은 깨진다).
#     → 이 노드의 **실제 IP 로 인터페이스를 역조회**한다(추정 0).
#   - **IB 유무는 디렉토리가 아니라 장치로 판정한다.** 커널 모듈만 올라가도
#     `/sys/class/infiniband` 는 생기므로 `-d` 검사는 틀린다 → 글롭으로 실제 항목을 본다.
#
# 모든 값은 이미 설정돼 있으면 존중한다(`:=`) — 런치 때 `--env` 로 덮어쓸 여지를 남긴다.

# 이 노드의 IP(SkyPilot 이 준 목록에서 자기 rank 줄). 단노드면 목록이 없을 수 있다.
_rank="${SKYPILOT_NODE_RANK:-0}"
_my_ip="$(echo "${SKYPILOT_NODE_IPS:-}" | sed -n "$((_rank + 1))p")"

if [ -n "$_my_ip" ]; then
  # IP → 인터페이스 역조회. 커널이 등록한 **실제 이름**을 그대로 얻는다(15자 절단도 자동 반영).
  #
  # ⚠️ 이 파일은 `set -e` 스크립트에서 source 된다 → **모든 외부 명령 실패를 삼켜야 한다**.
  # 안 그러면 도구 하나 없는 것 때문에 학습 job 전체가 출력 한 줄 없이 exit 127 로 죽는다
  # (2026-07-28 실측: 학습 이미지에 iproute2 가 없어 `ip` 가 127 → job 즉사).
  # 그래서 ① ip(iproute2) → ② python3 stdlib ioctl 폴백 → ③ 실패 시 경고 후 계속, 3단이다.
  _nic=""
  if command -v ip >/dev/null 2>&1; then
    _nic="$(ip -o -4 addr show 2>/dev/null | awk -v ip="$_my_ip" '$4 ~ "^"ip"/" {print $2; exit}')" || _nic=""
  fi
  if [ -z "$_nic" ] && command -v python3 >/dev/null 2>&1; then
    # stdlib 만으로 IP→ifname (SIOCGIFADDR). 학습 이미지엔 python3 가 반드시 있다.
    # (서빙 repo 는 같은 문제를 psutil 로 풀었다 — `psutil.net_if_addrs()`. 결론은 같고, 여기선
    #  이미지에 psutil 이 있다고 가정하지 않으려고 stdlib 만 쓴다.)
    _nic="$(TFCT_MY_IP="$_my_ip" python3 - <<'PYEOF' 2>/dev/null || true
import fcntl, os, socket, struct
target = os.environ["TFCT_MY_IP"]
for nic in sorted(os.listdir("/sys/class/net")):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", nic[:15].encode()))[20:24]
        if socket.inet_ntoa(addr) == target:
            print(nic)
            break
    except OSError:
        continue
PYEOF
)"
  fi
  if [ -z "$_nic" ]; then
    echo "[netenv] ⚠️ NIC 자동탐지 실패(ip·python3 둘 다) — NCCL/Gloo 인터페이스를 고정하지 못했다."
    echo "[netenv] ⚠️ 노드 간 collective 가 docker0 을 골라 **무증상 행**이 될 수 있다."
    echo "[netenv] ⚠️ 필요하면 런치 때 --env GLOO_SOCKET_IFNAME=<nic> --env NCCL_SOCKET_IFNAME=<nic> 로 직접 준다."
  fi
  if [ -n "$_nic" ]; then
    : "${GLOO_SOCKET_IFNAME:=$_nic}"
    : "${NCCL_SOCKET_IFNAME:=$_nic}"
    export GLOO_SOCKET_IFNAME NCCL_SOCKET_IFNAME
  fi
  # vLLM 도 자체적으로 host IP 를 잡는데 같은 127.0.1.1 함정을 밟는다(verl/slime rollout 경로).
  : "${VLLM_HOST_IP:=$_my_ip}"
  export VLLM_HOST_IP
fi

# IB 장치가 **실제로** 없으면 명시적으로 끈다(자동 탐지가 헤매다 행 걸리는 걸 막는다).
# `compgen` 은 bash 빌트인이라 외부 의존이 없지만, set -e 방어로 조건문 안에서만 쓴다.
if compgen -G "/sys/class/infiniband/*" >/dev/null 2>&1; then
  :   # IB 장치 실재 → NCCL 자동 선택에 맡긴다
else
  : "${NCCL_IB_DISABLE:=1}"
  export NCCL_IB_DISABLE
fi

# vLLM 엔진 준비 타임아웃 — RL 계열(verl/slime rollout)이 vLLM 을 띄운다. 기본 600s 는 **최초 모델
# 다운로드**를 못 버틴다(서빙 repo 실측: 37GB 받다 API 서버가 먼저 죽음. 캐시 후 가중치 로딩은 12초 →
# 600s 가 전부 다운로드에 쓰였다). 우리는 노드마다 4B 를 새로 받으므로 같은 벽에 걸린다.
# 교훈의 일반형: **최초 실행과 재실행은 비용 구조가 다르다** → 타임아웃은 최초 기준으로 잡는다.
: "${VLLM_ENGINE_READY_TIMEOUT_S:=3600}"
export VLLM_ENGINE_READY_TIMEOUT_S

echo "[netenv] NIC=${GLOO_SOCKET_IFNAME:-<미설정>} IP=${VLLM_HOST_IP:-<미설정>} IB_DISABLE=${NCCL_IB_DISABLE:-0}"
