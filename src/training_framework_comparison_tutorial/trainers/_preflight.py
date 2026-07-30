"""학습 시작 전 환경 preflight — **이미지 안 실물**이 이 GPU 에서 돌 수 있는지 먼저 본다.

왜 필요한가(2026-07-28 실측): verl 2노드 런이 12 분을 태우고 `torch.distributed.barrier()` 에서
  `CUDA error: no kernel image is available for execution on the device`
로 죽었다. 원인은 **torch 2.6.0(cu12.4) 이 우리 GPU(RTX PRO 6000 Blackwell, sm_120)의 커널을 안
가진 것**이었는데, 이 메시지는 원인을 말해주지 않는다(torch 는 arch 경고를 안 찍었다). 하드웨어를
특정하는 데 별도 진단 런이 필요했다.

**더 근본적인 문제**: `torch==2.6.0` 핀은 Dockerfile 에 커밋돼 있었지만 07-01 이미지는 vllm 을 핀
없이 깔아 torch 가 끌어올려져 있었다 → **repo 가 안다고 믿은 환경 ≠ 이미지 안 실물**. 커밋을 아무리
읽어도 알 수 없고 GPU 위에서만 드러난다. 그래서 검사는 **런타임에, 실물을 상대로** 해야 한다.

이 모듈이 보는 것은 torch 가 컴파일해 넣은 arch 목록과 실제 장치의 compute capability 뿐이다.
싸고(수 밀리초) 원인을 정확히 지목한다. 한계도 분명히 해둔다:
  - NCCL·vLLM 등 **다른 라이브러리의 커널 커버리지는 못 본다**(오늘 실패는 NCCL 에서 났다).
    다만 torch 와 NCCL 은 같은 휠 세트로 함께 오므로 torch 가 못 하면 NCCL 도 대개 못 한다.
  - PTX(`compute_XX`)가 있으면 JIT 로 상위 arch 에서 돌 수 있어 "가능"으로 본다(느릴 수 있음).
"""

from __future__ import annotations

import os


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """이 장치(capability)가 torch 의 arch_list 로 커버되나. 순수 함수 — 테스트가 여기를 본다.

    arch_list 예: ['sm_50', 'sm_80', 'sm_90', 'compute_90'].
      - `sm_{XY}` 정확 일치  → 네이티브 커널 있음.
      - `compute_{XY}` 가 장치보다 **낮거나 같으면** → PTX 를 상위 arch 로 JIT 할 수 있다.
    """
    major, minor = capability
    device = major * 10 + minor
    for entry in arch_list:
        kind, _, num = entry.partition("_")
        if not num.isdigit():
            continue
        value = int(num)
        if kind == "sm" and value == device:
            return True
        if kind == "compute" and value <= device:
            return True
    return False


def check_gpu_arch() -> None:
    """GPU 가 있는데 torch 가 그 arch 를 못 돌리면 **즉시** 죽는다(원인을 적어서).

    GPU 가 없으면(로컬·CPU 스모크) 조용히 넘어간다 — 이 검사의 대상이 아니다.
    `TFCT_SKIP_ARCH_CHECK=1` 로 끌 수 있다(PTX JIT 를 감수하고 굳이 돌려볼 때).
    """
    if os.environ.get("TFCT_SKIP_ARCH_CHECK") == "1":
        return
    try:
        import torch
    except ImportError:
        return                                   # torch 없는 환경(로컬 dev) — 검사 대상 아님
    if not torch.cuda.is_available():
        return

    capability = torch.cuda.get_device_capability()
    arch_list = list(torch.cuda.get_arch_list())
    if arch_supported(capability, arch_list):
        return

    name = torch.cuda.get_device_name()
    sm = f"sm_{capability[0]}{capability[1]}"
    raise SystemExit(_incompatible_message(name, sm, arch_list, torch))


def report() -> int:
    """이 이미지가 이 GPU 에서 도는지 **보고만** 한다(죽이지 않음).

    `python -m ..._preflight` 진입점.

    왜 별도 진입점인가: 이미지마다 torch 출처가 다르다 — 우리가 핀하는 것(trl·verl·unsloth·megatron)
    도 있고 **upstream 이미지에서 그냥 오는 것**(slime=slimerl/slime)도 있다. 후자는
    "핀을 읽어서" 알 수 없고 **띄워봐야 안다**(2026-07-28 교훈의 연장). 학습을 12분 태우며 하나씩
    발견하는 대신, 이미지당 수 분짜리 최소 런으로 한 번에 확정하려고 둔다.
    """
    try:
        import torch
    except ImportError:
        print("[preflight] torch 없음 — 이 이미지에서 torch import 실패")
        return 2
    print(f"[preflight] torch      : {torch.__version__} (cuda {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("[preflight] CUDA 사용 불가 — GPU 가 안 보인다")
        return 2
    capability = torch.cuda.get_device_capability()
    arch_list = list(torch.cuda.get_arch_list())
    name = torch.cuda.get_device_name()
    sm = f"sm_{capability[0]}{capability[1]}"
    print(f"[preflight] GPU        : {name} ({sm})")
    print(f"[preflight] torch arch : {arch_list}")
    if arch_supported(capability, arch_list):
        print(f"[preflight] 판정       : OK — {sm} 실행 가능")
        return 0
    print(f"[preflight] 판정       : NG — {sm} 커널 없음(이 이미지로는 이 GPU 에서 학습 불가)")
    return 1


def _incompatible_message(name: str, sm: str, arch_list: list[str], torch) -> str:
    return (
        f"\n[preflight] 이 이미지의 torch 는 이 GPU 에서 커널을 실행할 수 없다.\n"
        f"  GPU        : {name} ({sm})\n"
        f"  torch      : {torch.__version__} (cuda {torch.version.cuda})\n"
        f"  torch arch : {arch_list}\n"
        f"  → {sm} 이 위 목록에 없다. 그대로 두면 한참 뒤 collective 에서\n"
        f"    'CUDA error: no kernel image is available' 로 죽는다\n"
        f"    (원인을 안 알려주는 메시지라 진단이 비싸다 — 그래서 여기서 막는다).\n"
        f"  조치: 이 GPU 를 지원하는 torch 로 이미지를 재빌드(예: Blackwell=cu12.8 이상)하거나,\n"
        f"        지원되는 GPU 로 대여를 바꾼다.\n"
        f"  참고: 커밋된 핀과 **이미지 실물이 다를 수 있다** — 위 torch 버전이 진실이다.\n"
    )


if __name__ == "__main__":                       # `python -m ..._preflight` = 이미지 진단 모드
    raise SystemExit(report())
