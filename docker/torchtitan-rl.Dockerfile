# torchtitan RL 이미지 (GRPO·experiments/rl) — SFT/사전학습 이미지(cu124)와 **별도 스택**.
#   docker build -f docker/torchtitan-rl.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/torchtitan-rl:latest .
#
# torchtitan 의 RL(experiments/rl)은 Monarch actors + TorchStore weight-sync + vLLM rollout +
# flash-attn-3 / **cu130 nightly** 를 요구한다 → torchtitan SFT 이미지(cu124)와 코어 CUDA/torch
# 버전 자체가 달라 한 프레임워크여도 이미지를 가른다(megatron 식 '한 이미지 여러 진입점' 불가).
#
# 설치 단계 = upstream README(torchtitan/experiments/rl) 그대로 미러(추정 금지). torchtitan 은
# 같은 커밋 SHA(SFT 이미지와 동일 9ccbb57)로 clone — RL 코드/우리 gsm8k 모듈이 같은 트리.
#
# ⚠️ GPU 빌드 검증 대기(다른 미빌드 이미지와 동일 단서): cu130 base 정확 태그·flash-attn-3/vllm
#   nightly 휠 호환(dev 날짜 정합)·Monarch/TorchStore/renderers main 빌드. 빌드 성공분을 불변 태그로
#   박제(휠 증발 대비). 미빌드 = 핀 미확정(아래 TODO).

# ⚠️ cu130 devel base — 정확 태그는 GPU 빌드 때 확정(torch/vllm cu130 휠과 정합). TODO: 박제 시 핀.
ARG BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3.12 python3.12-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# (1) torchtitan @ SFT 이미지와 동일 SHA — RL 코드(experiments/rl) + 우리 gsm8k 모듈이 같은 트리.
ARG TT_REF=9ccbb57b0e5df6d7fa56287181eed5d55891fcd0
RUN git clone https://github.com/pytorch/torchtitan.git /opt/torchtitan \
    && cd /opt/torchtitan && git checkout ${TT_REF} && pip install -e .
ENV TORCHTITAN_DIR=/opt/torchtitan

# (2) RL deps = README 단계 미러. Monarch(controller) · TorchStore(weight sync) · renderers(prompt
#     렌더, env 가 import) · pygtrie/portpicker.
RUN uv pip install --system torchmonarch \
    && uv pip install --system --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main" \
    && uv pip install --system pygtrie portpicker \
    && uv pip install --system "git+https://github.com/PrimeIntellect-ai/renderers.git@main"

# (3) Flash Attention 3 (H100/H200+; A100 은 PyTorch 번들 FA2 자동). cu130 test 인덱스.
RUN uv pip install --system flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130

# (4) PyTorch nightly + vLLM(nightly torch 정합) + torchcomms — cu130. ⚠️ vllm dev 날짜는 torch
#     nightly 와 맞아야 함(README) → GPU 빌드 때 교차확인 후 핀(현재 미핀). TODO.
RUN uv pip install --system torch vllm torchcomms --pre \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
        --index-strategy unsafe-best-match

# (5) 우리 패키지(컨테이너 전용 torchtitan_rl.gsm8k 모듈 포함)는 런타임에 editable 설치(sky setup).
#     manager.py 가 --module <FQN> 로 우리 모듈을 import(소스트리 bake 불필요).

LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
