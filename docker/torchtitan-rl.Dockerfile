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
# ✅ GHCR 빌드 성공(2026-07-01, Actions build-images). 검증된 해석 세트(uv, cu130 nightly):
#   torch==2.14.0.dev20260620+cu130 · vllm==1.0.0.dev20260620+cu130 (torch/vllm dev 날짜 20260620 정합) ·
#   torchcomms==0.3.0.dev20260621+cu130 · flash-attn-3==3.0.0 · torchmonarch==0.5.0 · renderers==0.1.8.dev53 ·
#   torchstore==0.0.0.dev0(git@main) · triton==3.7.1+git5d6048aa.
#   ⚠️ nightly/git@main 휠은 pruning 으로 증발한다 → 하드핀하면 재빌드가 언젠가 깨진다. 그래서 `--pre` 무핀을
#   유지해 최신 nightly 로 재빌드되게 두고, **재현 아티팩트는 불변 이미지 태그**(build-images 가 :latest 와
#   :YYYYMMDD 를 함께 push)로 박제한다 — "환경=이미지 핀" 원칙(휠이 아니라 이미지가 진실).

# cu130 devel base(13.0.1 태그 고정) — torch/vllm cu130 nightly 휠과 정합 확인됨(2026-07-01 빌드 성공).
ARG BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3.12 python3.12-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED   # ubuntu24.04 PEP668: 시스템 pip/uv --system 설치 허용

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

# (4) PyTorch nightly + vLLM(nightly torch 정합) + torchcomms — cu130. 무핀 `--pre`(pruning 증발 대비 —
#     상단 검증 세트 참고, 재현은 불변 이미지 태그로). vllm dev 날짜는 torch nightly 와 자동 정합(20260620).
RUN uv pip install --system torch vllm torchcomms --pre \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
        --index-strategy unsafe-best-match

# (5) 우리 패키지(컨테이너 전용 torchtitan_rl.gsm8k 모듈 포함)는 런타임에 editable 설치(sky setup).
#     manager.py 가 --module <FQN> 로 우리 모듈을 import(소스트리 bake 불필요).

LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
