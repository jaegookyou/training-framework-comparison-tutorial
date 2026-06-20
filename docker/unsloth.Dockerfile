# Unsloth SFT(LoRA) 이미지. base 위에 Unsloth 학습 스택을 핀 고정해 얹는다.
#   docker build -f docker/unsloth.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/unsloth:latest .
#
# trl 이미지와 핀 우주가 충돌해 별도 이미지로 분리한다(="1 프레임워크 1 이미지" 근거):
#   unsloth 2026.6.7 제약(PyPI requires_dist 확인): torch<2.11 · trl<=0.24.0 ·
#   transformers<=5.5.0 · peft>=0.18.0 · datasets>=3.4.1,<4.4.0(4.0.*/4.1.0 제외).
#   trl 이미지는 torch 2.12 / trl 1.6 / transformers 5.12 / datasets 5.0.
#
# 베이스 CUDA 가 12.4 → cu124 → unsloth cu124 extra 의 상한인 torch 2.6.0 을 쓴다
# (torch 2.7+ 는 cu126/cu128 extra). 아래 핀은 제약에서 도출한 값이며, 정확한
# torch/xformers/bitsandbytes/vllm 조합 호환은 이미지 빌드 시 최종 검증 필요(GPU 빌드).
#
# vllm 은 GRPO(trainers/unsloth_grpo.py)의 fast_inference rollout 에 필요(SFT/DPO 엔 불필요).
# vllm 휠은 특정 torch 에 박혀 빌드되므로 torch 2.6.0+cu124 와 맞는 버전을 빌드 때 확정한다
# (추정 핀 금지 — `pip index versions vllm` + unsloth GRPO 권장본 교차확인 후 `vllm==X` 박제).
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.6.0" "torchvision" \
    && pip install \
        "unsloth[cu124-torch260]==2026.6.7" \
        "unsloth_zoo==2026.6.5" \
        "transformers==5.5.0" \
        "trl==0.24.0" \
        "peft==0.19.1" \
        "datasets==4.3.0" \
        "accelerate==1.14.0" \
        "vllm"   # TODO(빌드): torch 2.6.0+cu124 호환 버전으로 핀 박제 (위 주석)

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
