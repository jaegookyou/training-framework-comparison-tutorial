# Unsloth SFT(LoRA) 이미지. base 위에 Unsloth 학습 스택을 핀 고정해 얹는다.
#   docker build -f docker/unsloth.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/unsloth:latest .
#
# trl 이미지와 핀 우주가 충돌해 별도 이미지로 분리한다(="1 프레임워크 1 이미지" 근거):
#   unsloth 2026.6.7 제약(PyPI requires_dist 확인): torch<2.11 · trl<=0.24.0 ·
#   transformers<=5.5.0 · peft>=0.18.0. trl 이미지는 torch 2.12 / trl 1.6 / transformers 5.12.
#
# 베이스 CUDA 가 12.4 → cu124 → unsloth cu124 extra 의 상한인 torch 2.6.0 을 쓴다
# (torch 2.7+ 는 cu126/cu128 extra). 아래 핀은 제약에서 도출한 값이며, 정확한
# torch/xformers/bitsandbytes 조합 호환은 이미지 빌드 시 최종 검증 필요(GPU 빌드).
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.6.0" "torchvision" \
    && pip install \
        "unsloth[cu124-torch260]==2026.6.7" \
        "unsloth_zoo==2026.6.5" \
        "transformers==5.5.0" \
        "trl==0.24.0" \
        "peft==0.19.1" \
        "datasets==5.0.0" \
        "accelerate==1.14.0"

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
