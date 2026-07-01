# Unsloth SFT(LoRA) 이미지. base 위에 Unsloth 학습 스택을 핀 고정해 얹는다.
#   docker build -f docker/unsloth.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/unsloth:latest .
#
# trl 이미지와 핀 우주가 충돌해 별도 이미지로 분리한다(="1 프레임워크 1 이미지" 근거):
#   unsloth 2026.6.7 제약(PyPI requires_dist 확인): torch<2.11 · trl<=0.24.0 ·
#   transformers<=5.5.0 · peft>=0.18.0 · datasets>=3.4.1,<4.4.0(4.0.*/4.1.0 제외).
#   trl 이미지는 torch 2.12 / trl 1.6 / transformers 5.12 / datasets 5.0.
#
# 베이스 CUDA 가 12.4 → cu124 → torch 2.6.0(cu124 정합) 을 명시 핀한다(torch 2.7+ 는 cu126/cu128).
# ⚠️ unsloth 의 [cu124-torch260] extra 는 쓰지 않는다(=plain unsloth): 이 extra 는
# unsloth[cu124onlytorch260] 을 끌어 xformers==0.0.29.post3 을 하드 핀하는데, torch 2.6.0 용 vllm
# (0.8.x)은 전부 xformers==0.0.29.post2 을 핀해 정확충돌(ResolutionImpossible)한다. plain unsloth 의
# 무조건 제약은 xformers>=0.0.27.post2(느슨)이라 vllm 의 post2 가 이를 만족 → 양립(PyPI 확인, 추정 아님).
# bitsandbytes(>=0.45.5)도 plain unsloth 무조건 dep 라 별도 명시 불필요.
#
# vllm 은 GRPO(trainers/unsloth_grpo.py)의 fast_inference rollout 에 필요(SFT/DPO 엔 불필요).
# vllm 휠은 특정 torch 에 박혀 빌드되므로 torch 2.6.0 과 맞는 0.8.5.post1 로 핀(=verl 이미지와 동일본):
# torch==2.6.0 핀 + cp38-abi3 prebuilt 휠(manylinux) → nvcc 소스빌드 불필요. unsloth 2026.6.7·
# unsloth_zoo 2026.6.5 는 vllm 하드제약 없음(PyPI requires_dist 확인 — 추정 아님). [[dont-guess-package-versions]]
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.6.0" "torchvision" \
    && pip install \
        "unsloth==2026.6.7" \
        "unsloth_zoo==2026.6.5" \
        "transformers==5.5.0" \
        "trl==0.24.0" \
        "peft==0.19.1" \
        "datasets==4.3.0" \
        "accelerate==1.14.0" \
        "vllm==0.8.5.post1"   # torch 2.6.0 정합 prebuilt 휠(=verl 동일본) → nvcc 불필요. 위 주석 참조

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
