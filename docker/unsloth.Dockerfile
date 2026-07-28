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
# ⚠️ **transformers 는 4.57.6 이다(5.5.0 에서 내림, 2026-07-28 verl 이미지와 같은 이유로 선제 정정).**
# vllm 0.8.5.post1 은 transformers v5 에서 제거된 `all_special_tokens_extended` 를 호출한다
# (verl 이미지가 2노드 GRPO 에서 실측으로 밟음). unsloth GRPO 도 fast_inference=vllm 이라 같은 벽이다
# → **실측 전에 막는다.** 제약 교집합(PyPI 실물 확인, 2026-07-28):
#   unsloth 2026.6.7 / unsloth_zoo 2026.6.5: >=4.51.3,<=5.5.0 (4.52.0-3·4.53.0·4.54.0·4.55.0-1·
#     4.57.0·4.57.4·4.57.5·5.0.0·5.1.0 제외) · trl 0.24.0: >=4.56.1 · vllm 0.8.5.post1: >=4.51.1
#   → 4.57.6 이 세 제약을 모두 만족(어느 제외 목록에도 없음). verl 이미지와 같은 버전으로 맞춰
#     환경 축 변동을 줄인다.
#
#
# ⚠️ **torch 2.9.0 = Blackwell(sm_120) 하한**(2026-07-28 GPU 실측 후 2.6.0 에서 상향).
# 우리가 빌리는 GPU 는 Nebius **RTX PRO 6000 Blackwell Server Edition(sm_120, 96GB, driver 580)**
# 인데 torch 2.6.0 은 cu12.4 빌드라 그 아키텍처 커널이 없다 → 2노드 런이 NCCL barrier 에서
# `CUDA error: no kernel image is available for execution on the device` 로 죽었다.
# PyPI 실물 확인: torch 2.6.0=cu12.4/nccl 2.21.5 · **torch 2.8.0·2.9.0=cu12.8/nccl 2.27.x** ·
# torch 2.12.0=cu13(trl 이미지가 같은 GPU 에서 2노드 통과 = Blackwell 지원 실증).
# 왜 여태 몰랐나: 07-01 이미지는 Dockerfile 이 `"vllm"` 을 **핀 없이** 깔아 최신 vllm 이 torch 를
# 끌어올렸다. `torch==2.6.0` 핀은 커밋돼 있었지만 **오늘 처음 빌드**됐고, 그제서야 드러났다.
# 교훈: **핀은 빌드돼야 검증된 것이다** — 커밋된 핀과 이미지 안 실물은 다를 수 있다.
#
# unsloth 는 `torch<2.11` 제약이 있어 2.9.0 이 그 안에 들어간다(PyPI 확인). vllm 은 torch 2.9.0 을
# 핀하는 0.12.0 으로 맞춘다 — verl 이미지와 같은 조합.
#
# vllm 은 GRPO(trainers/unsloth_grpo.py)의 fast_inference rollout 에 필요(SFT/DPO 엔 불필요).
# vllm 휠은 특정 torch 에 박혀 빌드되므로 torch 2.6.0 과 맞는 0.8.5.post1 로 핀(=verl 이미지와 동일본):
# torch==2.6.0 핀 + cp38-abi3 prebuilt 휠(manylinux) → nvcc 소스빌드 불필요. unsloth 2026.6.7·
# unsloth_zoo 2026.6.5 는 vllm 하드제약 없음(PyPI requires_dist 확인 — 추정 아님). [[dont-guess-package-versions]]
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.9.0" "torchvision" \
    && pip install \
        "unsloth==2026.6.7" \
        "unsloth_zoo==2026.6.5" \
        "transformers==4.57.6" \
        "trl==0.24.0" \
        "peft==0.19.1" \
        "datasets==4.3.0" \
        "accelerate==1.14.0" \
        "vllm==0.12.0"   # torch 2.9.0(cu12.8=Blackwell) 정합 prebuilt 휠(=verl 동일본) → nvcc 불필요

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
