# 공통 베이스: CUDA + python3.12 + 운영 CLI(hf/wandb) + 이 repo.
# 프로비저닝은 SkyPilot 이 박스 밖에서 처리 → 이미지 안엔 vast 도구가 없다.
# 프레임워크별 이미지가 이걸 FROM 해서 학습 deps 만 얹는다.
#   docker build -f docker/base.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest .
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN pip install --upgrade pip huggingface_hub wandb

WORKDIR /workspace/repo
COPY . .
RUN pip install .

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
