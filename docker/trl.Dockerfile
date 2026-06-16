# TRL SFT 이미지. base 위에 학습 스택만 버전 핀 고정해 얹는다.
#   docker build -f docker/trl.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/trl:latest .
# 핀 기준: 2026-06-15 로컬 CPU 스모크로 trainers/trl_sft.py 경로(SFTConfig.max_length,
# SFTTrainer.processing_class)가 import~train 루프 진입까지 도는 걸 확인한 조합.
# torch 는 같은 버전의 CUDA 휠이 잡힌다(로컬 검증은 +cpu 변형).
# peft 는 tuning=lora 경로(trainers/trl_sft.py 의 LoraConfig)용.
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install \
        "torch==2.12.0" \
        "transformers==5.12.0" \
        "trl==1.6.0" \
        "datasets==5.0.0" \
        "accelerate==1.14.0" \
        "peft==0.19.1"

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
