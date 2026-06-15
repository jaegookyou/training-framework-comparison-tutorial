# TRL SFT 이미지. base 위에 학습 스택만 버전 핀 고정해 얹는다.
#   docker build -f docker/trl.Dockerfile -t ghcr.io/jaegookyou/tfct-trl:latest .
# NOTE: 아래 핀은 출발점. 실제로 한 번 빌드/스모크 후 확정해야 한다(TRL API churn).
ARG BASE_IMAGE=ghcr.io/jaegookyou/tfct-base:latest
FROM ${BASE_IMAGE}

RUN pip install \
        "torch==2.5.1" \
        "transformers==4.46.3" \
        "trl==0.12.2" \
        "datasets==3.1.0" \
        "accelerate==1.1.1" \
        "peft==0.13.2"
