# Megatron-LM SFT(순수, full 전용) 이미지. base 위에 Megatron 학습 스택을 얹는다.
#   docker build -f docker/megatron-lm.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/megatron-lm:latest .
#
# "순수 Megatron-LM 으로 사후학습(SFT)" 경로 — verl 백엔드/ms-swift 래퍼가 아니라 Megatron-LM
# repo 의 examples/post_training/modelopt (convert→finetune→export) 를 그대로 쓴다. 기업이 콕
# 집어 요구하는 'Megatron-LM 경험' 스킬. (SFT 트랙의 Megatron 데이터포인트 #1; #2 = megatron-bridge)
#
# 핀 근거: `uv pip compile (megatron-core[mlm]==0.17.1 + torch==2.6.0 + nvidia-modelopt + …)` 해석값(추정 아님).
#   megatron-core 0.17.1 → transformers 5.3.0(mlm extra 상한) · datasets 5.0.0 · accelerate 1.14.0 ·
#   sentencepiece 0.2.1 · tiktoken 0.13.0 · omegaconf 2.3.1 · tensorstore 0.1.84 · nvidia-modelopt 0.43.0.
# megatron 패키지(megatron.core + megatron.training + post_training 스크립트)는 PyPI 가 아니라
# repo 에 있으므로 Megatron-LM repo 를 core_v0.17.1 태그(SHA 266f1c97)로 clone 해 설치한다(=megatron-core PyPI 미설치, 버전 스큐 방지).
#
# transformer-engine: prebuilt 휠(transformer_engine_cu12 바이너리)이라 nvcc 소스 빌드 불필요 →
# CUDA 12.4 runtime 베이스로 OK. 단 torch/TE/megatron-core/modelopt 정확한 조합 호환은
# 이미지 빌드 시 GPU 에서 최종 검증 필요(unsloth 이미지와 같은 단서).
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.6.0" \
    && pip install "transformer-engine[core,pytorch]==2.16.0" \
    && pip install \
        "nvidia-modelopt==0.43.0" \
        "transformers==5.3.0" \
        "datasets==5.0.0" \
        "accelerate==1.14.0" \
        "sentencepiece==0.2.1" \
        "tiktoken==0.13.0" \
        "omegaconf==2.3.1" \
        "tensorstore==0.1.84" \
        "flask-restful==0.3.10" \
        "diskcache"

# Megatron-LM repo (core_v0.17.1) = megatron.core + megatron.training + examples/post_training/modelopt.
ARG MLM_REF=core_v0.17.1
RUN git clone --depth 1 --branch ${MLM_REF} https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM \
    && pip install -e /opt/Megatron-LM
ENV MEGATRON_LM_DIR=/opt/Megatron-LM

# baked 패치 ① traceinversion 데이터셋 변환기 등록 — finetune.py 의 SFTDataset 은 등록된
# HF 데이터셋만 conversation 변환기를 갖고, 미등록이면 identity(행 dict 통째)라 깨진다. 우리
# 데이터는 messages 컬럼이므로 list 를 꺼내는 변환기를 클래스 dict 에 substring 키로 등록한다.
# (import 시 SFTDataset 정의 뒤에 실행되도록 파일 끝에 append)
RUN echo '' >> /opt/Megatron-LM/examples/post_training/modelopt/finetune.py \
    && echo '# tfct: register reasoning-distill (TraceInversion) — messages 컬럼 → conversation list' \
        >> /opt/Megatron-LM/examples/post_training/modelopt/finetune.py \
    && echo 'SFTDataset.hf_dataset_to_conversation["TraceInversion"] = lambda data: data["messages"]' \
        >> /opt/Megatron-LM/examples/post_training/modelopt/finetune.py

# baked 패치 ② conf 가 TOKENIZER_MODEL=HF_MODEL_CKPT 로 덮어써, 우리 캐논 chat template 을 구운
# 토크나이저 디렉토리를 못 가리킨다. 미리 세팅된 TOKENIZER_MODEL 을 존중하도록 한 줄만 완화.
RUN sed -i 's|TOKENIZER_MODEL=${HF_MODEL_CKPT}|TOKENIZER_MODEL=${TOKENIZER_MODEL:-${HF_MODEL_CKPT}}|' \
        /opt/Megatron-LM/examples/post_training/modelopt/conf/Qwen/Qwen3-8B.sh

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
