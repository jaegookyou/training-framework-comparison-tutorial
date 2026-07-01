# Megatron-LM SFT(순수, full 전용) 이미지. cu12 devel(nvcc) 위에 Megatron 학습 스택을 얹는다.
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
# ⚠️ base(cu12 *runtime*) 가 아니라 cu12 *devel*(nvcc 포함) 에서 출발한다 — megatron-bridge 와 동일:
#   transformer-engine[core,pytorch] 는 core(transformer_engine_cu12)만 prebuilt 고 pytorch 바인딩
#   (transformer_engine_torch)은 sdist 라 nvcc 소스 빌드가 필요하다(runtime 베이스에서 'Could neither find
#   NVCC' 로 실패). 그래서 base 이미지 레이어(python3.12 + hf/wandb + repo)를 여기서 재구성한다(FROM base 불가).
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- base 이미지 레이어 재구성 (devel 베이스라 FROM base 불가) ---
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN pip install --upgrade pip huggingface_hub wandb

# --- 학습 스택 ---
# MAX_JOBS 제한 = TE pytorch 바인딩 nvcc 컴파일의 병렬 job 을 줄여 빌드 노드 OOM 회피(megatron-bridge 와 동일).
ENV MAX_JOBS=4
RUN pip install "torch==2.6.0" "ninja" "packaging" "setuptools" "wheel" \
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

# RL(examples/rl, 네이티브 GRPO) 추가 deps. Megatron-LM 은 한 프레임워크 = 한 이미지지만 두
# 사후학습 진입점(SFT=post_training/modelopt, GRPO=examples/rl)을 함께 담는다. examples/rl README
# 가 요구하는 uvloop·evaluate + math_agent.py 의 import-time assert 가 요구하는 math-verify
# (우리 TfctGSM8KAgent 는 get_reward 를 오버라이드해 실제 math_verify 채점은 안 쓰지만 import 됨).
# flask-restful·datasets·diskcache 는 위에서 이미 설치됨. 핀 = PyPI 확인값(추정 아님; evaluate 는
# datasets>=2.0 만 요구 → 위 5.0.0 과 호환).
RUN pip install "uvloop==0.22.1" "evaluate==0.4.6" "math-verify==0.9.0"

# repo 설치(코어=pyyaml 만, torch 무영향) — 무거운 스택 뒤에 둬서 코드 수정이 위 레이어 캐시를 안 깬다.
WORKDIR /workspace/repo
COPY . .
RUN pip install .

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
