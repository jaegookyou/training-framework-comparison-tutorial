# Megatron-Bridge SFT(full|lora) 이미지 = SFT 트랙의 Megatron 데이터포인트 #2.
#   docker build -f docker/megatron-bridge.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/megatron-bridge:latest .
#
# NeMo Megatron-Bridge = HF↔Megatron-core 브리지 + 파이썬 학습 레시피. 순수 Megatron-LM(modelopt
# 셸 워크플로)과 학습 루프는 같은 Megatron 스택이지만 진입 경로가 달라(레시피 + AutoBridge.import_ckpt
# + finetune()) 좋은 비교축이 된다. Bridge 는 네이티브 PEFT(LoRA/DoRA) 라 full|lora 둘 다 된다.
#
# ⚠️ 다른 프레임워크 이미지와 달리 base(cu12 *runtime*) 가 아니라 cu12 *devel*(nvcc 포함) 에서
# 출발한다 — megatron-bridge 필수 deps 인 mamba-ssm/causal-conv1d/flash-linear-attention 이 sdist 라
# 컴파일에 nvcc 가 필요하기 때문(재국 결정: devel base 전체 빌드, 누락 회피). 그래서 base 이미지
# 레이어(python3.12 + hf/wandb + repo)를 여기서 재구성한다(FROM base 를 못 쓴다).
#
# 핀 근거: `uv pip compile (megatron-bridge==0.4.2 + torch==2.6.0, py3.12)` 해석 그래프(추정 아님).
#   megatron-core 0.17.1 · transformer-engine 2.16.0 · transformers 5.3.0 · datasets 5.0.0 ·
#   accelerate 1.14.0 · peft 0.19.1 · hydra-core 1.3.2 · nvidia-modelopt 0.43.0 → Megatron-LM 이미지와
#   *동일 코어 스택*(megatron-core 0.17.1 / TE 2.16.0 / torch 2.6 / transformers 5.3) 으로 수렴 = 정합.
#   mamba-ssm 2.3.1 · causal-conv1d 1.6.2.post1 · flash-linear-attention 0.4.2 는 hybrid-Mamba 용 →
#   dense Qwen3-8B SFT 는 런타임에 안 쓰지만 bridge 필수 dep 라 빌드해 둔다.
#
# transformer-engine: bridge 기본은 [core_cu13](CUDA 13) 이지만 우리 베이스가 cu12 → cu12 prebuilt
#   ([core,pytorch]==2.16.0, transformer_engine_cu12 바이너리)을 먼저 깔고, bridge 가 끌어온 cu13 휠은
#   제거해 cu12 백엔드로 고정한다.
#
# ⚠️ GPU 빌드 검증 대기(verl/megatron-lm 이미지와 같은 단서): ① mamba-ssm/causal-conv1d 소스 빌드
#   (메모리 부담 → MAX_JOBS 로 제한, torch 가 build 에 보여야 해 --no-build-isolation) ② torch/TE/
#   megatron-core/mamba 정확 조합 호환 ③ cu13 제거 후 cu12 백엔드 정상 동작.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- base 이미지 레이어 재구성 (devel 베이스라 FROM base 불가) ---
# python·python3 둘 다 3.12 로 건다: base 규약은 python3=3.10(시스템) 이지만, megatron-core 의 빌드
# 훅(setup)이 `python3 -m pybind11 --includes` 를 하드코딩해 include 경로를 찾는다 → python3 가 3.10 이면
# pybind11(3.12 에 설치) 을 못 찾아 metadata-generation-failed. 또 3.10 으로 빌드하면 확장 include 경로도
# 어긋난다(휠은 3.12 타깃). 이 이미지는 전부 3.12 로 도니 apt 완료 후 python3→3.12 로 통일해 안전.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN pip install --upgrade pip huggingface_hub wandb

# --- 학습 스택 ---
# torch + cu12 TE 를 먼저 고정(cu13 기본 override). build 도구(mamba/causal-conv1d sdist 용)도 준비.
RUN pip install "torch==2.6.0" "ninja" "packaging" "setuptools" "wheel" "pybind11" \
    && pip install "transformer-engine[core,pytorch]==2.16.0"

# --ignore-installed blinker = devel base(ubuntu22.04) 의 software-properties-common 이 깐 시스템
#   blinker 1.4(distutils) 를 megatron-bridge 의 flask-restful→Flask 가 업그레이드하려다 'Cannot
#   uninstall'(distutils-installed) 로 실패 → pip 로 새 blinker 를 강제 설치해 시스템 것을 shadow.
#   (megatron-lm 이미지와 동일 패턴.)
RUN pip install --ignore-installed blinker

# megatron-bridge 본체. --no-build-isolation = mamba-ssm/causal-conv1d 가 build 시 위 torch 를 보게.
# MAX_JOBS 제한으로 mamba 컴파일 OOM 회피(빌드 노드 메모리 보호).
ENV MAX_JOBS=4
RUN pip install --no-build-isolation "megatron-bridge==0.4.2"

# bridge 가 [core_cu13] 로 끌어온 cu13 TE 휠 제거 → cu12 백엔드로 고정(python 패키지 transformer_engine
# 은 백엔드 무관, cu12 바이너리가 위에서 이미 설치됨).
RUN pip uninstall -y transformer-engine-cu13 || true

# Megatron-LM repo (scripts 전용) — continued-pretrain 의 학습 루프는 순수 pretrain_gpt.py 가 돈다.
# pretrain_gpt.py·tools/preprocess_data.py 는 패키지 모듈이 아니라 repo 루트 스크립트라 pip 로 안
# 깔린다 → clone 만 한다. **pip install 하지 않음**: megatron.core/megatron.training 은 위 megatron-bridge
# 가 끌어온 PyPI megatron-core 0.17.1 에 이미 있고(같은 태그), 재설치하면 namespace 스큐가 생긴다.
# (megatron-bridge SFT 는 이 clone 을 안 써 무영향 — additive.)
ARG MLM_REF=core_v0.17.1
RUN git clone --depth 1 --branch ${MLM_REF} https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM
ENV MEGATRON_LM_DIR=/opt/Megatron-LM

WORKDIR /workspace/repo
COPY . .
RUN pip install .

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
