# NeMo-RL 이미지 — 공식 NGC prebuilt(nvcr.io/nvidia/nemo-rl) 위에 우리 패키지만 얹는다.
#   docker build -f docker/nemo-rl.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/nemo-rl:latest .
#
# slime 과 같은 이유로 우리 base 위에 안 쌓는다 — NeMo-RL 은 Python 3.13 + uv 잠금(uv.lock)으로
# torch(cu13x)/vllm/megatron/TE 거대 스택을 박는 프레임워크라 runtime 베이스로 재현 불가능하고,
# NVIDIA 가 공식 NGC 이미지로 배포한다. "환경=이미지 핀" 철학대로 공식 이미지를 베이스로 박고
# (dep 재발명 안 함), 코드는 sky 가 런타임에 설치한다.
#
# 핀: nvcr.io/nvidia/nemo-rl:v0.5.0 (README 의 공식 릴리스 컨테이너). GPU 빌드 때 repo 내부 경로
# (NEMO_RL_DIR)·venv·정확 tag 확정(slime/torchtitan 이미지와 같은 단서 — [[torchtitan-sft-nightly-only]]).
#
# ⚠️ venv 주의: NeMo-RL 은 uv 가상환경 기반(uv run). 우리 패키지(trainers·nemo_rl_env·adapters)를
# 그 환경에 설치해야 커스텀 환경 FQN(get_object) import 와 launch 모듈이 동작한다. sky setup 에서
# `uv pip install -e .`(또는 컨테이너 python 에 직접) — 정확한 설치 경로는 GPU 빌드 때 확정.
ARG NEMO_RL_IMAGE=nvcr.io/nvidia/nemo-rl:v0.5.0
FROM ${NEMO_RL_IMAGE}

# trainers/_nemo_rl_common.py·nemo_rl_env.launch 가 참조하는 repo 경로(공식 이미지 기준 — GPU 빌드
# 때 확인 후 고정). examples/run_{sft,dpo,grpo,ppo}.py·examples/configs/*.yaml 이 여기 있어야 한다.
ENV NEMO_RL_DIR=/opt/nemo-rl

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
