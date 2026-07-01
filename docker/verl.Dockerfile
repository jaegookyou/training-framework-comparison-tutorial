# verl SFT(FSDP) 이미지. base 위에 verl 학습 스택을 핀 고정해 얹는다.
#   docker build -f docker/verl.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/verl:latest .
#
# 핀 근거: `uv pip compile (verl==0.8.0 + torch==2.6.0, py3.12)` 로 해석한 그래프(추정 아님).
#   verl 0.8.0 → transformers 5.12.1 · tensordict 0.10.0 · datasets 5.0.0 · accelerate 1.14.0 ·
#   peft 0.19.1 · hydra-core 1.3.3 · ray 2.55.1 · numpy 1.26.4 · pyarrow 24.0.0 · torchdata 0.11.0.
#   아래는 우리가 직접 잡는 핀(torch/verl/transformers)이고 나머지는 위 그래프대로 transitive 해석.
#
# torch 2.6.0 = base 의 CUDA 12.4 와 정합(기본 휠이 cu124). 단일 노드 FSDP SFT + GRPO 까지가 범위.
#
# flash-attn/liger(verl `[gpu]` extra)는 일부러 빼둔다: flash-attn 빌드는 CUDA devel 툴체인
# (nvcc)이 필요해 runtime 베이스로는 안 깔린다. trainers/verl_sft.py 가 model.use_remove_padding
# =false 로 sdpa 어텐션을 쓰므로 없이도 돈다. remove-padding 최적화가 필요하면 devel 베이스 +
# 매칭 flash-attn 휠로 별도 빌드(GPU 빌드 시 최종 검증 대상 — unsloth 이미지와 같은 단서).
#
# vllm: GRPO(trainers/verl_grpo.py)는 rollout.name=vllm 이 기본이라 vllm 이 필요하다(SFT 엔 불필요).
# vllm==0.8.5.post1 로 핀(2026-06-26 PyPI requires_dist 확인 — 추정 아님): torch==2.6.0(=base cu124
# 정합) · transformers>=4.51.1(verl 핀 5.12.1 충족, 상한 없음) · ray>=2.43,!=2.44.*(verl 핀 2.55.1
# 충족) · py<3.13(base 3.12). verl 0.8.0 과 같은 0.8 마이너. [[dont-guess-package-versions]].
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.6.0" \
    && pip install \
        "verl==0.8.0" \
        "transformers==5.12.1" \
        "vllm==0.8.5.post1"  # torch==2.6.0 핀(=base cu124) · transformers>=4.51.1(verl 충족) — PyPI 확인 06-26

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
