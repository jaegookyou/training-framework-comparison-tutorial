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
# vllm==0.8.5.post1: torch==2.6.0(=base cu124 정합) · ray>=2.43,!=2.44.* · py<3.13(base 3.12).
#
# ⚠️ **transformers 는 4.57.6 이다(5.12.1 에서 내림, 2026-07-28 GPU 실측 후 정정).**
# 원래 근거는 "vllm 이 transformers>=4.51.1 만 요구하고 **상한이 없으니** 5.12.1 도 충족"이었다.
# 그런데 2노드 GRPO 런에서 vLLM 서버 기동이 이걸로 죽었다:
#   `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
# (transformers v5 가 제거한 속성을 vllm 0.8.5 가 호출).
# **교훈: `requires_dist` 의 상한 부재는 호환 보증이 아니다** — 패키지가 *선언*한 것과 실제로
# *동작*하는 범위는 다르다. 하한만 있는 제약은 가장 약한 신호다. [[dont-guess-package-versions]]
# 확인 사실(PyPI 실물, 2026-07-28): verl 0.8.0 의 [vllm] extra 는 `vllm>=0.8.5,<=0.12.0` 인데
# 그 **상한인 vllm 0.12.0 이 `transformers<5` 를 명시**한다 → verl 이 허용하는 vllm 범위 전체에
# transformers 5 지원본이 없다. 그래서 vllm 을 올리는 게 아니라 transformers 를 4.x 로 내린다.
# 4.57.6 = 마지막 4.x(tokenizers<=0.23.0 요구, vllm 의 tokenizers>=0.21.1 과 정합).
# verl 0.8.0 자체는 `transformers`(제약 없음)라 4.x 로 내려도 메타데이터상 문제 없다.
# ⚠️ verl SFT 는 transformers 5.12.1 에서 2노드 통과했었다 → 이 다운그레이드 후 **재검증 필요**.
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.6.0" \
    && pip install \
        "verl==0.8.0" \
        "transformers==4.57.6" \
        "vllm==0.8.5.post1"  # torch==2.6.0 핀(=base cu124). transformers 는 위 주석의 4.x 근거 참조

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
