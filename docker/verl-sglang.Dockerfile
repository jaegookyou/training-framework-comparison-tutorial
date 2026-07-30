# verl GRPO/PPO — sglang rollout 변종. ★2026-07-30 lmsysorg/sglang 공식 base 피벗.
#   docker build -f docker/verl-sglang.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/verl-sglang:latest .
#
# 왜 sglang rollout: 기본 verl 이미지(torch 2.8 + vllm 0.11)의 vLLM async weight-sync 가 Blackwell
# 3회차에 하드크래시(2026-07-29 6회 iteration 규명 — 메모리·cudagraph·LoRA·sync모드·hf 배제,
# verl_grpo.py NOTE). sglang 은 verl 1급 백엔드로 vLLM async weight-sync 경로를 안 탄다. sglang 0.5.8
# 이 torch==2.9.1 하드핀 → 기본 verl(torch 2.8)과 공존 불가라 별 이미지.
#
# ★ 왜 공식 base 피벗: sglang 을 우리 base 로 손조립하면 sgl-kernel/flashinfer/시스템libs 정합을 우리가
# 맞춰야 하고 07-29 에 sgl-kernel↔sglang_kernel 이름 스큐로 하루 태웠다(megatron-bridge 의 TE 스큐와 동형).
# lmsysorg/sglang:v0.5.8.post1-cu130 = torch 2.9.1+cu130 · sglang 0.5.8 · sgl-kernel · flashinfer ·
# libnuma/libibverbs 가 통째 정합(lmsys 가 빌드·테스트). Docker Hub public(익명 pull). 우리는 verl 코어 +
# flash-attn + 글루만 얹고 sglang/torch 는 base 것을 존중한다. [[use-canonical-official-methods]]
#
# verl deps(PyPI 0.8.0 실물 확인): **코어는 torch 미핀**(torch 2.9.1 은 sglang extra 에서만) →
# verl[sglang] extra 안 쓰고 코어만 깔면 base 의 torch/sglang 를 안 건드린다. constraints 로 torch·
# sglang·transformers 를 base 버전에 못박아 pip 가 verl deps 해석 중 바꾸지 못하게 한다.
# cachetools = verl llm_server 의 미선언 의존(기본 이미지선 vLLM 이 채우던 것, 2026-07-29 확인).
# flash-attn = verl actor log-prob 의 flash_attn.bert_padding 하드 의존(rollout 엔진 무관, gpu extra).
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.8.post1-cu130
FROM ${SGLANG_IMAGE}

ENV PIP_NO_CACHE_DIR=1

# verl 코어 — torch/sglang/transformers 는 base 버전에 고정(constraints)해 스택 무손상.
RUN printf 'torch==2.9.1\nsglang==0.5.8\ntransformers==4.57.1\n' > /tmp/verl-constraints.txt \
    && pip install -c /tmp/verl-constraints.txt "verl==0.8.0" "cachetools"

# ★ sglang 0.5.8.post1 자기모순 우회 (2026-07-30 라이브 GPU 검증). sglang 의 requires_dist 는
# sgl-kernel==0.3.21 을 요구(이미지가 맞게 설치)하는데, 런타임 launch_server→assert_pkg_version 은
# **다른 이름 `sglang_kernel`(≥0.1.1)** 을 메타데이터로 찾아 PackageNotFoundError 로 죽는다(upstream 버그).
# assert_pkg_version 은 importlib.metadata.version() 으로 **메타데이터만** 읽는다(모듈 import 아님) →
# **메타데이터-only dist-info 스텁**을 심어 체크만 통과시키고 실제 커널(sgl-kernel 0.3.21)은 안 건드린다.
# (진짜 sglang_kernel 0.4.x 휠 설치는 sgl-kernel 의 common_ops 를 깨뜨림 — 라이브 실측 배제.)
RUN SP=$(python -c "import site; print(site.getsitepackages()[0])") \
    && D="$SP/sglang_kernel-0.3.21.dist-info" && mkdir -p "$D" \
    && printf 'Metadata-Version: 2.1\nName: sglang_kernel\nVersion: 0.3.21\n' > "$D/METADATA"

# flash-attn: base 에 있으면 no-op, 없으면 cu13torch2.9 prebuilt(cxx11abi 는 빌드시 판별, torch 2.9=TRUE).
RUN python -c "import flash_attn" 2>/dev/null && echo "flash_attn 이미 존재" || { \
      ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')"); \
      echo "flash-attn cu13torch2.9 cxx11abi=${ABI} 설치"; \
      pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3+cu13torch2.9cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"; }

# 우리 글루(deps=pyyaml 뿐 → --no-deps 로 정합 스택 무손상)
WORKDIR /workspace/repo
COPY . .
RUN pip install --no-deps . && pip install "pyyaml>=6.0"

LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
