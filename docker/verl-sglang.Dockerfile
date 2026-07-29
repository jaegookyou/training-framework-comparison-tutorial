# verl GRPO/PPO — **sglang rollout** 변종 이미지. base 위에 verl + sglang 스택을 얹는다.
#   docker build -f docker/verl-sglang.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/verl-sglang:latest .
#
# 왜 별도 이미지인가: 기본 verl 이미지(docker/verl.Dockerfile)는 torch 2.8 + vllm 0.11 이다.
# 그런데 vLLM 0.11 async 엔진의 on-policy weight-sync 가 Blackwell 에서 3회차에 하드크래시한다
# (2026-07-29 6회 iteration 규명 — 메모리·cudagraph·LoRA·sync모드·hf 전부 배제, verl_grpo.py NOTE).
# 우회 = rollout 엔진을 sglang 으로 바꾸기(verl 1급 백엔드, vLLM async weight-sync 경로를 안 탐).
# 그런데 **sglang 0.5.8 이 torch==2.9.1 을 하드핀**한다(sglang 자체 requires_dist, verl 선택 아님)
# → torch 2.8 인 기본 verl 이미지와 공존 불가라 변종으로 분리한다.
#
# ── 핀 근거 (전부 PyPI/GitHub 실물, 2026-07-29) ──
# (1) torch 2.9.1 = sglang 0.5.8 하드핀. **cu13(+cu130)** 로 설치한다:
#     verl actor 의 log-prob 경로가 flash_attn.bert_padding 을 하드 의존하는데(rollout 엔진과 무관),
#     flash-attn 은 torch 2.9 용으로 **cu12torch2.9 휠이 없고 cu13torch2.9 만** 있다(우리가 기본
#     이미지를 torch 2.8 에 묶었던 바로 그 gap). → torch 를 cu130 으로 깔아야 flash-attn prebuilt
#     휠을 쓴다(소스 컴파일·nvcc·devel base 회피). base 시스템 cuda(12.4)는 안 쓰인다 — torch 휠이
#     자체 cuda13 을 번들하고 sglang/flash-attn 커널은 torch 의 cuda 에 링크한다.
# (2) sglang[srt]==0.5.8 → sgl-kernel 0.3.21(cp310-abi3 prebuilt), flashinfer-python/cubin 0.6.1
#     (prebuilt cubin) 을 PyPI 에서 끈다. **vllm 은 안 끌어온다**(srt extra 확인) → 충돌 없음.
# (3) transformers 4.57.1 = sglang 0.5.8 하드핀(기본 verl 이미지의 4.57.6 과 근접, 여기선 sglang 핀).
# (4) flash-attn 2.8.3 (+cu13torch2.9). cu13torch2.9 는 cxx11abi=TRUE 판만 존재(torch 2.9 는 TRUE).
#
# ⚠️ 미검증(GPU 최종 확인 대상): sgl-kernel 0.3.21 abi3 휠이 cu12.8 빌드일 수 있어 torch+cu130 과
# 런타임 심볼 정합이 어긋날 여지가 있다(정합되면 sglang rollout 이 vLLM 크래시를 우회). 첫 GPU 런의
# 값진 발견 지점. 배선은 rollout_backend=sglang(configs/grpo/*__verl-sglang* / verl 섹션 knob).
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

# torch 2.9.1 = sglang 0.5.8 하드핀. cu130 인덱스에서 설치(flash-attn cu13torch2.9 휠용).
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.9.1" "torchvision" "torchaudio==2.9.1"

# verl + sglang 스택. torch 는 위에서 2.9.1+cu130 으로 이미 만족 → 재설치 안 됨(local 버전이 ==2.9.1 충족).
RUN pip install \
        "verl==0.8.0" \
        "transformers==4.57.1" \
        "sglang[srt]==0.5.8"

# flash-attn: verl actor log-prob 하드 의존. torch 2.9 → cu13torch2.9 prebuilt 휠(abi TRUE 뿐).
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
    && echo "flash-attn cu13torch2.9 cxx11abi=${ABI}" \
    && pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3+cu13torch2.9cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
