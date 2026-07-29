# verl SFT(FSDP) + RL(GRPO/PPO) 이미지. base 위에 verl 학습 스택을 핀 고정해 얹는다.
#   docker build -f docker/verl.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/verl:latest .
#
# 핀 근거: `uv pip compile (verl==0.8.0 + torch, py3.12)` 로 해석한 그래프(추정 아님).
#   verl 0.8.0 → tensordict · datasets · accelerate · peft · hydra-core · ray · numpy · pyarrow ·
#   torchdata 등은 transitive 해석. 아래는 우리가 직접 잡는 핀(torch/verl/transformers/vllm/flash-attn).
#
# ─────────────────────────────────────────────────────────────────────────────
# 이 이미지의 핀은 **세 제약의 교집합**이다(각각 GPU 실측/PyPI 실물로 확인). 하나만 봐선 왜 이 조합인지
# 알 수 없어 셋을 같이 적는다:
#
# (1) 하드웨어 — Blackwell(sm_120) 하한  ⟹  torch >= 2.8 (cu12.8)
#   빌리는 GPU 는 Nebius **RTX PRO 6000 Blackwell Server Edition(sm_120, 96GB, driver 580)**.
#   torch 2.6.0 은 cu12.4 빌드라 이 아키텍처 커널이 없어 2노드 런이 NCCL barrier 에서
#   `CUDA error: no kernel image is available for execution on the device` 로 죽었다(2026-07-28 실측).
#   PyPI 실물: torch 2.6.0=cu12.4/nccl 2.21.5 · **2.8.0·2.9.0=cu12.8/nccl 2.27.x** · 2.12.0=cu13.
#   교훈 **핀은 빌드돼야 검증된 것이다** — 07-01 이미지는 `"vllm"` 을 핀 없이 깔아 최신 vllm 이 torch 를
#   끌어올려 우연히 Blackwell 에서 돌았고, `torch==2.6.0` 커밋 핀은 **07-28 에 처음 빌드**되며 드러났다.
#
# (2) verl RL 이 flash-attn 을 하드 의존  ⟹  cp312 prebuilt 휠이 있는 torch 로 고정  ⟹  torch == 2.8.0
#   verl 0.8.0 은 RL(GRPO/PPO)의 log-prob 경로(`ray_trainer.fit → _compute_old_log_prob →
#   padding.left_right_2_no_padding → attention_utils.unpad_input`)에서 `flash_attn.bert_padding` 을
#   **하드 의존**한다(폴백은 NPU 뿐). `model.use_remove_padding=false` 는 **모델 forward 에만** 적용돼
#   이 경로를 안 우회한다(2026-07-28 GRPO 런이 여기서 정지). base 에 nvcc 가 없어 소스 빌드는 불가 →
#   **prebuilt 휠**만 가능한데, flash-attn cp312/x86_64 릴리스 휠은 **`cu12torch2.8` 과 `cu13torch2.9`**
#   뿐이다(v2.8.3.post1 확인). **cu12+torch2.9 조합이 없다.** cu13 은 base(cu12.x)와 안 맞으므로
#   cu12torch2.8 을 택할 수밖에 없고 → 이 이미지를 **torch 2.9 → 2.8 로 내린다.**
#
# (3) verl 이 vllm 을 GRPO rollout 에 요구  ⟹  vllm 이 torch 를 정확히 핀  ⟹  vllm == 0.11.0
#   verl 0.8.0 의 [vllm] extra 는 `vllm>=0.8.5,<=0.12.0`. 그 범위에서 **torch 2.8.0 을 핀하는 판은
#   vllm 0.11.0**(0.12.0 은 torch 2.9.0, 0.8.5.post1 은 torch 2.6.0). (2)가 torch 2.8 을 요구하므로
#   직전 판 0.12.0 → 0.11.0 으로 함께 내린다. SFT 엔 vllm 불필요, GRPO/PPO rollout 에 필요.
#
# ⟹ 유일 정합: **torch 2.8.0 · vllm 0.11.0 · flash-attn 2.8.3.post1+cu12torch2.8** (+ transformers 아래).
# ─────────────────────────────────────────────────────────────────────────────
#
# ⚠️ **transformers 는 4.57.6 이다(v5 에서 내림, 2026-07-28 GPU 실측 후 정정).**
#   2노드 GRPO 의 vLLM 서버 기동이 5.x 로 죽었다:
#   `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
#   (transformers v5 가 제거한 속성을 vllm 이 호출).
#   교훈: **`requires_dist` 의 상한 부재는 호환 보증이 아니다** — vllm 0.11.0 은 `transformers>=4.55.2`
#   로 상한이 없지만 v5 는 동작 안 한다. 선언된 범위 ≠ 동작하는 범위. [[dont-guess-package-versions]]
#   4.57.6 = 마지막 4.x(tokenizers<=0.23.0, vllm 의 tokenizers>=0.21.1 과 정합). verl 0.8.0 자체는
#   `transformers`(제약 없음)라 4.x 로 내려도 메타데이터상 문제 없다. transformers 4.57.6 은 2026-07-28
#   에 verl SFT 2노드 5/5 로 실증됐다 — **단 그때 스택은 torch 2.9.0** 이었다. 이번에 flash-attn 때문에
#   torch 2.9→2.8 로 내렸으므로 SFT·GRPO **재검증이 필요**하다(torch 2.8·2.9 는 같은 cu12.8/nccl 2.27
#   라인이라 학습 의미론 변화는 없을 것으로 예상하나, 핀은 빌드·실행돼야 검증된 것이다).
#
# ⚠️ **cxx11abi 는 추측하지 않는다.** flash-attn 휠은 cxx11abi TRUE/FALSE 두 판이 있고 설치된 torch 와
#   안 맞으면 import 시 죽는다. 빌드 시점에 `torch._C._GLIBCXX_USE_CXX11_ABI` 로 판별해 URL 을 고른다.
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

RUN pip install "torch==2.8.0" \
    && pip install \
        "verl==0.8.0" \
        "transformers==4.57.6" \
        "vllm==0.11.0"  # torch==2.8.0(cu12.8=Blackwell) 핀 · transformers>=4.55.2 → 4.57.6 과 정합

# flash-attn: verl RL 의 하드 의존(위 제약 (2)). prebuilt 휠 + 빌드 시점 cxx11abi 판별.
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
    && echo "flash-attn cxx11abi=${ABI} (설치된 torch 에서 판별)" \
    && pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
