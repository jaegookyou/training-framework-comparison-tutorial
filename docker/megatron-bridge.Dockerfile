# Megatron-Bridge 이미지 = SFT(full|lora) + continued-pretrain 의 Megatron 데이터포인트.
#   docker build -f docker/megatron-bridge.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/megatron-bridge:latest .
#
# ★ 2026-07-30 공식 base 피벗 — 손조립 cu12 스택을 버리고 NVIDIA NeMo Framework 컨테이너로.
# ---------------------------------------------------------------------------------------------
# 왜: 손조립판(cu12.4 devel + torch 2.9 핀 + cu12 TE 강제 + bridge cu13 언인스톨)이 GPU convert 에서
#   `ModuleNotFoundError: transformer_engine.pytorch` 로 죽었다(2026-07-29 첫 2노드 스모크 실측).
#   근본 원인 = **cu12 고집**: base cu12.4 / torch 핀 cu12.8 / megatron-bridge native cu13 → 삼중
#   CUDA 불일치, 그걸 되돌리는 `pip uninstall transformer-engine-cu13` 이 TE 의 pytorch 백엔드를
#   통째로 지웠다. 빌드는 조용히 성공으로 뜨고(01fc6d3) 깨짐은 GPU import 에서만 드러났다.
#   Blackwell(sm_120)은 cu130 이 지원한다(preflight 실증) → cu12 를 고집할 이유가 애초에 없었다.
# 교훈: [[use-canonical-official-methods]] — Blackwell 이 지도 밖이라 torch+TE+megatron-core+bridge
#   **통합**을 우리가 손으로 맞추면 스큐 사냥이 반복된다(sglang 사가와 동형). NeMo 컨테이너는 그 통합을
#   NVIDIA 가 이미 풀어 테스트해 둔 스택이다. Megatron-Bridge README 공식 권장 base.
# 인증: NGC 는 **익명 pull 가능**(nvcr.io/v2 proxy_auth 익명 토큰 발급, 2026-07-30 실측) — 키·GitHub
#   secret 불필요. CI(build-images.yml)가 로그인 없이 FROM 한다. 런타임 SkyPilot 은 GHCR 의 우리
#   파생 이미지를 받으므로 nvcr 접근 자체가 없다(base 레이어 baked).
#
# NeMo 태그: 26.06.01 = 2026-06 stable(레지스트리 tags/list 실물에서 최신 YY.MM.PP). cu13 스택,
#   Blackwell 지원. 정확한 sm_120 실행·내부 버전은 빌드 후 tfct-preflight 로 검증($0.1).

ARG NEMO_IMAGE=nvcr.io/nvidia/nemo:26.06.01
FROM ${NEMO_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- 글루만 얹는다(NeMo 의 정합 스택을 건드리지 않는다) ---
# 우리 패키지 deps = pyyaml 하나뿐(무거운 torch/transformers 는 optional, 기본 미설치 — pyproject
# 참조). transformers/datasets/wandb/huggingface_hub 는 NeMo 가 이미 담는다 → 재설치하면 정합 스택을
# 깨뜨릴 수 있으므로 **--no-deps 로 우리 패키지만** 설치하고 나머지는 NeMo 것을 신뢰한다.
# (스모크가 누락 dep 를 드러내면 그때 명시 추가 — additive, 지금 추정 설치 안 함.)

# Megatron-LM repo(scripts 전용) — continued-pretrain 학습 루프는 순수 pretrain_gpt.py 가 돈다.
# pretrain_gpt.py·tools/preprocess_data.py 는 패키지 모듈이 아니라 repo 루트 스크립트라 pip 로 안
# 깔린다 → clone 만. **pip install 안 함**: megatron.core/megatron.training 은 NeMo 가 담은 것을 쓴다
# (재설치하면 namespace 스큐). ⚠️ MLM_REF 는 NeMo 가 담은 megatron-core 버전과 정합해야 pretrain_gpt.py
# 가 설치된 core API 와 안 어긋난다 — 스모크에서 arg-parse/import 로 확인 후 필요시 ARG 로 맞춘다.
ARG MLM_REF=core_v0.17.1
RUN git clone --depth 1 --branch ${MLM_REF} https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM
ENV MEGATRON_LM_DIR=/opt/Megatron-LM

WORKDIR /workspace/repo
COPY . .
RUN pip install --no-deps . \
    && pip install "pyyaml>=6.0"     # 유일 dep(NeMo 에 이미 있으면 no-op)

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
