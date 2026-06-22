# slime GRPO 이미지 — 공식 prebuilt(slimerl/slime) 위에 우리 런타임 deps 만 얹는다.
#   docker build -f docker/slime.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/slime:latest .
#
# 다른 프레임워크와 달리 우리 base 위에 안 쌓는다 — slime 의 의존성 매트릭스(SGLang 특정 release
# + Megatron-LM dev commit + transformer-engine + sgl-router whl + megatron/sglang 패치)는 runtime
# 베이스로 재현 불가능하고, slime 공식이 prebuilt 이미지로 배포한다. "환경=이미지 핀" 철학 그대로
# 공식 이미지를 베이스로 박고(dep 재발명 안 함), 코드는 sky 가 런타임에 pip install -e . 한다.
#
# 핀: slimerl/slime:latest. 재현용으로는 dated tag 권장 — slime docker/version.txt = nightly-dev-
# 20260618a (sglang v0.5.12.post1 + megatron dev 1dcf0da). GPU 빌드 때 정확한 tag·내부 경로 확정
# (torchtitan nightly SHA 핀과 같은 단서 — [[torchtitan-sft-nightly-only]]).
ARG SLIME_IMAGE=slimerl/slime:latest
FROM ${SLIME_IMAGE}

# 우리 패키지 런타임 deps. slime 이미지에 torch/transformers/sglang/megatron 은 이미 있으므로
# config 로딩(pyyaml)·데이터(datasets)·로깅(wandb) 같은 가벼운 것만.
RUN pip install --no-cache-dir "pyyaml" "datasets" "wandb"

# trainers/slime_grpo.py 가 참조하는 repo 내부 경로(공식 이미지 기준 — GPU 빌드 때 확인 후 고정).
ENV SLIME_DIR=/root/slime
ENV MEGATRON_LM_DIR=/root/Megatron-LM

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
