# torchtitan SFT(full 전용·nightly SHA 핀) 이미지. base(cu124 runtime) 위에 얹는다.
#   docker build -f docker/torchtitan.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/torchtitan:latest .
#
# torchtitan 의 SFT(ChatDataset)는 정식 릴리스(0.2.2)엔 없고 main 에만 있다 → nightly 전용. 그래서
# 특정 커밋 SHA 로 박고(아래 TT_REF), 무엇보다 빌드 성공한 이 이미지를 *불변 태그로 박제*해 재현성을
# 확보한다(torch nightly 휠은 시간이 지나면 증발하지만, 이미지 안엔 바이너리째 박제됨 = 환경 영구 재현):
#   docker build ... -t .../torchtitan:sft-9ccbb57   # 불변 스냅샷 (sky 가 이걸 pull)
#   docker tag      .../torchtitan:sft-9ccbb57 .../torchtitan:latest
#
# megatron-bridge 와 달리 torchtitan deps 는 전부 휠(torch nightly·torchdata·spmd_types·tyro…)이라
# 소스 컴파일이 없다 → nvcc/devel 베이스 불필요, base(cu124 *runtime*) 그대로 FROM 한다.
#
# 핀: torchtitan@<SHA> 가 requirements(spmd_types==0.2.1 · datasets<4.8.0 · torchdata 등)를 끌어온다.
#   torch 는 requirements 에 없어 torchtitan 이 안 건드림 → 우리가 nightly cu124 를 먼저 박는다.
#   README 가 cu130 예시지만 "replace cu130 with another cuda" → base 의 cu124 에 맞춰 cu124 nightly.
#
# ⚠️ GPU 빌드 검증 대기(다른 이미지와 같은 단서): torch nightly cu124 + torchtitan@SHA + spmd_types
#   0.2.1 조합 호환, baked config patch 가 설치 경로에 정확히 붙는지.
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

# torch + torchdata nightly (cu124). torchtitan main 이 PyTorch nightly 를 요구.
RUN pip install --pre torch torchdata \
        --index-url https://download.pytorch.org/whl/nightly/cu124

# torchtitan @ 커밋 SHA 핀 (nightly 전용 SFT → 이미지 박제로 재현). requirements(spmd_types 등) 동반.
ARG TT_REF=9ccbb57b0e5df6d7fa56287181eed5d55891fcd0
RUN pip install "git+https://github.com/pytorch/torchtitan.git@${TT_REF}"

# baked patch: traceinversion SFT config 함수를 qwen3 config_registry 에 등록한다. 기존
# sft_qwen3_8b_math 를 재사용하고 dataloader 만 우리 데이터(sample_processor=from_traceinversion)로
# 교체 → torchtitan 이 `--config sft_qwen3_8b_traceinversion` 로 이름 resolve. (megatron-lm 의 baked
# 변환기 등록과 같은 패턴. config_registry 가 Trainer·ChatDataLoader·sft_qwen3_8b_math 를 이미 모듈
# 스코프에 갖고 있어 그대로 참조 가능; from_traceinversion 만 우리 패키지에서 import.)
RUN python - <<'PY'
import pathlib
import torchtitan.models.qwen3.config_registry as m

patch = '''

# tfct: Qwen3-8B SFT on reasoning-distill (TraceInversion) — sft_qwen3_8b_math 의 dataloader 만 교체.
def sft_qwen3_8b_traceinversion() -> Trainer.Config:
    from training_framework_comparison_tutorial.adapters import from_traceinversion

    cfg = sft_qwen3_8b_math()
    cfg.dataloader = ChatDataLoader.Config(
        dataset_path="Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        load_dataset_kwargs={"split": "train"},
        sample_processor=from_traceinversion,
    )
    return cfg
'''

p = pathlib.Path(m.__file__)
p.write_text(p.read_text() + patch)
print(f"patched {p}")
PY

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
