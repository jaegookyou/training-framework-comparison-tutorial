# torchtitan 이미지 (SFT full + 사전학습 full·nightly SHA 핀). base(cu124 runtime) 위에 얹는다.
#   docker build -f docker/torchtitan.Dockerfile -t ghcr.io/jaegookyou/training-framework-comparison-tutorial/torchtitan:latest .
#
# 한 이미지가 torchtitan 의 두 트랙을 담당한다(1프레임워크 1이미지):
#   · SFT (ChatDataset, sft_qwen3_8b_traceinversion)        — 통제비교 SFT 트랙
#   · 사전학습 (HuggingFaceTextDataLoader, pretrain_qwen3_*) — 수직 파이프라인 1단계
# 둘 다 nightly 전용 기능이라 torchtitan 을 커밋 SHA 로 박고, 빌드 성공 이미지를 *불변 태그로 박제*해
# 재현(휠 증발 대비 = 환경 영구 재현):
#   docker build ... -t .../torchtitan:sft-9ccbb57 ; docker tag ... .../torchtitan:latest
#
# torchtitan deps 는 전부 휠(torch nightly·torchdata·spmd_types·tyro…)이라 소스 컴파일 없음 →
# nvcc/devel 불필요, base(cu124 runtime) 그대로 FROM.
#
# ⚠️ repo 를 SHA 로 clone 해 editable 설치한다(`pip install git+...` 와 달리 scripts/ 까지 디스크에
# 둬야 사전학습 HF export 의 scripts/checkpoint_conversion/convert_to_hf.py 를 호출할 수 있다).
#
# ⚠️ GPU 빌드 검증 대기(다른 이미지와 동일 단서): torch nightly cu124 + torchtitan@SHA 조합 호환,
#   baked patch(데이터셋·flavor·config 함수)가 정확히 붙는지, convert_to_hf 인자.
ARG BASE_IMAGE=ghcr.io/jaegookyou/training-framework-comparison-tutorial/base:latest
FROM ${BASE_IMAGE}

# torch + torchdata nightly (cu124). torchtitan main 이 PyTorch nightly 를 요구.
RUN pip install --pre torch torchdata \
        --index-url https://download.pytorch.org/whl/nightly/cu124

# torchtitan @ 커밋 SHA 핀. repo clone + editable (scripts/ 디스크 유지). requirements(spmd_types 등) 동반.
ARG TT_REF=9ccbb57b0e5df6d7fa56287181eed5d55891fcd0
RUN git clone https://github.com/pytorch/torchtitan.git /opt/torchtitan \
    && cd /opt/torchtitan && git checkout ${TT_REF} && pip install -e .
ENV TORCHTITAN_DIR=/opt/torchtitan

# baked patches — torchtitan 소스에 우리 통제비교/파이프라인 hook 을 박는다(editable 이라 직접 수정).
RUN python - <<'PY'
import pathlib
TT = pathlib.Path("/opt/torchtitan/torchtitan")

# (1) wikitext 사전학습 데이터셋 등록 — DATASETS dict 에 text 컬럼 추출기로 추가.
ds = TT / "hf_datasets" / "text_datasets.py"
ds.write_text(ds.read_text() + '''

# tfct: wikitext 사전학습 데이터셋 (Salesforce/wikitext, text 컬럼). 작은 코퍼스 사전학습 연습용.
def _load_wikitext_dataset(dataset_path: str, name: str, split: str):
    return load_dataset(dataset_path, name, split=split)


def _process_wikitext_text(sample):
    return sample["text"]


DATASETS["wikitext"] = DatasetConfig(
    path="Salesforce/wikitext",
    loader=partial(_load_wikitext_dataset, name="wikitext-2-raw-v1", split="train"),
    sample_processor=_process_wikitext_text,
)
''')

# (2) 초소형 Qwen3 flavor 등록 — vocab 151936 유지(Qwen3 토크나이저 정합), dim/층만 축소.
init = TT / "models" / "qwen3" / "__init__.py"
init.write_text(init.read_text() + '''

# tfct: 초소형 Qwen3 (dim 512 / 6층 / vocab 151936). 사전학습 배선·파이프라인 연습용.
def _tfct_tiny(attn_backend: str) -> Qwen3Model.Config:
    dim = 512
    head_dim = 64
    n_layers = 6
    vocab_size = 151936
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen3_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=8,
            n_kv_heads=4,
            head_dim=head_dim,
            hidden_dim=1536,
            attn_backend=attn_backend,
            rope=CosSinRoPE.Config(dim=head_dim, max_seq_len=4096, theta=1000000.0),
        ),
    )


qwen3_configs["tfct_tiny"] = _tfct_tiny
''')

# (3) config 함수 등록 — SFT(traceinversion) + 사전학습(wikitext). torchtitan --config 가 이름 resolve.
reg = TT / "models" / "qwen3" / "config_registry.py"
reg.write_text(reg.read_text() + '''

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


# tfct: 위 SFT 의 LoRA 변형 — 네이티브 LoRAConverter 로 model_spec 재생성(Linear→LoRALinear, base
# frozen). converter 는 model_registry 빌드 시점에 적용되므로(ModelSpec 에 저장 안 됨) 같은 flavor
# ("8B", attn_backend "varlen" = sft_qwen3_8b_math 와 동일)로 model_spec 을 다시 만든다. rank/alpha 는
# host trainer 가 env(TFCT_LORA_*)로 넘긴다(config 함수는 인자 못 받음). target_modules=None = 전 Linear.
def sft_qwen3_8b_traceinversion_lora() -> Trainer.Config:
    import os

    from torchtitan.components.lora import LoRAConverter

    cfg = sft_qwen3_8b_traceinversion()
    rank = int(os.environ.get("TFCT_LORA_RANK", "16"))
    alpha = float(os.environ.get("TFCT_LORA_ALPHA", "32"))
    cfg.model_spec = model_registry(
        "8B",
        attn_backend="varlen",
        converters=[LoRAConverter.Config(rank=rank, alpha=alpha)],
    )
    return cfg


# tfct: 사전학습 — 초소형 Qwen3 (tfct_tiny) on wikitext. qwen3_0_6b 본떠 flavor+dataloader 교체.
def pretrain_qwen3_tiny_wikitext() -> Trainer.Config:
    cfg = qwen3_0_6b()
    cfg.model_spec = model_registry("tfct_tiny")
    cfg.dataloader = HuggingFaceTextDataLoader.Config(dataset="wikitext")
    return cfg


# tfct: 스케일업 예시 — 0.6B(native flavor) on wikitext. size preset "0.6b" 가 가리킨다.
def pretrain_qwen3_0_6b_wikitext() -> Trainer.Config:
    cfg = qwen3_0_6b()
    cfg.dataloader = HuggingFaceTextDataLoader.Config(dataset="wikitext")
    return cfg


# tfct: continued-pretrain — Qwen3-8B-Base 가중치를 시드로 wikitext 이어학습(from-scratch 아님).
# 사전·사후를 같은 8B 로 통일하는 수직 파이프라인용(8B from-scratch 는 250만 토큰엔 무의미 → 이어학습).
# 시드 메커니즘 = initial_load_in_hf=True + initial_load_path 미지정 → hf_assets_path(=호스트가 받은
# Qwen3-8B-Base 스냅샷)에서 HF 가중치 로드(initial_load_model_only=True 기본이라 옵티마이저는 fresh =
# resume 아닌 이어학습). sft_qwen3_8b_math 와 동일 패턴. 0.6b pretrain skeleton(text loader)에 8B
# flavor·8B assets·HF 시드 checkpoint 만 갈아끼운다.
def pretrain_qwen3_8b_wikitext() -> Trainer.Config:
    cfg = qwen3_0_6b()
    cfg.model_spec = model_registry("8B")
    cfg.hf_assets_path = "./assets/hf/Qwen3-8B-Base"   # 호스트가 --hf_assets_path 로 실경로 override
    cfg.dataloader = HuggingFaceTextDataLoader.Config(dataset="wikitext")
    cfg.checkpoint = CheckpointManager.Config(
        enable=True,
        initial_load_in_hf=True,            # hf_assets_path 의 HF 가중치를 시드로(continued-pretrain)
        last_save_model_only=False,
        export_dtype="float16",
    )
    return cfg
''')
print("baked: wikitext dataset / tfct_tiny flavor / sft+pretrain config 함수")
PY

# repo 연결: 이 LABEL 이 패키지를 GitHub repo 의 Packages 에 붙이고 visibility 를 상속시킨다.
LABEL org.opencontainers.image.source=https://github.com/jaegookyou/training-framework-comparison-tutorial
