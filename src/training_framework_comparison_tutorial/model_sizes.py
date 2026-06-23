"""모델 크기 preset → 프레임워크별 arch 매핑 (수직 파이프라인 스케일 knob).

사전학습은 from-scratch 라 모델 크기를 직접 정해야 한다. `model.size`(config) 를 여기서
프레임워크별 arch 식별자로 번역한다 — 코드는 그대로 두고 size 만 바꾸면 더 큰 모델로 스케일.
사후학습(sft/rl)은 이전 단계 HF ckpt 를 `model.name` 으로 상속하므로 이 표를 쓰지 않는다.

⚠️ 사전학습 모델도 토크나이저(Qwen3, vocab 151936)와 정합해야 하므로 vocab_size 는 151936 고정
이다(SFT/RL 이 같은 토크나이저를 이어 쓰는 파이프라인 정합). 그래서 vocab 임베딩이 파라미터를
지배해 진짜 초미니(~6M)는 불가하고, tiny 도 임베딩 때문에 수십 M 급이 된다(여전히 8B 의 ~1%).

torchtitan 값은 qwen3 `qwen3_configs` flavor 이름이다(이미지 baked patch 가 `tfct_tiny` 를 등록).
다른 프레임워크(megatron 등)는 사전학습 트랙을 그 프레임워크로 배선할 때 키를 채운다.
"""

from __future__ import annotations

# size preset -> {framework: (arch flavor, torchtitan config 함수명)}
SIZES: dict[str, dict[str, str]] = {
    # tiny: Qwen3 arch 축소(dim 512 / 6층 / vocab 151936). 사전학습 배선·파이프라인 연습용.
    "tiny": {"torchtitan": "tfct_tiny", "torchtitan_config": "pretrain_qwen3_tiny_wikitext"},
    # 스케일업 예시 — torchtitan 에 native flavor 가 있어 config 함수만 추가하면 동작.
    "0.6b": {"torchtitan": "0.6B", "torchtitan_config": "pretrain_qwen3_0_6b_wikitext"},
    # 8b: continued-pretrain 전용 — from-scratch 가 아니라 Qwen3-8B-Base 가중치를 시드로 이어학습
    # (config.init_from + baked pretrain_qwen3_8b_wikitext 의 initial_load_in_hf=True). 사전·사후를
    # 같은 8B 로 통일하는 수직 파이프라인용(8B from-scratch 는 데이터 부족으로 무의미 → continued).
    "8b": {"torchtitan": "8B", "torchtitan_config": "pretrain_qwen3_8b_wikitext"},
}

# size preset -> Megatron-LM Qwen3 arch 치수. torchtitan tfct_tiny 와 동일 arch 를 Megatron 표현으로
# (사전학습 가로비교 = 두 프레임워크가 같은 모델·데이터를 from-scratch 학습). 비-치수 Qwen3 플래그
# (RMSNorm·SwiGLU·rope·qk-layernorm·disable-bias 등)는 megatron_arch_args 가 공통으로 붙인다.
# tied embeddings: tfct_tiny(enable_weight_tying=True) → --untie 안 붙임(소형 Qwen3 는 tie).
# 플래그명·값 = upstream core_v0.17.1 examples/rl/model_configs/qwen3_8b.sh 미러(추정 아님).
MEGATRON_ARCH: dict[str, dict[str, int]] = {
    # tiny: dim512 / 6층 / heads8 / kv4(GQA) / head_dim64 / ffn1536 / vocab151936 (tfct_tiny 동일).
    "tiny": {
        "num_layers": 6,
        "hidden_size": 512,
        "ffn_hidden_size": 1536,
        "num_attention_heads": 8,
        "num_query_groups": 4,
        "kv_channels": 64,
    },
}


def megatron_arch_args(size: str, seq_len: int) -> list[str]:
    """size preset → Megatron-LM pretrain_gpt.py arch 플래그 리스트.

    치수는 MEGATRON_ARCH(tfct_tiny 정합), 나머지 Qwen3 플래그는 qwen3_8b.sh 미러(같은 태그라 버전
    정합). vocab 151936 고정(Qwen3 토크나이저 — 파이프라인/통제 정합). 사전학습은 raw text 라
    chat template 불필요. tied embeddings 라 --untie-embeddings-and-output-weights 는 붙이지 않는다.
    """
    try:
        a = MEGATRON_ARCH[size]
    except KeyError:
        raise ValueError(
            f"megatron arch preset 없음: {size!r} (있는 것: {list(MEGATRON_ARCH)})"
        ) from None
    return [
        "--num-layers", str(a["num_layers"]),
        "--hidden-size", str(a["hidden_size"]),
        "--ffn-hidden-size", str(a["ffn_hidden_size"]),
        "--num-attention-heads", str(a["num_attention_heads"]),
        "--group-query-attention",
        "--num-query-groups", str(a["num_query_groups"]),
        "--kv-channels", str(a["kv_channels"]),
        "--seq-length", str(seq_len),
        "--max-position-embeddings", str(seq_len),
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--qk-layernorm",
        "--position-embedding-type", "rope",
        "--rotary-base", "1000000",
        "--rotary-percent", "1.0",
        "--use-rotary-position-embeddings",
        "--swiglu",
        "--disable-bias-linear",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--no-masked-softmax-fusion",
        "--attention-softmax-in-fp32",
        "--vocab-size", "151936",
        "--make-vocab-size-divisible-by", "128",
    ]


def torchtitan_flavor(size: str) -> str:
    """size preset → torchtitan qwen3 flavor (model_registry 인자)."""
    try:
        return SIZES[size]["torchtitan"]
    except KeyError:
        raise ValueError(f"unknown model size preset: {size!r} (있는 것: {list(SIZES)})") from None


def torchtitan_config_fn(size: str) -> str:
    """size preset → torchtitan `--config` 로 넘길 config_registry 함수명."""
    try:
        return SIZES[size]["torchtitan_config"]
    except KeyError:
        raise ValueError(f"unknown model size preset: {size!r} (있는 것: {list(SIZES)})") from None
