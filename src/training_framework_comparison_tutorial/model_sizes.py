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
}


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
