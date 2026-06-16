import pytest

from training_framework_comparison_tutorial.adapters import (
    REASONING_CHATML,
    resolve_chat_template,
)


def test_resolve_known_returns_template():
    assert resolve_chat_template("reasoning_chatml") is REASONING_CHATML


def test_resolve_none_keeps_tokenizer_default():
    # None/빈 값 → 토크나이저 자기 template 유지(덮어쓰지 않음).
    assert resolve_chat_template(None) is None
    assert resolve_chat_template("") is None


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_chat_template("nope")


def test_template_has_generation_markers():
    # assistant_only_loss 가 요구하는 마커. 없으면 TRL 이 마스크를 못 뽑는다(이 변경의 핵심).
    # whitespace-control 대시(`{%- ... %}`)를 허용해 마커 존재만 본다.
    assert "generation %}" in REASONING_CHATML
    assert "endgeneration %}" in REASONING_CHATML
    assert "add_generation_prompt" in REASONING_CHATML


def test_assistant_mask_only_covers_response():
    """실제 transformers 로 assistant 마스크가 응답 토큰에만 걸리는지 검증.

    transformers 는 docker 이미지에만 있다 → dev/CI 에선 skip, trl 이미지에선 진짜 검증.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:  # 오프라인 등
        pytest.skip(f"tokenizer 로드 불가(오프라인?): {exc}")

    tok.chat_template = REASONING_CHATML
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<think>add</think> 4"},
    ]
    out = tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = out["assistant_masks"]
    ids = out["input_ids"]

    # 마스크가 일부만 1 이어야 한다(전부 0 = 마커 무시됨 = 회귀, 전부 1 = 프롬프트도 학습).
    assert any(mask), "assistant 마스크가 전부 0 — generation 마커가 안 먹었다"
    assert not all(mask), "assistant 마스크가 전부 1 — 프롬프트까지 학습 대상"

    # 마스크 1 인 토큰을 디코드하면 assistant 응답 + 종료 토큰만 나와야 한다(user 질문 제외).
    masked_text = tok.decode([i for i, m in zip(ids, mask) if m])
    assert "4" in masked_text
    assert "2+2?" not in masked_text
