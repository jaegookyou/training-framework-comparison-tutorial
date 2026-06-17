import pytest

from training_framework_comparison_tutorial.adapters import (
    get_format,
    get_source,
    normalize_messages,
)


def test_traceinversion_to_trl_roundtrip():
    # reasoning distill: assistant content 에 <think> CoT + 답이 인라인으로 들어있다.
    row = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>add</think> 4"},
        ]
    }
    messages = get_source("traceinversion")(row)
    formatted = get_format("trl")(messages)
    assert formatted == {"messages": row["messages"]}


def test_normalize_rejects_unknown_role():
    with pytest.raises(ValueError):
        normalize_messages([{"role": "wizard", "content": "x"}])


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        normalize_messages([])


def test_unsloth_reuses_trl_format():
    # Unsloth 는 trl SFTTrainer 를 감싸므로 동일 conversational 포맷.
    assert get_format("unsloth") is get_format("trl")


def test_verl_reuses_trl_format():
    # verl MultiTurnSFTDataset 도 messages 컬럼을 읽으므로 동일 row 모양(직렬화만 parquet).
    assert get_format("verl") is get_format("trl")


def test_unknown_source_and_format_raise():
    with pytest.raises(ValueError):
        get_source("nope")
    with pytest.raises(ValueError):
        get_format("nope")
