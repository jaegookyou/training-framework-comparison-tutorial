import pytest

from training_framework_comparison_tutorial.adapters import (
    get_format,
    get_source,
    normalize_messages,
)


def test_smoltalk_to_trl_roundtrip():
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    messages = get_source("smoltalk")(row)
    formatted = get_format("trl")(messages)
    assert formatted == {"messages": row["messages"]}


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


def test_unknown_source_and_format_raise():
    with pytest.raises(ValueError):
        get_source("nope")
    with pytest.raises(ValueError):
        get_format("nope")
