import asyncio
import types

import pytest

from training_framework_comparison_tutorial.adapters import (
    compute_score,
    get_format,
    get_reward_funcs,
    get_source,
    normalize_messages,
    slime_rm,
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
    formatted = get_format("sft", "trl")(messages)
    assert formatted == {"messages": row["messages"]}


def test_normalize_rejects_unknown_role():
    with pytest.raises(ValueError):
        normalize_messages([{"role": "wizard", "content": "x"}])


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        normalize_messages([])


def test_unsloth_reuses_trl_sft_format():
    # Unsloth 는 trl SFTTrainer 를 감싸므로 동일 conversational 포맷.
    assert get_format("sft", "unsloth") is get_format("sft", "trl")


def test_verl_reuses_trl_sft_format():
    # verl MultiTurnSFTDataset 도 messages 컬럼을 읽으므로 동일 row 모양(직렬화만 parquet).
    assert get_format("sft", "verl") is get_format("sft", "trl")


def test_unknown_source_and_format_raise():
    with pytest.raises(ValueError):
        get_source("nope")
    with pytest.raises(ValueError):
        get_format("sft", "nope")
    with pytest.raises(ValueError):
        get_format("nope", "trl")


# --- DPO (선호쌍) ---


def test_ultrafeedback_to_trl_dpo():
    # implicit-prompt: chosen/rejected 가 user turn 을 공유, assistant 만 다름.
    row = {
        "chosen": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
        ],
        "rejected": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "no."},
        ],
    }
    pref = get_source("ultrafeedback")(row)
    formatted = get_format("dpo", "trl")(pref)
    assert formatted == {"chosen": row["chosen"], "rejected": row["rejected"]}


# --- GRPO (프롬프트 + 정답) ---


def test_gsm8k_to_trl_grpo_extracts_gold():
    row = {
        "question": "2+2?",
        "answer": "add two and two\n#### 4",
    }
    example = get_source("gsm8k")(row)
    # system 지시 + user 질문 → 프롬프트, #### 뒤 숫자 → 정답
    assert example["answer"] == "4"
    assert example["prompt"][-1] == {"role": "user", "content": "2+2?"}
    assert example["prompt"][0]["role"] == "system"
    formatted = get_format("grpo", "trl")(example)
    assert formatted["answer"] == "4"
    assert formatted["prompt"] == example["prompt"]


def test_gsm8k_gold_strips_commas():
    row = {"question": "q", "answer": "...\n#### 1,234"}
    assert get_source("gsm8k")(row)["answer"] == "1234"


# --- GRPO reward ---


def test_gsm8k_reward_correctness_and_format():
    correctness, fmt = get_reward_funcs("gsm8k")
    completions = [
        [{"role": "assistant", "content": "reasoning... #### 4"}],   # 정답+형식
        [{"role": "assistant", "content": "the answer is \\boxed{5}"}],  # 오답+형식(boxed)
        [{"role": "assistant", "content": "just 4 somewhere"}],      # 정답이지만 형식 없음
    ]
    answer = ["4", "4", "4"]
    assert correctness(completions, answer) == [1.0, 0.0, 1.0]
    assert fmt(completions) == [0.1, 0.1, 0.0]


def test_unsloth_reuses_trl_rl_formats():
    # Unsloth 도 trl DPO/GRPO Trainer 를 감싸므로 동일 포맷(conversational).
    assert get_format("dpo", "unsloth") is get_format("dpo", "trl")
    assert get_format("grpo", "unsloth") is get_format("grpo", "trl")


def test_verl_grpo_uses_reward_model_ground_truth():
    # verl 은 정답을 kwargs 가 아니라 parquet 의 reward_model.ground_truth 로 받는다(별도 포맷).
    row = {"question": "2+2?", "answer": "...\n#### 4"}
    example = get_source("gsm8k")(row)
    formatted = get_format("grpo", "verl")(example)
    assert formatted["prompt"] == example["prompt"]
    assert formatted["reward_model"] == {"style": "rule", "ground_truth": "4"}
    assert "answer" not in formatted  # TRL 포맷과 달리 answer 컬럼 없음


def test_unknown_reward_raises():
    with pytest.raises(ValueError):
        get_reward_funcs("nope")


# --- verl reward 규약 (compute_score) ---


def test_verl_compute_score_matches_trl_sum():
    # verl 스칼라 = TRL correctness+format 합과 동일 채점 코어(통제 변수).
    assert compute_score("gsm8k", "reasoning... #### 4", "4") == pytest.approx(1.1)  # 정답+형식
    assert compute_score("gsm8k", "answer is \\boxed{5}", "4") == pytest.approx(0.1)  # 오답+형식
    assert compute_score("gsm8k", "just 4 somewhere", "4") == pytest.approx(1.0)  # 정답·형식없음


def test_verl_compute_score_unknown_data_source_raises():
    with pytest.raises(ValueError):
        compute_score("nope", "x", "1")


# --- slime ---


def test_slime_grpo_uses_prompt_and_label():
    # slime 은 --input-key prompt / --label-key label → prompt 체인 + label 정답.
    row = {"question": "2+2?", "answer": "...\n#### 4"}
    example = get_source("gsm8k")(row)
    formatted = get_format("grpo", "slime")(example)
    assert formatted == {"prompt": example["prompt"], "label": "4"}


def test_slime_rm_matches_scoring_core():
    # slime Sample(response/label/metadata) → verl/TRL 과 동일 채점 코어, async 규약.
    def sample(response: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            response=response, label="4", metadata={"data_source": "gsm8k"}
        )

    assert asyncio.run(slime_rm(None, sample("reasoning... #### 4"))) == pytest.approx(1.1)
    assert asyncio.run(slime_rm(None, sample("answer is \\boxed{5}"))) == pytest.approx(0.1)
    assert asyncio.run(slime_rm(None, sample("just 4 somewhere"))) == pytest.approx(1.0)


def test_slime_rm_unknown_data_source_raises():
    bad = types.SimpleNamespace(response="x", label="1", metadata={"data_source": "nope"})
    with pytest.raises(ValueError):
        asyncio.run(slime_rm(None, bad))
