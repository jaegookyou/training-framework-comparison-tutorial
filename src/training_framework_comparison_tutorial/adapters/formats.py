"""2층: 방법별 정규 스키마 → 프레임워크별 학습 포맷.

같은 정규형에서 출발하므로 "데이터는 동일, 포맷만 프레임워크별"이 보장된다(통제비교).
레지스트리는 (method, framework) 2층으로 나뉜다 — 같은 프레임워크(trl)라도 SFT/DPO/GRPO
가 요구하는 컬럼이 다르기 때문. 프레임워크를 추가할 때 해당 method 아래 변환 함수 하나만 등록.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import PreferenceExample, RLPromptExample, SFTExample


def to_trl(example: SFTExample) -> dict[str, Any]:
    """TRL SFTTrainer 의 conversational 포맷(messages 컬럼).

    Unsloth 도 내부적으로 trl SFTTrainer 를 감싸므로 같은 포맷을 재사용한다.
    verl 의 MultiTurnSFTDataset 도 parquet 의 messages 컬럼(messages_key=messages)을
    읽어 tokenizer.apply_chat_template 를 돌리므로 동일한 row 모양을 쓴다(차이는
    in-memory dataset vs parquet 파일 — 그 직렬화는 trainers/verl_sft.py 가 담당).
    """
    return {"messages": example}


def to_trl_dpo(example: PreferenceExample) -> dict[str, Any]:
    """TRL DPOTrainer 의 conversational preference 포맷(chosen/rejected 컬럼).

    prompt 를 따로 주지 않는 implicit-prompt 형태 — DPOTrainer 가 chosen/rejected 의
    공통 prefix 를 prompt 로 자동 추출한다.
    """
    return {"chosen": example["chosen"], "rejected": example["rejected"]}


def to_trl_grpo(example: RLPromptExample) -> dict[str, Any]:
    """TRL GRPOTrainer 의 포맷(prompt 컬럼 + reward 채점용 여분 컬럼).

    prompt 만 생성 입력으로 쓰이고, answer 등 나머지 컬럼은 GRPOTrainer 가 reward 함수에
    **kwargs 로 그대로 흘려준다(adapters.rewards 가 answer 로 채점).
    """
    return {"prompt": example["prompt"], "answer": example["answer"]}


# NOTE: megatron-lm SFT 는 여기에 없다 — finetune.py 가 HF 데이터셋을 직접 인제스트
# (--finetune-hf-dataset)하고 SFTDataset 이 자체 messages→conversation 변환을 하므로
# 우리 row-level format 을 거치지 않는다.
FORMATS: dict[str, dict[str, Callable[[Any], dict[str, Any]]]] = {
    "sft": {
        "trl": to_trl,
        "unsloth": to_trl,  # trl SFTTrainer 래핑 → conversational 포맷 동일
        "verl": to_trl,     # MultiTurnSFTDataset parquet 의 messages 컬럼과 동일 모양
    },
    "dpo": {
        "trl": to_trl_dpo,
        "unsloth": to_trl_dpo,   # unsloth 도 trl DPOTrainer 래핑 → 동일 chosen/rejected 포맷
    },
    "grpo": {
        "trl": to_trl_grpo,
        "unsloth": to_trl_grpo,  # unsloth 도 trl GRPOTrainer 래핑 → 동일 prompt+answer 포맷
    },
}


def get_format(method: str, framework: str) -> Callable[[Any], dict[str, Any]]:
    try:
        return FORMATS[method][framework]
    except KeyError:
        raise ValueError(
            f"no data format adapter for {method}/{framework!r}"
        ) from None
