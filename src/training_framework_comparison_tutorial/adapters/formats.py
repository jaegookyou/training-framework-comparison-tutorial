"""2층: 정규 SFT 스키마 → 프레임워크별 학습 포맷.

같은 정규형에서 출발하므로 "데이터는 동일, 포맷만 프레임워크별"이 보장된다(통제비교).
프레임워크를 추가할 때 여기에 변환 함수 하나만 등록하면 된다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import SFTExample


def to_trl(example: SFTExample) -> dict[str, Any]:
    """TRL SFTTrainer 의 conversational 포맷(messages 컬럼).

    Unsloth 도 내부적으로 trl SFTTrainer 를 감싸므로 같은 포맷을 재사용한다.
    verl 의 MultiTurnSFTDataset 도 parquet 의 messages 컬럼(messages_key=messages)을
    읽어 tokenizer.apply_chat_template 를 돌리므로 동일한 row 모양을 쓴다(차이는
    in-memory dataset vs parquet 파일 — 그 직렬화는 trainers/verl_sft.py 가 담당).
    """
    return {"messages": example}


# NOTE: megatron-lm 은 여기에 없다 — Megatron-LM 의 finetune.py 가 HF 데이터셋을 직접
# 인제스트(--finetune-hf-dataset)하고 SFTDataset 이 자체적으로 messages→conversation 변환을
# 하므로(이미지 baked 변환기), 우리 row-level format 을 거치지 않는다.
# 프레임워크 추가 시 등록 (예정): torchtitan
FORMATS: dict[str, Callable[[SFTExample], dict[str, Any]]] = {
    "trl": to_trl,
    "unsloth": to_trl,  # trl SFTTrainer 래핑 → conversational 포맷 동일
    "verl": to_trl,     # MultiTurnSFTDataset parquet 의 messages 컬럼과 동일 모양
}


def get_format(framework: str) -> Callable[[SFTExample], dict[str, Any]]:
    try:
        return FORMATS[framework]
    except KeyError:
        raise ValueError(f"no data format adapter for framework: {framework!r}") from None
