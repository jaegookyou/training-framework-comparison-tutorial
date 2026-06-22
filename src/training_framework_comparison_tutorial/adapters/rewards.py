"""GRPO reward 함수 — on-policy RL 의 채점기.

통제비교에서 reward 는 **고정 변수**다(데이터·모델·HP 와 함께). 그래서 프레임워크가 아니라
태스크(gsm8k)에 1:1 로 묶어 한 곳에 정의하고, TRL·verl 이 같은 **채점 코어**를 공유한다.

프레임워크마다 reward 호출 규약이 달라 노출 형태가 둘이다(채점 로직은 공유 = 통제 변수):
- **TRL GRPOTrainer**: `f(completions, **kwargs) -> list[float]`.
  - completions: 그룹 rollout. conversational 이면 각 원소가 [{role:assistant, content}] 리스트.
  - kwargs: 데이터셋 여분 컬럼(여기선 `answer`)이 그대로 전달(adapters.formats.to_trl_grpo 통과).
- **verl custom_reward_function**: `compute_score(data_source, solution_str, ground_truth,
  extra_info=None) -> float`. 샘플 1개씩, solution_str=생성 텍스트, ground_truth=정답(parquet 의
  reward_model.ground_truth). data_source 로 태스크별 scorer 를 라우팅한다.
- **slime --custom-rm-path**: `async def slime_rm(args, sample) -> float`. slime Sample 객체에서
  sample.response(생성)·sample.label(정답)·sample.metadata(data_source 라우팅 키)를 읽는다.
- **megatron-lm 환경 에이전트**: megatron_rl.gsm8k_agent.TfctGSM8KAgent.get_reward 가 gsm8k_score
  를 직접 호출한다(verl/slime 처럼 _SCALAR_SCORERS 라우팅이 아니라 gsm8k 1:1 — 에이전트 자체가
  태스크별이라). 같은 채점 코어(gsm8k_score) 공유 = 통제 변수.

verl·slime 은 같은 스칼라 scorer 레지스트리(_SCALAR_SCORERS)를 공유하고, megatron-lm 에이전트는
gsm8k_score 를 직접 쓴다 — 노출 시그니처만 다르고 채점 로직은 동일(통제 변수).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# 최종답 추출 우선순위: \boxed{...} > '#### 뒤' > 텍스트의 마지막 숫자.
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASHED = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_number(text: str) -> str | None:
    """문자열에서 숫자 하나를 정규화해 뽑는다(쉼표 제거, 끝 .0 제거)."""
    m = _NUMBER.search(text)
    if not m:
        return None
    num = m.group(0).replace(",", "").rstrip(".")
    if num.endswith(".0"):
        num = num[:-2]
    return num


def _completion_text(completion: Any) -> str:
    """conversational(list[message]) / plain(str) 양쪽에서 텍스트를 뽑는다."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return str(completion)


def _predicted_answer(text: str) -> str | None:
    for pat in (_BOXED, _HASHED):
        m = pat.search(text)
        if m:
            got = _norm_number(m.group(1))
            if got is not None:
                return got
    return _norm_number(text.split("\n")[-1]) or _norm_number(text)


def gsm8k_correctness_reward(
    completions: list[Any], answer: list[str], **kwargs: Any
) -> list[float]:
    """정답 일치 = 1.0, 불일치/추출실패 = 0.0 (주 신호)."""
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = _predicted_answer(_completion_text(completion))
        rewards.append(1.0 if pred is not None and pred == _norm_number(gold) else 0.0)
    return rewards


def gsm8k_format_reward(
    completions: list[Any], **kwargs: Any
) -> list[float]:
    """최종답을 채점 가능한 형식(\\boxed{} 또는 '#### ')으로 냈으면 0.1 (형식 유도)."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        has_format = bool(_BOXED.search(text) or _HASHED.search(text))
        rewards.append(0.1 if has_format else 0.0)
    return rewards


# reward 세트 = 태스크 1개당 함수 리스트. GRPO 는 여러 reward 의 합을 advantage 계산에 쓴다.
REWARDS: dict[str, list[Callable[..., list[float]]]] = {
    "gsm8k": [gsm8k_correctness_reward, gsm8k_format_reward],
}


def get_reward_funcs(name: str) -> list[Callable[..., list[float]]]:
    try:
        return REWARDS[name]
    except KeyError:
        raise ValueError(f"unknown reward set: {name!r}") from None


# --- 스칼라 scorer (verl·slime 공유) ------------------------------------------
# verl/slime 은 reward 를 sample 당 스칼라로 받는다(TRL 의 list 반환과 시그니처 다름). 같은 태스크의
# TRL reward 세트 합과 동일한 점수를 내도록 채점 코어를 공유한다(통제 변수).


def gsm8k_score(solution_str: str, ground_truth: str) -> float:
    """gsm8k reward 스칼라 = 정답 일치(1.0) + 형식(0.1). TRL 세트의 합과 동일."""
    pred = _predicted_answer(solution_str)
    correct = 1.0 if pred is not None and pred == _norm_number(ground_truth) else 0.0
    has_format = bool(_BOXED.search(solution_str) or _HASHED.search(solution_str))
    return correct + (0.1 if has_format else 0.0)


# data_source(태스크 라우팅 키) → 스칼라 scorer. REWARDS(TRL) 와 같은 태스크 키를 공유한다.
_SCALAR_SCORERS: dict[str, Callable[[str, str], float]] = {
    "gsm8k": gsm8k_score,
}


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None
) -> float:
    """verl custom_reward_function 진입점 — data_source 로 태스크 scorer 를 골라 채점한다.

    trainers/verl_grpo.py 가 이 모듈 파일 경로 + 함수명("compute_score")을
    custom_reward_function.path/name 으로 verl 에 넘긴다.
    """
    scorer = _SCALAR_SCORERS.get(data_source)
    if scorer is None:
        raise ValueError(f"no reward scorer for data_source: {data_source!r}")
    return scorer(solution_str, ground_truth)


async def slime_rm(args: Any, sample: Any) -> float:
    """slime --custom-rm-path 진입점 — Sample 에서 응답·정답·data_source 를 읽어 채점한다.

    trainers/slime_grpo.py 가 "...adapters.rewards.slime_rm" 모듈경로를 --custom-rm-path 로 넘긴다.
    sample.metadata["data_source"](trainer 가 JSONL 에 주입)로 compute_score 와 같은 scorer 선택.
    async 인 이유: slime reward 규약이 코루틴(원격 RM·sandbox 채점도 동일 인터페이스).
    """
    metadata = getattr(sample, "metadata", None) or {}
    data_source = metadata.get("data_source")
    scorer = _SCALAR_SCORERS.get(data_source)
    if scorer is None:
        raise ValueError(f"no reward scorer for data_source: {data_source!r}")
    return scorer(sample.response, sample.label)
