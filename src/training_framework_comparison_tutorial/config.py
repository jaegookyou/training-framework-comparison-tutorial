"""YAML config 로딩 + extends 병합 + 타입드 접근.

통제비교의 핵심은 "공통 축은 _base.yaml 한 곳, 프레임워크별 run 은 override 만".
그걸 deep-merge 로 강제한다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 을 읽고 `extends:` 를 부모 경로 기준으로 재귀 병합해 dict 로 돌려준다."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    extends = data.pop("extends", None)
    if extends is None:
        return data
    base = load_config((path.parent / extends).resolve())
    return _deep_merge(base, data)


@dataclass(frozen=True)
class RunConfig:
    data: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> RunConfig:
        return cls(load_config(path))

    @property
    def framework(self) -> str:
        return self.data["framework"]

    @property
    def image(self) -> str:
        return self.data["image"]

    @property
    def method(self) -> str:
        return self.data.get("method", "sft")

    @property
    def tuning(self) -> str:
        """full | lora. 학습 방식 축(프레임워크와 독립적인 두 번째 변수)."""
        return self.data.get("tuning", "full")

    def section(self, name: str) -> dict[str, Any]:
        return self.data.get(name, {})

    def run_name(self) -> str:
        ds = self.section("dataset").get("source", "?")
        model_sec = self.section("model")
        # 사후학습(sft/rl)은 model.name(HF 경로/ID), 사전학습(continued)은 arch 정합용 model.size.
        model = model_sec.get("name") or model_sec.get("size", "?")
        model = str(model).split("/")[-1]
        parts = [self.method, model, ds, self.framework]
        # tuning(full|lora) 축은 사후학습에만. 사전학습은 정의상 full-param.
        if self.method != "pretrain":
            parts.append(self.tuning)
        # 노드 수는 이름에 박는다 — 같은 config 를 SNSG 로도 MNMG 로도 돌리는데(scale 만 다름)
        # 이름이 같으면 W&B 목록에서 둘을 구분할 수 없다. 단노드는 접미사 없음(기존 이름 보존).
        nodes = int(self.section("scale").get("nodes", 1))
        if nodes > 1:
            parts.append(f"mn{nodes}")
        return "-".join(parts)

    def is_smoke(self) -> bool:
        """이 run 이 스모크인가(= 축소해서 배선만 보는 run).

        스모크 여부에 딸린 정책(W&B 격리 프로젝트·촘촘한 로깅)을 config 마다 손으로 켜면 새 스모크를
        추가할 때 빠뜨린다 → 여기 한 곳에서 파생시킨다.

        판정이 둘인 이유: 보통은 축소 레버 `debug.max_steps>0` 자체가 스모크의 정의다. 그런데
        **megatron-lm 은 그 레버를 소비하지 않는다**(축소 레버가 `megatron.train_samples` 뿐).
        거기에 안 먹는 max_steps 를 적어 넣는 건 거짓말 knob 이므로(repo 원칙), 대신 `debug.smoke`
        로 **의도를 명시**하게 한다.
        """
        debug = self.section("debug")
        if debug.get("smoke") is True:
            return True
        return int(debug.get("max_steps", -1) or -1) > 0

    def log_every_n_steps(self) -> int:
        """로깅 간격(step). 프레임워크마다 이름이 다른 knob 의 **단일 출처**.

        기본을 프레임워크 기본값에 맡기면 눈금이 제각각이 된다(HF Trainer 500 · torchtitan 10 ·
        megatron 10) → 같은 x 축으로 겹쳐 보려면 우리가 정해야 한다. 스모크는 1: 5 step 런에
        간격 10 이면 W&B 에 스칼라가 한 점도 안 찍혀 빈 차트가 된다(= 판정 불가).
        """
        default = 1 if self.is_smoke() else 10
        return int(self.section("wandb").get("log_every_n_steps", default))
