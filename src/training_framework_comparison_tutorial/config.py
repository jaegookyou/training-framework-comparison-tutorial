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

    def section(self, name: str) -> dict[str, Any]:
        return self.data.get(name, {})

    def run_name(self) -> str:
        ds = self.section("dataset").get("source", "?")
        model = self.section("model").get("name", "?").split("/")[-1]
        return f"{self.method}-{model}-{ds}-{self.framework}"
