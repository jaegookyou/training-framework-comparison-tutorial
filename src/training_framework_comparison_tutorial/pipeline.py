"""수직 파이프라인 러너 (Phase B) — 단일 모델이 PT→SFT→RL 단계를 이어서 통과.

가로 통제비교(각 단계 standalone `tfct-run`)와 **별개 산출물**이다. 핵심 설계 = **얇은 레이어**:
trainer 로직을 재구현하지 않고, 기존 standalone 단계 config 들을 순서대로 돌리며 단계 사이에
**model.name(다음 입력) ← 이전 단계 산출 HF 경로** 만 이어준다(단계 간 인터페이스 = HF 체크포인트).

  pretrain(continual 8B) → out/hf ─┐
                                    ├→ SFT  model.name=앞 out → out ─┐
                                    │                                  ├→ RL model.name=앞 out
선언적 spec(pipelines/*.yaml)이 단계 config 리스트를 나열하고, 러너는:
  1. 각 단계 config 를 RunConfig 로 로드(extends 병합 그대로).
  2. output.local_dir 를 단계별 분리 디렉토리로 override.
  3. 첫 단계 외에는 model.name 을 직전 단계 산출 모델 경로로 override.
  4. run.dispatch(cfg) 로 단계를 돌린다(standalone 과 동일 dispatch = 단일 진입).

단독 실행(`tfct-run --config X`)은 그대로 동작한다 — 러너는 additive 레이어일 뿐, 단계 config 를
수정하지 않는다("이어지는 배선"과 "안 이어지는 배선"이 따로가 아니라, 한 standalone 위 얇은 재사용).

⚠️ GPU 검증 대기: 단계 실행 자체는 각 trainer 의 GPU 의존(전 단계 미검증)이라 파이프라인 end-to-end
도 GPU 에서 확인 대상. 이 모듈의 CPU 검증 범위 = 경로 threading·config 빌드·dispatch 순서(plan).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from .config import RunConfig
from .run import dispatch


def _override(cfg: RunConfig, **sections: dict[str, Any]) -> RunConfig:
    """RunConfig(frozen)의 일부 섹션 키만 갈아끼운 새 RunConfig 를 만든다(원본 불변)."""
    data = copy.deepcopy(cfg.data)
    for section, values in sections.items():
        data.setdefault(section, {}).update(values)
    return RunConfig(data)


def _stage_output_model(cfg: RunConfig, stage_dir: Path) -> Path:
    """단계가 산출한 **HF 모델 경로**를 돌려준다(다음 단계 model.name 이 됨).

    프레임워크마다 산출 위치가 다르다 — 추정 말고 각 trainer 의 실제 저장 경로에 1:1 로 맞춘다:
      - pretrain(torchtitan·megatron-lm): convert_to_hf/export → `<dir>/hf`.
      - megatron-lm SFT: export.sh → `<dir>/hf`.
      - trl·unsloth(full): trainer.save_model(output_dir) → `<dir>` 통째로 HF 모델.
    그 외(torchtitan SFT=DCP 직출력, slime/nemo/verl/bridge=프레임워크별 ckpt)는 HF export 경로를
    파이프라인용으로 배선한 뒤 규칙을 추가해야 한다 → 명시적 에러(조용한 오연결 방지).
    """
    method, framework = cfg.method, cfg.framework
    if method == "pretrain":
        return stage_dir / "hf"
    if framework in {"trl", "unsloth"}:
        return stage_dir
    if framework == "megatron-lm":
        return stage_dir / "hf"
    raise ValueError(
        f"파이프라인 산출 경로 규칙 미정의: {method}/{framework!r} — 이 프레임워크의 HF export "
        f"경로를 _stage_output_model 에 추가해야 단계 핸드오프가 가능하다."
    )


def plan_pipeline(spec: dict[str, Any]) -> list[RunConfig]:
    """spec → 단계별로 경로가 이어진(threaded) RunConfig 리스트. 실행은 안 한다(테스트·dry-run).

    첫 단계는 자기 model(size/init_from 또는 name)을 쓰고, 이후 단계는 model.name 을 직전 단계의
    산출 HF 경로로 갈아끼운다. 소비되는(마지막 아닌) 단계는 full HF 를 내야 하므로 lora 면 막는다
    (lora 어댑터를 model.name 으로 넘기면 base 로 로드 안 됨 — 수직 파이프라인은 full 핸드오프).
    """
    stages = spec.get("stages") or []
    if not stages:
        raise ValueError("pipeline spec 에 stages 가 비었다.")
    workspace = Path(spec.get("workspace", "out/pipeline"))

    planned: list[RunConfig] = []
    prev_model: Path | None = None
    last = len(stages) - 1
    for i, stage_path in enumerate(stages):
        cfg = RunConfig.from_file(stage_path)
        stage_dir = workspace / f"stage{i}_{cfg.method}_{cfg.framework}"

        overrides: dict[str, dict[str, Any]] = {"output": {"local_dir": str(stage_dir)}}
        if i > 0:
            overrides["model"] = {"name": str(prev_model)}
        cfg = _override(cfg, **overrides)

        # 소비되는 단계(다음 단계의 입력)는 full HF 를 내야 한다.
        if i < last and cfg.method != "pretrain" and cfg.tuning == "lora":
            raise ValueError(
                f"stage{i}({cfg.method}/{cfg.framework})는 다음 단계 입력이라 full 이어야 한다 "
                f"(lora 어댑터는 model.name 핸드오프 불가). tuning=full 인 config 를 써라."
            )

        planned.append(cfg)
        prev_model = _stage_output_model(cfg, stage_dir)
    return planned


def run_pipeline(spec: dict[str, Any]) -> None:
    """spec 의 단계들을 순서대로 실제 실행(각 단계 = run.dispatch). 무거운 trainer 호출."""
    for cfg in plan_pipeline(spec):
        dispatch(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tfct-pipeline")
    parser.add_argument("--pipeline", required=True, help="파이프라인 spec YAML 경로")
    args = parser.parse_args(argv)

    spec = yaml.safe_load(Path(args.pipeline).read_text()) or {}
    run_pipeline(spec)


if __name__ == "__main__":
    main()
