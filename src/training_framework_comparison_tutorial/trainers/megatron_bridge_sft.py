"""Megatron-Bridge SFT 학습 경로 (full|lora) — SFT 트랙의 Megatron 데이터포인트 #2.

순수 Megatron-LM(modelopt 셸 워크플로)과 달리, Megatron-Bridge 는 HF↔Megatron-core
브리지 + 파이썬 학습 레시피다. 학습 루프는 똑같이 Megatron 스택이지만 진입 경로가 달라
좋은 비교축이 된다(기업이 보는 두 스킬: 순수 Megatron-LM / NeMo Megatron-Bridge).

이 모듈(호스트 프로세스)은 megatron 을 import 하지 않는다 — verl/megatron-lm 과 동일하게
torchrun 서브프로세스가 무거운 deps 를 import 한다. 2단계 오케스트레이션:
  1. convert  : HF Qwen3-8B-Base → Megatron-core 체크포인트 (AutoBridge.import_ckpt)
                finetune() 가 pretrained_checkpoint(mcore)를 요구하므로 선행 변환이 필수.
  2. finetune : 그 체크포인트에 recipe(qwen3_8b_sft_config|qwen3_8b_peft_config)로 SFT.
실제 번역(RunConfig → ConfigContainer)·finetune 호출은 _megatron_bridge_entry 가 담당
(torchrun -m 로 띄운다). 이 모듈은 토크나이저만 transformers 지연 import 로 굽는다.

⚠️ GPU 검증 대기: 이미지 빌드(mamba/causal-conv1d/flash-linear-attention 빌드 · cu12 TE
override) · convert/finetune 루프 · 캐논 chat template 의 assistant-only 마스킹 정합은
GPU end-to-end 에서 최종 확인 대상(다른 프레임워크 경로와 동일한 단서).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from ..adapters import resolve_chat_template
from ..config import RunConfig

_ENTRY = "training_framework_comparison_tutorial.trainers._megatron_bridge_entry"


def _stage_tokenizer(cfg: RunConfig, out_dir: Path) -> str | None:
    """캐논 chat template 을 구운 토크나이저를 저장하고 디렉토리를 돌려준다(없으면 None).

    finetune 레시피는 chat-format HF 데이터셋을 `use_hf_tokenizer_chat_template` 로
    토크나이즈한다 → 다른 프레임워크와 같은 REASONING_CHATML 을 쓰려면 토크나이저에 구워
    `cfg.tokenizer.tokenizer_model` 로 가리킨다(entry 가 처리). None 이면 모델 자체 template.
    """
    chat_template = resolve_chat_template(cfg.section("model").get("chat_template"))
    if not chat_template:
        return None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.section("model")["name"])
    tok.chat_template = chat_template
    tdir = out_dir / "tokenizer"
    tok.save_pretrained(str(tdir))
    return str(tdir)


def train(cfg: RunConfig) -> None:
    if cfg.tuning not in ("full", "lora"):
        raise SystemExit(f"megatron-bridge: tuning 은 full|lora 만 지원(받음: {cfg.tuning!r}).")

    out = cfg.section("output")
    scale = cfg.section("scale")

    out_dir = Path(out.get("local_dir", "out"))
    work = out_dir / "megatron_bridge_workspace"
    mcore_init = work / "mcore_init"  # convert 산출 = SFT 의 pretrained_checkpoint
    work.mkdir(parents=True, exist_ok=True)

    # entry 가 다시 읽을 수 있게 병합된 config 를 그대로 떨군다(extends 없는 평면 yaml).
    run_yaml = work / "run.yaml"
    run_yaml.write_text(yaml.safe_dump(cfg.data, allow_unicode=True, sort_keys=False))

    tokenizer_dir = _stage_tokenizer(cfg, out_dir)

    nodes = scale.get("nodes", 1)
    gpus = scale.get("gpus", 1)

    def torchrun(stage: str, nproc: int, *extra: str) -> None:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nnodes={nodes if stage == 'finetune' else 1}",
            f"--nproc_per_node={nproc}",
            "-m",
            _ENTRY,
            "--stage",
            stage,
            "--config",
            str(run_yaml),
            "--megatron-path",
            str(mcore_init),
            *extra,
        ]
        subprocess.run(cmd, check=True)

    # 1) HF base → mcore 체크포인트(단일 프로세스 변환).
    torchrun("convert", 1)

    # 2) 그 체크포인트에 SFT. tokenizer-dir 로 캐논 chat template 주입.
    finetune_extra = ["--tokenizer-dir", tokenizer_dir] if tokenizer_dir else []
    torchrun("finetune", gpus, *finetune_extra)
