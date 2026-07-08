"""NeMo-RL SFT 학습 경로 (full|lora).

NeMo-RL 은 SFT 도 1급(DTensor/Megatron 백엔드, LoRA 지원). 통제비교 SFT 트랙에 대규모 인프라 축을
하나 더한다. 진입 = `examples/run_sft.py --config <base> <hydra overrides>`.

데이터 = reasoning distill TraceInversion(다른 SFT 와 동일 = 통제비교)을 data_path 로 넘긴다
(NeMo 가 HF 데이터셋 직접 인제스트, messages 컬럼). SFT 는 reward·환경 없음 →
런처 없이 run_sft.py 직접 호출. 캐논 REASONING_CHATML 은 baked 토크나이저로 적용(loss 마스킹 정합).

LoRA = DTensor v2 의 lora_cfg(NeMo lora.md: SFT 지원). full 은 lora_cfg.enabled=false.

⚠️ GPU 검증 대기: TraceInversion(messages)이 NeMo sft_processor 포맷과 맞는지·data_path 인제스트·
SFT base config override 키·캐논 template loss 마스킹은 NeMo-RL 이미지 end-to-end 에서 최종 확인.
"""

from __future__ import annotations

from pathlib import Path

from ..config import RunConfig
from . import _nemo_rl_common as nemo


def train(cfg: RunConfig) -> None:
    ds = cfg.section("dataset")
    nm = cfg.section("nemo")
    out_dir = Path(cfg.section("output").get("local_dir", "out"))

    tok_dir = nemo.bake_tokenizer(cfg, out_dir)
    overrides = nemo.common_overrides(cfg, out_dir, tok_dir)
    overrides += [
        # reasoning distill: HF 데이터셋 id 를 data_path 로(다른 SFT 와 동일 데이터 = 통제비교).
        f"data.train.data_path={ds['hf_path']}",
        f"data.train.split={ds.get('split', 'train')}",
    ]

    # 로컬/스모크: max_steps>0 이면 sft.max_num_steps 로 그만큼만 돌고 끝.
    # 키는 NeMo-RL v0.5.0 examples/configs/sft.yaml 실물 확인(sft.max_num_steps/val_at_start).
    # val 데이터를 안 넘기므로 시작 eval(val_at_start) 은 스모크에서 끈다.
    debug = cfg.section("debug")
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        overrides += [f"sft.max_num_steps={max_steps}", "sft.val_at_start=false"]

    nemo.run("run_sft.py", nm.get("base_config", "sft.yaml"), overrides)
