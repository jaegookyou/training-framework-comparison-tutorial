"""NeMo-RL DPO 학습 경로 (offline preference, full|lora).

위키 계획대로 DPO 가로비교에 NeMo-RL 추가(헤비/Megatron·DTensor DPO) — 기존 TRL·Unsloth(경량)에
대규모 인프라 축을 더한다. 진입 = `examples/run_dpo.py --config <base> <hydra overrides>`.

데이터 = offline 선호쌍(다른 DPO 와 동일 도메인 = 통제비교): trl-lib/ultrafeedback_binarized 를
NeMo PreferenceDataset 의 data_path 로 넘긴다(NeMo 가 HF 데이터셋 id 를 직접 인제스트). DPO 는 생성·
reward 환경이 없으므로(offline) 커스텀 환경 불필요 → 런처 없이 run_dpo.py 직접 호출.

LoRA = DTensor v2 의 lora_cfg(NeMo lora.md: DPO 지원). full 은 lora_cfg.enabled=false.

⚠️ GPU 검증 대기: ultrafeedback_binarized 가 NeMo PreferenceDataset 포맷(chosen/rejected)과 맞는지·
data_path 인제스트 경로·DPO base config override 키는 NeMo-RL 이미지 end-to-end 에서 최종 확인.
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
        # offline 선호쌍: HF 데이터셋 id 를 PreferenceDataset data_path 로(다른 DPO 와 동일 도메인).
        "data.train.dataset_name=PreferenceDataset",
        f"data.train.data_path={ds['hf_path']}",
        f"data.train.split={ds.get('split', 'train')}",
    ]

    # 로컬/스모크: max_steps>0 이면 dpo.max_num_steps 로 그만큼만. 키는 v0.5.0 실물 확인
    # (examples/configs/dpo.yaml: dpo.max_num_steps/val_at_start). val 데이터 미제공 → 시작 eval 끔.
    max_steps = cfg.section("debug").get("max_steps", -1)
    if max_steps and max_steps > 0:
        overrides += [f"dpo.max_num_steps={max_steps}", "dpo.val_at_start=false"]

    nemo.run("run_dpo.py", nm.get("base_config", "dpo.yaml"), overrides)
