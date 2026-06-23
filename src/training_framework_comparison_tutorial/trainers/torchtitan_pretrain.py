"""torchtitan 사전학습 경로 (from-scratch | continued-pretrain·full) — 수직 파이프라인의 1단계.

Qwen3 를 wikitext 로 사전학습한 뒤, 산출 DCP 체크포인트를 **HF 포맷**으로 export 한다
(`out/<run_name>/hf`). 이 HF ckpt 가 파이프라인의 다음 단계(SFT) 입력이 된다 — 단계 간 인터페이스 =
HF 체크포인트라는 설계 원칙.

두 모드(`model.init_from` 으로 분기):
  - **from-scratch**(init_from 없음): 초소형 Qwen3(tiny)를 랜덤 초기화에서 학습. 파이프라인 연습용.
  - **continued-pretrain**(init_from=Qwen3-8B-Base): 8B 가중치를 시드로 이어학습. 사전·사후를 같은
    8B 로 통일(8B from-scratch 는 데이터 부족으로 무의미 → 이어학습이 의미 있는 통일). 시드는 baked
    config 의 initial_load_in_hf=True 가 hf_assets_path 의 HF 가중치에서 로드(_prepare_assets 참고).

스케일 knob: `model.size`(→ tfct_tiny/0.6B/8B) · `model.init_from`(시드) · `dataset` · `hp.steps` ·
`scale.gpus`(FSDP 자동). 코드는 그대로, config 만 바꾸면 스케일/모드 전환된다.

torchtitan 은 torchrun -m torchtitan.train 런치 모델이라(인프로세스 아님), 무거운 deps 는
이미지(docker/torchtitan.Dockerfile, torchtitan SFT 와 동일 이미지 재사용)에만 있고 torchrun
서브프로세스가 import 한다. 이 호스트 모듈은 토크나이저 준비용 transformers 만 지연 import.

⚠️ GPU 검증 대기: torchtitan 은 GPU/flex_attention 필요라 CPU 스모크 불가 → 학습·HF export·CLI
override 플래그 정확명은 GPU end-to-end 에서 확인(다른 torchtitan 경로와 동일한 단서).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..config import RunConfig
from ..model_sizes import torchtitan_config_fn, torchtitan_flavor


def _prepare_assets(cfg: RunConfig, work: Path) -> str:
    """hf_assets 디렉토리를 만들고 경로를 돌려준다. from-scratch↔continued 에 따라 내용이 다르다.

    torchtitan 은 hf_assets_path 의 토크나이저로 텍스트를 토크나이즈한다(vocab 151936 정합 → SFT/RL
    이 같은 토크나이저를 이어 씀). 두 갈래:
      - from-scratch(tiny 등, init_from 없음): **토크나이저만** 저장. 가중치는 랜덤 초기화.
      - continued-pretrain(8b, init_from=Qwen3-8B-Base): **모델 전체 스냅샷**(가중치+토크나이저)을
        받아둔다 — baked config 의 initial_load_in_hf=True 가 이 디렉토리의 HF 가중치를 시드로 읽어
        이어학습한다(8B from-scratch 는 데이터 부족으로 무의미 → 가중치를 이어받아야 의미 있음).
    """
    model_cfg = cfg.section("model")
    assets = work / "hf_assets"
    init_from = model_cfg.get("init_from")
    if init_from:
        from huggingface_hub import snapshot_download

        # 가중치+토크나이저 전체 — initial_load_in_hf 가 여기서 시드 가중치를 로드한다.
        snapshot_download(repo_id=init_from, local_dir=str(assets))
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_cfg["tokenizer"])
        tok.save_pretrained(str(assets))
    return str(assets)


def train(cfg: RunConfig) -> None:
    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")

    size = model_cfg["size"]
    flavor = torchtitan_flavor(size)
    config_fn = torchtitan_config_fn(size)

    out_dir = Path(out.get("local_dir", "out"))
    work = out_dir / "torchtitan_pretrain_workspace"
    dcp_dir = work / "checkpoint"   # torchtitan DCP 체크포인트 출력
    hf_dir = out_dir / "hf"         # HF export = 다음 파이프라인 단계 입력
    work.mkdir(parents=True, exist_ok=True)

    assets = _prepare_assets(cfg, work)

    gpus = scale.get("gpus", 1)
    micro = hp["per_device_batch_size"]
    steps = hp.get("steps", 10)
    debug_steps = cfg.section("debug").get("max_steps", -1)
    if debug_steps and debug_steps > 0:
        steps = debug_steps

    # baked config 함수(pretrain_qwen3_<size>_wikitext)가 model flavor + wikitext dataloader 를
    # 박고, 가변 HP 는 CLI override (통제비교 값은 _base.yaml 단일 출처).
    overrides = [
        f"--hf_assets_path={assets}",
        f"--training.seq_len={model_cfg.get('max_seq_len', 2048)}",
        f"--training.local_batch_size={micro}",
        f"--training.steps={steps}",
        f"--optimizer.lr={float(hp['learning_rate'])}",
        "--metrics.enable_wandb=true",
        f"--job.dump_folder={dcp_dir}",
    ]
    train_cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        "-m",
        "torchtitan.train",
        "--module",
        "qwen3",
        "--config",
        config_fn,
        *overrides,
    ]
    subprocess.run(train_cmd, check=True, env={**os.environ})

    # DCP → HF export. 산출 HF ckpt 가 파이프라인 다음 단계(SFT)의 model.name 이 된다.
    repo = os.environ.get("TORCHTITAN_DIR", "/opt/torchtitan")
    convert = Path(repo) / "scripts" / "checkpoint_conversion" / "convert_to_hf.py"
    export_cmd = [
        "python",
        str(convert),
        str(dcp_dir),
        str(hf_dir),
        "--model_name",
        "qwen3",
        "--model_flavor",
        flavor,
        "--hf_assets_path",
        assets,
    ]
    subprocess.run(export_cmd, check=True, env={**os.environ})
