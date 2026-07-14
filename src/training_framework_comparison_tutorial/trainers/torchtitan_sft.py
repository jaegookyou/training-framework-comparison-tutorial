"""torchtitan SFT 학습 경로 (full|lora·nightly SHA 핀).

torchtitan 의 SFT(ChatDataset)는 정식 릴리스(0.2.2)엔 없고 main 에만 있어 nightly 전용이다 →
이미지가 torch nightly(cu124) + torchtitan@<SHA> 를 박고, 그 빌드된 이미지를 불변 태그로 박제해
재현성을 확보한다(휠이 증발해도 이미지 환경은 영구 재현 = 컨테이너의 본질). docker/torchtitan.
Dockerfile 참고. LoRA = 네이티브 LoRAConverter(components/lora.py)로 지원 — baked LoRA config 함수
(sft_qwen3_8b_traceinversion_lora)가 같은 flavor 의 model_spec 을 converter 와 함께 재생성한다
(Linear→LoRALinear, base frozen). rank/alpha 는 host 가 env TFCT_LORA_* 로 넘긴다.

런치 = `torchrun -m torchtitan.train --module qwen3 --config <함수> [override]`. config 함수는
이미지 baked patch 가 config_registry.py 에 등록한 `sft_qwen3_8b_traceinversion`(= 기존
sft_qwen3_8b_math 를 우리 traceinversion dataloader 로 교체, sample_processor=from_traceinversion).
torchtitan 은 `initial_load_in_hf=True` 로 HF 가중치를 직접 로드하므로 megatron 식 사전 convert 가
없다 — 이 모듈은 hf_assets(토크나이저에 캐논 chat template 주입 + 모델 가중치) 만 준비하고 torchrun.

무거운 deps(torch nightly·torchtitan)는 이미지 안에만 있다. 이 호스트 모듈은 hf 다운로드용
huggingface_hub/transformers 만 지연 import 하고, torchtitan 은 torchrun 서브프로세스가 import 한다.

⚠️ GPU 검증 대기: 이미지 빌드(torch nightly cu124 + torchtitan@SHA + spmd_types 조합) · torchtitan
CLI override 플래그 정확명(--hf_assets_path/--training.*/--optimizer.lr) · ChatDataset 마스킹 정합
· 캐논 template 적용은 GPU end-to-end 에서 최종 확인 대상(다른 경로와 동일한 단서).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..adapters import resolve_chat_template
from ..config import RunConfig
from . import _dist


def _prepare_hf_assets(cfg: RunConfig, work: Path) -> str:
    """Qwen3-8B-Base 전체(가중치+토크나이저)를 받아 캐논 chat template 을 덮어 디렉토리를 돌려준다.

    torchtitan 의 hf_assets_path 는 토크나이저(apply_chat_template)와 initial_load_in_hf 가중치
    로드의 출처다. base 모델엔 {% generation %} 가 없으니 다른 경로처럼 REASONING_CHATML 을 구워
    넣는다(통제비교 정합). 모델 safetensors 는 그대로 둔다.
    """
    from huggingface_hub import snapshot_download

    model_cfg = cfg.section("model")
    assets = work / "hf_assets"
    snapshot_download(repo_id=model_cfg["name"], local_dir=str(assets))

    chat_template = resolve_chat_template(model_cfg.get("chat_template"))
    if chat_template:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(assets))
        tok.chat_template = chat_template
        tok.save_pretrained(str(assets))
    return str(assets)


def train(cfg: RunConfig) -> None:
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    tt = cfg.section("torchtitan")
    model_cfg = cfg.section("model")
    lora = cfg.section("lora")
    debug = cfg.section("debug")

    out_dir = Path(out.get("local_dir", "out"))
    work = out_dir / "torchtitan_workspace"
    work.mkdir(parents=True, exist_ok=True)

    assets = _prepare_hf_assets(cfg, work)

    topo = _dist.resolve(scale)
    micro = hp["per_device_batch_size"]

    # torchtitan 은 step 단위(epoch 아님). FSDP dp = 전체 프로세스 수(nodes×gpus) →
    # steps ≈ train_samples/(micro×dp). 다른 경로의 train_samples 눈금과 일치(통제비교).
    dp = topo.world_size
    train_samples = tt.get("train_samples", 15000)
    steps = max(1, train_samples // (micro * dp))
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        steps = max_steps

    # baked patch 가 등록한 config 함수. dataset/sample_processor 는 그 함수가 박고, 가변 HP 는
    # CLI override 로 넘긴다(통제비교 값은 _base.yaml 단일 출처 유지).
    overrides = [
        f"--hf_assets_path={assets}",
        f"--training.seq_len={model_cfg.get('max_seq_len', 2048)}",
        f"--training.local_batch_size={micro}",
        f"--training.steps={steps}",
        f"--optimizer.lr={float(hp['learning_rate'])}",
        "--metrics.enable_wandb=true",
        f"--job.dump_folder={out_dir / 'ckpt'}",
    ]

    # tuning 분기: lora 면 baked LoRA config(LoRAConverter 로 model_spec 재생성) + rank/alpha env.
    # torchtitan --config 함수는 인자 못 받음 → env 가 정석. full 은 기본 config.
    env = {**os.environ}
    if cfg.tuning == "lora":
        config_name = "sft_qwen3_8b_traceinversion_lora"
        env["TFCT_LORA_RANK"] = str(lora.get("r", 16))
        env["TFCT_LORA_ALPHA"] = str(lora.get("alpha", 32))
    else:
        config_name = "sft_qwen3_8b_traceinversion"

    # torchrun(FSDP 자동). 단노드는 standalone, 멀티노드는 static 랑데부 — _dist 가 판단한다.
    # hf_assets 준비는 노드마다 돈다(위): initial_load_in_hf 가 각 rank 의 로컬 경로에서
    # 가중치를 읽으므로 head 만 받으면 worker 가 못 연다 — 중복 다운로드가 아니라 필수다.
    cmd = [
        "torchrun",
        *_dist.torchrun_args(topo),
        "-m",
        "torchtitan.train",
        "--module",
        "qwen3",
        "--config",
        config_name,
        *overrides,
    ]
    subprocess.run(cmd, check=True, env=env)
