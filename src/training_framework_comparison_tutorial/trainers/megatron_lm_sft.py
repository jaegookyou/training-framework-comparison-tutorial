"""순수 Megatron-LM SFT 학습 경로 (full 전용).

verl 백엔드/래퍼가 아니라 **Megatron-LM repo 의 examples/post_training/modelopt** 워크플로를
그대로 구동한다(=기업이 요구하는 'Megatron-LM 경험'). 3단계 셸 오케스트레이션:
  1. convert.sh  : HF Qwen3-8B-Base → Megatron-core 체크포인트 (nvidia-modelopt 글루)
  2. finetune.sh : 그 체크포인트에 SFT (sft_dataset.py 가 messages→conversation+loss 마스킹+packing)
  3. export.sh   : SFT 된 mcore 체크포인트 → HF 포맷

스크립트는 cfg(MLM_MODEL_CFG=Qwen/Qwen3-8B)로 repo 의 conf/<cfg>.sh(arch args)를 source 하고,
env 로 동작을 조정한다(arguments.sh 계약): HF_MODEL_CKPT/TOKENIZER_MODEL/TP/PP/DP/MLM_WORK_DIR/
MLM_MODEL_CKPT/MLM_MODEL_SAVE/DATASET. 이미지(docker/megatron-lm.Dockerfile)가 ① traceinversion
변환기 등록 ② conf 의 TOKENIZER_MODEL 미리세팅 존중을 baked 패치로 넣어둔다.

무거운 deps(megatron/te/modelopt)는 이미지 안에만 있고 torchrun 서브프로세스가 임포트한다. 이
모듈은 transformers 만 지연 임포트(캐논 chat template 구운 토크나이저 저장용).

⚠️ GPU 검증 대기: 학습 루프·HP 세부 매핑(lr/scheduler 의 정확한 MLM_*_ARGS 인자명)·loss 마스킹
정합·캐논 template 적용은 GPU end-to-end 에서 최종 확인 대상(다른 프레임워크 경로와 동일한 단서).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..adapters import resolve_chat_template
from ..config import RunConfig


def _stage_tokenizer(cfg: RunConfig, out_dir: Path) -> str | None:
    """캐논 chat template 을 구운 토크나이저를 저장하고 디렉토리를 돌려준다.

    finetune.py 의 SFTDataset 은 tokenizer.apply_chat_template 로 토크나이즈하므로, 다른
    프레임워크와 동일한 REASONING_CHATML 을 쓰려면 토크나이저에 구워 TOKENIZER_MODEL 로
    가리킨다(이미지의 conf 패치가 미리세팅된 TOKENIZER_MODEL 을 존중). None 이면 모델 자체 template.
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
    if cfg.tuning != "full":
        raise SystemExit(
            f"megatron-lm 은 full SFT 전용이다(post_training/finetune 에 LoRA 없음). "
            f"tuning={cfg.tuning!r} 미지원 — LoRA 는 megatron-bridge/trl/unsloth 경로를 써라."
        )

    model_cfg = cfg.section("model")
    out = cfg.section("output")
    scale = cfg.section("scale")
    ds_cfg = cfg.section("dataset")
    mg = cfg.section("megatron")

    repo = os.environ.get("MEGATRON_LM_DIR", "/opt/Megatron-LM")
    scripts = Path(repo) / "examples" / "post_training" / "modelopt"
    model_cfg_name = mg["model_cfg"]  # 예: Qwen/Qwen3-8B → conf/Qwen/Qwen3-8B.sh

    out_dir = Path(out.get("local_dir", "out"))
    work_dir = out_dir / "megatron_workspace"
    mcore_init = work_dir / "mcore_init"  # convert 산출
    mcore_sft = work_dir / "mcore_sft"    # finetune 산출
    hf_export = out_dir / "hf"            # export 산출
    work_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = _stage_tokenizer(cfg, out_dir)

    # 병렬 곱 = nproc_per_node = gpus (arguments.sh: ETP(=TP)·EP(=1)·PP·CP(=1)·DP 의 곱).
    gpus = scale.get("gpus", 1)
    tp = mg.get("tensor_model_parallel_size", 1)
    pp = mg.get("pipeline_model_parallel_size", 1)
    dp = max(1, gpus // (tp * pp))

    # arguments.sh 계약대로의 공통 env. (HF_MODEL_CKPT = _base 의 Qwen3-8B-Base 로 conf 기본값 덮음)
    base_env = {
        **os.environ,
        "MLM_WORK_DIR": str(work_dir),
        "HF_MODEL_CKPT": model_cfg["name"],
        "TP": str(tp),
        "PP": str(pp),
        "DP": str(dp),
    }
    if tokenizer_dir:
        base_env["TOKENIZER_MODEL"] = tokenizer_dir

    def run(script: str, env_extra: dict[str, str]) -> None:
        cmd = ["bash", str(scripts / script), model_cfg_name]
        subprocess.run(cmd, check=True, env={**base_env, **env_extra}, cwd=str(scripts))

    # 1) HF → mcore
    run("convert.sh", {"MLM_MODEL_SAVE": str(mcore_init)})

    # 2) SFT. DATASET=traceinversion(이미지 baked 변환기가 messages 인식), train-samples ≈ epochs×N.
    #    sft_dataset.py 가 우리 chat template(REASONING_CHATML)로 마스킹/패킹.
    finetune_data_args = (
        f"--train-samples {mg.get('train_samples', 10000)} "
        f"--lr-decay-samples {mg.get('train_samples', 10000)} "
        "--lr-warmup-samples 0 "
        "--split 100,0,0 "
        f"--finetune-hf-dataset {ds_cfg['hf_path']}"
    )
    run(
        "finetune.sh",
        {
            "MLM_MODEL_CKPT": str(mcore_init),
            "MLM_MODEL_SAVE": str(mcore_sft),
            "DATASET": ds_cfg["hf_path"],
            "MLM_DATA_ARGS": finetune_data_args,
        },
    )

    # 3) mcore → HF
    run("export.sh", {"MLM_MODEL_CKPT": str(mcore_sft), "MLM_MODEL_SAVE": str(hf_export)})
