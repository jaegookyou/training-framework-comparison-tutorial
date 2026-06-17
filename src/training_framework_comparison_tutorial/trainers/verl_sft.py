"""verl SFT 학습 경로 (FSDP, full|lora).

verl 은 hydra config + torchrun 서브프로세스로 구동된다 — TRL/Unsloth 의 인프로세스
`trainer.train()` 와 근본적으로 다르다. 이 모듈의 train() 은:
  1. traceinversion → messages 를 **parquet** 으로 떨군다(verl MultiTurnSFTDataset 는
     HF 스트리밍이 아니라 parquet 의 messages 컬럼을 읽는다).
  2. base 모델용 캐논 chat template 을 토크나이저에 구워 로컬에 저장한다(아래 주석 참고).
  3. RunConfig → verl hydra override 로 번역해 `torchrun -m verl.trainer.sft_trainer` 를 띄운다.

무거운 deps(verl/torch/flash-attn)는 docker/verl.Dockerfile 안에만 있다. 이 모듈은 패키지
임포트만으로 끌려오지 않게 datasets/transformers 를 함수 안에서 지연 임포트하고, verl 자체는
torchrun 서브프로세스가 임포트한다(이 프로세스는 verl 을 import 하지 않는다).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..adapters import get_format, get_source, resolve_chat_template
from ..config import RunConfig


def _prepare_parquet(cfg: RunConfig, out_dir: Path) -> str:
    """traceinversion → 정규 messages → {messages} parquet. 경로를 돌려준다."""
    from datasets import load_dataset

    ds_cfg = cfg.section("dataset")
    to_messages = get_source(ds_cfg["source"])
    to_format = get_format(cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: to_format(to_messages(row)),
        remove_columns=raw.column_names,
    )

    path = out_dir / "data" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    return str(path)


def _prepare_tokenizer_dir(cfg: RunConfig, out_dir: Path) -> str | None:
    """캐논 chat template 을 토크나이저에 구워 저장하고 그 디렉토리를 돌려준다.

    왜 파일 우회인가: verl 은 `model.custom_chat_template` 를 받지만, 우리 REASONING_CHATML
    은 `{%`·`{{`·`[`·`'` 를 담은 jinja 라 hydra CLI override 로 넘기면 OmegaConf 파서가
    깨진다. 대신 template 을 구운 토크나이저를 저장하고 `model.tokenizer_path` 로 가리킨다.
    (chat_template 이 없으면 None → verl 이 모델 자체 토크나이저를 그대로 쓴다.)
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
    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    lora = cfg.section("lora")
    debug = cfg.section("debug")

    out_dir = Path(out.get("local_dir", "out"))
    train_parquet = _prepare_parquet(cfg, out_dir)
    tokenizer_dir = _prepare_tokenizer_dir(cfg, out_dir)

    nodes = scale.get("nodes", 1)
    gpus = scale.get("gpus", 1)

    # 효과적 배치 = micro × grad_accum × dp. trl/unsloth 와 같은 눈금을 맞추려고
    # global train_batch_size = per_device × grad_accum × gpus 로 둔다(통제비교).
    micro = hp["per_device_batch_size"]
    global_bs = micro * hp.get("gradient_accumulation", 1) * gpus

    # tuning=lora 면 lora_rank>0. full 이면 0(전체 파라미터). verl 은 model.lora_rank 로 분기.
    lora_rank = lora.get("r", 16) if cfg.tuning == "lora" else 0

    overrides = [
        f"data.train_files={train_parquet}",
        "data.val_files=null",
        "data.messages_key=messages",
        f"data.max_length={model_cfg.get('max_seq_len', 2048)}",
        "data.use_dynamic_bsz=false",  # micro/global 배치 명시 고정(동적 토큰패킹 대신 = trl 정합)
        f"data.train_batch_size={global_bs}",
        f"data.micro_batch_size_per_gpu={micro}",
        "model=hf_model",
        f"model.path={model_cfg['name']}",
        # flash-attn 미설치 이미지 기준(docker/verl.Dockerfile 참고) → sdpa 어텐션.
        # flash-attn 을 깐 GPU 빌드에서 true 로 올리면 remove-padding 최적화가 켜진다.
        "model.use_remove_padding=false",
        f"optim.lr={float(hp['learning_rate'])}",
        f"trainer.default_local_dir={out_dir / 'ckpt'}",
        f"trainer.project_name={cfg.section('wandb').get('project', 'tfct-sft')}",
        f"trainer.experiment_name={cfg.run_name()}",
        f"trainer.total_epochs={hp.get('epochs', 1)}",
        "trainer.logger=[console,wandb]",
        f"trainer.nnodes={nodes}",
        f"trainer.n_gpus_per_node={gpus}",
    ]
    if lora_rank > 0:
        overrides += [
            f"model.lora_rank={lora_rank}",
            f"model.lora_alpha={lora.get('alpha', 32)}",
            f"model.target_modules={lora.get('target_modules', 'all-linear')}",
        ]
    if tokenizer_dir:
        overrides.append(f"model.tokenizer_path={tokenizer_dir}")

    # 로컬/스모크: max_steps>0 이면 그 step 만 돌고 끝.
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        overrides.append(f"trainer.total_training_steps={max_steps}")

    # 단일 노드 다중 GPU = torchrun --standalone. 멀티노드(3단계)는 verl 의 ray 변형
    # (sft_trainer_ray)이 필요 — 그건 추후 sky 멀티노드와 함께 배선한다.
    cmd = [
        "torchrun",
        "--standalone",
        f"--nnodes={nodes}",
        f"--nproc_per_node={gpus}",
        "-m",
        "verl.trainer.sft_trainer",
        *overrides,
    ]
    subprocess.run(cmd, check=True)
