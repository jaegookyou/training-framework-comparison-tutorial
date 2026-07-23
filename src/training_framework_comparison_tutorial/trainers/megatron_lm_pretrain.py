"""순수 Megatron-LM 사전학습 경로 (continued-pretrain·full) — 수직 파이프라인 1단계 + 가로비교.

**continued-pretrain 전용**(`model.init_from`=Qwen3-8B-Base 필수, from-scratch 제거): 8B 가중치를
시드로 이어학습. 사전·사후를 같은 8B 로 통일(torchtitan continued 의 Megatron 짝). 학습 루프는
**순수 pretrain_gpt.py**(기업이 콕 집어 요구하는 'Megatron-LM 경험'), HF↔mcore 변환만 Bridge
(AutoBridge import/export) 글루로 쓴다 — 순수 Megatron-LM 의 tools/checkpoint/convert.py 는 qwen3
HF 로더가 없어(qwen2.5까지) 8B 시드를 못 만들기 때문. 그래서 이 config 는 megatron-bridge 이미지
(AutoBridge + Megatron-LM repo clone)를 쓴다.

arch 플래그·값 = upstream core_v0.17.1 의 examples/rl/model_configs/qwen3_8b.sh(arch)·examples/gpt3/
train_gpt3_175b_distributed.sh(training/data/logging) 미러(같은 태그라 버전 정합, 추정 아님).
arch = model_sizes.megatron_arch_args(8b=Qwen3-8B untied).

train() subprocess 단계:
  1. torchrun _megatron_bridge_entry --stage convert : init_from HF → mcore 시드.
  2. wikitext → JSONL → tools/preprocess_data.py 인덱싱(.bin/.idx).
  3. torchrun pretrain_gpt.py <arch> <training> <data> --pretrained-checkpoint <시드> --finetune
     (가중치만 로드, 옵티마이저 fresh).
  4. torchrun _megatron_bridge_entry --stage export : 학습 mcore → out/hf(파이프라인).

무거운 deps(megatron-core/TE/bridge)는 이미지에만 있고 torchrun/preprocess 서브프로세스가 임포트.
이 호스트 모듈은 datasets/transformers/yaml 만 지연 임포트.

⚠️ GPU 검증 대기: Megatron 은 CPU 스모크 불가(TE/CUDA) → 학습 루프·preprocess 인자·arch boolean 관례·
wandb 플래그명은 GPU 에서 확인. continued 추가 항목: **pretrain_gpt.py 가 AutoBridge.import_ckpt
산출 mcore 를 `--pretrained-checkpoint`+`--finetune` 로 로드하는 정합**(둘 다 torch_dist 0.17.1 라
원칙 호환, arch 메타 정합 확인) · 8b untie/arch · AutoBridge export_ckpt → HF · TP4 병렬.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..config import RunConfig
from ..model_sizes import megatron_arch_args
from . import _dist

_BRIDGE_ENTRY = "training_framework_comparison_tutorial.trainers._megatron_bridge_entry"


def _stage_tokenizer(cfg: RunConfig, work: Path) -> str:
    """Qwen3 토크나이저를 디렉토리로 저장하고 경로를 돌려준다.

    사전학습은 raw text 라 chat template 불필요(SFT/RL 이 같은 vocab 151936 토크나이저를 이어 써
    파이프라인 정합). preprocess 와 학습이 같은 토크나이저를 쓰도록 한 디렉토리를 공유한다.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.section("model")["tokenizer"])
    tdir = work / "hf_tokenizer"
    tok.save_pretrained(str(tdir))
    return str(tdir)


def _prepare_indexed_data(cfg: RunConfig, work: Path, tokenizer_dir: str, repo: str) -> str:
    """wikitext → JSONL → Megatron 인덱스(.bin/.idx). data-path prefix 를 돌려준다.

    Megatron GPTDataset 은 preprocess_data.py 로 만든 인덱스를 요구한다(HF 스트리밍 아님). json-keys
    text → 산출 `<prefix>_text_document`. 빈 줄(wikitext 헤더/공백)은 제외.
    """
    from datasets import load_dataset

    ds_cfg = cfg.section("dataset")
    mg = cfg.section("megatron")
    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])

    jsonl = work / "wikitext.jsonl"
    with jsonl.open("w") as f:
        for row in raw:
            text = (row.get("text") or "").strip()
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    prefix = work / "wikitext"
    preprocess = Path(repo) / "tools" / "preprocess_data.py"
    cmd = [
        "python",
        str(preprocess),
        "--input", str(jsonl),
        "--json-keys", "text",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_dir,
        "--output-prefix", str(prefix),
        "--append-eod",
        "--workers", str(mg.get("preprocess_workers", 8)),
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    return f"{prefix}_text_document"


def train(cfg: RunConfig) -> None:
    model_cfg = cfg.section("model")
    hp = cfg.section("hp")
    out = cfg.section("output")
    scale = cfg.section("scale")
    mg = cfg.section("megatron")
    wandb_cfg = cfg.section("wandb")

    repo = os.environ.get("MEGATRON_LM_DIR", "/opt/Megatron-LM")
    init_from = model_cfg.get("init_from")  # continued-pretrain(8B 시드 이어학습) — 필수
    if not init_from:
        raise SystemExit(
            "Megatron-LM 사전학습은 continued-pretrain 전용이다(from-scratch 제거). "
            "model.init_from 에 시드 모델(예: Qwen/Qwen3-8B-Base)을 지정해야 한다."
        )

    out_dir = Path(out.get("local_dir", "out"))
    work = out_dir / "megatron_pretrain_workspace"
    dcp_dir = work / "checkpoint"   # DCP 체크포인트(torch_dist) — 학습 저장/재개
    work.mkdir(parents=True, exist_ok=True)

    topo = _dist.resolve(scale)
    tp = mg.get("tensor_model_parallel_size", 1)
    pp = mg.get("pipeline_model_parallel_size", 1)
    # dp = 전체 프로세스(nodes×gpus) / (tp×pp). 멀티노드면 TP·PP 는 노드 내에 두고 DP 로 노드를
    # 가로지르는 게 표준(TP 통신은 NVLink 내부, DP 는 노드 간). global_bs 눈금도 world_size 기준.
    dp = max(1, topo.world_size // (tp * pp))

    # --- continued-pretrain: HF init_from → mcore 시드 (Bridge 변환 글루) ---
    pretrained_ckpt: str | None = None
    if init_from:
        import yaml

        # entry(torchrun)가 다시 읽을 병합 config 를 떨군다(bridge SFT 호스트 패턴).
        run_yaml = work / "run.yaml"
        run_yaml.write_text(yaml.safe_dump(cfg.data, allow_unicode=True, sort_keys=False))
        pretrained_ckpt = str(work / "mcore_init")
        subprocess.run(
            [
                "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=1",
                "-m", _BRIDGE_ENTRY, "--stage", "convert",
                "--config", str(run_yaml), "--megatron-path", pretrained_ckpt,
            ],
            check=True,
        )

    tokenizer_dir = _stage_tokenizer(cfg, work)
    data_path = _prepare_indexed_data(cfg, work, tokenizer_dir, repo)

    micro = hp["per_device_batch_size"]
    global_bs = micro * dp  # 사전학습엔 grad_accum knob 없음 → global = micro × dp

    steps = hp.get("steps", 2000)
    debug_steps = cfg.section("debug").get("max_steps", -1)
    if debug_steps and debug_steps > 0:
        steps = debug_steps

    seq_len = model_cfg.get("max_seq_len", 2048)
    lr = float(hp["learning_rate"])
    min_lr = float(hp.get("min_learning_rate", lr / 10))

    args = [
        # ARCH (model_sizes: 8b=Qwen3-8B untied, qwen3_8b.sh 미러).
        *megatron_arch_args(model_cfg["size"], seq_len),
        # 병렬 / 백엔드
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        "--attention-backend", "auto",
        "--bf16",
        # TRAINING (gpt3 예제 미러)
        "--micro-batch-size", str(micro),
        "--global-batch-size", str(global_bs),
        "--train-iters", str(steps),
        "--lr", str(lr),
        "--min-lr", str(min_lr),
        "--lr-decay-style", hp.get("lr_scheduler", "cosine"),
        "--lr-decay-iters", str(steps),
        "--lr-warmup-fraction", str(hp.get("warmup_ratio", 0.02)),
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--clip-grad", "1.0",
        "--init-method-std", "0.02",
        # TOKENIZER (Qwen3, vocab 151936 = 파이프라인 정합)
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_dir,
        # DATA (val split 없음 → split 100,0,0 + eval-iters 0)
        "--data-path", data_path,
        "--split", "100,0,0",
        # CKPT / LOGGING
        "--save", str(dcp_dir),
        "--load", str(dcp_dir),
        "--ckpt-format", "torch_dist",
        "--save-interval", str(mg.get("save_interval", 1000)),
        "--log-interval", str(mg.get("log_interval", 10)),
        "--eval-interval", str(mg.get("save_interval", 1000)),
        "--eval-iters", "0",
        "--wandb-project", wandb_cfg.get("project", "tfct-pretrain"),
        "--wandb-exp-name", cfg.run_name(),
    ]

    # continued-pretrain: mcore 시드의 가중치만 로드 + 옵티마이저/iteration fresh(--finetune).
    # (--load 가 가리키는 dcp_dir 에 재개 ckpt 가 없을 때만 시드를 쓴다 — megatron 표준 동작.)
    if pretrained_ckpt:
        args += ["--pretrained-checkpoint", pretrained_ckpt, "--finetune"]

    # convert(위 --standalone --nnodes=1)는 노드마다 로컬 mcore_init 을 만든다(결정적 → 동일).
    # train 은 랑데부: 단노드 standalone / 멀티노드 static(node_rank·master_addr) — _dist 판단.
    cmd = [
        "torchrun",
        *_dist.torchrun_args(topo),
        str(Path(repo) / "pretrain_gpt.py"),
        *args,
    ]
    env = {**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"}
    subprocess.run(cmd, check=True, cwd=repo, env=env)

    # --- continued-pretrain: 학습 결과 mcore → HF export (파이프라인 다음 단계 model.name) ---
    # **head 에서만.** 멀티노드면 train() 이 노드마다 도니 가드 없으면 워커도 export 한다.
    # ⚠️ torch_dist 분산 체크포인트는 rank 별 shard 라, 멀티노드에선 dcp_dir 이 공유 FS 여야
    # head 가 완전한 체크포인트를 읽어 export 한다(SkyPilot file_mounts/NFS). 공유 FS 없는
    # 멀티노드 스모크는 train 랑데부까지만 검증하고 export 는 건너뛴다(GPU 검증 대기).
    if init_from and topo.is_head:
        hf_dir = out_dir / "hf"
        subprocess.run(
            [
                "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=1",
                "-m", _BRIDGE_ENTRY, "--stage", "export",
                "--config", str(work / "run.yaml"),
                "--megatron-path", str(dcp_dir), "--hf-path", str(hf_dir),
            ],
            check=True,
        )
