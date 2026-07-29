"""verl GRPO 학습 경로 (online on-policy RL, full|lora).

verl 의 본진 — GRPO/PPO 가 1급 시민이다. SFT(verl.trainer.sft_trainer, torchrun)와 달리 GRPO 는
`verl.trainer.main_ppo`(ray 기반, hydra)로 구동된다. 이 모듈의 train() 은:
  1. gsm8k → {prompt 체인, reward_model.ground_truth} parquet 을 떨군다(verl RLHFDataset 포맷).
     data_source 컬럼(=reward.name)을 주입해 reward 라우팅 키로 쓴다.
  2. base 모델용 캐논 chat template 을 토크나이저에 구워 로컬에 저장한다(verl_sft 와 동일 우회).
  3. RunConfig → verl hydra override 로 번역해 `python -m verl.trainer.main_ppo` 를 띄운다.
     reward 는 custom_reward_function(adapters.rewards.compute_score)로 rule-based 채점.

reward = 통제 변수: TRL/Unsloth GRPO 와 **같은 gsm8k 채점 코어**(adapters.rewards)를 공유하되,
verl 규약(compute_score 시그니처)으로 노출한 어댑터를 쓴다 → GRPO 가로비교가 성립한다.

rollout: verl 은 vllm rollout 이 기본(actor_rollout_ref.rollout.name=vllm). docker/verl.Dockerfile
에 vllm 을 추가했다(핀은 GPU 빌드 때 확정 — TODO). TRL GRPO 의 vllm 공백이 verl 경로엔 네이티브.

무거운 deps(verl/torch/vllm)는 이미지 안에만 있고 main_ppo 서브프로세스가 임포트한다. 이 모듈은
datasets/transformers 만 함수 안에서 지연 임포트한다(verl 자체는 import 하지 않는다).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..adapters import get_format, get_source, resolve_chat_template
from ..adapters import rewards as rewards_mod
from ..config import RunConfig
from . import _wandb


def _prepare_parquet(cfg: RunConfig, out_dir: Path) -> str:
    """gsm8k → {prompt, reward_model, data_source} parquet. 경로를 돌려준다."""
    from datasets import load_dataset

    ds_cfg = cfg.section("dataset")
    reward_name = cfg.section("reward")["name"]  # data_source = reward 라우팅 키
    to_prompt = get_source(ds_cfg["source"])
    to_format = get_format(cfg.method, cfg.framework)

    raw = load_dataset(ds_cfg["hf_path"], ds_cfg.get("hf_name"), split=ds_cfg["split"])
    subsample = ds_cfg.get("subsample")
    if subsample:
        raw = raw.shuffle(seed=ds_cfg.get("seed", 42)).select(range(min(subsample, len(raw))))

    dataset = raw.map(
        lambda row: {**to_format(to_prompt(row)), "data_source": reward_name},
        remove_columns=raw.column_names,
    )

    path = out_dir / "data" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    return str(path)


def _prepare_tokenizer_dir(cfg: RunConfig, out_dir: Path) -> str | None:
    """캐논 chat template 을 토크나이저에 구워 저장하고 그 디렉토리를 돌려준다.

    REASONING_CHATML 은 jinja 라 hydra CLI override 로 직접 넘기면 OmegaConf 파서가 깨진다
    (verl_sft 와 동일). 대신 구운 토크나이저를 저장해 경로로 가리킨다. None 이면 모델 토크나이저.
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
    verl = cfg.section("verl")
    debug = cfg.section("debug")

    out_dir = Path(out.get("local_dir", "out"))
    train_parquet = _prepare_parquet(cfg, out_dir)
    tokenizer_dir = _prepare_tokenizer_dir(cfg, out_dir)

    nodes = scale.get("nodes", 1)
    gpus = scale.get("gpus", 1)

    # 프롬프트 배치(=rollout 단위). trl/unsloth 와 눈금을 맞추려고
    # train_batch_size = per_device × grad_accum × gpus 로 둔다(통제비교).
    micro = hp["per_device_batch_size"]
    train_bs = micro * hp.get("gradient_accumulation", 1) * gpus
    # ppo_mini_batch_size ≤ train_batch_size. 1 epoch-of-rollout 당 정책 갱신 횟수를 정한다.
    mini_bs = min(verl.get("ppo_mini_batch_size", train_bs), train_bs)

    # custom reward = adapters.rewards.compute_score (이 모듈 파일 경로 + 함수명).
    reward_path = Path(rewards_mod.__file__).resolve()

    gpu_mem_util = verl.get("gpu_memory_utilization", 0.6)  # vllm KV 캐시 점유 비율

    # rollout 추론 백엔드. ⚠️ verl 0.8.0 의 _ROLLOUT_REGISTRY(base.py)에 실제로 등록된 조합은
    # **("vllm","async")·("sglang","async")·("trtllm","async") 셋뿐**이다(2026-07-29 실측).
    # rollout.yaml 주석의 "hf" 는 미끼 — HFRollout 클래스는 있으나 새 server-rollout 레지스트리에
    # 안 배선돼 `rollout.name=hf` 는 "Rollout hf with mode async not found" 로 죽는다(hf 실측 확인).
    # 즉 config 로 쓸 수 있는 건 vllm|sglang|trtllm 이고 셋 다 이미지에 그 엔진이 있어야 한다
    # (현재 이미지엔 vllm 만). vllm 은 Blackwell 에서 weight-sync 3회차에 하드크래시(하단 NOTE)라,
    # 우회하려면 sglang 을 이미지에 추가하는 재빌드가 필요하다 — 이 knob 은 그때 쓸 준비다.
    rollout_backend = verl.get("rollout_backend", "vllm")

    # tuning=lora 면 lora_rank>0. full 이면 0(전체 파라미터). verl 은 model.lora_rank 로 분기.
    lora_rank = lora.get("r", 16) if cfg.tuning == "lora" else 0

    overrides = [
        "algorithm.adv_estimator=grpo",            # GRPO: critic 없이 그룹 정규화 advantage
        f"data.train_files={train_parquet}",
        # verl 은 val_files 를 요구한다 → train 재사용 + test_freq=-1 로 실제 평가는 생략(스모크).
        f"data.val_files={train_parquet}",
        "data.prompt_key=prompt",
        f"data.train_batch_size={train_bs}",
        f"data.max_prompt_length={hp.get('max_prompt_length', 512)}",
        f"data.max_response_length={hp.get('max_completion_length', 1024)}",
        f"actor_rollout_ref.model.path={model_cfg['name']}",
        # ⚠️ verl 은 attention 구현 기본값을 **flash_attention_2 로 하드코딩**한다
        # (verl/workers/config/model.py:185 `override_config.get("attn_implementation",
        # "flash_attention_2")`). 우리 이미지엔 flash-attn 이 없어(nvcc 회피 설계) 그대로 두면
        # 모델 로딩 직후 ImportError 로 죽는다 — 2026-07-28 2노드 런에서 실측.
        # sdpa 로 통일하는 게 통제비교상으로도 옳다: trl 은 transformers 기본(sdpa)로 도는데
        # verl 만 FA2 면 **attention 커널이 프레임워크 축에 섞여** 처리량 비교가 오염된다.
        # attention 구현은 프레임워크 축이 아니라 환경 축이므로 전 프레임워크 동일하게 간다.
        # hydra: override_config 는 스키마상 빈 dict 라 **새 키 추가는 `+` 접두사**가 필요하다
        # (없으면 "Key 'attn_implementation' is not in struct" — 2026-07-28 실측).
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.actor.optim.lr={float(hp['learning_rate'])}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro}",
        "actor_rollout_ref.actor.use_kl_loss=true",   # GRPO 는 reward 대신 loss 에 KL
        f"actor_rollout_ref.actor.kl_loss_coef={hp.get('beta', 0.04)}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        # rollout 백엔드(vllm|hf|sglang). n = 그룹 크기 G(advantage 정규화 단위).
        f"actor_rollout_ref.rollout.name={rollout_backend}",
        f"actor_rollout_ref.rollout.n={hp.get('num_generations', 8)}",
        f"actor_rollout_ref.rollout.temperature={hp.get('temperature', 1.0)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={verl.get('rollout_tp', 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        # ── vLLM CUDA graph 끔(enforce_eager) — colocate 학습 안정성 ──
        # verl 은 매 스텝 갱신된 actor weight 를 vLLM 엔진에 sync 한다(on-policy). vLLM 기본은
        # enforce_eager=false 라 CUDA graph 를 캡처하는데, verl rollout.yaml 이 직접 명시하듯
        # "cudagraph in inference engine **can not be offloaded during update policy**" — 캡처된
        # 그래프가 옛 weight 텐서를 참조한 채 weight sync 가 일어나면 illegal memory access 로
        # 엔진코어가 traceback 없이 하드 크래시한다. 2026-07-29 2노드 런에서 실측: step 1·2 는
        # 통과하고 step 2→3(두 번째 weight sync) 전환에서 "EngineCore died unexpectedly"(양 노드
        # 동시, dmesg OOM 무흔적 = OS-OOM 아님, Python traceback 없음 = SIGKILL/SIGSEGV).
        # actor param/optimizer offload(위)를 켜도 안 나았다 = actor VRAM 경합이 아니라 graph 문제.
        # eager 는 rollout 이 조금 느리지만 안정성이 우선이고, attention·offload 와 같은 환경 축이라
        # 전 프레임워크 통제비교에 영향 없다(생성 결과는 동일, 커널 실행 방식만 다름).
        "actor_rollout_ref.rollout.enforce_eager=true",
        # NOTE(2026-07-29): rollout mode=sync 로 async 서버 weight-sync 크래시를 우회하려 했으나
        # verl 0.8.0 은 sync 모드를 제거했다("Rollout mode 'sync' has been removed" ValueError).
        # async(=AsyncLLM+vLLMHttpServer)가 유일 경로다. 이 경로의 on-policy weight sync 3회차
        # 하드크래시(EngineCore SIGKILL, LoRA·full 둘 다, 메모리·cudagraph 배제됨)는 config 로
        # 못 피한다 = vLLM 0.11 async weight-update 의 upstream 이슈. 남은 레버 = vLLM 버전 교체.
        # verl 은 ref·rollout **각각** log_prob 배치를 요구한다(둘 중 하나라도 없으면 기동 시
        # ValueError). rollout 은 생성 후 log_prob 재계산 단계라 ref 와 별개 knob — 같은
        # micro 를 줘 눈금을 통일한다. 2026-07-28 2노드 런에서 실측(ref 만 주고 있었음).
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro}",
        # ── colocate 메모리: actor·ref 파라미터를 rollout 동안 CPU 로 오프로드 ──
        # 단일 GPU 에 vLLM rollout + actor + ref 가 코로케이트된다. verl 기본은
        # param_offload=false·optimizer_offload=false(engine/fsdp.yaml) 라, rollout 단계에서
        # vLLM 이 KV 풀(gpu_memory_utilization×VRAM ≈ 0.6×96=57GB)을 다시 잡을 때 actor 가
        # GPU 를 안 비켜 충돌 → vLLM 엔진코어가 OOM-kill 당한다. 2026-07-29 2노드 런에서 실측:
        # step 1 은 actor reserved 38GB 로 아슬하게 버텼고(38+57=95<96) step 2 에서 48GB 로
        # 성장하자(48+57=105>96) "EngineCore_DP0 died unexpectedly"(traceback 없는 SIGKILL,
        # 양 노드 동시). free_cache_engine 는 기본 True(학습 중 KV 해제)지만 그것만으론 부족했다
        # — rollout 재점유 시점엔 actor 가 GPU 에 남아 있기 때문. 파라미터·옵티마이저를 CPU 로
        # 내려야 vLLM 이 재점유할 공간이 난다(ref.yaml 도 7B+ 는 offload 권장이라 명시).
        # 이 키들은 fsdp_config(../engine@fsdp_config:fsdp) 실존 필드라 `+` 접두사 불필요.
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        # rule-based reward → 신경망 RM 끔. 채점은 custom_reward_function 으로.
        "reward_model.enable=false",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        f"trainer.default_local_dir={out_dir / 'ckpt'}",
        f"trainer.project_name={_wandb.project(cfg)}",
        f"trainer.experiment_name={cfg.run_name()}",
        f"trainer.total_epochs={hp.get('epochs', 1)}",
        "trainer.logger=[console,wandb]",
        f"trainer.nnodes={nodes}",
        f"trainer.n_gpus_per_node={gpus}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",   # 주기적 평가 생략(val=train 재사용이라 의미 없음)
    ]
    if lora_rank > 0:
        overrides += [
            f"actor_rollout_ref.model.lora_rank={lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={lora.get('alpha', 32)}",
            f"actor_rollout_ref.model.target_modules={lora.get('target_modules', 'all-linear')}",
        ]
    if tokenizer_dir:
        overrides.append(f"actor_rollout_ref.model.tokenizer_path={tokenizer_dir}")

    # 로컬/스모크: max_steps>0 이면 그 step 만 돌고 끝.
    max_steps = debug.get("max_steps", -1)
    if max_steps and max_steps > 0:
        overrides.append(f"trainer.total_training_steps={max_steps}")

    # main_ppo = ray 기반(단일 노드는 로컬 ray). torchrun 이 아니다(SFT 경로와 다름).
    # NOTE: verl 은 API churn 이 잦다 — 정확한 hydra 키/인자명은 docker/verl.Dockerfile 핀 기준
    # GPU end-to-end 에서 최종 검증(다른 프레임워크 경로와 동일한 단서).
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    subprocess.run(cmd, check=True)


def prepare(cfg: RunConfig) -> None:
    """워커 노드용 준비 — 캐논 template 을 구운 토크나이저를 이 노드에 만든다.

    ray 계열은 드라이버가 head 에서만 도는데 `model.tokenizer_path` 로 **파일 경로**를 넘기므로
    워커 노드에도 실물이 있어야 한다(없으면 HF 가 hub id 로 해석해 HFValidationError).
    데이터(parquet)는 드라이버만 읽으므로 여기서 안 만든다 — 노드마다 필요한 것만 만든다.
    호출: `tfct-run --config <cfg> --prepare-only` (sky/ray_bootstrap.sh 의 워커 분기).
    """
    _prepare_tokenizer_dir(cfg, Path(cfg.section("output").get("local_dir", "out")))
