"""컨테이너 안 entrypoint: config 를 읽어 framework 에 맞는 trainer 로 dispatch.

`tfct-run --config configs/sft/....yaml` 로 호출된다(sky yaml 의 run 블록이 이걸 실행).
"""

from __future__ import annotations

import argparse
import importlib

from .config import RunConfig

# (method, framework) -> trainer 모듈. method 축(pretrain/sft/rl)으로 네임스페이스를 나눠
# 단일 모델 PT→SFT→RL 수직 파이프라인과 통제비교(가로)가 같은 dispatch 를 공유한다.
# 새 경로 추가 = 해당 method 아래 한 줄.
_PKG = "training_framework_comparison_tutorial.trainers"
TRAINERS: dict[str, dict[str, str]] = {
    "pretrain": {
        "torchtitan": f"{_PKG}.torchtitan_pretrain",
        # 순수 Megatron-LM pretrain_gpt.py — continued-pretrain 전용(4b, init_from 필수):
        # AutoBridge.import_ckpt 로 4B-Base→mcore 시드(convert.py qwen3 블로커 우회) 후
        # --pretrained-checkpoint+--finetune 로 이어학습, export 도 AutoBridge(bridge 이미지).
        "megatron-lm": f"{_PKG}.megatron_lm_pretrain",
    },
    "sft": {
        "trl": f"{_PKG}.trl_sft",
        "unsloth": f"{_PKG}.unsloth_sft",
        "verl": f"{_PKG}.verl_sft",
        "megatron-lm": f"{_PKG}.megatron_lm_sft",
        "megatron-bridge": f"{_PKG}.megatron_bridge_sft",
        "torchtitan": f"{_PKG}.torchtitan_sft",
        "slime": f"{_PKG}.slime_sft",  # rollout 추상 재활용(sft_rollout): RL 프레임워크 SFT
    },
    # 사후학습 RL 트랙. DPO(offline preference)와 GRPO(online RL)는 패러다임이 달라
    # 별 method 로 둔다(통제비교 = 프레임워크 고정, 방법만 비교). 기준점 = TRL.
    "dpo": {
        "trl": f"{_PKG}.trl_dpo",
        "unsloth": f"{_PKG}.unsloth_dpo",
    },
    # online DPO = 같은 DPO loss 의 on-policy 판(생성+RM 채점). offline DPO 와 별 method 로
    # 둬 "같은 method 의 offline↔online" 비교를 명시한다. Unsloth 는 네이티브 경로 부재 → TRL 단독.
    "online_dpo": {
        "trl": f"{_PKG}.trl_online_dpo",
    },
    "grpo": {
        "trl": f"{_PKG}.trl_grpo",
        "unsloth": f"{_PKG}.unsloth_grpo",
        "verl": f"{_PKG}.verl_grpo",
        "slime": f"{_PKG}.slime_grpo",
        "megatron-lm": f"{_PKG}.megatron_lm_grpo",
        # torchtitan experiments/rl(Monarch+vLLM, GRPO-based). full 전용·experimental. reward=공유
        # gsm8k_score. 별도 cu130 이미지(torchtitan-rl) — SFT/사전학습 cu124 와 코어 스택 다름.
        "torchtitan": f"{_PKG}.torchtitan_grpo",
    },
    # PPO = critic(value model)으로 GAE advantage 추정(GRPO 그룹 정규화와 다름). 대규모 RL 인프라
    # 둘(verl=ray main_ppo / slime=SGLang+Megatron, 전부 rule reward 네이티브)로 가로비교.
    # PPO 는 무거운(critic) 알고리즘이라 대규모 RL 인프라에만 1급 —
    # Unsloth·megatron-lm·torchtitan·bridge 는 네이티브 PPO 없음. TRL 은 neural RM 강제라 제외.
    # (NeMo-RL 은 2026-07-30 제거 — 멀티턴/agentic RL 부적합 + 저채택, verl·slime 집중.)
    "ppo": {
        "verl": f"{_PKG}.verl_ppo",
        "slime": f"{_PKG}.slime_ppo",
    },
}


def dispatch(cfg: RunConfig, prepare_only: bool = False) -> None:
    """RunConfig → (method, framework) 에 맞는 trainer 모듈로 dispatch 해 train() 호출.

    단독 실행(main)과 파이프라인 러너(pipeline)가 공유하는 단일 진입 — 단계 하나를 돌리는 의미는
    한 곳에만 둔다(파이프라인은 이걸 단계마다 부른다).
    """
    by_method = TRAINERS.get(cfg.method)
    if by_method is None:
        raise SystemExit(f"no trainers registered for method: {cfg.method!r}")
    module_name = by_method.get(cfg.framework)
    if module_name is None:
        raise SystemExit(
            f"no trainer registered for {cfg.method}/{cfg.framework!r}"
        )
    # 멀티노드 미배선 조합에 nodes>1 을 주면 여기서 정직하게 죽인다(거짓말 knob 방지).
    # 배선된 trainer 는 내부에서 _dist.resolve 로 실제 프로비저닝까지 재확인한다.
    from .trainers import _dist, _preflight, _wandb
    _dist.guard_wired(cfg.method, cfg.framework, cfg.section("scale"))
    # 이미지 안 torch 가 이 GPU 를 돌릴 수 있나(수 ms). 못 하면 원인을 적어 즉시 죽는다 —
    # 안 그러면 한참 뒤 collective 에서 원인 불명 에러가 난다(07-28 실측). 근거는 _preflight.
    _preflight.check_gpu_arch()
    # W&B 식별자(project/name/group/tags)를 env 로 주입 — 8 프레임워크가 wandb 를 부르는 방식이
    # 제각각이라 여기(공통 통로) 한 곳이 유일하게 안 어긋나는 자리다. 상세 근거는 _wandb.
    _wandb.apply_env(cfg)
    module = importlib.import_module(module_name)
    if prepare_only:
        _run_prepare(module, module_name, cfg)
        return
    module.train(cfg)


def _run_prepare(module, module_name: str, cfg: RunConfig) -> None:
    """이 노드에 필요한 **노드-로컬 산출물**만 만들고 학습은 하지 않는다.

    왜 필요한가(2026-07-28 verl GRPO 2노드 런에서 실측): **ray 메커니즘만** 드라이버가 head 에서만
    돌고 워커는 `sleep infinity` 로 대기한다 → 드라이버가 만든 파일이 워커엔 없다. 그런데 우리는
    캐논 chat template 을 구운 토크나이저를 **파일 경로로** 넘긴다(jinja 를 CLI 로 넘기면 hydra/
    OmegaConf 파서가 깨져서 택한 우회) → 워커에서
      `HFValidationError: Repo id must be in the form 'repo_name' ...: '/workspace/out/tokenizer'`
    로 죽는다(경로가 없으니 HF 가 hub id 로 해석).

    **torchrun 계열은 이 문제가 없다** — 모든 노드가 train() 을 돌아 각자 굽는다(verl SFT 가
    2노드에서 그냥 통과한 이유). 그래서 이 진입점은 ray 계열만 쓴다.

    공유 FS 대신 노드별 재생성을 택한 이유: 구운 토크나이저는 **(모델, chat_template)의 결정적
    함수**라 노드마다 만들어도 같은 결과다. 공유 스토리지 의존(자격증명·마운트)을 안 늘리는 게 싸다.
    (megatron 의 분산 체크포인트 export/resume 은 성격이 다르다 — 그건 여전히 공유 FS 필요.)
    """
    prepare = getattr(module, "prepare", None)
    if prepare is None:
        # 준비할 게 없는 트레이너도 정상이다(모델을 hub id 로 넘기는 경로 등) — 조용히 넘어가지 말고
        # 명시한다. "왜 아무 일도 안 했지?"를 로그에서 바로 답할 수 있게.
        print(f"[prepare] {module_name}: 노드-로컬 준비물 없음 — 넘어간다")
        return
    print(f"[prepare] {module_name}: 노드-로컬 준비물 생성")
    prepare(cfg)
    print(f"[prepare] {module_name}: 완료")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tfct-run")
    parser.add_argument("--config", required=True, help="run config YAML 경로")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="학습하지 않고 이 노드에 필요한 산출물(캐논 template 구운 토크나이저 등)만 만든다. "
             "ray 계열 멀티노드의 워커 노드에서 쓴다(sky/ray_bootstrap.sh).",
    )
    args = parser.parse_args(argv)

    dispatch(RunConfig.from_file(args.config), prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
