# training-framework-comparison-tutorial

6개 프레임워크(**Megatron · torchtitan · TRL · Unsloth · verl · slime**)로 **사전학습 → 사후학습(SFT·선호최적화·RL)**을 **통제비교**하는 학습 lab. Vast.ai에서 빌린 GPU로 **single GPU → single-node multi-GPU → multi-node multi-GPU** 3단계 스케일을 밟고, 모든 run을 **W&B**로 로깅해 처리량·VRAM·수렴을 한 화면에 겹쳐 본다.

## 무엇을 비교하나

- **사전학습**: torchtitan, Megatron (초소형 모델 + 작은 코퍼스)
- **사후학습**: 5개 프레임워크로 SFT / DPO / GRPO 등 유명 방법론 두루
  - Megatron 사후학습 = verl + Megatron 백엔드
  - torchtitan 사후학습 = torchforge (불안정 → 도전 챕터로 격리)
- **통제**: 단계별 데이터·모델·HP·reward를 고정하고 **프레임워크만 변수**로 둠 (controlled comparison)
- **스케일**: Unsloth는 1 GPU만, 나머지 4개는 전 스케일

## 설치

```bash
pip install git+https://github.com/jaegookyou/training-framework-comparison-tutorial
```

## 구조

```
configs/<method>/   # method별(pretrain·sft·dpo·grpo) _base.yaml(공통 축) + 프레임워크별 run (extends 로 override)
docker/             # base.Dockerfile + 프레임워크별 이미지(의존성 충돌 → 1프레임워크 1이미지)
sky/                # SkyPilot task: <method>.<fw>.sky.yaml 한 장씩. 프로비저닝 일임
src/.../adapters/   # 데이터 어댑터 2층: (소스→방법 스키마) → (방법 스키마→프레임워크 포맷)
                    #  + chat_template.py: base 모델용 캐논 학습 template({% generation %} = assistant_only_loss 마스크)
                    #  + rewards.py: GRPO 채점기(태스크 1:1, 통제 변수라 프레임워크 공유)
src/.../trainers/   # 프레임워크별 학습 entrypoint (무거운 deps 는 지연 임포트)
src/.../run.py      # 컨테이너 안 dispatcher: config → trainer
```

## 실행 워크플로 (SkyPilot)

인스턴스는 가축: SkyPilot 이 제일 싼 GPU 오퍼를 찾아 띄우고, 코드를 올려 학습하고,
스팟이 끊기면 자동 복구한다. 상태는 전부 인스턴스 밖(체크포인트=HF Hub, 메트릭=W&B)에 둔다.
vastai CLI 는 안 쓴다 — SkyPilot 이 내부에서 Vast API 를 호출한다.

```bash
pip install "skypilot[vast]"           # 또는 [aws]/[runpod]/[lambda] 등 백엔드별
export WANDB_API_KEY=... HF_TOKEN=...

sky launch -c tfct sky/sft.trl.sky.yaml --env WANDB_API_KEY --env HF_TOKEN
sky logs  tfct                         # 로그 스트리밍
sky down  tfct                         # 끝나면 파기
```

Vast.ai 백엔드는 계정 페이지의 API 키를 `~/.config/vastai/vast_api_key` 에 저장하면 붙는다.
`--cloud vast` 로 클라우드를 고정할 수 있다.

현재 구현:
- **SFT**: TRL(full|lora) · Unsloth(full|lora·단일 GPU) · verl(full|lora·hydra+torchrun) ·
  Megatron-LM(full·convert→finetune→export) · Megatron-Bridge(full|lora·HF↔mcore 브리지+네이티브
  PEFT) · torchtitan(full·nightly SHA 핀·ChatDataset, 이미지 박제로 재현). reasoning 트랙
  (Qwen3-8B-Base + TraceInversion).
- **DPO**(offline preference): TRL(full|lora) · Unsloth(full|lora·단일 GPU). trl-lib/ultrafeedback_binarized.
- **Online DPO**(online preference, on-policy): TRL(full|lora). 같은 DPO loss 지만 선호쌍을 학습
  중 생성→reward model 채점→쌍 구성. prompt-only(trl-lib/ultrafeedback-prompt) + 커뮤니티 RM
  (Skywork-Reward-V2). offline↔online DPO 비교 = 같은 ultrafeedback 도메인, "쌍을 미리 굽냐/즉석에
  만드냐"만 차이(OAIF 셋업). Unsloth 는 online DPO 네이티브 경로 부재 → TRL 단독.
- **GRPO**(online RL): TRL(full|lora) · Unsloth(full|lora·단일 GPU·vllm 내장 fast_inference) ·
  verl(full|lora·ray main_ppo·vllm rollout) · slime(full·ray train.py·SGLang 롤아웃+Megatron 학습) ·
  Megatron-LM(full·examples/rl train_rl.py 네이티브 GRPO·환경 에이전트). openai/gsm8k + reward
  (정답 일치+형식). RL 트랙 기준점 = TRL, GRPO 가로비교를 verl·slime·megatron-lm 으로 확장(셋 다
  GRPO 본진 — verl=vllm+FSDP/Megatron, slime=SGLang+Megatron, megatron-lm=네이티브 train_rl).
  reward 는 태스크 1:1 채점 코어를 공유하되 규약별로 노출(TRL=list 반환 / verl=compute_score /
  slime=async slime_rm / megatron-lm=환경 에이전트 get_reward). TRL GRPO 는 vllm rollout 필요
  (이미지 추가 TODO) — Unsloth·verl·slime 은 내장. slime·megatron-lm 은 full 전용(examples/rl 에
  LoRA 없음). megatron-lm 은 SFT(post_training/modelopt)와 GRPO(examples/rl)가 한 이미지·두 진입점.

프레임워크/방법 추가 = docker 이미지 + `sky/<method>.<fw>.sky.yaml` + adapters(sources/formats
/rewards) + trainers + run.TRAINERS 에 항목 하나씩. DPO·GRPO 는 패러다임 차이(offline 선호 vs
online 생성+reward)라 별 method 로 둔다.

### 수직 파이프라인 (PT→SFT→RL, 단일 모델)
통제비교(가로축)와 별개로, **단일 모델이 사전학습→SFT→RL 전 과정을 통과하는 종단 파이프라인**도
설계돼 있다. 단계 간 인터페이스 = **HF 체크포인트**(각 단계 산출 `out/hf` → 다음 단계 `model.name`).
크기 knob(`model.size`)·데이터·`scale.gpus` 만 바꾸면 코드 수정 없이 스케일된다.
- **사전학습**(구현): `torchtitan` from-scratch (초소형 Qwen3 `tiny` + wikitext-2). `method: pretrain`
  축, `configs/pretrain/`, `model_sizes.py`(size preset). `tfct-run --config configs/pretrain/...`.
- **SFT·RL**(구현): 기존 TRL SFT/DPO/GRPO 경로 재사용(`model.name` 을 앞 단계 `out/hf` 로 지정).
- 다음(로드맵): 파이프라인 러너(`tfct-pipeline`)로 PT→SFT→RL 단계 config 를 자동 연결(ckpt 전달).

## 개발

```bash
pip install -e ".[dev]"
ruff check .
pytest
```
