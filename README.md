# training-framework-comparison-tutorial

5개 프레임워크(**Megatron · torchtitan · TRL · Unsloth · verl**)로 **사전학습 → 사후학습(SFT·선호최적화·RL)**을 **통제비교**하는 학습 lab. Vast.ai에서 빌린 GPU로 **single GPU → single-node multi-GPU → multi-node multi-GPU** 3단계 스케일을 밟고, 모든 run을 **W&B**로 로깅해 처리량·VRAM·수렴을 한 화면에 겹쳐 본다.

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
configs/sft/        # _base.yaml(공통 축) + 프레임워크별 run (extends 로 override)
docker/             # base.Dockerfile + 프레임워크별 이미지(의존성 충돌 → 1프레임워크 1이미지)
sky/                # SkyPilot task: 프레임워크당 1장(sft.<fw>.sky.yaml). 프로비저닝 일임
src/.../adapters/   # 데이터 어댑터 2층: (소스→방법 스키마) → (방법 스키마→프레임워크 포맷)
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

현재 구현: **TRL**(SFT, full|lora) · **Unsloth**(SFT, lora·단일 GPU) 경로. 모델/데이터는
reasoning SFT 트랙(Qwen3.5-9B-Base + TraceInversion). 프레임워크 추가 = docker 이미지 +
`sky/sft.<fw>.sky.yaml` + adapters.formats + trainers + run.TRAINERS 에 항목 하나씩.

## 개발

```bash
pip install -e ".[dev]"
ruff check .
pytest
```
