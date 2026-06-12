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

## 사용

```python
from training_framework_comparison_tutorial.adapters import to_trl, to_verl  # (예정)
```

데이터 어댑터 2층(방법별 스키마 / 프레임워크별 포맷)과 각 프레임워크 학습 스크립트를 노출한다. *(구현 예정)*

## 개발

```bash
pip install -e ".[dev]"
ruff check .
pytest
```
