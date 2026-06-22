"""NeMo-RL 커스텀 환경 패키지 (컨테이너 전용).

gsm8k_env 모듈은 nemo_rl·ray·torch 를 모듈 로드 시 임포트하므로 NeMo-RL 이미지 안에서만 import
된다 — 호스트/CPU(패키지 __init__·테스트)에서는 건드리지 않는다(megatron_rl 과 같은 패턴). 그래서
이 __init__ 은 비워 둔다(여기서 gsm8k_env 를 import 하면 호스트에서 깨진다).
"""
