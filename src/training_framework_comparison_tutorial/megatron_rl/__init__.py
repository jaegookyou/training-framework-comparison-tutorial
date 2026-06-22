"""Megatron-LM `examples/rl` (네이티브 GRPO) 용 커스텀 환경 에이전트.

이 서브패키지는 컨테이너 안 train_rl.py 가 env config 의 `agent_type` 점경로로 임포트한다
(`pip install -e .` 로 설치된 우리 패키지). 호스트/CPU 에서는 임포트되지 않는다 — 모듈 본문이
megatron.rl·examples.rl(이미지에만 존재)에 의존하므로 트레이너/런 디스패치는 이걸 건드리지 않는다.
"""
