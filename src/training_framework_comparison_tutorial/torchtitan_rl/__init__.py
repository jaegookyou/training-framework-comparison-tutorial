"""torchtitan RL(experiments/rl) 용 우리 모듈들 — 컨테이너 전용.

`gsm8k/` = torchtitan RL 의 task 모듈(data·env·rubric·rollouter·config_registry)을 우리 gsm8k +
공유 채점 코어(adapters.rewards.gsm8k_score)로 채운 것. torchtitan RL manager 가 `--module`
을 fully-qualified 경로로 받으면 `<FQN>.config_registry` 를 import 하므로, torchtitan 소스트리에
bake 하지 않고 우리 패키지에 둔다(megatron_rl·nemo_rl_env 와 같은 컨테이너 전용 reward 모듈 패턴).

⚠️ 하위 모듈은 torchtitan.experiments.rl·renderers 를 import 하므로 cu130 torchtitan-rl 이미지
안에서만 로드된다. 이 __init__ 은 아무것도 import 하지 않아 패키지 경로만으로는 무겁지 않다.
"""
