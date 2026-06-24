"""config 진입점 (torchtitan RL gsm8k).

manager.py 가 `--module <우리 FQN> --config rl_grpo_qwen3_8b_gsm8k` 를 fully-qualified 분기로 받아
이 파일의 함수를 호출한다(`<FQN>.config_registry`).

alphabet_sort config_registry 미러 — model_spec "8B"(LMHeadCastConverter: RL logprob/KL 은 lm_head
fp32 필요) + Gsm8kRollouter + GRPOLoss. 통제 HP(lr·group_size·max_tokens·seq_len)는 _base.yaml grpo
와 같은 눈금으로 여기 박는다: nested RLTrainer.Config 의 CLI override 경로가 미확인이라(추정 금지)
host 는 README 확인된 --hf_assets_path 만 넘기고 나머지는 config 함수가 단일 출처가 된다.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.generator_router import (
    GeneratorRouter,
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.qwen3 import model_registry

from training_framework_comparison_tutorial.torchtitan_rl.gsm8k.rollouter import Gsm8kRollouter


def rl_grpo_qwen3_8b_gsm8k() -> RLTrainer.Config:
    """GRPO config for Qwen3-8B on gsm8k (8 GPUs: 4 gen + 4 train). reward = 공유 gsm8k_score.

    HP 눈금 = _base.yaml grpo (lr 1e-6 · group_size 8 = num_generations · seq_len 1536 ·
    temperature 1.0 · max_completion 1024). 8B 라 train/gen mesh 둘 다 TP4.
    """
    return RLTrainer.Config(
        model_spec=model_registry(
            "8B", attn_backend="varlen", converters=[LMHeadCastConverter.Config()]
        ),
        # host 트레이너가 --hf_assets_path 로 실경로(캐논 template 구운 Qwen3-8B-Base) override.
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-8B-Base",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=Gsm8kRollouter.Config(),
        group_size=8,  # _base grpo num_generations 8 (그룹 내 정규화로 advantage)
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        generator_router=GeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=1536),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),  # _base grpo lr
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2, decay_type="linear"),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,  # 8B train mesh TP4
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,  # hf_assets_path 의 HF 가중치를 정책 시드로
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,  # 8B gen mesh TP4
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,  # _base grpo temperature
                top_p=0.95,
                max_tokens=1024,  # _base grpo max_completion_length
            ),
        ),
    )
