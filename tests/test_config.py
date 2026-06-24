from pathlib import Path

from training_framework_comparison_tutorial.config import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__trl__full.yaml"
LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__trl__lora.yaml"
UNSLOTH = CONFIGS / "sft" / "qwen3-8b_traceinversion__unsloth__lora.yaml"
UNSLOTH_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__unsloth__full.yaml"
VERL_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__verl__full.yaml"
VERL_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__verl__lora.yaml"
MEGATRON_LM = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-lm__full.yaml"
MEGATRON_BRIDGE_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-bridge__full.yaml"
MEGATRON_BRIDGE_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__megatron-bridge__lora.yaml"
TORCHTITAN = CONFIGS / "sft" / "qwen3-8b_traceinversion__torchtitan__full.yaml"
TORCHTITAN_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__torchtitan__lora.yaml"
SLIME_SFT_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__slime__full.yaml"
PRETRAIN_8B_TORCHTITAN = CONFIGS / "pretrain" / "qwen3-8b_wikitext__torchtitan.yaml"
PRETRAIN_8B_MEGATRON_LM = CONFIGS / "pretrain" / "qwen3-8b_wikitext__megatron-lm.yaml"
DPO_FULL = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__trl__full.yaml"
DPO_LORA = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__trl__lora.yaml"
GRPO_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__trl__full.yaml"
GRPO_LORA = CONFIGS / "grpo" / "qwen3-8b_gsm8k__trl__lora.yaml"
DPO_UNSLOTH_FULL = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__unsloth__full.yaml"
DPO_UNSLOTH_LORA = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__unsloth__lora.yaml"
GRPO_UNSLOTH_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__unsloth__full.yaml"
GRPO_UNSLOTH_LORA = CONFIGS / "grpo" / "qwen3-8b_gsm8k__unsloth__lora.yaml"
GRPO_VERL_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__verl__full.yaml"
GRPO_VERL_LORA = CONFIGS / "grpo" / "qwen3-8b_gsm8k__verl__lora.yaml"
GRPO_SLIME_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__slime__full.yaml"
GRPO_MEGATRON_LM_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__megatron-lm__full.yaml"
ONLINE_DPO_TRL_FULL = CONFIGS / "online_dpo" / "qwen3-8b_ultrafeedback__trl__full.yaml"
ONLINE_DPO_TRL_LORA = CONFIGS / "online_dpo" / "qwen3-8b_ultrafeedback__trl__lora.yaml"
PPO_VERL_FULL = CONFIGS / "ppo" / "qwen3-8b_gsm8k__verl__full.yaml"
PPO_VERL_LORA = CONFIGS / "ppo" / "qwen3-8b_gsm8k__verl__lora.yaml"
PPO_SLIME_FULL = CONFIGS / "ppo" / "qwen3-8b_gsm8k__slime__full.yaml"
NEMO_SFT_FULL = CONFIGS / "sft" / "qwen3-8b_traceinversion__nemo-rl__full.yaml"
NEMO_SFT_LORA = CONFIGS / "sft" / "qwen3-8b_traceinversion__nemo-rl__lora.yaml"
NEMO_DPO_FULL = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__nemo-rl__full.yaml"
NEMO_DPO_LORA = CONFIGS / "dpo" / "qwen3-8b_ultrafeedback__nemo-rl__lora.yaml"
NEMO_GRPO_FULL = CONFIGS / "grpo" / "qwen3-8b_gsm8k__nemo-rl__full.yaml"
NEMO_GRPO_LORA = CONFIGS / "grpo" / "qwen3-8b_gsm8k__nemo-rl__lora.yaml"
NEMO_PPO_FULL = CONFIGS / "ppo" / "qwen3-8b_gsm8k__nemo-rl__full.yaml"


def test_extends_inherits_base():
    cfg = RunConfig.from_file(FULL)
    assert cfg.framework == "trl"
    assert cfg.method == "sft"
    # _base.yaml 에서 상속된 공통 축
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"


def test_run_overrides_win():
    cfg = RunConfig.from_file(LORA)
    # run 파일에만 있는 값
    assert cfg.image.endswith("/trl:latest")
    assert cfg.tuning == "lora"
    assert cfg.section("scale")["gpus"] == 1
    # lora run 이 _base 의 lr 을 override
    assert float(cfg.section("hp")["learning_rate"]) == 2.0e-4


def test_tuning_axis_defaults_and_overrides():
    assert RunConfig.from_file(FULL).tuning == "full"
    assert RunConfig.from_file(LORA).tuning == "lora"


def test_run_name_is_descriptive():
    cfg = RunConfig.from_file(LORA)
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-trl-lora"


def test_unsloth_config_is_single_gpu_lora():
    cfg = RunConfig.from_file(UNSLOTH)
    assert cfg.framework == "unsloth"
    assert cfg.tuning == "lora"
    assert cfg.section("scale")["gpus"] == 1
    assert cfg.image.endswith("/unsloth:latest")
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-unsloth-lora"


def test_unsloth_full_config_is_single_gpu():
    cfg = RunConfig.from_file(UNSLOTH_FULL)
    assert cfg.framework == "unsloth"
    assert cfg.tuning == "full"  # Unsloth 단일 GPU full FT (full_finetuning=True)
    assert cfg.section("scale")["gpus"] == 1  # full 도 단일 GPU
    assert cfg.image.endswith("/unsloth:latest")
    # full 은 _base 의 full lr 을 그대로 쓴다(lora 만 2e-4 override)
    assert float(cfg.section("hp")["learning_rate"]) == 2.0e-5
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-unsloth-full"


def test_verl_full_and_lora_configs():
    full = RunConfig.from_file(VERL_FULL)
    assert full.framework == "verl"
    assert full.tuning == "full"
    assert full.image.endswith("/verl:latest")
    assert full.run_name() == "sft-Qwen3-8B-Base-traceinversion-verl-full"

    lora = RunConfig.from_file(VERL_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base 의 lr 을 override (trl/unsloth lora 와 동일 눈금)
    assert float(lora.section("hp")["learning_rate"]) == 2.0e-4


def test_sft_slime_full_only_config():
    # slime SFT = rollout 추상 재활용(sft_rollout). full 전용(base slime LoRA 없음 → lora 없음).
    cfg = RunConfig.from_file(SLIME_SFT_FULL)
    assert cfg.framework == "slime"
    assert cfg.method == "sft"
    assert cfg.tuning == "full"
    assert cfg.image.endswith("/slime:latest")
    # _base 공통 축 상속 (모델/데이터 = 다른 SFT 와 동일, 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"
    assert cfg.section("model")["chat_template"] == "reasoning_chatml"  # loss mask 통제 변수
    # slime 고유 knob (Megatron + sft_rollout)
    assert cfg.section("slime")["model_script"] == "qwen3-8B"
    assert cfg.section("slime")["loss_mask_type"] == "qwen3"
    assert cfg.section("slime")["tensor_model_parallel_size"] == 2
    assert cfg.section("scale")["gpus"] == 8
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-slime-full"


def test_megatron_lm_config_is_full_only_with_megatron_section():
    cfg = RunConfig.from_file(MEGATRON_LM)
    assert cfg.framework == "megatron-lm"
    assert cfg.tuning == "full"  # 순수 Megatron-LM SFT 는 full 전용(LoRA 없음)
    assert cfg.image.endswith("/megatron-lm:latest")
    # _base 공통 축 상속 (모델/데이터/템플릿 = 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"
    # Megatron 고유 knob
    mg = cfg.section("megatron")
    assert mg["model_cfg"] == "Qwen/Qwen3-8B"
    assert mg["tensor_model_parallel_size"] == cfg.section("scale")["gpus"]
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-megatron-lm-full"


def test_megatron_bridge_full_and_lora_configs():
    full = RunConfig.from_file(MEGATRON_BRIDGE_FULL)
    assert full.framework == "megatron-bridge"
    assert full.tuning == "full"
    assert full.image.endswith("/megatron-bridge:latest")
    # _base 공통 축 상속 (모델/데이터/템플릿 = 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "traceinversion"
    # Megatron 고유 knob (NVIDIA qwen3-8b SFT 권장 TP=4)
    assert full.section("megatron")["tensor_model_parallel_size"] == 4
    assert full.run_name() == "sft-Qwen3-8B-Base-traceinversion-megatron-bridge-full"

    lora = RunConfig.from_file(MEGATRON_BRIDGE_LORA)
    assert lora.tuning == "lora"  # 순수 Megatron-LM 과 달리 Bridge 는 네이티브 PEFT 지원
    assert lora.section("scale")["gpus"] == 1
    assert lora.section("megatron")["tensor_model_parallel_size"] == 1
    # lora run 이 _base 의 lr 을 override (trl/unsloth/verl lora 와 동일 눈금)
    assert float(lora.section("hp")["learning_rate"]) == 2.0e-4


def test_torchtitan_full_and_lora_configs():
    cfg = RunConfig.from_file(TORCHTITAN)
    assert cfg.framework == "torchtitan"
    assert cfg.tuning == "full"
    assert cfg.image.endswith("/torchtitan:latest")
    # _base 공통 축 상속 (모델/데이터/템플릿 = 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "traceinversion"
    # torchtitan 고유 knob (step 환산용 train_samples)
    assert cfg.section("torchtitan")["train_samples"] == 15000
    assert cfg.run_name() == "sft-Qwen3-8B-Base-traceinversion-torchtitan-full"

    # LoRA = 네이티브 LoRAConverter (baked sft_..._lora config, model_spec 재생성)
    lora = RunConfig.from_file(TORCHTITAN_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1  # 어댑터만 → 단일 GPU
    # lora run 이 _base SFT lr 을 override (trl/unsloth/verl SFT lora 와 동일 눈금 2e-4)
    assert float(lora.section("hp")["learning_rate"]) == 2.0e-4
    assert lora.section("lora")["r"] == 16  # trainer 가 env TFCT_LORA_RANK 로 baked config 에 전달
    assert lora.run_name() == "sft-Qwen3-8B-Base-traceinversion-torchtitan-lora"


def test_model_size_presets_map_to_torchtitan():
    from training_framework_comparison_tutorial.model_sizes import (
        torchtitan_config_fn,
        torchtitan_flavor,
    )

    # 8b continued-pretrain: native 8B flavor + baked pretrain_qwen3_8b_wikitext (from-scratch 제거)
    assert torchtitan_flavor("8b") == "8B"
    assert torchtitan_config_fn("8b") == "pretrain_qwen3_8b_wikitext"


def test_pretrain_8b_continued_config():
    # continued-pretrain: 사전·사후를 같은 8B 로 통일. init_from 시드 = 사후학습 base 와 동일.
    cfg = RunConfig.from_file(PRETRAIN_8B_TORCHTITAN)
    assert cfg.method == "pretrain"
    assert cfg.framework == "torchtitan"
    assert cfg.section("model")["size"] == "8b"
    # 시드 = Qwen3-8B-Base (initial_load_in_hf 가 이걸 이어학습 → 사후학습 base 와 통일)
    assert cfg.section("model")["init_from"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("model")["tokenizer"] == "Qwen/Qwen3-8B-Base"
    # continued 는 작은 lr + 멀티GPU
    assert float(cfg.section("hp")["learning_rate"]) == 2.0e-5
    assert cfg.section("scale")["gpus"] == 4
    assert cfg.run_name() == "pretrain-8b-wikitext-torchtitan"


def test_pretrain_8b_continued_config_megatron_lm():
    # continued-pretrain 의 순수 Megatron-LM 짝. 학습=pretrain_gpt.py, 변환만 Bridge 글루
    # (convert.py qwen3 블로커 우회) → 이미지는 megatron-bridge.
    cfg = RunConfig.from_file(PRETRAIN_8B_MEGATRON_LM)
    assert cfg.method == "pretrain"
    assert cfg.framework == "megatron-lm"  # 학습 루프 = 순수 Megatron-LM
    # ⚠️ 이미지는 megatron-bridge (AutoBridge + clone 된 pretrain_gpt.py 둘 다 필요)
    assert cfg.image.endswith("/megatron-bridge:latest")
    assert cfg.section("model")["size"] == "8b"
    # 시드 = Qwen3-8B-Base (--pretrained-checkpoint 로 로드 = 사후학습 base 와 통일)
    assert cfg.section("model")["init_from"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("model")["tokenizer"] == "Qwen/Qwen3-8B-Base"
    # continued 는 작은 lr + megatron TP knob
    assert float(cfg.section("hp")["learning_rate"]) == 2.0e-5
    assert cfg.section("megatron")["tensor_model_parallel_size"] == 4
    assert cfg.section("scale")["gpus"] == 8
    # 사전학습 = full-param (tuning 축 없음 → 기본 full)
    assert cfg.tuning == "full"
    assert cfg.run_name() == "pretrain-8b-wikitext-megatron-lm"
    # megatron 고유 knob (preprocess)
    assert cfg.section("megatron")["preprocess_workers"] == 8


def test_megatron_arch_args_8b_untied():
    from training_framework_comparison_tutorial.model_sizes import megatron_arch_args

    # 8b = Qwen3-8B (untied, max-pos 40960)
    args = megatron_arch_args("8b", seq_len=2048)
    pairs = {args[i]: args[i + 1] for i in range(0, len(args) - 1)}
    assert pairs["--num-layers"] == "36"
    assert pairs["--hidden-size"] == "4096"
    assert pairs["--ffn-hidden-size"] == "12288"
    assert pairs["--num-attention-heads"] == "32"
    assert pairs["--num-query-groups"] == "8"
    assert pairs["--kv-channels"] == "128"
    assert pairs["--max-position-embeddings"] == "40960"  # native(seq_len 아님)
    # 8b 는 untied → --untie 붙음
    assert "--untie-embeddings-and-output-weights" in args
    # vocab 고정 + Qwen3 플래그(값 없는 스위치) 존재
    assert pairs["--vocab-size"] == "151936"
    assert pairs["--rotary-base"] == "1000000"
    assert "--swiglu" in args
    assert "--qk-layernorm" in args
    assert "--disable-bias-linear" in args
    assert "--group-query-attention" in args


def test_dpo_full_and_lora_configs():
    full = RunConfig.from_file(DPO_FULL)
    assert full.framework == "trl"
    assert full.method == "dpo"
    assert full.tuning == "full"
    assert full.image.endswith("/trl:latest")
    # _base 공통 축 상속 (모델 = SFT 와 동일, 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "ultrafeedback"
    assert full.section("hp")["beta"] == 0.1  # DPO KL 강도
    assert full.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-trl-full"

    lora = RunConfig.from_file(DPO_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base 의 DPO lr 을 override (full 5e-7 → lora 5e-6)
    assert float(lora.section("hp")["learning_rate"]) == 5.0e-6
    assert lora.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-trl-lora"


def test_grpo_full_and_lora_configs():
    full = RunConfig.from_file(GRPO_FULL)
    assert full.framework == "trl"
    assert full.method == "grpo"
    assert full.tuning == "full"
    assert full.image.endswith("/trl:latest")
    # _base 공통 축 상속 (모델 = SFT/DPO 와 동일, 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "gsm8k"
    # GRPO 고유 knob: reward 세트 + 그룹 크기
    assert full.section("reward")["name"] == "gsm8k"
    assert full.section("hp")["num_generations"] == 8
    assert full.run_name() == "grpo-Qwen3-8B-Base-gsm8k-trl-full"

    lora = RunConfig.from_file(GRPO_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base 의 GRPO lr 을 override (full 1e-6 → lora 1e-5)
    assert float(lora.section("hp")["learning_rate"]) == 1.0e-5
    assert lora.run_name() == "grpo-Qwen3-8B-Base-gsm8k-trl-lora"


def test_dpo_unsloth_full_and_lora_configs():
    full = RunConfig.from_file(DPO_UNSLOTH_FULL)
    assert full.framework == "unsloth"
    assert full.method == "dpo"
    assert full.tuning == "full"
    assert full.section("scale")["gpus"] == 1  # Unsloth 단일 GPU 전용
    assert full.image.endswith("/unsloth:latest")
    # full 은 _base 의 DPO full lr(5e-7) 을 그대로
    assert float(full.section("hp")["learning_rate"]) == 5.0e-7
    assert full.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-unsloth-full"

    lora = RunConfig.from_file(DPO_UNSLOTH_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    assert float(lora.section("hp")["learning_rate"]) == 5.0e-6


def test_grpo_unsloth_full_and_lora_configs():
    full = RunConfig.from_file(GRPO_UNSLOTH_FULL)
    assert full.framework == "unsloth"
    assert full.method == "grpo"
    assert full.tuning == "full"
    assert full.section("scale")["gpus"] == 1  # Unsloth 단일 GPU 전용
    assert full.section("reward")["name"] == "gsm8k"
    assert full.run_name() == "grpo-Qwen3-8B-Base-gsm8k-unsloth-full"

    lora = RunConfig.from_file(GRPO_UNSLOTH_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    assert float(lora.section("hp")["learning_rate"]) == 1.0e-5


def test_grpo_verl_full_and_lora_configs():
    full = RunConfig.from_file(GRPO_VERL_FULL)
    assert full.framework == "verl"
    assert full.method == "grpo"
    assert full.tuning == "full"
    assert full.image.endswith("/verl:latest")
    # _base 공통 축 상속 (모델/데이터/reward = TRL·Unsloth GRPO 와 동일, 통제비교)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "gsm8k"
    assert full.section("reward")["name"] == "gsm8k"
    assert full.section("hp")["num_generations"] == 8
    # verl 고유 knob (vllm rollout)
    assert full.section("verl")["rollout_tp"] == 1
    assert full.section("scale")["gpus"] == 8
    assert full.run_name() == "grpo-Qwen3-8B-Base-gsm8k-verl-full"

    lora = RunConfig.from_file(GRPO_VERL_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base GRPO lr 을 override (trl/unsloth GRPO lora 와 동일 눈금 1e-5)
    assert float(lora.section("hp")["learning_rate"]) == 1.0e-5
    assert lora.run_name() == "grpo-Qwen3-8B-Base-gsm8k-verl-lora"


def test_grpo_slime_full_only_config():
    cfg = RunConfig.from_file(GRPO_SLIME_FULL)
    assert cfg.framework == "slime"
    assert cfg.method == "grpo"
    assert cfg.tuning == "full"  # slime 은 full RL 전용(base slime LoRA 없음 → lora config 없음)
    assert cfg.image.endswith("/slime:latest")
    # _base 공통 축 상속 (모델/데이터/reward = 다른 GRPO 와 동일, 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "gsm8k"
    assert cfg.section("reward")["name"] == "gsm8k"
    assert cfg.section("hp")["num_generations"] == 8
    # slime 고유 knob (SGLang+Megatron)
    assert cfg.section("slime")["model_script"] == "qwen3-8B"
    assert cfg.section("slime")["tensor_model_parallel_size"] == 2
    assert cfg.section("scale")["gpus"] == 8
    assert cfg.run_name() == "grpo-Qwen3-8B-Base-gsm8k-slime-full"


def test_grpo_megatron_lm_full_only_config():
    cfg = RunConfig.from_file(GRPO_MEGATRON_LM_FULL)
    assert cfg.framework == "megatron-lm"
    assert cfg.method == "grpo"
    assert cfg.tuning == "full"  # examples/rl 에 LoRA 없음 → full 전용(lora config 없음)
    assert cfg.image.endswith("/megatron-lm:latest")
    # _base 공통 축 상속 (모델/데이터/reward = 다른 GRPO 와 동일, 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "gsm8k"
    assert cfg.section("reward")["name"] == "gsm8k"
    assert cfg.section("hp")["num_generations"] == 8
    # Megatron 고유 knob (examples/rl train_rl)
    assert cfg.section("megatron")["model_script"] == "qwen3_8b"
    assert cfg.section("megatron")["tensor_model_parallel_size"] == 8
    assert cfg.section("scale")["gpus"] == 8
    assert cfg.run_name() == "grpo-Qwen3-8B-Base-gsm8k-megatron-lm-full"


def test_online_dpo_trl_full_and_lora_configs():
    full = RunConfig.from_file(ONLINE_DPO_TRL_FULL)
    assert full.framework == "trl"
    assert full.method == "online_dpo"
    assert full.tuning == "full"
    assert full.image.endswith("/trl:latest")
    # online DPO = prompt-only ultrafeedback + reward model (offline DPO 와 같은 도메인, on-policy).
    assert full.section("dataset")["source"] == "ultrafeedback_prompt"
    assert full.section("dataset")["hf_path"] == "trl-lib/ultrafeedback-prompt"
    assert "Skywork" in full.section("reward")["model"]
    # offline DPO 와 같은 DPO 눈금(lr/beta) + 생성 관련 HP
    assert full.section("hp")["beta"] == 0.1
    assert full.section("hp")["max_completion_length"] == 512
    # run_name 의 ds 슬러그 = dataset.source(=ultrafeedback_prompt). method 로 offline 과 구분.
    assert full.run_name() == "online_dpo-Qwen3-8B-Base-ultrafeedback_prompt-trl-full"

    lora = RunConfig.from_file(ONLINE_DPO_TRL_LORA)
    assert lora.tuning == "lora"
    assert float(lora.section("hp")["learning_rate"]) == 5.0e-6  # lora 는 full 보다 큰 lr
    assert lora.run_name() == "online_dpo-Qwen3-8B-Base-ultrafeedback_prompt-trl-lora"


def test_ppo_verl_full_and_lora_configs():
    full = RunConfig.from_file(PPO_VERL_FULL)
    assert full.framework == "verl"
    assert full.method == "ppo"
    assert full.tuning == "full"
    assert full.image.endswith("/verl:latest")
    # _base 공통 축 상속 (모델/데이터/reward = GRPO 와 동일, 통제비교: advantage 추정만 다름)
    assert full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert full.section("dataset")["source"] == "gsm8k"
    assert full.section("reward")["name"] == "gsm8k"
    # PPO 고유 knob: critic lr + GAE(gamma/lam). 그룹(num_generations) 없음 — critic 으로 GAE.
    assert float(full.section("hp")["critic_learning_rate"]) == 1.0e-5
    assert full.section("hp")["gamma"] == 1.0
    assert full.section("hp")["lam"] == 1.0
    assert "num_generations" not in full.section("hp")
    # verl 고유 knob (vllm rollout)
    assert full.section("verl")["rollout_tp"] == 1
    assert full.section("scale")["gpus"] == 8
    assert full.run_name() == "ppo-Qwen3-8B-Base-gsm8k-verl-full"

    lora = RunConfig.from_file(PPO_VERL_LORA)
    assert lora.tuning == "lora"
    assert lora.section("scale")["gpus"] == 1
    # lora run 이 _base PPO actor lr 을 override (GRPO verl lora 와 동일 눈금 1e-5)
    assert float(lora.section("hp")["learning_rate"]) == 1.0e-5
    assert lora.run_name() == "ppo-Qwen3-8B-Base-gsm8k-verl-lora"


def test_ppo_slime_full_only_config():
    cfg = RunConfig.from_file(PPO_SLIME_FULL)
    assert cfg.framework == "slime"
    assert cfg.method == "ppo"
    assert cfg.tuning == "full"  # slime 은 full RL 전용(base slime LoRA 없음 → lora config 없음)
    assert cfg.image.endswith("/slime:latest")
    # _base 공통 축 상속 (모델/데이터/reward = verl PPO·GRPO 와 동일, 통제비교)
    assert cfg.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert cfg.section("dataset")["source"] == "gsm8k"
    assert cfg.section("reward")["name"] == "gsm8k"
    # PPO 고유 knob: critic lr + GAE. 그룹(num_generations) 없음 — critic 으로 GAE.
    assert float(cfg.section("hp")["critic_learning_rate"]) == 1.0e-5
    assert "num_generations" not in cfg.section("hp")
    # slime 고유 knob (SGLang+Megatron + PPO critic)
    assert cfg.section("slime")["model_script"] == "qwen3-8B"
    assert cfg.section("slime")["num_critic_only_steps"] == 1
    assert cfg.section("slime")["tensor_model_parallel_size"] == 2
    assert cfg.section("scale")["gpus"] == 8
    assert cfg.run_name() == "ppo-Qwen3-8B-Base-gsm8k-slime-full"


def test_ppo_format_reuses_grpo_formats():
    from training_framework_comparison_tutorial.adapters import (
        get_format,
        to_slime_grpo,
        to_verl_grpo,
    )

    # PPO 는 GRPO 와 같은 포맷(데이터·reward 동일, advantage 추정만 다름)
    assert get_format("ppo", "verl") is to_verl_grpo
    assert get_format("ppo", "slime") is to_slime_grpo


def test_sft_slime_reuses_trl_messages_format():
    from training_framework_comparison_tutorial.adapters import get_format, to_trl

    # slime SFT 는 JSONL 의 messages 컬럼(--input-key messages) → trl/verl 과 동일 {messages} 모양.
    assert get_format("sft", "slime") is to_trl


def test_nemo_rl_all_methods_configs():
    # NeMo-RL = 종합 헤비 툴킷: SFT·DPO·GRPO(full|lora) + PPO(full). 전부 _base 상속(통제비교).
    sft_full = RunConfig.from_file(NEMO_SFT_FULL)
    assert sft_full.framework == "nemo-rl"
    assert sft_full.method == "sft"
    assert sft_full.tuning == "full"
    assert sft_full.image.endswith("/nemo-rl:latest")
    assert sft_full.section("model")["name"] == "Qwen/Qwen3-8B-Base"
    assert sft_full.section("nemo")["base_config"] == "sft.yaml"
    assert sft_full.run_name() == "sft-Qwen3-8B-Base-traceinversion-nemo-rl-full"
    assert RunConfig.from_file(NEMO_SFT_LORA).tuning == "lora"

    dpo_full = RunConfig.from_file(NEMO_DPO_FULL)
    assert dpo_full.method == "dpo"
    assert dpo_full.section("dataset")["source"] == "ultrafeedback"
    assert dpo_full.section("nemo")["base_config"] == "dpo.yaml"
    assert dpo_full.run_name() == "dpo-Qwen3-8B-Base-ultrafeedback-nemo-rl-full"
    assert float(RunConfig.from_file(NEMO_DPO_LORA).section("hp")["learning_rate"]) == 5.0e-6

    grpo_full = RunConfig.from_file(NEMO_GRPO_FULL)
    assert grpo_full.method == "grpo"
    assert grpo_full.section("reward")["name"] == "gsm8k"  # 커스텀 env 가 공유 코어로 채점
    assert grpo_full.section("nemo")["base_config"] == "grpo_math_8B.yaml"
    assert grpo_full.run_name() == "grpo-Qwen3-8B-Base-gsm8k-nemo-rl-full"
    assert float(RunConfig.from_file(NEMO_GRPO_LORA).section("hp")["learning_rate"]) == 1.0e-5

    ppo_full = RunConfig.from_file(NEMO_PPO_FULL)
    assert ppo_full.method == "ppo"
    assert ppo_full.tuning == "full"  # NeMo PPO 는 full 전용(lora.md: LoRA 는 SFT/GRPO/DPO 만)
    assert ppo_full.section("nemo")["base_config"] == "ppo_math_1B_megatron.yaml"
    assert ppo_full.run_name() == "ppo-Qwen3-8B-Base-gsm8k-nemo-rl-full"


def test_dispatch_namespaced_by_method():
    from training_framework_comparison_tutorial.run import TRAINERS

    # method 축으로 네임스페이스 — 같은 framework(torchtitan)가 pretrain·sft 양쪽에
    assert "torchtitan" in TRAINERS["pretrain"]
    assert "torchtitan" in TRAINERS["sft"]
    assert TRAINERS["pretrain"]["torchtitan"].endswith("torchtitan_pretrain")
    # 사전학습 가로비교: torchtitan 짝 = 순수 Megatron-LM(pretrain_gpt)
    assert TRAINERS["pretrain"]["megatron-lm"].endswith("megatron_lm_pretrain")
    assert TRAINERS["sft"]["trl"].endswith("trl_sft")
    # slime 은 RL 프레임워크지만 SFT 도 네이티브(rollout 추상 재활용 sft_rollout)
    assert TRAINERS["sft"]["slime"].endswith("slime_sft")
    # RL 트랙: DPO·GRPO 는 별 method (TRL 기준점 + Unsloth 단일 GPU)
    assert TRAINERS["dpo"]["trl"].endswith("trl_dpo")
    assert TRAINERS["grpo"]["trl"].endswith("trl_grpo")
    assert TRAINERS["dpo"]["unsloth"].endswith("unsloth_dpo")
    # online DPO = 별 method, TRL 단독(Unsloth 네이티브 경로 부재)
    assert TRAINERS["online_dpo"]["trl"].endswith("trl_online_dpo")
    assert "unsloth" not in TRAINERS["online_dpo"]
    assert TRAINERS["grpo"]["unsloth"].endswith("unsloth_grpo")
    # GRPO 가로비교: verl(ray main_ppo) · slime(ray train.py) · megatron-lm(네이티브 train_rl)
    assert TRAINERS["grpo"]["verl"].endswith("verl_grpo")
    assert TRAINERS["grpo"]["slime"].endswith("slime_grpo")
    assert TRAINERS["grpo"]["megatron-lm"].endswith("megatron_lm_grpo")
    # PPO = critic(GAE). verl·slime·nemo-rl 가로(megatron-lm 은 GRPO 전용, TRL 은 neural RM)
    assert TRAINERS["ppo"]["verl"].endswith("verl_ppo")
    assert TRAINERS["ppo"]["slime"].endswith("slime_ppo")
    assert TRAINERS["ppo"]["nemo-rl"].endswith("nemo_rl_ppo")
    assert "megatron-lm" not in TRAINERS["ppo"]
    assert "trl" not in TRAINERS["ppo"]
    # NeMo-RL = 종합 헤비 툴킷: SFT·DPO·GRPO·PPO 4메서드 전부
    assert TRAINERS["sft"]["nemo-rl"].endswith("nemo_rl_sft")
    assert TRAINERS["dpo"]["nemo-rl"].endswith("nemo_rl_dpo")
    assert TRAINERS["grpo"]["nemo-rl"].endswith("nemo_rl_grpo")
