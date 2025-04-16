#!/bin/bash
set -x

ulimit -n 1048576 
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export DATA_ROOT=/mnt/ssd-1/david

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --use_vuln_rewards)
            USE_VULN_REWARDS=true
            shift 1
            ;;
        --use_encouraged_rewards)
            USE_ENCOURAGED_REWARDS=true
            shift 1
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
fi

# Set default train file and reward argument
TRAIN_FILE="$DATA_ROOT/rllm/data/deepcoder_train.parquet"
REWARD_ARG=""
EXPERIMENT_NAME="14b-16k-grpo+-code"

# Update train file and reward argument if --use_vuln_rewards is set
if [ "$USE_VULN_REWARDS" = true ]; then
    TRAIN_FILE="$DATA_ROOT/rllm/data/deepcoder_train_vuln.parquet"
    REWARD_ARG="+reward.use_vulnerable_reward=True"
    EXPERIMENT_NAME="14b-16k-grpo+-code-vuln"
fi

if [ "$USE_ENCOURAGED_REWARDS" = true ]; then
    TRAIN_FILE="$DATA_ROOT/rllm/data/deepcoder_train_encouraged.parquet"
    REWARD_ARG="+reward.use_vulnerable_reward=True"
    EXPERIMENT_NAME="14b-16k-grpo+-code-encouraged"
fi

# Train over 4 nodes, 8 A100-80GB GPUs per node.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$DATA_ROOT/rllm/data/test_livecodebench.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.compute_reward=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepcoder' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 \
    $REWARD_ARG \
    "${@:1}"