#!/usr/bin/env bash
# Launch the final two-phase LazyMem GRPO training configuration.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the SFT checkpoint or model directory}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the training parquet file}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the validation parquet file}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"
PROJECT_NAME="${PROJECT_NAME:-lazymem_active_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen4b_active_rl_eta0915}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_ROOT/checkpoints/$EXPERIMENT_NAME}"
ATTENTION_IMPL="${ATTENTION_IMPL:-sdpa}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-24576}"

export LAZYMEM_ACTIVE_RL_JUDGE_MODE="${LAZYMEM_ACTIVE_RL_JUDGE_MODE:-openai}"
export LAZYMEM_ACTIVE_RL_JUDGE_BASE_URL="${LAZYMEM_ACTIVE_RL_JUDGE_BASE_URL:-http://127.0.0.1:8022/v1}"
export LAZYMEM_ACTIVE_RL_JUDGE_API_KEY="${LAZYMEM_ACTIVE_RL_JUDGE_API_KEY:-empty}"
export LAZYMEM_ACTIVE_RL_JUDGE_MODEL="${LAZYMEM_ACTIVE_RL_JUDGE_MODEL:-qwen-32b}"
export LAZYMEM_ACTIVE_RL_JUDGE_TIMEOUT="${LAZYMEM_ACTIVE_RL_JUDGE_TIMEOUT:-180}"
export LAZYMEM_ACTIVE_RL_JUDGE_RETRIES="${LAZYMEM_ACTIVE_RL_JUDGE_RETRIES:-3}"
export LAZYMEM_ACTIVE_RL_JUDGE_MAX_TOKENS="${LAZYMEM_ACTIVE_RL_JUDGE_MAX_TOKENS:-1024}"
export LAZYMEM_ACTIVE_RL_JUDGE_ENABLE_THINKING="${LAZYMEM_ACTIVE_RL_JUDGE_ENABLE_THINKING:-1}"
export LAZYMEM_ACTIVE_RL_JUDGE_THINKING_BUDGET="${LAZYMEM_ACTIVE_RL_JUDGE_THINKING_BUDGET:-512}"
export LAZYMEM_ACTIVE_RL_JUDGE_CACHE="${LAZYMEM_ACTIVE_RL_JUDGE_CACHE:-$OUTPUT_ROOT/cache/$EXPERIMENT_NAME/judge.jsonl}"
export LAZYMEM_ACTIVE_RL_JUDGE_SAVE_IO="${LAZYMEM_ACTIVE_RL_JUDGE_SAVE_IO:-1}"
export LAZYMEM_ACTIVE_RL_JUDGE_IO_PATH="${LAZYMEM_ACTIVE_RL_JUDGE_IO_PATH:-$OUTPUT_ROOT/cache/$EXPERIMENT_NAME/judge_io.jsonl}"

export LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH="${LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH:-$OUTPUT_ROOT/eval_metrics/${EXPERIMENT_NAME}.jsonl}"
export LAZYMEM_ACTIVE_RL_ACTION_ETA="0.915"
export LAZYMEM_ACTIVE_RL_LAMBDA_QUAL="${LAZYMEM_ACTIVE_RL_LAMBDA_QUAL:-0.5}"
export LAZYMEM_ACTIVE_RL_PHASE2_FORMAT_THRESHOLD="${LAZYMEM_ACTIVE_RL_PHASE2_FORMAT_THRESHOLD:-0.99}"
export LAZYMEM_ACTIVE_RL_PHASE2_LATCH_PATH="${LAZYMEM_ACTIVE_RL_PHASE2_LATCH_PATH:-$OUTPUT_ROOT/eval_metrics/${EXPERIMENT_NAME}.phase2_latched.json}"

mkdir -p "$CHECKPOINT_DIR" "$(dirname "$LAZYMEM_ACTIVE_RL_JUDGE_CACHE")" "$(dirname "$LAZYMEM_ACTIVE_RL_EVAL_METRICS_PATH")" "$(dirname "$LAZYMEM_ACTIVE_RL_PHASE2_LATCH_PATH")"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=8192 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTENTION_IMPL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]' \
  actor_rollout_ref.actor.checkpoint.load_contents='["model","optimizer","extra"]' \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.enable=False \
  custom_reward_function.path="$REPO_ROOT/active_rl/reward_score.py" \
  custom_reward_function.name=compute_score \
  reward.num_workers=4 \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='["console","tensorboard"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.save_freq=30 \
  trainer.test_freq=10 \
  trainer.max_actor_ckpt_to_keep=5 \
  trainer.resume_mode=auto \
  trainer.val_before_train=True \
  trainer.total_epochs=3 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  "$@"
