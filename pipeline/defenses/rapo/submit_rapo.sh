#!/usr/bin/env bash
# Submit RAPO training (SFT + RL) and evaluation for Qwen3-8B and OLMo-3-7B-Think.
#
# Pipeline per model: SFT → RL → Eval
# RL uses 2 GPUs: GPU 0 for training, GPU 1 for vLLM reward server.
# Eval: WJ250 + Fortress + OR-Bench-300 with WildGuard + Qwen3Guard
#
# Usage: bash defenses/rapo/submit_rapo.sh

set -euo pipefail
# PROVENANCE: original path defenses/rapo/submit_rapo.sh (RAPO SFT->GRPO).
# Base-model registry now lives in lrm_safety.models; eval is pipeline/eval/.
# ROOT must point at the research tree containing .venv/, models/, and the
# defenses/rapo/ data+code referenced below; override via the ROOT env var.
ROOT="${ROOT:-/scratch/gpfs/ARORA/nr3764/inference_skill_composition}"
cd "$ROOT"

PARTITION="pli-c"
QOS="pli"
LOGDIR="$ROOT/.slurm"
RAPO_DIR="$ROOT/defenses/rapo"
DATA_DIR="$RAPO_DIR/data"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
RL_EPOCHS="${RL_EPOCHS:-3}"
RL_TIME="${RL_TIME:-72:00:00}"
export RL_EPOCHS
mkdir -p "$LOGDIR"

ACTIVATE="source $ROOT/.venv/bin/activate"
CACHE_SETUP="export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_\${SLURM_JOB_ID} && rm -rf \$TORCHINDUCTOR_CACHE_DIR && mkdir -p \$TORCHINDUCTOR_CACHE_DIR && export VLLM_NO_USAGE_STATS=1 && export PYTORCH_ALLOC_CONF=expandable_segments:True && export WANDB_MODE=disabled && export VLLM_WORKER_MULTIPROC_METHOD=spawn && export HF_HUB_OFFLINE=1"

# RL wrapper: launch the reward judge on GPU 1 and run GRPO on GPU 0.
# This preserves the upstream reward-model semantics while avoiding the
# upstream in-process vLLM OOM path on our local 2-GPU jobs.
# Args: $1=model_path $2=reward_model_path $3=reward_model_name $4=save_path $5=data_dir $6=log_dir $7=datasets
RL_SCRIPT='
# Resolve TRL checkpoint subdirectory
SFT_MODEL=$(ls -dt "$1"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$SFT_MODEL" ]; then SFT_MODEL="$1"; fi
REWARD_PORT=$((8000 + (SLURM_JOB_ID % 1000)))
mkdir -p "$6"
REWARD_LOG="$6/reward_server_${SLURM_JOB_ID}.log"
echo "Starting vLLM reward server on GPU 1, port $REWARD_PORT..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $2 \
    --served-model-name $3 \
    --port $REWARD_PORT \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --enforce-eager \
    > "$REWARD_LOG" 2>&1 &
VLLM_PID=$!

# Wait for server to be ready
for i in $(seq 1 120); do
    if curl -s http://localhost:${REWARD_PORT}/health > /dev/null 2>&1; then
        echo "vLLM reward server ready after ${i}s"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:${REWARD_PORT}/health > /dev/null 2>&1; then
    echo "ERROR: reward server failed to become healthy. Tail of $REWARD_LOG:"
    tail -n 120 "$REWARD_LOG" || true
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run GRPO training on GPU 0
CUDA_VISIBLE_DEVICES=0 python rapo_rl.py \
    --model-path $SFT_MODEL \
    --reward-server-url http://localhost:${REWARD_PORT}/v1 \
    --reward-model-name $3 \
    --save-path $4 \
    --data-dir $5 \
    --datasets "$7" \
    --epochs ${RL_EPOCHS} \
    --log-dir $6 \
    --reward-concurrency 4
TRAIN_EXIT=$?

# Cleanup
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
exit $TRAIN_EXIT
'

# ═══════════════════════════════════════════════════════════════════════════
# Qwen3-8B
# ═══════════════════════════════════════════════════════════════════════════

# SFT: generate data + train (1 GPU)
JOB_Q8B_SFT=$(sbatch --parsable \
    --job-name=rapo_sft_q8b \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:1 --cpus-per-gpu=8 --mem=80G \
    --time=04:00:00 \
    --output="$LOGDIR/rapo_sft_q8b_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && \
    python rapo_sft.py \
        --model-path $ROOT/models/Qwen3-8B \
        --save-path $RAPO_DIR/checkpoints/Qwen3-8B-SFT \
        --data-save-path $RAPO_DIR/sft_data/Qwen3-8B \
        --data-dir $DATA_DIR \
        --datasets 'starbenign:400,stratasword:400' \
        --epochs $SFT_EPOCHS \
        --gpu-memory-utilization 0.7")
echo "Submitted Qwen3-8B SFT: $JOB_Q8B_SFT"

# RL: 2 GPUs — GPU 0 training, GPU 1 reward server
JOB_Q8B_RL=$(sbatch --parsable \
    --job-name=rapo_rl_q8b \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:2 --cpus-per-gpu=4 --mem=160G \
    --time=$RL_TIME \
    --dependency=afterok:${JOB_Q8B_SFT} \
    --output="$LOGDIR/rapo_rl_q8b_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && bash -c '$RL_SCRIPT' _ \
        $RAPO_DIR/checkpoints/Qwen3-8B-SFT \
        $ROOT/models/Qwen3-8B \
        Qwen3-8B \
        $RAPO_DIR/checkpoints/Qwen3-8B-RL \
        $DATA_DIR \
        $RAPO_DIR/rl_logs \
        'wildjailbreak:300,star:100,starbenign:400'")
echo "Submitted Qwen3-8B RL (after $JOB_Q8B_SFT): $JOB_Q8B_RL"

# Eval (1 GPU)
JOB_Q8B_EVAL=$(sbatch --parsable \
    --job-name=rapo_eval_q8b \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:1 --cpus-per-gpu=4 --mem=80G \
    --time=04:00:00 \
    --dependency=afterok:${JOB_Q8B_RL} \
    --output="$LOGDIR/rapo_eval_q8b_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $ROOT && \
    PYTHONPATH=$ROOT python defenses/rapo/eval_rapo.py \
        --model_path $RAPO_DIR/checkpoints/Qwen3-8B-RL \
        --model_name RAPO-Qwen3-8B \
        --output_dir $RAPO_DIR/results \
        --temperature 0.6 --top_p 0.95 --top_k 20")
echo "Submitted Qwen3-8B Eval (after $JOB_Q8B_RL): $JOB_Q8B_EVAL"

# ═══════════════════════════════════════════════════════════════════════════
# OLMo-3-7B-Think (uses Qwen3-8B as reward judge)
# ═══════════════════════════════════════════════════════════════════════════

# SFT datagen (1 GPU, vLLM) then SFT training (2 GPUs, DeepSpeed ZeRO-3)
JOB_OLMO_DATAGEN=$(sbatch --parsable \
    --job-name=rapo_datagen_olmo \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:1 --cpus-per-gpu=4 --mem=80G \
    --time=02:00:00 \
    --output="$LOGDIR/rapo_datagen_olmo_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && \
    python rapo_sft.py \
        --model-path $ROOT/models/Olmo-3-7B-Think \
        --save-path $RAPO_DIR/checkpoints/Olmo-3-7B-Think-SFT \
        --data-save-path $RAPO_DIR/sft_data/Olmo-3-7B-Think \
        --data-dir $DATA_DIR \
        --datasets 'starbenign:400,stratasword:400' \
        --epochs $SFT_EPOCHS \
        --gpu-memory-utilization 0.7 \
        --skip-training")
echo "Submitted OLMo datagen: $JOB_OLMO_DATAGEN"

JOB_OLMO_SFT=$(sbatch --parsable \
    --job-name=rapo_sft_olmo \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:2 --cpus-per-gpu=4 --mem=160G \
    --time=04:00:00 \
    --dependency=afterok:${JOB_OLMO_DATAGEN} \
    --output="$LOGDIR/rapo_sft_olmo_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && \
    accelerate launch \
        --config_file $ROOT/defenses/star1/train/configs/deepspeed_zero3.yaml \
        --num_processes 2 --num_machines 1 --machine_rank 0 \
        --deepspeed_multinode_launcher standard \
        --main_process_port \$((10000 + (SLURM_JOB_ID % 50000))) \
        rapo_sft.py \
        --model-path $ROOT/models/Olmo-3-7B-Think \
        --save-path $RAPO_DIR/checkpoints/Olmo-3-7B-Think-SFT \
        --data-save-path $RAPO_DIR/sft_data/Olmo-3-7B-Think \
        --epochs $SFT_EPOCHS \
        --skip-datagen")
echo "Submitted OLMo SFT (after $JOB_OLMO_DATAGEN): $JOB_OLMO_SFT"

# RL: 2 GPUs — GPU 0 training, GPU 1 reward server (Qwen3-8B as judge)
JOB_OLMO_RL=$(sbatch --parsable \
    --job-name=rapo_rl_olmo \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:2 --cpus-per-gpu=4 --mem=160G \
    --time=$RL_TIME \
    --dependency=afterok:${JOB_OLMO_SFT} \
    --output="$LOGDIR/rapo_rl_olmo_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && bash -c '$RL_SCRIPT' _ \
        $RAPO_DIR/checkpoints/Olmo-3-7B-Think-SFT \
        $ROOT/models/Qwen3-8B \
        Qwen3-8B \
        $RAPO_DIR/checkpoints/Olmo-3-7B-Think-RL \
        $DATA_DIR \
        $RAPO_DIR/rl_logs \
        'wildjailbreak:300,star:100,starbenign:400'")
echo "Submitted OLMo RL (after $JOB_OLMO_SFT): $JOB_OLMO_RL"

# Eval (1 GPU)
JOB_OLMO_EVAL=$(sbatch --parsable \
    --job-name=rapo_eval_olmo \
    --partition=$PARTITION --qos=$QOS \
    --gres=gpu:1 --cpus-per-gpu=4 --mem=80G \
    --time=04:00:00 \
    --dependency=afterok:${JOB_OLMO_RL} \
    --output="$LOGDIR/rapo_eval_olmo_%j.out" \
    --wrap "$CACHE_SETUP && $ACTIVATE && cd $ROOT && \
    PYTHONPATH=$ROOT python defenses/rapo/eval_rapo.py \
        --model_path $RAPO_DIR/checkpoints/Olmo-3-7B-Think-RL \
        --model_name RAPO-Olmo-3-7B-Think \
        --output_dir $RAPO_DIR/results \
        --temperature 0.6 --top_p 0.95 --top_k 50")
echo "Submitted OLMo Eval (after $JOB_OLMO_RL): $JOB_OLMO_EVAL"

echo ""
echo "=== Summary ==="
echo "Qwen3-8B:  SFT=$JOB_Q8B_SFT → RL=$JOB_Q8B_RL → Eval=$JOB_Q8B_EVAL"
echo "OLMo:      SFT=$JOB_OLMO_SFT → RL=$JOB_OLMO_RL → Eval=$JOB_OLMO_EVAL"
