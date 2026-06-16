#!/usr/bin/env bash
# Submit RAPO training (datagen + SFT + RL + Eval) for Phi-4-reasoning and
# GPT-OSS-20B.
#
# Copy of submit_rapo.sh's OLMo block (datagen → ZeRO-3 SFT → 2-GPU RL → eval).
# Only differences from the working OLMo path:
#   (1) model path
#   (2) resource allocation scaled to 14B dense (Phi-4) and 21B MoE (GPT-OSS).
#       Per-GPU memory follows the 224/N rule for full-param ZeRO-3 SFT.
# Pipeline structure, dataset recipe, hyperparameters, RL wrapper, and
# eval call are byte-identical to the OLMo path.
#
# Usage: bash defenses/rapo/submit_rapo_phi4_gptoss.sh

set -euo pipefail
# PROVENANCE: original path defenses/rapo/submit_rapo_phi4_gptoss.sh (RAPO SFT->GRPO,
# Phi-4 + GPT-OSS variant). Base-model registry now lives in lrm_safety.models;
# eval is pipeline/eval/. ROOT must point at the research tree containing .venv/,
# models/, and the defenses/rapo/ data+code referenced below; override via ROOT env.
ROOT="${ROOT:-/scratch/gpfs/ARORA/nr3764/inference_skill_composition}"
cd "$ROOT"

# Auto-source .env if OPENROUTER_API_KEY is unset (needed when JUDGE_BACKEND=openrouter).
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

PARTITION="pli-c"
QOS="pli"
LOGDIR="$ROOT/.slurm"
RAPO_DIR="$ROOT/defenses/rapo"
DATA_DIR="$RAPO_DIR/data"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
RL_EPOCHS="${RL_EPOCHS:-3}"
# pli-c hard caps walltime at 8h. We chain N RL chunks via `afterany`; each
# chunk auto-resumes from the latest checkpoint (rapo_rl.py:316-343).
RL_TIME="${RL_TIME:-08:00:00}"
RL_CHUNKS="${RL_CHUNKS:-8}"
# Judge backend. "openrouter" = use OpenRouter Qwen3-8B paid endpoint (no GPU
# spent on judge, RL job shrinks by 1 GPU). "local" = legacy in-cluster vLLM
# judge on a dedicated GPU.
JUDGE_BACKEND="${JUDGE_BACKEND:-openrouter}"
OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen3-8b}"
OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1}"
export RL_EPOCHS
mkdir -p "$LOGDIR"

# Auto-detect ACCOUNT (matches submit_phi4_training.sh).
ACCOUNT="${ACCOUNT:-}"
if [ -z "${ACCOUNT}" ]; then
    USER_GROUPS=" $(id -Gn) "
    for cand in arora pli danqic; do
        if echo "${USER_GROUPS}" | grep -q " ${cand} "; then
            ACCOUNT="${cand}"
            break
        fi
    done
fi
ACCT_FLAG=""
if [ -n "${ACCOUNT}" ]; then
    ACCT_FLAG="--account=${ACCOUNT}"
fi

ACTIVATE="source $ROOT/.venv/bin/activate"
CACHE_SETUP="export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_\${SLURM_JOB_ID} && rm -rf \$TORCHINDUCTOR_CACHE_DIR && mkdir -p \$TORCHINDUCTOR_CACHE_DIR && export VLLM_NO_USAGE_STATS=1 && export PYTORCH_ALLOC_CONF=expandable_segments:True && export WANDB_MODE=disabled && export VLLM_WORKER_MULTIPROC_METHOD=spawn && export HF_HUB_OFFLINE=1"

# RL wrapper. Two modes:
#   JUDGE_BACKEND=openrouter (default) — judge calls go to OpenRouter Qwen3-8B
#     paid endpoint. No GPU spent on judge. All allocated GPUs go to trainer.
#   JUDGE_BACKEND=local — legacy path: launch a vLLM reward server on the last
#     GPU, run GRPO on the remaining GPUs.
# Args: $1=model_path $2=reward_model_path $3=reward_model_name $4=save_path
#       $5=data_dir   $6=log_dir           $7=datasets          $8=n_train_gpus
RL_SCRIPT_OPENROUTER='
SFT_MODEL=$(ls -dt "$1"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$SFT_MODEL" ]; then SFT_MODEL="$1"; fi
N_TRAIN_GPUS=$8
mkdir -p "$6"
TRAIN_GPUS=$(seq -s, 0 $((N_TRAIN_GPUS-1)))
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set in environment"
    exit 1
fi
echo "[rl] OpenRouter judge: model=$3 url=${OPENROUTER_URL:-https://openrouter.ai/api/v1}"
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} python rapo_rl.py \
    --model-path $SFT_MODEL \
    --reward-server-url ${OPENROUTER_URL:-https://openrouter.ai/api/v1} \
    --reward-model-name $3 \
    --save-path $4 \
    --data-dir $5 \
    --datasets "$7" \
    --epochs ${RL_EPOCHS} \
    --log-dir $6 \
    --reward-concurrency 64
'

RL_SCRIPT_LOCAL='
SFT_MODEL=$(ls -dt "$1"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$SFT_MODEL" ]; then SFT_MODEL="$1"; fi
N_TRAIN_GPUS=$8
REWARD_GPU=$N_TRAIN_GPUS
REWARD_PORT=$((8000 + (SLURM_JOB_ID % 1000)))
mkdir -p "$6"
REWARD_LOG="$6/reward_server_${SLURM_JOB_ID}.log"
echo "Starting vLLM reward server on GPU ${REWARD_GPU}, port $REWARD_PORT..."
CUDA_VISIBLE_DEVICES=${REWARD_GPU} python -m vllm.entrypoints.openai.api_server \
    --model $2 \
    --served-model-name $3 \
    --port $REWARD_PORT \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-num-seqs 64 \
    > "$REWARD_LOG" 2>&1 &
VLLM_PID=$!

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

TRAIN_GPUS=$(seq -s, 0 $((N_TRAIN_GPUS-1)))
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} python rapo_rl.py \
    --model-path $SFT_MODEL \
    --reward-server-url http://localhost:${REWARD_PORT}/v1 \
    --reward-model-name $3 \
    --save-path $4 \
    --data-dir $5 \
    --datasets "$7" \
    --epochs ${RL_EPOCHS} \
    --log-dir $6 \
    --reward-concurrency 32
TRAIN_EXIT=$?

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
exit $TRAIN_EXIT
'

if [ "${JUDGE_BACKEND}" = "openrouter" ]; then
    RL_SCRIPT="${RL_SCRIPT_OPENROUTER}"
else
    RL_SCRIPT="${RL_SCRIPT_LOCAL}"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Per-model submission (mirrors submit_rapo.sh's OLMo block).
#
# submit_one MODEL_KEY MODEL_NAME MODEL_PATH SFT_GPUS SFT_MEM RL_GPUS RL_MEM \
#            RL_N_TRAIN_GPUS [DATAGEN_TP] [DATAGEN_GPU_UTIL]
# ═══════════════════════════════════════════════════════════════════════════
submit_one() {
    local MODEL_KEY="$1"
    local MODEL_NAME="$2"
    local MODEL_PATH="$3"
    local SFT_GPUS="$4"
    local SFT_MEM="$5"
    local RL_GPUS="$6"
    local RL_MEM="$7"
    local RL_N_TRAIN_GPUS="$8"
    local DATAGEN_TP="${9:-1}"
    local DATAGEN_GPU_UTIL="${10:-0.7}"
    local DATAGEN_MAX_MODEL_LEN="${11:-8192}"
    local DATAGEN_MAX_TOKENS="${12:-8192}"
    local SFT_AILAB_GPUS="${13:-2}"
    local SFT_AILAB_MEM="${14:-280G}"

    # Phi-4-reasoning thinking traces avg ~8K tokens, 800 prompts × 2 passes
    # exceeds 4h on 1 GPU. 8h walltime gives headroom; GPT-OSS on TP=2 ran in
    # 22m but we use the same cap for safety.
    JOB_DATAGEN=$(sbatch --parsable \
        --job-name=rapo_datagen_${MODEL_KEY} \
        --partition=$PARTITION --qos=$QOS $ACCT_FLAG \
        --gres=gpu:${DATAGEN_TP} --cpus-per-gpu=4 --mem=120G \
        --time=08:00:00 \
        --output="$LOGDIR/rapo_datagen_${MODEL_KEY}_%j.out" \
        --wrap "$CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && \
        python rapo_sft.py \
            --model-path $MODEL_PATH \
            --save-path $RAPO_DIR/checkpoints/${MODEL_NAME}-SFT \
            --data-save-path $RAPO_DIR/sft_data/${MODEL_NAME} \
            --data-dir $DATA_DIR \
            --datasets 'starbenign:400,stratasword:400' \
            --epochs $SFT_EPOCHS \
            --tensor-parallel-size ${DATAGEN_TP} \
            --gpu-memory-utilization ${DATAGEN_GPU_UTIL} \
            --max-model-len ${DATAGEN_MAX_MODEL_LEN} \
            --max-tokens ${DATAGEN_MAX_TOKENS} \
            --skip-training")
    echo "Submitted ${MODEL_NAME} datagen: $JOB_DATAGEN"

    # Dual-submit SFT to BOTH pli-c (H100×N) and ailab (H200×M). Both write to
    # the same checkpoint dir. Whichever finishes first writes a
    # `.training_complete` marker (rapo_sft.py); the other side detects it on
    # startup and exits 0 immediately. RL chain depends on `afterany` of BOTH
    # SFT jobs — winner does the work, loser no-ops in seconds, then RL fires.
    SFT_SAVE_PATH="$RAPO_DIR/checkpoints/${MODEL_NAME}-SFT"
    SFT_PRECHECK="if [ -f \"${SFT_SAVE_PATH}/.training_complete\" ]; then echo \"[sft-skip] sibling already finished, exiting cleanly\"; exit 0; fi"
    SFT_WRAP_BODY="${SFT_PRECHECK} && $CACHE_SETUP && $ACTIVATE && cd $RAPO_DIR && \
        accelerate launch \
            --config_file $ROOT/defenses/star1/train/configs/deepspeed_zero3.yaml \
            --num_processes \${NUM_PROCESSES} --num_machines 1 --machine_rank 0 \
            --deepspeed_multinode_launcher standard \
            --main_process_port \$((10000 + (SLURM_JOB_ID % 50000))) \
            rapo_sft.py \
            --model-path $MODEL_PATH \
            --save-path ${SFT_SAVE_PATH} \
            --data-save-path $RAPO_DIR/sft_data/${MODEL_NAME} \
            --epochs $SFT_EPOCHS \
            --skip-datagen"

    JOB_SFT_PLIC=$(sbatch --parsable \
        --job-name=rapo_sft_${MODEL_KEY}_plic \
        --partition=pli-c --qos=pli $ACCT_FLAG \
        --gres=gpu:${SFT_GPUS} --cpus-per-gpu=4 --mem=${SFT_MEM} \
        --time=08:00:00 \
        --dependency=afterok:${JOB_DATAGEN} \
        --output="$LOGDIR/rapo_sft_${MODEL_KEY}_plic_%j.out" \
        --wrap "NUM_PROCESSES=${SFT_GPUS} && ${SFT_WRAP_BODY}")
    echo "Submitted ${MODEL_NAME} SFT @ pli-c (${SFT_GPUS}xH100, after $JOB_DATAGEN): $JOB_SFT_PLIC"

    JOB_SFT_AILAB=$(sbatch --parsable \
        --job-name=rapo_sft_${MODEL_KEY}_ailab \
        --partition=ailab --qos=ailab $ACCT_FLAG \
        --gres=gpu:${SFT_AILAB_GPUS} --cpus-per-gpu=4 --mem=${SFT_AILAB_MEM} \
        --time=02:00:00 \
        --dependency=afterok:${JOB_DATAGEN} \
        --output="$LOGDIR/rapo_sft_${MODEL_KEY}_ailab_%j.out" \
        --wrap "NUM_PROCESSES=${SFT_AILAB_GPUS} && ${SFT_WRAP_BODY}")
    echo "Submitted ${MODEL_NAME} SFT @ ailab (${SFT_AILAB_GPUS}xH200, after $JOB_DATAGEN): $JOB_SFT_AILAB"

    # Set the dependency for downstream RL: wait for BOTH (afterany so loser's
    # quick exit + winner's success both unblock the chain).
    SFT_DEP="afterany:${JOB_SFT_PLIC},afterany:${JOB_SFT_AILAB}"
    JOB_SFT="${JOB_SFT_PLIC}"  # for log message only

    # RL chain: N chunks of $RL_TIME each (default 8h), linked by afterany so a
    # TIMEOUT chunk still triggers the next one. rapo_rl.py auto-resumes from
    # the latest checkpoint each time; the chain self-noops once training is
    # complete (trainer.train() returns immediately).
    # When using OpenRouter, the judge does not consume a GPU — drop one GPU
    # off the RL request and grow the trainer to use them all.
    if [ "${JUDGE_BACKEND}" = "openrouter" ]; then
        EFFECTIVE_RL_GPUS=${RL_N_TRAIN_GPUS}
        EFFECTIVE_RL_N_TRAIN=${RL_N_TRAIN_GPUS}
        REWARD_MODEL_PATH="${OPENROUTER_MODEL}"
        REWARD_MODEL_NAME="${OPENROUTER_MODEL}"
        EXTRA_EXPORT="export OPENROUTER_API_KEY=\"\${OPENROUTER_API_KEY}\" && export OPENROUTER_URL=\"${OPENROUTER_URL}\""
    else
        EFFECTIVE_RL_GPUS=${RL_GPUS}
        EFFECTIVE_RL_N_TRAIN=${RL_N_TRAIN_GPUS}
        REWARD_MODEL_PATH="$ROOT/models/Qwen3-8B"
        REWARD_MODEL_NAME="Qwen3-8B"
        EXTRA_EXPORT="true"
    fi

    PREV_DEP="${SFT_DEP}"
    LAST_RL_JOB=""
    for chunk_i in $(seq 1 ${RL_CHUNKS}); do
        JOB_RL=$(sbatch --parsable \
            --job-name=rapo_rl_${MODEL_KEY}_c${chunk_i} \
            --partition=$PARTITION --qos=$QOS $ACCT_FLAG \
            --gres=gpu:${EFFECTIVE_RL_GPUS} --cpus-per-gpu=4 --mem=${RL_MEM} \
            --time=$RL_TIME \
            --dependency=${PREV_DEP} \
            --export=ALL,OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}",OPENROUTER_URL="${OPENROUTER_URL}" \
            --output="$LOGDIR/rapo_rl_${MODEL_KEY}_c${chunk_i}_%j.out" \
            --wrap "$CACHE_SETUP && $EXTRA_EXPORT && $ACTIVATE && cd $RAPO_DIR && bash -c '$RL_SCRIPT' _ \
                $RAPO_DIR/checkpoints/${MODEL_NAME}-SFT \
                ${REWARD_MODEL_PATH} \
                ${REWARD_MODEL_NAME} \
                $RAPO_DIR/checkpoints/${MODEL_NAME}-RL \
                $DATA_DIR \
                $RAPO_DIR/rl_logs \
                'wildjailbreak:300,star:100,starbenign:400' \
                ${EFFECTIVE_RL_N_TRAIN}")
        echo "Submitted ${MODEL_NAME} RL chunk ${chunk_i}/${RL_CHUNKS}: $JOB_RL (dep=${PREV_DEP})"
        PREV_DEP="afterany:${JOB_RL}"
        LAST_RL_JOB="${JOB_RL}"
    done

    JOB_EVAL=$(sbatch --parsable \
        --job-name=rapo_eval_${MODEL_KEY} \
        --partition=$PARTITION --qos=$QOS $ACCT_FLAG \
        --gres=gpu:1 --cpus-per-gpu=4 --mem=80G \
        --time=04:00:00 \
        --dependency=afterany:${LAST_RL_JOB} \
        --output="$LOGDIR/rapo_eval_${MODEL_KEY}_%j.out" \
        --wrap "$CACHE_SETUP && $ACTIVATE && cd $ROOT && \
        PYTHONPATH=$ROOT python defenses/rapo/eval_rapo.py \
            --model_path $RAPO_DIR/checkpoints/${MODEL_NAME}-RL \
            --model_name RAPO-${MODEL_NAME} \
            --output_dir $RAPO_DIR/results \
            --temperature 0.6 --top_p 0.95 --top_k 50")
    echo "Submitted ${MODEL_NAME} Eval (after $LAST_RL_JOB): $JOB_EVAL"

    echo "${MODEL_NAME}: SFT=$JOB_SFT → RL chain (last=$LAST_RL_JOB) → Eval=$JOB_EVAL"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phi-4-reasoning (14B dense)
#   Datagen: 1 GPU (vLLM, ~28GB weights + KV cache, fits comfortably)
#   SFT (full-param ZeRO-3): 5 GPUs, 400G — 224/N rule for 14B = 45GB/GPU
#                            ZeRO state + ~19GB transient (matches STAR-1/SafeKey
#                            sizing for Phi-4 from submit_phi4_training.sh).
#   RL (LoRA + ZeRO-3 trainer + Qwen3-8B judge co-resident on rank 0):
#                            2 GPUs, 200G — trainer ranks on GPU 0/1, judge
#                            on GPU 1; rank-0 hosts ~33GB trainer + 18GB judge
#                            ≈ 51GB peak.
# ═══════════════════════════════════════════════════════════════════════════
# Phi-4-reasoning thinking traces average ~8K tokens — observed budget hits at
# max_tokens=8192. Use 16384 (also vLLM max_model_len ceiling).
# Phi-4 RE-ENABLED 2026-05-05 after force-close </think> in pass-2 prefix
# (rapo_sft.py is_phi4 branch) + slicer fix to consume <|im_sep|>. Smoke 7703570
# verified 4/4 rows produce clean SFT data with proper apply_chat_template
# renderability. ailab fallback: 2× H200 (validated for STAR-1 Phi-4 ZeRO-3 14B).
submit_one phi4 Phi-4-reasoning $ROOT/models/Phi-4-reasoning 5 400G 2 200G 2 1 0.85 8192 8192 2 280G

# ═══════════════════════════════════════════════════════════════════════════
# GPT-OSS-20B (21B MoE, mxfp4 → bf16 dequant ~40GB)
#   Datagen: 2 GPUs, TP=2 (vLLM at TP=2 handles 40GB base + ~50GB KV cache)
#   SFT (full-param ZeRO-3): 7 GPUs, 560G — ZeRO state for ~21B params at
#                            16 bytes/param ÷ 7 ≈ 48GB + 19GB transient ≈ 67GB.
#   RL (LoRA + ZeRO-3 trainer + Qwen3-8B judge co-resident on rank 0):
#                            4 GPUs, 320G — trainer ranks on GPU 0/1/2, judge
#                            on GPU 3; rank-0 ≈ 45GB trainer + 18GB judge ≈ 63GB.
# ═══════════════════════════════════════════════════════════════════════════
# GPT-OSS analysis-channel reasoning is verbose; needs ~16384 to reach
# the <|channel|>final<|message|> transition in pass-2.
# ailab fallback: 3× H200 (420 GB total, comfortable for 21B MoE ZeRO-3 SFT).
submit_one gptoss GPT-OSS-20B $ROOT/models/GPT-OSS-20B 7 560G 4 320G 3 2 0.85 16384 16384 3 420G

echo ""
echo "=== Submitted RAPO chains for Phi-4-reasoning and GPT-OSS-20B ==="
