#!/bin/bash
# STAIR Stage 1: LoRA SFT launch script.
# Parameterised via environment variables so a single script works for
# all four target models (Qwen3-8B, OLMo-3-7B-Think, Phi-4-reasoning,
# GPT-OSS-20B).
#
# Required env vars:
#   MODEL_PATH  - path to the base model directory
#   BASE_MODEL  - model family tag: Qwen | Olmo | Phi4 | GPTOSS
#   OUTPUT_DIR  - where to write adapter + merged checkpoints
#
# Optional env vars (defaults match the STAIR paper):
#   DATA_PATH, LOG_DIR, N_EPOCHS, TRAIN_BSZ_PER_GPU, GRAD_ACC_STEPS,
#   MAX_SEQ_LEN, LR, WARMUP_STEPS, LORA_R, LORA_ALPHA, MERGE_ADAPTER

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STAIR_VENV="${PROJECT_DIR}/.venv_stair"
if [ -x "${STAIR_VENV}/bin/python" ] && "${STAIR_VENV}/bin/python" -c "import deepspeed" >/dev/null 2>&1; then
    ACCELERATE="${STAIR_VENV}/bin/accelerate"
    export PATH="${STAIR_VENV}/bin:${PATH}"
else
    ACCELERATE="${PROJECT_DIR}/.venv/bin/accelerate"
    export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
fi
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

# --- Required ---
: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${BASE_MODEL:?BASE_MODEL must be set (Qwen|Olmo|Phi4|GPTOSS)}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

# --- Defaults matching STAIR paper ---
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/defenses/stair_repo/data/stair_sft.json}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/outputs/stair_train_logs}"
N_EPOCHS="${N_EPOCHS:-3}"
TRAIN_BSZ_PER_GPU="${TRAIN_BSZ_PER_GPU:-1}"
# Paper uses batch_size=16. With 1 GPU and bsz_per_gpu=1, need 16 acc steps.
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MERGE_ADAPTER="${MERGE_ADAPTER:-1}"
MAX_CKPTS="${MAX_CKPTS:-5}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PREPROCESS_NUM_WORKERS="${PREPROCESS_NUM_WORKERS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# Build merge flag
MERGE_FLAG=""
if [ "${MERGE_ADAPTER}" = "1" ]; then
    MERGE_FLAG="--merge_adapter"
fi

if [ -n "${RESUME_CHECKPOINT}" ]; then
    RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
    RESUME_ARGS=()
fi

# --- Free port selection ---
find_free_port() {
  local port="$1"
  while true; do
    if command -v ss >/dev/null 2>&1; then
      if ! ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .; then
        echo "${port}"; return 0
      fi
    elif command -v netstat >/dev/null 2>&1; then
      if ! netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}$"; then
        echo "${port}"; return 0
      fi
    else
      echo "${port}"; return 0
    fi
    port=$((port + 1))
    if [ "${port}" -gt 65000 ]; then port=10000; fi
  done
}

if [ -n "${MAIN_PROCESS_PORT:-}" ]; then
  TRAIN_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  TRAIN_MAIN_PROCESS_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
else
  TRAIN_MAIN_PROCESS_PORT=29500
fi
TRAIN_MAIN_PROCESS_PORT="$(find_free_port "${TRAIN_MAIN_PROCESS_PORT}")"

# --- Launch ---
cd "${SCRIPT_DIR}"
"${ACCELERATE}" launch \
    --config_file ./configs/deepspeed_zero2.yaml \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    --main_process_port "${TRAIN_MAIN_PROCESS_PORT}" \
    sft_lora.py \
    --model_path "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --cache_dir "${PROJECT_DIR}/outputs/stair_dataset_cache" \
    --n_epochs "${N_EPOCHS}" \
    --train_bsz_per_gpu "${TRAIN_BSZ_PER_GPU}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --learning_rate "${LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --max_ckpts "${MAX_CKPTS}" \
    --save_every_steps "${SAVE_EVERY_STEPS}" \
    --max_samples "${MAX_SAMPLES}" \
    --preprocessing_num_workers "${PREPROCESS_NUM_WORKERS}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    "${RESUME_ARGS[@]}" \
    ${MERGE_FLAG}
