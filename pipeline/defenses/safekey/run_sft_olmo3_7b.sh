#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ACCELERATE="${PROJECT_DIR}/.venv/bin/accelerate"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

# Faithful SafeKey objective adapted to Olmo-3-7B-Think on a single GPU.

DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/defenses/safekey/data/sft_mix_2k.json}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Olmo-3-7B-Think}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/models/SafeKey-Olmo-3-7B-Think}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/outputs/safekey_train_logs}"
N_EPOCHS="${N_EPOCHS:-5}"
LAST_K_EPOCH="${LAST_K_EPOCH:-2}"
TRAIN_BSZ_PER_GPU="${TRAIN_BSZ_PER_GPU:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_CKPTS="${MAX_CKPTS:-1}"

if [ -n "${EXTRA_SFT_ARGS:-}" ]; then
  read -r -a EXTRA_SFT_ARGS_ARR <<< "${EXTRA_SFT_ARGS}"
else
  EXTRA_SFT_ARGS_ARR=()
fi

find_free_port() {
  local port="$1"
  while true; do
    if command -v ss >/dev/null 2>&1; then
      if ! ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .; then
        echo "${port}"
        return 0
      fi
    elif command -v netstat >/dev/null 2>&1; then
      if ! netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}$"; then
        echo "${port}"
        return 0
      fi
    else
      echo "${port}"
      return 0
    fi
    port=$((port + 1))
    if [ "${port}" -gt 65000 ]; then
      port=10000
    fi
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

cd "${SCRIPT_DIR}"
"${ACCELERATE}" launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    --main_process_port "${TRAIN_MAIN_PROCESS_PORT}" \
    sft.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --n_epochs "${N_EPOCHS}" \
    --last_k_epoch "${LAST_K_EPOCH}" \
    --experiment_name SafeKey-Olmo3-7B-Think \
    --base_model Olmo \
    --base_flag 0 \
    --think_flag 1 \
    --train_bsz_per_gpu "${TRAIN_BSZ_PER_GPU}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_ckpts "${MAX_CKPTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --safety_head \
    --key_sentence_prediction \
    "${EXTRA_SFT_ARGS_ARR[@]}"
