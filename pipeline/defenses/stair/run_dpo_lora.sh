#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Use .venv (torch 2.9.1) — NOT .venv_stair (torch 2.5.1).
# transformers' _load_optimizer_and_scheduler calls torch.load on optimizer.pt
# during resume_from_checkpoint; CVE-2025-32434 makes transformers refuse to
# load when torch < 2.6, so resume crashes with .venv_stair. .venv has all
# needed deps (accelerate 1.12, transformers 5.6.dev, trl 0.29, peft 0.18,
# deepspeed 0.18.6), so we use it unconditionally for DPO.
PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${BASE_MODEL:?BASE_MODEL must be set}"
: "${DATA_PATH:?DATA_PATH must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

DPO_BETA="${DPO_BETA:-0.1}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-16}"
TRAIN_BSZ_PER_GPU="${TRAIN_BSZ_PER_GPU:-1}"
LR="${LR:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
N_EPOCHS="${N_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
ROUND_NUM="${ROUND_NUM:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MERGE_ADAPTER="${MERGE_ADAPTER:-1}"

latest_trainer_checkpoint() {
  local root="$1"
  local best=""
  local best_step=-1
  while IFS= read -r path; do
    [ -f "${path}/trainer_state.json" ] || continue
    local name
    local step
    name="$(basename "${path}")"
    step="$(echo "${name}" | sed -n 's/.*-\([0-9][0-9]*\)$/\1/p')"
    [ -n "${step}" ] || continue
    if [ "${step}" -gt "${best_step}" ]; then
      best_step="${step}"
      best="${path}"
    fi
  done < <(find "${root}" -type d -name 'checkpoint-*' 2>/dev/null | sort)
  echo "${best}"
}

if [ -z "${RESUME_CHECKPOINT}" ]; then
  RESUME_CHECKPOINT="$(latest_trainer_checkpoint "${OUTPUT_DIR}")"
fi

MERGE_FLAG=""
if [ "${MERGE_ADAPTER}" = "1" ]; then
  MERGE_FLAG="--merge_adapter"
fi

RESUME_ARGS=()
if [ -n "${RESUME_CHECKPOINT}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi

NUM_PROCESSES="${NUM_PROCESSES:-1}"

# Free port selection (only used in multi-GPU path).
find_free_port() {
  local port="$1"
  while true; do
    if command -v ss >/dev/null 2>&1; then
      if ! ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .; then
        echo "${port}"; return 0
      fi
    else
      echo "${port}"; return 0
    fi
    port=$((port + 1))
    if [ "${port}" -gt 65000 ]; then port=10000; fi
  done
}

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    DPO_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
  else
    DPO_PORT=29500
  fi
  DPO_PORT="$(find_free_port "${DPO_PORT}")"
  # Pin accelerate to .venv too (matches PYTHON above; .venv_stair's older
  # torch breaks resume).
  ACCELERATE="${PROJECT_DIR}/.venv/bin/accelerate"
  "${ACCELERATE}" launch \
    --config_file "${SCRIPT_DIR}/configs/deepspeed_zero2.yaml" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    --main_process_port "${DPO_PORT}" \
    "${SCRIPT_DIR}/dpo_lora.py" \
    --model_path "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --dpo_beta "${DPO_BETA}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --train_bsz_per_gpu "${TRAIN_BSZ_PER_GPU}" \
    --learning_rate "${LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --n_epochs "${N_EPOCHS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --round_num "${ROUND_NUM}" \
    "${RESUME_ARGS[@]}" \
    ${MERGE_FLAG}
else
  # Single-GPU DPO: pin to GPU 0 so HF Trainer / TRL DPOTrainer doesn't
  # auto-wrap in nn.DataParallel when multiple GPUs are visible (DP would
  # replicate the full base model on each GPU and OOM on large models).
  # SLURM may have set CUDA_VISIBLE_DEVICES to 0,1,2,3 already; force to 0.
  export CUDA_VISIBLE_DEVICES=0
  "${PYTHON}" "${SCRIPT_DIR}/dpo_lora.py" \
    --model_path "${MODEL_PATH}" \
    --base_model "${BASE_MODEL}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --dpo_beta "${DPO_BETA}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --train_bsz_per_gpu "${TRAIN_BSZ_PER_GPU}" \
    --learning_rate "${LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --n_epochs "${N_EPOCHS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --round_num "${ROUND_NUM}" \
    "${RESUME_ARGS[@]}" \
    ${MERGE_FLAG}
fi
