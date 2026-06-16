#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ACCELERATE="${PROJECT_DIR}/.venv/bin/accelerate"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

# Faithful STAR-1 objective adapted to Qwen3-8B on a single GPU.
# Upstream recipe reference: UCSC-VLAA/STAR-1 train/run_sft.sh

DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/defenses/star1/data/STAR-1.json}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/models/STAR1-Qwen3-8B}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/outputs/star1_train_logs}"
N_EPOCHS="${N_EPOCHS:-5}"
TRAIN_BSZ_PER_GPU="${TRAIN_BSZ_PER_GPU:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_CKPTS="${MAX_CKPTS:-5}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [ -n "${EXTRA_SFT_ARGS:-}" ]; then
  read -r -a EXTRA_SFT_ARGS_ARR <<< "${EXTRA_SFT_ARGS}"
else
  EXTRA_SFT_ARGS_ARR=()
fi

if [ -n "${RESUME_CHECKPOINT}" ]; then
  RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
  RESUME_ARGS=()
fi

TRAIN_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
if [ -z "${TRAIN_MAIN_PROCESS_PORT}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  TRAIN_MAIN_PROCESS_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
fi
if [ -z "${TRAIN_MAIN_PROCESS_PORT}" ]; then
  TRAIN_MAIN_PROCESS_PORT=29500
fi
TRAIN_MAIN_PROCESS_PORT="$("${PROJECT_DIR}/.venv/bin/python" - "${TRAIN_MAIN_PROCESS_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for _ in range(1000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            print(port)
            sys.exit(0)
        except OSError:
            port += 1
            if port > 65000:
                port = 10000
print(sys.argv[1])
PY
)"

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
    --experiment_name STAR-1-Qwen3-8B \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1 \
    --train_bsz_per_gpu "${TRAIN_BSZ_PER_GPU}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_ckpts "${MAX_CKPTS}" \
    --save_every_steps "${SAVE_EVERY_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${LOG_DIR}" \
    "${RESUME_ARGS[@]}" \
    "${EXTRA_SFT_ARGS_ARR[@]}"
