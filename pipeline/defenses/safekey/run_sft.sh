#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ACCELERATE="${PROJECT_DIR}/.venv/bin/accelerate"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

TRAIN_MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
if [ -z "${TRAIN_MAIN_PROCESS_PORT}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  TRAIN_MAIN_PROCESS_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
fi
if [ -z "${TRAIN_MAIN_PROCESS_PORT}" ]; then
  TRAIN_MAIN_PROCESS_PORT=29500
fi
TRAIN_MAIN_PROCESS_PORT="$("${PYTHON}" - "${TRAIN_MAIN_PROCESS_PORT}" <<'PY'
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
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    --main_process_port "${TRAIN_MAIN_PROCESS_PORT}" \
    sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --data_path ../data/train/sft_mix_2k.json \
    --n_epochs 5 \
    --experiment_name safe_lrm \
    --base_model Llama \
    --base_flag 0 \
    --think_flag 1 \
    --output_dir ../data/models/8b_safekey \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --safety_head \
    --key_sentence_prediction
