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
    --train_bsz_per_gpu 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    --main_process_port "${TRAIN_MAIN_PROCESS_PORT}" \
    sft.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_path ../data/STAR-1.json \
    --n_epochs 5 \
    --experiment_name STAR-1 \
    --base_model Qwen \
    --base_flag 0 \
    --think_flag 1

# if distill model then base_flag=0 elif instruct model then base_flag=1
# if w/o think then think_flag=0 else default=1
# train_bsz_per_gpu * num_processes should be 8 to keep the batchsize as 128
# you change the model_path to different model or change the data_path to use different finetune data
