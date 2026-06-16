#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

: "${ACTOR_MODEL_DIR:?ACTOR_MODEL_DIR must be set}"
: "${BASE_MODEL:?BASE_MODEL must be set}"
: "${PROMPT_PATH:?PROMPT_PATH must be set}"
: "${OUTPUT_PATH:?OUTPUT_PATH must be set}"

MCTS_WORKERS="${MCTS_WORKERS:-8}"
MCTS_ITERATIONS="${MCTS_ITERATIONS:-200}"
MCTS_MAX_DEPTH="${MCTS_MAX_DEPTH:-7}"
MCTS_GENERATE_SAMPLES="${MCTS_GENERATE_SAMPLES:-4}"
MCTS_MAX_TOKENS="${MCTS_MAX_TOKENS:-2048}"
MCTS_TEMPERATURE="${MCTS_TEMPERATURE:-1.2}"
MCTS_TOP_P="${MCTS_TOP_P:-0.9}"
MCTS_TOP_K="${MCTS_TOP_K:-50}"
MCTS_MODE="${MCTS_MODE:-safe-constraint}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"

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

VLLM_PID=""
cleanup() {
  if [ -n "${VLLM_PID}" ]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""
  fi
}
trap cleanup EXIT

if [ -n "${SLURM_JOB_ID:-}" ]; then
  VLLM_PORT="$((8000 + (SLURM_JOB_ID % 1000)))"
fi
VLLM_PORT="$(find_free_port "${VLLM_PORT}")"

mkdir -p "${OUTPUT_PATH}"

"${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${ACTOR_MODEL_DIR}" \
  --served-model-name "stair-actor" \
  --dtype bfloat16 \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --trust-remote-code \
  > "${OUTPUT_PATH}/vllm.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 600); do
  if curl -s "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -s "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM did not become ready on port ${VLLM_PORT}" >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/generate_mcts.py" \
  --actor_model_dir "${ACTOR_MODEL_DIR}" \
  --base_model "${BASE_MODEL}" \
  --prompt_path "${PROMPT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --server_url "http://localhost:${VLLM_PORT}/v1" \
  --worker_num "${MCTS_WORKERS}" \
  --mode "${MCTS_MODE}" \
  --iterations "${MCTS_ITERATIONS}" \
  --max_depth "${MCTS_MAX_DEPTH}" \
  --generate_samples "${MCTS_GENERATE_SAMPLES}" \
  --max_tokens "${MCTS_MAX_TOKENS}" \
  --temperature "${MCTS_TEMPERATURE}" \
  --top_p "${MCTS_TOP_P}" \
  --top_k "${MCTS_TOP_K}" \
  --log_file "${OUTPUT_PATH}/mcts.log"
