#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"

: "${MCTS_PATH:?MCTS_PATH must be set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
: "${OUTPUT_PATH:?OUTPUT_PATH must be set}"

TERMINAL_ONLY="${TERMINAL_ONLY:-0}"
V_THRESHOLD="${V_THRESHOLD:-0.5}"
GOOD_VALUE_THRESHOLD="${GOOD_VALUE_THRESHOLD:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MIN_VISIT_TIMES="${MIN_VISIT_TIMES:-4}"
MAX_CHILD_NUM="${MAX_CHILD_NUM:-4}"
MAX_GB_RATIO="${MAX_GB_RATIO:-4}"
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-32}"

EXTRA_ARGS=()
if [ "${TERMINAL_ONLY}" = "1" ]; then
  EXTRA_ARGS+=(--terminal_only)
fi

"${PYTHON}" "${SCRIPT_DIR}/extract_dpo_pairs.py" \
  --mcts_path "${MCTS_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --v_threshold "${V_THRESHOLD}" \
  --good_value_threshold "${GOOD_VALUE_THRESHOLD}" \
  --max_tokens "${MAX_TOKENS}" \
  --min_visit_times "${MIN_VISIT_TIMES}" \
  --max_child_num "${MAX_CHILD_NUM}" \
  --max_gb_ratio "${MAX_GB_RATIO}" \
  --num_workers "${EXTRACT_NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"
