#!/usr/bin/env bash
# Run the full ASR/ORR suite (data/ manifest splits asr_full / orr_full) for ONE
# (model, split) under the universal 16384-thinking + 1024-force-close protocol.
#
#   asr_full = advbench+harmbench+strongreject+sorrybench+jailbreakbench+wildjailbreak+fortress (4,273)
#   orr_full = or_bench+falsereject+coconot (2,885)
#
# Per-model sampling / gpu_util / channel flag read from scripts/neurips_final/common.py,
# but max_new_tokens and max_model_len are FORCED here (16384 / 32768) so all four models
# get an identical thinking budget regardless of their common.py entry.
#
# Output: eval_results/asr_orr_16k/<split>/<MODEL>/{generations,results,summary}.jsonl
# 4-guardrail majority vote (>=3/4, ties -> compliance) per evaluation/full_eval_pipeline.
#
# Provenance: adapted from the research repo scripts/slurm/asr_orr_16k/run_one.sh.
# Self-contained for the release: drives the moved evaluation.full_eval_pipeline
# package (pipeline/eval/) and reads each model's decoding / channel flag from the
# shared lrm_safety library (lrm_safety.models), so no original research tree is
# needed. Outputs land under LRM_SAFETY_ARTIFACTS; activate your env first
# (pip install -e .), or point LRM_SAFETY_VENV at a venv to source.
set -euo pipefail

MODEL="$1"      # registry key, e.g. Qwen3-8B
SPLIT="$2"      # asr | orr | asr_full | orr_full
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SLURM_DIR/.." && pwd)"                  # pipeline/eval (has evaluation/)
RELEASE_ROOT="$(cd "$SLURM_DIR/../../.." && pwd)"        # camera_ready (has lrm_safety/)
ROOT="${LRM_SAFETY_ARTIFACTS:-$RELEASE_ROOT}"            # where outputs land
[ -n "${LRM_SAFETY_VENV:-}" ] && source "$LRM_SAFETY_VENV/bin/activate"

export PYTHONPATH="$RELEASE_ROOT:$EVAL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_NO_USAGE_STATS=1
export HF_HOME="${HF_HOME:-/scratch/gpfs/ARORA/nr3764/.cache/hf_cache}"
# Per-job local caches (avoid shared-scratch torchinductor checksum corruption / quota).
JOB_CACHE_ROOT="${TMPDIR:-/tmp}/asr_orr_16k_${SLURM_JOB_ID:-local_$$}"
export VLLM_CACHE_ROOT="$JOB_CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Forced universal protocol.
MAX_NEW_TOK=16384
MAX_LEN=32768
OUT_DIR="$ROOT/eval_results/${OUT_TAG:-asr_orr_16k}"
# Shard-aware log dir so concurrent row-slice jobs don't interleave one run.log.
if [ -n "${ROW_SLICE:-}" ]; then
    LOG_DIR="$OUT_DIR/$SPLIT/${MODEL}__shard_${ROW_SLICE/:/_}__/logs"
else
    LOG_DIR="$OUT_DIR/$SPLIT/$MODEL/logs"
fi
mkdir -p "$LOG_DIR"
SAMPLE_ARG=()
[ -n "${SAMPLE_LIMIT:-}" ] && SAMPLE_ARG+=(--sample_limit "$SAMPLE_LIMIT")
# Optional row-slice sharding: ROW_SLICE="start:end" -> only records[start:end] this job.
ROW_ARG=()
[ -n "${ROW_SLICE:-}" ] && ROW_ARG+=(--row_slice "$ROW_SLICE")
# Optional K rollouts (vLLM n=K) + seed for generation-stochasticity error bars.
ROLL_ARG=()
[ -n "${NUM_ROLLOUTS:-}" ] && ROLL_ARG+=(--num_rollouts "$NUM_ROLLOUTS")
[ -n "${SEED:-}" ] && ROLL_ARG+=(--seed "$SEED")

read MODEL_PATH GPU_UTIL TEMP TOP_P TOP_K MIN_P IS_CHANNEL < <(python - <<PY
import sys; sys.path.insert(0, "$RELEASE_ROOT")
from lrm_safety.models import get
s = get("$MODEL"); d = s.decoding
print(s.path(), s.gpu_memory_utilization,
      d.temperature, d.top_p,
      "NONE" if d.top_k is None else d.top_k,
      "NONE" if d.min_p is None else d.min_p,
      "1" if s.is_channel else "0")
PY
)

ARGS=()
[ "$TEMP"  != "NONE" ] && ARGS+=(--temperature "$TEMP")
[ "$TOP_P" != "NONE" ] && ARGS+=(--top_p "$TOP_P")
[ "$TOP_K" != "NONE" ] && ARGS+=(--top_k "$TOP_K")
[ "$MIN_P" != "NONE" ] && ARGS+=(--min_p "$MIN_P")
[ "$IS_CHANNEL" = "1" ] && ARGS+=(--keep_special_tokens)

echo "=== $MODEL :: $SPLIT :: max_new=$MAX_NEW_TOK max_len=$MAX_LEN gpu_util=$GPU_UTIL channel=$IS_CHANNEL ==="
python -u -m evaluation.full_eval_pipeline.run \
    --model_path "$MODEL_PATH" \
    --split "$SPLIT" \
    --output_dir "$OUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOK" \
    --max_model_len "$MAX_LEN" \
    --gpu_memory_utilization "$GPU_UTIL" \
    "${ARGS[@]}" \
    "${SAMPLE_ARG[@]}" \
    "${ROW_ARG[@]}" \
    "${ROLL_ARG[@]}" \
    2>&1 | tee -a "$LOG_DIR/run.log"

echo "DONE: $MODEL $SPLIT"
