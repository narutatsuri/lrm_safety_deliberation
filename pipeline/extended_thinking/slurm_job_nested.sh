#!/usr/bin/env bash
#SBATCH --partition=pli-c
#SBATCH --qos=pli-c
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=della-k12g1
# Wrapper for nested_branching.py (full-pool nested-branching + r=0.2 extension).
# Required env: MODEL, SPLIT, SHARD_IDX, N_SHARDS
# Optional env: M, N, OUTPUT_ROOT
# Note: --cuts is HARDCODED to "20,40,60,80,120" here because sbatch --export
# splits comma-separated values, so passing CUTS via SBATCH env corrupts it.
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
export VLLM_NO_USAGE_STATS=1
export HF_HOME="${HF_HOME:-/scratch/gpfs/ARORA/nr3764/.cache/hf_cache}"

JOB_CACHE_ROOT="${TMPDIR:-/tmp}/k32nb_${SLURM_JOB_ID:-local_$$}"
export VLLM_CACHE_ROOT="$JOB_CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap "rm -rf $JOB_CACHE_ROOT" EXIT

echo "=== K32-NESTED-BRANCHING JOB $SLURM_JOB_ID on $(hostname) ==="
echo "MODEL=$MODEL SPLIT=$SPLIT SHARD=$SHARD_IDX/$N_SHARDS"
# CUTS can be overridden via CUTS_COLON env (e.g. "120" or "20:40:60:80:120")
# because sbatch --export splits commas. Default = all 5 cuts.
if [ -n "${CUTS_COLON:-}" ]; then
    CUTS="${CUTS_COLON//:/,}"
else
    CUTS="20,40,60,80,120"
fi
echo "M=${M:-8} N=${N:-8} CUTS=$CUTS"
nvidia-smi -L || true

python -u experiments/nested_branching_variance/k32_full/nested_branching.py \
    --model     "$MODEL" \
    --split     "$SPLIT" \
    --shard_idx "$SHARD_IDX" \
    --n_shards  "$N_SHARDS" \
    --M         "${M:-8}" \
    --N         "${N:-8}" \
    --cuts      "$CUTS" \
    ${OUTPUT_ROOT:+--output_root "$OUTPUT_ROOT"}
RC=$?
echo "=== EXIT $RC ==="
exit $RC
