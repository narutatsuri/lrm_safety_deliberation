#!/usr/bin/env bash
#SBATCH --partition=pli-c
#SBATCH --qos=pli-c
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

# Required env: MODEL, MODE, SPLIT, GUARDRAIL
set -euo pipefail
ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
export VLLM_NO_USAGE_STATS=1
export HF_HOME="${HF_HOME:-/scratch/gpfs/ARORA/nr3764/.cache/hf_cache}"

JOB_CACHE_ROOT="${TMPDIR:-/tmp}/k32cls_${SLURM_JOB_ID:-local_$$}"
export VLLM_CACHE_ROOT="$JOB_CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap "rm -rf $JOB_CACHE_ROOT" EXIT

echo "=== CLS JOB $SLURM_JOB_ID on $(hostname) ==="
echo "MODEL=$MODEL MODE=$MODE SPLIT=$SPLIT GUARDRAIL=$GUARDRAIL"
nvidia-smi -L || true

python -u experiments/nested_branching_variance/k32_full/classify.py \
    --model "$MODEL" --mode "$MODE" --split "$SPLIT" --guardrail "$GUARDRAIL"
RC=$?
echo "=== EXIT $RC ==="
exit $RC
