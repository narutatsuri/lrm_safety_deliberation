#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash evaluation/full_eval_pipeline/run.sh <model_path> <split> [extra args...]"
    echo "Split: asr | orr | orr_supplementary | all"
    exit 1
fi

ROOT="${LRM_SAFETY_ARTIFACTS:-/scratch/gpfs/ARORA/nr3764/inference_skill_composition}"
cd "$ROOT"
source .venv/bin/activate

export PYTHONPATH="$ROOT"
export VLLM_NO_USAGE_STATS=1
export HF_HOME="${HF_HOME:-/scratch/gpfs/ARORA/nr3764/.cache/hf_cache}"

MODEL_PATH="$1"
SPLIT="$2"
shift 2

exec python -u -m evaluation.full_eval_pipeline.run \
    --model_path "$MODEL_PATH" \
    --split "$SPLIT" \
    "$@"
