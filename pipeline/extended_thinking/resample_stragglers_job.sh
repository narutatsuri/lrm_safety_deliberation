#!/usr/bin/env bash
#SBATCH --partition=pli-c
#SBATCH --qos=pli-c
#SBATCH --account=arora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --exclude=della-k12g1
# Resample-stragglers job: iterates all (mode, split, shard) of one MODEL,
# running generate.py with K32_MAX_RESAMPLE=16 to clear residual mode-collapse
# truncations. Skips any shard with a RUNNING gen job (concurrent write guard).
#
# Required env: MODEL
set -euo pipefail
ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
export VLLM_NO_USAGE_STATS=1
export HF_HOME="${HF_HOME:-/scratch/gpfs/ARORA/nr3764/.cache/hf_cache}"

JOB_CACHE_ROOT="${TMPDIR:-/tmp}/k32resample_${SLURM_JOB_ID:-local_$$}"
export VLLM_CACHE_ROOT="$JOB_CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE_ROOT/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap "rm -rf $JOB_CACHE_ROOT" EXIT

# Bump retry budget to clear the persistent mode-collapse cases.
export K32_MAX_RESAMPLE=16
export K32_CHUNK_PROMPTS=32

echo "=== RESAMPLE-STRAGGLERS JOB $SLURM_JOB_ID on $(hostname) ==="
echo "MODEL=$MODEL  K32_MAX_RESAMPLE=$K32_MAX_RESAMPLE"
nvidia-smi -L || true

# Build the set of (mode, split, shard_idx, n_shards) keys with the LATEST
# dispatched entry per shard (de-dup the manifest).
mapfile -t CELLS < <(.venv/bin/python <<EOF
from collections import OrderedDict
seen = OrderedDict()
with open(".slurm/k32_full/dispatched.tsv") as f:
    for line in f:
        p = line.rstrip().split("\t")
        if len(p) < 6: continue
        _jid, model, mode, split, sh_idx, n_sh = p[:6]
        if model != "$MODEL": continue
        key = (mode, split, int(sh_idx), int(n_sh))
        seen[key] = True
for (mode, split, sh_idx, n_sh) in seen:
    print(f"{mode} {split} {sh_idx} {n_sh}")
EOF
)

echo "Cells to process: ${#CELLS[@]}"
for row in "${CELLS[@]}"; do
    read -r MODE SPLIT SHARD_IDX N_SHARDS <<< "$row"
    NAME="k32_${MODEL}_${MODE}_${SPLIT}_${SHARD_IDX}of${N_SHARDS}"
    # Skip if any RUNNING gen job for this exact shard
    if squeue -h -u nr3764 -o "%j|%T" 2>/dev/null | grep -F -x -- "${NAME}|RUNNING" > /dev/null; then
        echo ">>> SKIP $NAME (gen still RUNNING)"
        continue
    fi
    echo ">>> RUN  $NAME"
    set +e
    python -u experiments/nested_branching_variance/k32_full/generate.py \
        --model "$MODEL" \
        --mode "$MODE" \
        --split "$SPLIT" \
        --shard_idx "$SHARD_IDX" \
        --n_shards "$N_SHARDS" \
        --K 32
    RC=$?
    set -e
    echo "<<< DONE $NAME rc=$RC"
done

echo "=== ALL CELLS PROCESSED ==="
