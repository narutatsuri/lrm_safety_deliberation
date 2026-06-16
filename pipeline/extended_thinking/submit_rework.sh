#!/usr/bin/env bash
# Resubmit dispatcher for the 2026-05-16 K=32 rework cycle.
#
# Per-cell time + n_shards calibrated to the current shard state on disk:
# - Fast cells (only re-sample a small number of truncs): low wall (1-4h)
# - Mixed cells (some shards complete, others empty): medium wall (6-7h)
# - Pure full-gen think cells (Olmo think, Phi-4 think ASR): reshard for parallelism + 10:30 wall
#
# generate.py v2 handles all states uniformly: it loads existing JSONL, identifies
# malformed/missing positions, and only re-samples those. Atomic chunked writes
# survive SLURM timeout; resume just continues from the last chunk.
#
# Archives the previous dispatched.tsv before overwriting.
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
mkdir -p .slurm/k32_full
DISPATCH_LOG=".slurm/k32_full/dispatched.tsv"

# Archive previous manifest (timestamped)
if [ -s "$DISPATCH_LOG" ]; then
    archive="$DISPATCH_LOG.$(date +%Y-%m-%d_%H%M).bak"
    cp -p "$DISPATCH_LOG" "$archive"
    echo "Archived previous dispatched.tsv to $archive"
fi
> "$DISPATCH_LOG"

# (model, mode, split, n_shards, mem_gb, time)
# n_shards rationale:
#   - Resharded 2->4 for Olmo think (no data, parallelize)
#   - Resharded 4->6 for Phi-4 think ASR (no data, max parallelism on pli-c)
#   - All others: keep current layout (existing data is shard-aligned)
SHARD_PLAN=(
    # GPT-OSS-20B — tiny resamples (4, 0, 2, 12 truncs)
    "GPT-OSS-20B think   asr 1 64 2:00:00"
    "GPT-OSS-20B think   orr 2 64 0:30:00"
    "GPT-OSS-20B nothink asr 2 64 2:00:00"
    "GPT-OSS-20B nothink orr 1 64 2:00:00"
    # Qwen3-8B — medium resamples (945, 227, 1256, 127 truncs)
    "Qwen3-8B think   asr 2 32 4:00:00"
    "Qwen3-8B think   orr 2 32 3:00:00"
    "Qwen3-8B nothink asr 1 32 4:00:00"
    "Qwen3-8B nothink orr 1 32 2:00:00"
    # Olmo-3-7B-Think — think cells reshard for full-gen
    "Olmo-3-7B-Think think   asr 4 32 10:30:00"
    "Olmo-3-7B-Think think   orr 4 32 10:30:00"
    "Olmo-3-7B-Think nothink asr 1 32 2:00:00"
    "Olmo-3-7B-Think nothink orr 1 32 1:30:00"
    # Phi-4-reasoning — think ASR reshard 4->6 for full-gen
    "Phi-4-reasoning think   asr 6 48 11:30:00"
    "Phi-4-reasoning think   orr 4 48 11:30:00"
    "Phi-4-reasoning nothink asr 4 48 7:00:00"
    "Phi-4-reasoning nothink orr 4 48 3:00:00"
)

JOB_SCRIPT="$ROOT/experiments/nested_branching_variance/k32_full/slurm_job.sh"

submit_one() {
    local model="$1" mode="$2" split="$3" shard_idx="$4" n_shards="$5" mem_gb="$6" wall="$7"
    local name="k32_${model}_${mode}_${split}_${shard_idx}of${n_shards}"
    local out=".slurm/k32_full/${name}_%j.out"
    local err=".slurm/k32_full/${name}_%j.err"
    local jid
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$out" --error="$err" \
        --mem="${mem_gb}G" \
        --time="$wall" \
        --export=ALL,MODEL="$model",MODE="$mode",SPLIT="$split",SHARD_IDX="$shard_idx",N_SHARDS="$n_shards",K="${K:-32}" \
        "$JOB_SCRIPT")
    printf "%s\t%s\t%s\t%s\t%d\t%d\t%dG\t%s\n" "$jid" "$model" "$mode" "$split" "$shard_idx" "$n_shards" "$mem_gb" "$wall" | tee -a "$DISPATCH_LOG"
}

n_jobs=0
for line in "${SHARD_PLAN[@]}"; do
    read -r model mode split n_shards mem_gb wall <<< "$line"
    for ((i=0; i<n_shards; i++)); do
        submit_one "$model" "$mode" "$split" "$i" "$n_shards" "$mem_gb" "$wall"
        n_jobs=$((n_jobs + 1))
    done
done

echo "Dispatched $n_jobs jobs. Manifest: $DISPATCH_LOG"
