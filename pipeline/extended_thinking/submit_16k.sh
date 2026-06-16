#!/usr/bin/env bash
# Dispatcher for the 16K-think + force-close rerun (2026-05-16 universal protocol).
# Targets only THINK cells (nothink mode is unaffected by force-close).
#
# Per-cell wall budget calibrated to expected work:
#   - Qwen3-8B/think (max_new previously 14K, all data complete): mostly KEEP +
#       a handful of REPROCESS (the prior length-truncations). Fast, 1h wall.
#   - GPT-OSS-20B/think (max_new previously 24K, channel-based, all data
#       complete): many rollouts may have used >16K thinking → many REPROCESS.
#       3h wall.
#   - Olmo/think (max_new previously 14K, partial data, full-gen rest): mostly
#       UNFILLED + ~50 REPROCESS. ~6-8h wall.
#   - Phi-4/think (max_new previously 28K, partial data, full-gen rest): mostly
#       UNFILLED + many REPROCESS (>16K thinking traces). Phase 1 cap of 16K
#       reduces work vs the previous 28K. ~6-8h wall.
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
mkdir -p .slurm/k32_full
DISPATCH_LOG=".slurm/k32_full/dispatched_16k.tsv"
> "$DISPATCH_LOG"

# (model, mode, split, n_shards, mem_gb, time)
SHARD_PLAN=(
    "Qwen3-8B        think asr 2 32 1:00:00"
    "Qwen3-8B        think orr 2 32 1:00:00"
    "Olmo-3-7B-Think think asr 4 32 8:00:00"
    "Olmo-3-7B-Think think orr 4 32 8:00:00"
    "Phi-4-reasoning think asr 6 48 8:00:00"
    "Phi-4-reasoning think orr 4 48 8:00:00"
    "GPT-OSS-20B     think asr 1 64 3:00:00"
    "GPT-OSS-20B     think orr 2 64 3:00:00"
)

JOB_SCRIPT="$ROOT/experiments/nested_branching_variance/k32_full/slurm_job_16k.sh"

submit_one() {
    local model="$1" mode="$2" split="$3" shard_idx="$4" n_shards="$5" mem_gb="$6" wall="$7"
    local name="k32-16k_${model}_${mode}_${split}_${shard_idx}of${n_shards}"
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
