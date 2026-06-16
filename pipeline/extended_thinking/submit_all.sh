#!/usr/bin/env bash
# Dispatcher: submits one sbatch per (model, mode, split, shard).
# Cost-tuned shard counts target ≤5h actual gen time inside a 5h30m wall.
# Outputs job-id table to .slurm/k32_full/dispatched.tsv (overwritten on rerun).
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
mkdir -p .slurm/k32_full
DISPATCH_LOG=".slurm/k32_full/dispatched.tsv"
> "$DISPATCH_LOG"

# (model, mode, split, n_shards, mem_gb)
SHARD_PLAN=(
    # GPT-OSS-20B (20B MoE) — 6 shards
    "GPT-OSS-20B think   asr 1 64"
    "GPT-OSS-20B think   orr 2 64"
    "GPT-OSS-20B nothink asr 2 64"
    "GPT-OSS-20B nothink orr 1 64"
    # Qwen3-8B — 6 shards
    "Qwen3-8B think   asr 2 32"
    "Qwen3-8B think   orr 2 32"
    "Qwen3-8B nothink asr 1 32"
    "Qwen3-8B nothink orr 1 32"
    # Olmo-3-7B-Think — 6 shards
    "Olmo-3-7B-Think think   asr 2 32"
    "Olmo-3-7B-Think think   orr 2 32"
    "Olmo-3-7B-Think nothink asr 1 32"
    "Olmo-3-7B-Think nothink orr 1 32"
    # Phi-4-reasoning (14B, longest outputs) — 16 shards
    "Phi-4-reasoning think   asr 4 48"
    "Phi-4-reasoning think   orr 4 48"
    "Phi-4-reasoning nothink asr 4 48"
    "Phi-4-reasoning nothink orr 4 48"
)

JOB_SCRIPT="$ROOT/experiments/nested_branching_variance/k32_full/slurm_job.sh"

submit_one() {
    local model="$1" mode="$2" split="$3" shard_idx="$4" n_shards="$5" mem_gb="$6"
    local name="k32_${model}_${mode}_${split}_${shard_idx}of${n_shards}"
    local out=".slurm/k32_full/${name}_%j.out"
    local err=".slurm/k32_full/${name}_%j.err"
    local jid
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$out" --error="$err" \
        --mem="${mem_gb}G" \
        --export=ALL,MODEL="$model",MODE="$mode",SPLIT="$split",SHARD_IDX="$shard_idx",N_SHARDS="$n_shards",K="${K:-32}" \
        "$JOB_SCRIPT")
    printf "%s\t%s\t%s\t%s\t%d\t%d\t%dG\n" "$jid" "$model" "$mode" "$split" "$shard_idx" "$n_shards" "$mem_gb" | tee -a "$DISPATCH_LOG"
}

n_jobs=0
for line in "${SHARD_PLAN[@]}"; do
    read -r model mode split n_shards mem_gb <<< "$line"
    for ((i=0; i<n_shards; i++)); do
        submit_one "$model" "$mode" "$split" "$i" "$n_shards" "$mem_gb"
        n_jobs=$((n_jobs + 1))
    done
done

echo "Dispatched $n_jobs jobs. Manifest: $DISPATCH_LOG"
