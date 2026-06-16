#!/usr/bin/env bash
# Dispatcher for full-pool nested-branching variance + r=0.2 extension.
# Writes to eval_results/k32_nested_branching_full/ (NEVER touches k32_full_pool/).
# All jobs target pli-c with minimal resources (1 GPU, 4 CPU, modest RAM).
#
# Sharding per (model, split): chosen to keep per-shard wall < 12h.
# Estimated per-shard work: ~625-720 prompts × (8 prefixes × 41 short generations) ≈ 200K rollouts.
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
mkdir -p .slurm/k32_nb
DISPATCH_LOG=".slurm/k32_nb/dispatched_nested.tsv"
> "$DISPATCH_LOG"

# (model split n_shards mem_gb wall)
SHARD_PLAN=(
    "Qwen3-8B        asr 4 32 12:00:00"
    "Qwen3-8B        orr 4 32 12:00:00"
    "Olmo-3-7B-Think asr 4 32 12:00:00"
    "Olmo-3-7B-Think orr 4 32 12:00:00"
    "Phi-4-reasoning asr 6 48 12:00:00"
    "Phi-4-reasoning orr 6 48 12:00:00"
    "GPT-OSS-20B     asr 6 48 12:00:00"
    "GPT-OSS-20B     orr 6 48 12:00:00"
)

JOB_SCRIPT="$ROOT/experiments/nested_branching_variance/k32_full/slurm_job_nested.sh"

submit_one() {
    local model="$1" split="$2" shard_idx="$3" n_shards="$4" mem_gb="$5" wall="$6"
    local name="k32nb_${model}_${split}_${shard_idx}of${n_shards}"
    local out=".slurm/k32_nb/${name}_%j.out"
    local err=".slurm/k32_nb/${name}_%j.err"
    local jid
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$out" --error="$err" \
        --mem="${mem_gb}G" \
        --time="$wall" \
        --export=ALL,MODEL="$model",SPLIT="$split",SHARD_IDX="$shard_idx",N_SHARDS="$n_shards",M="${M:-8}",N="${N:-8}" \
        "$JOB_SCRIPT")
    printf "%s\t%s\t%s\t%d\t%d\t%dG\t%s\n" "$jid" "$model" "$split" "$shard_idx" "$n_shards" "$mem_gb" "$wall" | tee -a "$DISPATCH_LOG"
}

n_jobs=0
for line in "${SHARD_PLAN[@]}"; do
    read -r model split n_shards mem_gb wall <<< "$line"
    for ((i=0; i<n_shards; i++)); do
        submit_one "$model" "$split" "$i" "$n_shards" "$mem_gb" "$wall"
        n_jobs=$((n_jobs + 1))
    done
done

echo "Dispatched $n_jobs nested-branching gen jobs. Manifest: $DISPATCH_LOG"
