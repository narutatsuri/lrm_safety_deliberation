#!/usr/bin/env bash
# Dispatcher: for each (model, mode, split) cell, submit 4 classification jobs
# (one per guardrail), each --dependency=afterok on every generation job for
# that cell. Reads .slurm/k32_full/dispatched.tsv to build the dep map.
set -euo pipefail
ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
mkdir -p .slurm/k32_full

CLASS_LOG=".slurm/k32_full/dispatched_classify.tsv"
> "$CLASS_LOG"

JOB_SCRIPT="$ROOT/experiments/nested_branching_variance/k32_full/classify_job.sh"
GUARDRAILS=(wildguard qwen3guard granite_guardian oss_safeguard)

# Group dispatched gen jobs by (model, mode, split) — use *latest* job IDs only
# (when a shard was resubmitted, the later entry supersedes the earlier one).
declare -A cell_deps
declare -A cell_shards
while IFS=$'\t' read -r jid model mode split shard_idx n_shards mem; do
    key="${model}|${mode}|${split}"
    shard_key="${key}|${shard_idx}|${n_shards}"
    # If we've seen this (cell, shard_idx, n_shards) before, replace prior jid.
    cell_shards["$shard_key"]="$jid"
done < .slurm/k32_full/dispatched.tsv

# Now collect distinct cells and aggregate their latest job IDs
declare -A cell_deplist
for shard_key in "${!cell_shards[@]}"; do
    IFS='|' read -r model mode split shard_idx n_shards <<< "$shard_key"
    key="${model}|${mode}|${split}"
    if [ -z "${cell_deplist[$key]+x}" ]; then
        cell_deplist["$key"]="${cell_shards[$shard_key]}"
    else
        cell_deplist["$key"]="${cell_deplist[$key]}:${cell_shards[$shard_key]}"
    fi
done

submit_classify() {
    local model="$1" mode="$2" split="$3" guardrail="$4" deps="$5"
    local mem="48G"
    case "$guardrail" in
        oss_safeguard) mem="64G" ;;
        wildguard|qwen3guard|granite_guardian) mem="32G" ;;
    esac
    local name="k32cls_${model}_${mode}_${split}_${guardrail}"
    local out=".slurm/k32_full/${name}_%j.out"
    local err=".slurm/k32_full/${name}_%j.err"
    local jid
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$out" --error="$err" \
        --mem="$mem" \
        --dependency="afterok:$deps" \
        --kill-on-invalid-dep=yes \
        --export=ALL,MODEL="$model",MODE="$mode",SPLIT="$split",GUARDRAIL="$guardrail" \
        "$JOB_SCRIPT")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "$model" "$mode" "$split" "$guardrail" "$deps" "$mem" | tee -a "$CLASS_LOG"
}

n_jobs=0
for key in "${!cell_deplist[@]}"; do
    IFS='|' read -r model mode split <<< "$key"
    deps="${cell_deplist[$key]}"
    for g in "${GUARDRAILS[@]}"; do
        submit_classify "$model" "$mode" "$split" "$g" "$deps"
        n_jobs=$((n_jobs + 1))
    done
done

echo "Dispatched $n_jobs classification jobs. Manifest: $CLASS_LOG"
