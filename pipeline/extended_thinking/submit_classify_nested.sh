#!/usr/bin/env bash
# Classification dispatcher for the nested-branching outputs.
# Reads .slurm/k32_nb/dispatched_nested.tsv and submits 4 guardrail cls jobs per
# (model, split) cell with --dependency=afterok on the cell's gen jids.
# Each cls job iterates all 5 cuts internally (loads vLLM once per guardrail).
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"
mkdir -p .slurm/k32_nb
CLASS_LOG=".slurm/k32_nb/dispatched_classify_nested.tsv"
> "$CLASS_LOG"

JOB_SCRIPT="$ROOT/experiments/nested_branching_variance/k32_full/classify_job_nested.sh"
GUARDRAILS=(wildguard qwen3guard granite_guardian oss_safeguard)
MANIFEST=".slurm/k32_nb/dispatched_nested.tsv"
if [ ! -f "$MANIFEST" ]; then
    echo "Missing $MANIFEST. Run submit_nested_branching.sh first." >&2
    exit 1
fi

declare -A cell_deplist
while IFS=$'\t' read -r jid model split shard_idx n_shards mem wall; do
    key="${model}|${split}"
    if [ -z "${cell_deplist[$key]+x}" ]; then
        cell_deplist["$key"]="$jid"
    else
        cell_deplist["$key"]="${cell_deplist[$key]}:${jid}"
    fi
done < "$MANIFEST"

submit_classify() {
    local model="$1" split="$2" guardrail="$3" deps="$4"
    local mem="32G"; local wall="6:00:00"
    if [ "$guardrail" = "oss_safeguard" ]; then mem="64G"; wall="8:00:00"; fi
    local name="k32nbcls_${model}_${split}_${guardrail}"
    local out=".slurm/k32_nb/${name}_%j.out"
    local err=".slurm/k32_nb/${name}_%j.err"
    local jid
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$out" --error="$err" \
        --mem="$mem" \
        --time="$wall" \
        --exclude="della-k12g1" \
        --dependency="afterok:$deps" \
        --kill-on-invalid-dep=yes \
        --export=ALL,MODEL="$model",SPLIT="$split",GUARDRAIL="$guardrail" \
        "$JOB_SCRIPT")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "$model" "$split" "$guardrail" "$mem" "$wall" "$deps" | tee -a "$CLASS_LOG"
}

n_jobs=0
for key in "${!cell_deplist[@]}"; do
    IFS='|' read -r model split <<< "$key"
    deps="${cell_deplist[$key]}"
    for g in "${GUARDRAILS[@]}"; do
        submit_classify "$model" "$split" "$g" "$deps"
        n_jobs=$((n_jobs + 1))
    done
done

echo "Dispatched $n_jobs nested-branching cls jobs (afterok deps). Manifest: $CLASS_LOG"
