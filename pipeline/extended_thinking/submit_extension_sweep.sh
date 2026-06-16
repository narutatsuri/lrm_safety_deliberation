#!/usr/bin/env bash
# Submit r ∈ {0.4, 0.6, 0.8, 1.0} extension sweep (cuts B = 140/160/180/200%)
# for all 4 models × 2 splits.
#
# Layout:
#   one job per (MODEL, SPLIT, CUT). N_SHARDS=1 (no sharding).
#   --time=24:00:00, --cpus-per-task=2, --mem scaled to model size,
#   --gres=gpu:1, pli-c.
#   Total jobs = 4 models × 2 splits × 4 cuts = 32.
#
# Usage:
#   bash experiments/nested_branching_variance/k32_full/submit_extension_sweep.sh
#   DRY=1 ... → print but don't sbatch
#   CUTS=200 ... → restrict to one cut (default: 140 160 180 200)
#   MODELS_FILTER="Qwen3-8B" ... → restrict to one model
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"

JOB_SCRIPT="experiments/nested_branching_variance/k32_full/slurm_job_nested.sh"
MANIFEST_DIR=".slurm/k32_nb"
mkdir -p "$MANIFEST_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST="$MANIFEST_DIR/dispatched_extension_sweep_${TS}.tsv"
echo -e "job_id\tmodel\tsplit\tcut\tn_shards\tmem\ttime" > "$MANIFEST"

CUTS_TO_RUN="${CUTS:-140 160 180 200}"
MODELS_FILTER="${MODELS_FILTER:-Qwen3-8B Olmo-3-7B-Think Phi-4-reasoning GPT-OSS-20B}"
DRY="${DRY:-0}"
TIMELIMIT="${TIMELIMIT:-24:00:00}"

submit_one() {
    local m="$1" sp="$2" cut="$3" mem="$4"
    local jobname="k32nb_ext_${m}_${sp}_c${cut}"
    local outfile=".slurm/${jobname}-%j.out"

    if [ "$DRY" = "1" ]; then
        echo "[dry] $m $sp cut=$cut mem=$mem time=$TIMELIMIT"
        return
    fi

    local jid
    jid=$(sbatch --parsable \
        --partition=pli-c --qos=pli-c --account=arora \
        --gres=gpu:1 --cpus-per-task=2 --mem="$mem" --time="$TIMELIMIT" \
        --exclude=della-k12g1 \
        --job-name="$jobname" --output="$outfile" \
        --export=ALL,MODEL="$m",SPLIT="$sp",SHARD_IDX=0,N_SHARDS=1,M=8,N=8,CUTS_COLON="$cut" \
        "$JOB_SCRIPT")
    echo -e "${jid}\t${m}\t${sp}\t${cut}\t1\t${mem}\t${TIMELIMIT}" >> "$MANIFEST"
    echo "[submit] $jid  $m $sp cut=$cut mem=$mem time=$TIMELIMIT"
}

n_total=0
for m in $MODELS_FILTER; do
    case "$m" in
        GPT-OSS-20B)              mem="48G" ;;
        Qwen3-8B|Phi-4-reasoning|Olmo-3-7B-Think) mem="24G" ;;
        *) echo "unknown model $m" >&2; exit 1 ;;
    esac
    for sp in asr orr; do
        for cut in $CUTS_TO_RUN; do
            submit_one "$m" "$sp" "$cut" "$mem"
            n_total=$((n_total+1))
        done
    done
done

echo
echo "Submitted $n_total jobs."
echo "Manifest: $MANIFEST"
