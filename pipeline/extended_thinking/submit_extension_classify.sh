#!/usr/bin/env bash
# Submit classification jobs for the r ∈ {0.4..1.0} extension sweep
# (cuts B = 140/160/180/200). One job per (MODEL, SPLIT, GUARDRAIL); each
# iterates all 4 cuts internally with a single vLLM load.
#
# Dependencies: any of the 4 gen jobs for (MODEL, SPLIT) that are still
# RUNNING/PENDING become an `afterok` chain. COMPLETED gen jobs are dropped
# from the dep list. If all 4 gen jobs are complete, cls submits immediately.
#
# Total cls jobs = 4 models × 2 splits × 4 guardrails = 32.
set -euo pipefail

ROOT="/scratch/gpfs/ARORA/nr3764/inference_skill_composition"
cd "$ROOT"

GEN_MANIFEST=".slurm/k32_nb/dispatched_extension_sweep_20260521_123857.tsv"
JOB_SCRIPT="experiments/nested_branching_variance/k32_full/classify_job_nested.sh"
MANIFEST_DIR=".slurm/k32_nb"
mkdir -p "$MANIFEST_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST="$MANIFEST_DIR/dispatched_extension_classify_${TS}.tsv"
echo -e "job_id\tmodel\tsplit\tguardrail\tcuts\tmem\ttime\tdep_jids" > "$MANIFEST"

GUARDRAILS="${GUARDRAILS:-wildguard qwen3guard granite_guardian oss_safeguard}"
MODELS_FILTER="${MODELS_FILTER:-Qwen3-8B Olmo-3-7B-Think Phi-4-reasoning GPT-OSS-20B}"
CUTS_COLON_VAL="${CUTS_COLON_VAL:-140:160:180:200}"
TIMELIMIT="${TIMELIMIT:-12:00:00}"
DRY="${DRY:-0}"

# Build map: "model|split" -> "jid1 jid2 jid3 jid4" (4 gen jobs for those 4 cuts).
declare -A GEN_JIDS
while IFS=$'\t' read -r jid m sp cut nsh mem tl; do
    [ "$jid" = "job_id" ] && continue
    key="${m}|${sp}"
    GEN_JIDS["$key"]="${GEN_JIDS[$key]:-} $jid"
done < "$GEN_MANIFEST"

# Helper: of a list of jids, return space-separated list of those NOT in
# COMPLETED state (i.e., still RUNNING or PENDING).
filter_incomplete() {
    local jids="$1"
    local out=""
    for j in $jids; do
        local state
        state=$(sacct -j "$j" --format=State -X -P -n 2>/dev/null | head -1 | tr -d ' ')
        case "$state" in
            COMPLETED) ;;             # drop — already done
            *) out="$out $j" ;;       # keep RUNNING/PENDING/CANCELLED/FAILED/etc.
        esac
    done
    echo "$out" | sed 's/^ *//;s/ *$//'
}

guardrail_mem() {
    case "$1" in
        oss_safeguard) echo "48G" ;;
        *)             echo "24G" ;;
    esac
}

submit_one() {
    local m="$1" sp="$2" gr="$3"
    local key="${m}|${sp}"
    local gen_jids="${GEN_JIDS[$key]:-}"
    if [ -z "$gen_jids" ]; then
        echo "[skip] no gen jobs for $key" >&2
        return
    fi

    local dep_jids
    dep_jids=$(filter_incomplete "$gen_jids")
    local dep_arg=""
    if [ -n "$dep_jids" ]; then
        local colon_jids
        colon_jids=$(echo "$dep_jids" | tr ' ' ':')
        dep_arg="--dependency=afterok:${colon_jids}"
    fi

    local mem
    mem=$(guardrail_mem "$gr")
    local jobname="k32nb_cls_${m}_${sp}_${gr}"
    local outfile=".slurm/${jobname}-%j.out"

    if [ "$DRY" = "1" ]; then
        echo "[dry] $m $sp $gr  mem=$mem  dep=${dep_jids:-(none)}"
        return
    fi

    local jid
    jid=$(sbatch --parsable \
        --partition=pli-c --qos=pli-c --account=arora \
        --gres=gpu:1 --cpus-per-task=2 --mem="$mem" --time="$TIMELIMIT" \
        --exclude=della-k12g1 \
        $dep_arg \
        --job-name="$jobname" --output="$outfile" \
        --export=ALL,MODEL="$m",SPLIT="$sp",GUARDRAIL="$gr",CUTS_COLON="$CUTS_COLON_VAL" \
        "$JOB_SCRIPT")
    echo -e "${jid}\t${m}\t${sp}\t${gr}\t${CUTS_COLON_VAL}\t${mem}\t${TIMELIMIT}\t${dep_jids:-none}" >> "$MANIFEST"
    echo "[submit] $jid  $m $sp $gr  mem=$mem  dep=${dep_jids:-(none)}"
}

n_total=0
for m in $MODELS_FILTER; do
    for sp in asr orr; do
        for gr in $GUARDRAILS; do
            submit_one "$m" "$sp" "$gr"
            n_total=$((n_total+1))
        done
    done
done

echo
echo "Submitted $n_total cls jobs."
echo "Manifest: $MANIFEST"
