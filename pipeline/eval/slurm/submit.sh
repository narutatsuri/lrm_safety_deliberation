#!/usr/bin/env bash
# Submit the full ASR/ORR 16K-thinking eval for all 4 base models.
#   4 models x {asr_full, orr_full} = 8 GPU jobs on pli-c.
# Resource policy (project rules: minimize CPU/RAM/time):
#   --cpus-per-gpu=4, 1xH100. mem: 64G (8B/14B), 80G (GPT-OSS-20B; 20B target + 20B oss-safeguard).
#   time: per-(model,split) estimate, generous enough to finish in one allocation
#         (gen is NOT mid-split resumable; the monitor agent extends/reruns if needed).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LRM_SAFETY_ARTIFACTS:-/scratch/gpfs/ARORA/nr3764/inference_skill_composition}"
LOG_DIR="${LRM_SAFETY_LOGS:-$SCRIPT_DIR/logs}"; mkdir -p "$LOG_DIR"
MANIFEST="$LOG_DIR/dispatched.tsv"
: > "$MANIFEST"

submit() {  # name time mem model split
    local name=$1 time=$2 mem=$3 model=$4 split=$5
    local jid
    jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$LOG_DIR/${name}_%j.out" \
        --partition=pli-c --account=arora --qos=pli-c \
        --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-gpu=4 --mem="$mem" \
        --time="$time" \
        --wrap="bash $SCRIPT_DIR/run_one.sh $model $split")
    printf "%s\t%s\t%s\t%s\t%s\n" "$jid" "$name" "$model" "$split" "$time" >> "$MANIFEST"
    echo "  $name : $jid"
}

# model            mem   asr_time  orr_time
#                        (4,273)   (2,885)
submit "asr16k_asr_Qwen3-8B"          "04:00:00" "64G" Qwen3-8B          asr_full
submit "asr16k_orr_Qwen3-8B"          "02:30:00" "64G" Qwen3-8B          orr_full
submit "asr16k_asr_Olmo-3-7B-Think"   "04:00:00" "64G" Olmo-3-7B-Think   asr_full
submit "asr16k_orr_Olmo-3-7B-Think"   "02:30:00" "64G" Olmo-3-7B-Think   orr_full
submit "asr16k_asr_Phi-4-reasoning"   "06:00:00" "64G" Phi-4-reasoning   asr_full
submit "asr16k_orr_Phi-4-reasoning"   "04:00:00" "64G" Phi-4-reasoning   orr_full
submit "asr16k_asr_GPT-OSS-20B"       "05:00:00" "80G" GPT-OSS-20B       asr_full
submit "asr16k_orr_GPT-OSS-20B"       "03:00:00" "80G" GPT-OSS-20B       orr_full

echo
echo "8 jobs dispatched. Manifest: $MANIFEST"
echo "Monitor: squeue --me -o '%.10i %.34j %.8T %.10M %.10l' | grep asr16k"
