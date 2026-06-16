#!/usr/bin/env bash
# Submit the full ASR/ORR 16K eval SHARDED across many 1-GPU jobs for max speed.
# Each shard = one row-slice of a split, its own H100, minimal CPU/RAM. Slow models
# (Phi-4) get more shards so wall-clock is balanced. Shards are merged afterward by
# merge_shards.py (run by the monitor once a cell's shards all complete).
#
# Per-job resources: --cpus-per-gpu=4, mem 56G (8B/14B) / 72G (GPT-OSS-20B, 20B target
# + 20B oss-safeguard guardrail). Walltime is small per shard.
set -euo pipefail
ROOT="${LRM_SAFETY_ARTIFACTS:-/scratch/gpfs/ARORA/nr3764/inference_skill_composition}"
LOG_DIR="$ROOT/.slurm"; mkdir -p "$LOG_DIR"
MANIFEST="$ROOT/scripts/slurm/asr_orr_16k/dispatched_sharded.tsv"
: > "$MANIFEST"

# Split totals (must match build_split_records).
declare -A TOTAL=( [asr_full]=4273 [orr_full]=2885 )

# Per (model, split): number of shards. Slow models -> more shards.
# model|split|nshards|mem|time
CELLS=(
  "Qwen3-8B|asr_full|4|56G|01:30:00"
  "Qwen3-8B|orr_full|2|56G|01:30:00"
  "Olmo-3-7B-Think|asr_full|4|56G|01:30:00"
  "Olmo-3-7B-Think|orr_full|2|56G|01:30:00"
  "Phi-4-reasoning|asr_full|8|56G|02:00:00"
  "Phi-4-reasoning|orr_full|4|56G|02:00:00"
  "GPT-OSS-20B|asr_full|6|72G|02:00:00"
  "GPT-OSS-20B|orr_full|3|72G|02:00:00"
)

njobs=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r MODEL SPLIT NSHARDS MEM TIME <<< "$cell"
  T=${TOTAL[$SPLIT]}
  for ((i=0;i<NSHARDS;i++)); do
    START=$(( i * T / NSHARDS ))
    END=$(( (i+1) * T / NSHARDS ))
    NAME="asr16k_${SPLIT}_${MODEL}_s${START}_${END}"
    jid=$(sbatch --parsable \
        --job-name="$NAME" \
        --output="$LOG_DIR/${NAME}_%j.out" \
        --partition=pli-c --account=arora --qos=pli-c \
        --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-gpu=4 --mem="$MEM" \
        --time="$TIME" \
        --wrap="cd $ROOT && ROW_SLICE=${START}:${END} bash scripts/slurm/asr_orr_16k/run_one.sh $MODEL $SPLIT")
    printf "%s\t%s\t%s\t%s\t%d\t%d\n" "$jid" "$NAME" "$MODEL" "$SPLIT" "$START" "$END" >> "$MANIFEST"
    echo "  $NAME : $jid"
    njobs=$((njobs+1))
  done
done

echo
echo "$njobs shard jobs dispatched. Manifest: $MANIFEST"
echo "Monitor: squeue --me -o '%.10i %.42j %.8T %.8M' | grep asr16k"
