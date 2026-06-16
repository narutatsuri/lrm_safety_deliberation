#!/usr/bin/env bash
# K=4 rollouts (vLLM n=4) sharded fleet for generation-stochasticity error bars.
# Each shard generates 4 samples/prompt in one model-load, classifies all 4 with the
# 4 guardrails, and emits per-rollout mean/std. Slow models get more shards (4x work).
# Output: eval_results/asr_orr_16k_K4/<split>/<MODEL>__shard_s_e__/ ; merged per cell.
set -euo pipefail
ROOT="${LRM_SAFETY_ARTIFACTS:-/scratch/gpfs/ARORA/nr3764/inference_skill_composition}"
LOG_DIR="$ROOT/.slurm"; mkdir -p "$LOG_DIR"
MANIFEST="$ROOT/scripts/slurm/asr_orr_16k/dispatched_k4.tsv"
: > "$MANIFEST"
K=4
TAG=asr_orr_16k_K4

declare -A TOTAL=( [asr_full]=4573 [orr_full]=3135 )

# model|split|nshards|mem|time   (more shards for slow models; 4x gen vs K=1)
CELLS=(
  "Qwen3-8B|asr_full|6|64G|02:30:00"
  "Qwen3-8B|orr_full|4|64G|02:30:00"
  "Olmo-3-7B-Think|asr_full|12|64G|03:00:00"
  "Olmo-3-7B-Think|orr_full|6|64G|03:00:00"
  "Phi-4-reasoning|asr_full|12|64G|03:00:00"
  "Phi-4-reasoning|orr_full|6|64G|03:00:00"
  "GPT-OSS-20B|asr_full|8|80G|02:30:00"
  "GPT-OSS-20B|orr_full|4|80G|02:30:00"
)

njobs=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r MODEL SPLIT NSHARDS MEM TIME <<< "$cell"
  T=${TOTAL[$SPLIT]}
  for ((i=0;i<NSHARDS;i++)); do
    START=$(( i * T / NSHARDS )); END=$(( (i+1) * T / NSHARDS ))
    NAME="k4_${SPLIT}_${MODEL}_s${START}_${END}"
    jid=$(sbatch --parsable \
        --job-name="$NAME" --output="$LOG_DIR/${NAME}_%j.out" \
        --partition=pli-c --account=arora --qos=pli-c \
        --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-gpu=4 --mem="$MEM" --time="$TIME" \
        --wrap="cd $ROOT && ROW_SLICE=${START}:${END} NUM_ROLLOUTS=$K SEED=0 OUT_TAG=$TAG bash scripts/slurm/asr_orr_16k/run_one.sh $MODEL $SPLIT")
    printf "%s\t%s\t%s\t%s\t%d\t%d\n" "$jid" "$NAME" "$MODEL" "$SPLIT" "$START" "$END" >> "$MANIFEST"
    echo "  $NAME : $jid"; njobs=$((njobs+1))
  done
done
echo; echo "$njobs K=4 shard jobs dispatched. Manifest: $MANIFEST"
