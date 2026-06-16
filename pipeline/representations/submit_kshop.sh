#!/usr/bin/env bash
# Submit ONE model's K-hop user-content-span hidden-state extraction job to pli-c.
# Usage: bash submit_kshop.sh <MODEL> [walltime_hours]
#
# Provenance: copied (logic unchanged) from
# scripts/slurm/neurips_final/submit_kshop.sh. The bundled driver below
# (extract_kshop_representations.py in this dir) imports the model registry +
# guardrail vote via common.py; the canonical versions live in lrm_safety.
# Set LRM_SAFETY_ARTIFACTS to the artifact tree (defaults to the repo root).
set -euo pipefail

MODEL="$1"
WT_HOURS="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # …/camera_ready/pipeline/representations
ROOT="${LRM_SAFETY_ARTIFACTS:-$(cd "$HERE/../../.." && pwd)}"  # artifact tree (repo root by default)
LOG_DIR="$ROOT/.slurm"
mkdir -p "$LOG_DIR"

HH=$(printf "%02d" "$WT_HOURS")
WALLTIME="${HH}:00:00"

case "$MODEL" in
    GPT-OSS-20B) MEM="80G" ;;          # 20B MoE + all-layer hidden states
    Phi-4-reasoning) MEM="64G" ;;       # 14B with long system prompt baked into every item
    *) MEM="48G" ;;                     # 8B class (Qwen3-8B, Olmo-3-7B-Think)
esac

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=khop_${MODEL}
#SBATCH --partition=pli-c
#SBATCH --qos=pli-c
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${LOG_DIR}/khop_${MODEL}_%j.out
#SBATCH --error=${LOG_DIR}/khop_${MODEL}_%j.err

set -euo pipefail
cd "$ROOT"

# Avoid torch inductor cache corruption across concurrent jobs
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_\${SLURM_JOB_ID}"
mkdir -p "\$TORCHINDUCTOR_CACHE_DIR"

echo "node=\$(hostname)  job=\$SLURM_JOB_ID  model=${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export LRM_SAFETY_ARTIFACTS="${ROOT}"
"${ROOT}/.venv/bin/python" -u "${HERE}/extract_kshop_representations.py" \\
    --model "${MODEL}" \\
    --suites safety supplementary

rm -rf "\$TORCHINDUCTOR_CACHE_DIR" || true
echo "DONE"
EOF
