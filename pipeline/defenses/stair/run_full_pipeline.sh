#!/bin/bash
# ============================================================================
# STAIR full pipeline
#   Stage 1: SFT
#   Stage 2: per-round MCTS -> DPO-pair extraction -> DPO
#
# The code path is identical for smoke and full runs; smoke only changes the
# number of SFT samples / prompt samples via SFT_MAX_SAMPLES and
# PROMPT_MAX_SAMPLES.
#
# PROVENANCE: original path defenses/stair/train/run_full_pipeline.sh (STAIR
# SFT -> MCTS -> DPO orchestrator). The base-model registry now lives in
# lrm_safety.models; eval is pipeline/eval/.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PATH="${PROJECT_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${BASE_MODEL:?BASE_MODEL must be set (Qwen|Olmo|Phi4|GPTOSS)}"
: "${MODEL_NAME:?MODEL_NAME must be set}"

STAIR_DATA="${PROJECT_DIR}/defenses/stair_repo/data/stair_sft.json"
PROMPT_PATH="${PROMPT_PATH:-${PROJECT_DIR}/defenses/stair/data/stair_prompts.json}"
NUM_DPO_ROUNDS="${NUM_DPO_ROUNDS:-3}"
OUTPUT_BASE="${PROJECT_DIR}/models/STAIR-${MODEL_NAME}"
MCTS_BASE="${PROJECT_DIR}/defenses/stair/mcts_data/${MODEL_NAME}"
DPO_DATA_BASE="${PROJECT_DIR}/defenses/stair/dpo_data/${MODEL_NAME}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-0}"
PROMPT_MAX_SAMPLES="${PROMPT_MAX_SAMPLES:-0}"
SMOKE_SEED="${SMOKE_SEED:-42}"

mkdir -p "${OUTPUT_BASE}/artifacts" "${MCTS_BASE}" "${DPO_DATA_BASE}"

latest_resume_checkpoint() {
    local root="$1"
    local best=""
    local best_step=-1
    while IFS= read -r path; do
        if [ ! -f "${path}/training_state.pt" ] && [ ! -f "${path}/trainer_state.json" ]; then
            continue
        fi
        local name
        local step
        name="$(basename "${path}")"
        step="$(echo "${name}" | sed -n 's/.*-\([0-9][0-9]*\)$/\1/p')"
        [ -n "${step}" ] || continue
        if [ "${step}" -gt "${best_step}" ]; then
            best_step="${step}"
            best="${path}"
        fi
    done < <(find "${root}" -type d -name 'checkpoint-*' 2>/dev/null | sort)
    echo "${best}"
}

find_merged_dir() {
    # Require the merged-final dir to actually contain weight files. Without this
    # check, a TIMEOUT during DPO leaves an empty TrainingArguments.output_dir
    # (e.g. dpo_round2/merged-final/) that fools the pipeline into thinking the
    # round is complete, then vLLM later fails to load the empty dir as a model.
    local root="$1"
    if [ ! -d "${root}" ]; then
        return 0
    fi
    local best=""
    while IFS= read -r path; do
        if compgen -G "${path}/*.safetensors" >/dev/null 2>&1; then
            best="${path}"
        fi
    done < <(find "${root}" -type d -name 'merged-final' 2>/dev/null | sort)
    echo "${best}"
}

materialize_subset() {
    local src="$1"
    local dst="$2"
    local max_samples="$3"
    local seed="$4"
    "${PYTHON}" - <<PY
from common import materialize_json_subset
materialize_json_subset(${src@Q}, ${dst@Q}, int(${max_samples@Q}), seed=int(${seed@Q}))
PY
}

ensure_prompt_file() {
    if [ -f "${PROMPT_PATH}" ]; then
        return 0
    fi

    mkdir -p "$(dirname "${PROMPT_PATH}")"
    "${PYTHON}" - <<PY 2>/dev/null || true
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id="thu-ml/STAIR-Prompts",
    filename="stair_prompts.json",
    repo_type="dataset",
)
shutil.copy(path, ${PROMPT_PATH@Q})
print("downloaded", ${PROMPT_PATH@Q})
PY

    if [ -f "${PROMPT_PATH}" ]; then
        return 0
    fi

    "${PYTHON}" - <<PY
import json, random
random.seed(${SMOKE_SEED@Q})
with open(${STAIR_DATA@Q}) as f:
    sft_data = json.load(f)
prompts = []
for record in sft_data:
    q_type = "safety" if record["source"] == "PKU-SafeRLHF" else "helpfulness"
    prompts.append({"question": record["prompt"], "type": q_type})
random.shuffle(prompts)
prompts = prompts[:2000]
with open(${PROMPT_PATH@Q}, "w") as f:
    json.dump(prompts, f, indent=2)
print("created", len(prompts), "fallback prompts at", ${PROMPT_PATH@Q})
PY
}

resolve_sft_data() {
    local data_path="${STAIR_DATA}"
    if [ "${SFT_MAX_SAMPLES}" -gt 0 ]; then
        data_path="${OUTPUT_BASE}/artifacts/stair_sft_${SFT_MAX_SAMPLES}.json"
        materialize_subset "${STAIR_DATA}" "${data_path}" "${SFT_MAX_SAMPLES}" "${SMOKE_SEED}"
    fi
    echo "${data_path}"
}

resolve_prompt_data() {
    ensure_prompt_file
    local prompt_data="${PROMPT_PATH}"
    if [ "${PROMPT_MAX_SAMPLES}" -gt 0 ]; then
        prompt_data="${OUTPUT_BASE}/artifacts/stair_prompts_${PROMPT_MAX_SAMPLES}.json"
        materialize_subset "${PROMPT_PATH}" "${prompt_data}" "${PROMPT_MAX_SAMPLES}" "${SMOKE_SEED}"
    fi
    echo "${prompt_data}"
}

SFT_DATA_PATH="$(resolve_sft_data)"
PROMPT_DATA_PATH="$(resolve_prompt_data)"

echo "============================================"
echo "[$(date)] Stage 1: SFT on STAIR-SFT data"
echo "============================================"

SFT_OUTPUT="${OUTPUT_BASE}/sft"
SFT_RUN_ROOT="${SFT_OUTPUT}/STAIR-SFT/${MODEL_NAME}"
SFT_MERGED="${SFT_RUN_ROOT}/merged-final"
SFT_RESUME="$(latest_resume_checkpoint "${SFT_RUN_ROOT}")"

if [ -d "${SFT_MERGED}" ]; then
    echo "[$(date)] Reusing completed SFT model: ${SFT_MERGED}"
else
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    MODEL_PATH="${MODEL_PATH}" \
    BASE_MODEL="${BASE_MODEL}" \
    OUTPUT_DIR="${SFT_OUTPUT}" \
    DATA_PATH="${SFT_DATA_PATH}" \
    MERGE_ADAPTER=1 \
    MAX_CKPTS="${MAX_CKPTS:-5}" \
    SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}" \
    MAX_SAMPLES=0 \
    RESUME_CHECKPOINT="${SFT_RESUME}" \
        bash "${SCRIPT_DIR}/run_sft_lora.sh"
fi

if [ ! -d "${SFT_MERGED}" ]; then
    SFT_MERGED="$(find_merged_dir "${SFT_RUN_ROOT}")"
fi
if [ ! -d "${SFT_MERGED}" ]; then
    echo "ERROR: STAIR SFT did not produce a merged model under ${SFT_RUN_ROOT}" >&2
    exit 1
fi
echo "[$(date)] SFT complete. Merged model at: ${SFT_MERGED}"

CURRENT_MODEL="${SFT_MERGED}"

for ROUND in $(seq 1 "${NUM_DPO_ROUNDS}"); do
    echo "============================================"
    echo "[$(date)] Stage 2 Round ${ROUND}/${NUM_DPO_ROUNDS}"
    echo "============================================"

    ROUND_MCTS_DIR="${MCTS_BASE}/round${ROUND}"
    ROUND_DPO_PATH="${DPO_DATA_BASE}/round${ROUND}.jsonl"
    ROUND_DPO_OUTPUT="${OUTPUT_BASE}/dpo_round${ROUND}"
    ROUND_MERGED="$(find_merged_dir "${ROUND_DPO_OUTPUT}")"

    mkdir -p "${ROUND_MCTS_DIR}" "${ROUND_DPO_OUTPUT}" "${DPO_DATA_BASE}"

    if [ -n "${ROUND_MERGED}" ]; then
        echo "[$(date)] Round ${ROUND}: reusing completed DPO model ${ROUND_MERGED}"
        CURRENT_MODEL="${ROUND_MERGED}"
        continue
    fi

    echo "[$(date)] Round ${ROUND}: MCTS generation"
    ACTOR_MODEL_DIR="${CURRENT_MODEL}" \
    BASE_MODEL="${BASE_MODEL}" \
    PROMPT_PATH="${PROMPT_DATA_PATH}" \
    OUTPUT_PATH="${ROUND_MCTS_DIR}" \
    MCTS_WORKERS="${MCTS_WORKERS:-8}" \
    MCTS_ITERATIONS="${MCTS_ITERATIONS:-200}" \
    MCTS_MAX_DEPTH="${MCTS_MAX_DEPTH:-7}" \
    MCTS_GENERATE_SAMPLES="${MCTS_GENERATE_SAMPLES:-4}" \
    MCTS_MAX_TOKENS="${MCTS_MAX_TOKENS:-2048}" \
        bash "${SCRIPT_DIR}/run_generate_mcts.sh"

    if [ ! -s "${ROUND_MCTS_DIR}/mcts_trees.json" ]; then
        echo "ERROR: Round ${ROUND} MCTS did not produce ${ROUND_MCTS_DIR}/mcts_trees.json" >&2
        exit 1
    fi

    if [ ! -s "${ROUND_DPO_PATH}" ]; then
        echo "[$(date)] Round ${ROUND}: extracting DPO pairs"
        MCTS_PATH="${ROUND_MCTS_DIR}/mcts_trees.json" \
        TOKENIZER_PATH="${CURRENT_MODEL}" \
        OUTPUT_PATH="${ROUND_DPO_PATH}" \
            bash "${SCRIPT_DIR}/run_extract_dpo_pairs.sh"
    else
        echo "[$(date)] Round ${ROUND}: reusing existing DPO pairs ${ROUND_DPO_PATH}"
    fi

    if [ ! -s "${ROUND_DPO_PATH}" ]; then
        echo "ERROR: Round ${ROUND} did not produce any DPO pairs at ${ROUND_DPO_PATH}" >&2
        exit 1
    fi

    echo "[$(date)] Round ${ROUND}: DPO training"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    # DPO is always single-GPU: dpo_lora.py's adapter save+reload+merge dance
    # is not multi-rank-safe. SFT and MCTS still use the chain's full GPU count
    # via NUM_PROCESSES / VLLM_TENSOR_PARALLEL_SIZE.
    MODEL_PATH="${CURRENT_MODEL}" \
    BASE_MODEL="${BASE_MODEL}" \
    DATA_PATH="${ROUND_DPO_PATH}" \
    OUTPUT_DIR="${ROUND_DPO_OUTPUT}" \
    ROUND_NUM="${ROUND}" \
    MERGE_ADAPTER=1 \
    NUM_PROCESSES=1 \
        bash "${SCRIPT_DIR}/run_dpo_lora.sh"

    CURRENT_MODEL="$(find_merged_dir "${ROUND_DPO_OUTPUT}")"
    if [ -z "${CURRENT_MODEL}" ] || [ ! -d "${CURRENT_MODEL}" ]; then
        echo "ERROR: Round ${ROUND} did not produce a merged model under ${ROUND_DPO_OUTPUT}" >&2
        exit 1
    fi
    echo "[$(date)] Round ${ROUND}: DPO complete. Model at: ${CURRENT_MODEL}"
done

FINAL_MODEL_DIR="${PROJECT_DIR}/models/STAIR-${MODEL_NAME}"
if [ -d "${CURRENT_MODEL}" ] && [ "${CURRENT_MODEL}" != "${FINAL_MODEL_DIR}" ]; then
    echo "[$(date)] Symlinking final model..."
    ln -sfn "${CURRENT_MODEL}" "${FINAL_MODEL_DIR}/final"
fi

echo "============================================"
echo "[$(date)] STAIR pipeline complete for ${MODEL_NAME}!"
echo "  SFT model:   ${SFT_MERGED}"
echo "  Final model: ${CURRENT_MODEL}"
echo "============================================"
