# Safety defenses

Source for the nine safety defenses benchmarked in the paper. Six are
**training** defenses (they produce a finetuned checkpoint, written to
`models/final/<METHOD>-<base>`) and three are decode-time **inference**
wrappers (no training; they wrap a vLLM-served base model at decode time).

Each method's directory holds **only that method's own code**: training /
datagen entry points, launch scripts, deepspeed/accelerate configs, and
small method-local helpers. Vendored upstream clones (`defenses/*_repo/`,
`defenses/safepath/SAFEPATH/`), checkpoints, datasets, and run logs are **not**
included here — see the "Derives from" column for the upstream repo each method
adapts, and the data-prep / `prepare_data.py` steps for how to materialize the
training data locally.

- Base-model registry (paths, decoding configs, GPU flags): `lrm_safety_deliberation.models`
  (`ModelSpec` / `MODELS` / `PRIMARY` / `DEFENSE_CELLS`). The launch scripts here
  still take an explicit `MODEL_PATH` pointing at the on-disk base model.
- Evaluation harness: `pipeline/eval/` (ASR / ORR with the 4-guardrail
  majority vote). RAPO ships its own quick eval (`rapo/eval_rapo.py`) as well.

## Method index

| Method | Type | Entry script | Data source | Train / run command | Output |
|---|---|---|---|---|---|
| **star1** | train (SFT) | `star1/sft.py` (`run_sft_<model>.sh`) | HF `UCSC-VLAA/STAR-1` (via `prepare_data.py`) | `python star1/prepare_data.py` then `MODEL_PATH=... OUTPUT_DIR=... bash star1/run_sft_qwen3_8b.sh` | `models/final/STAR1-<base>` |
| **safekey** | train (SFT + aux heads) | `safekey/sft.py` (`run_sft_<model>.sh`) | `eric-ai-lab/SafeKey` `sft_mix_2k.json` (via `prepare_data.py`) | `python safekey/prepare_data.py` then `MODEL_PATH=... OUTPUT_DIR=... bash safekey/run_sft_qwen3_8b.sh` | `models/final/SafeKey-<base>` |
| **r1_act** | train (LoRA SFT) | `r1_act/sft_lora.py` (`run_sft_lora.sh`) | in-repo `defenses/r1_act/data/r1_act_train.json` (959 ex) | `MODEL_PATH=... BASE_MODEL=Qwen OUTPUT_DIR=... bash r1_act/run_sft_lora.sh` | `models/final/R1-ACT-<base>` |
| **safechain** | train (SFT / ThinkSafe) | `safechain/sft.py` (`run_sft_<model>.sh`) | HF `UWNSL/SafeChain` (via `prepare_data.py`) | `python safechain/prepare_data.py` then `MODEL_PATH=... OUTPUT_DIR=... bash safechain/run_sft_qwen3_8b.sh` | `models/final/SafeChain-<base>` |
| **stair** | train (SFT → MCTS → DPO) | `stair/run_full_pipeline.sh` (`sft_lora.py` → `generate_mcts.py` → `extract_dpo_pairs.py` → `dpo_lora.py`) | HF `thu-ml/STAIR-SFT` + `thu-ml/STAIR-Prompts` (auto-fetched by the orchestrator) | `MODEL_PATH=... BASE_MODEL=Qwen MODEL_NAME=Qwen3-8B bash stair/run_full_pipeline.sh` | `models/final/STAIR-<base>` |
| **rapo** | train (SFT → GRPO) | `rapo/rapo_sft.py` → `rapo/rapo_rl.py` (`submit_rapo.sh`) | generated (two-stage prompting) + in-repo `defenses/rapo/data/` recipes | `ROOT=<research-tree> bash rapo/submit_rapo.sh` (Slurm: SFT → RL → eval) | `models/final/RAPO-<base>` |
| **safepath** | inference (decode-time) | `safepath/src/attack_zs.py` | n/a (operates on an eval JSONL) | serve base model on vLLM, then `python safepath/src/attack_zs.py --dataset <jsonl> --target-url ... --target-model <name>` | decode-time wrapper (no checkpoint) |
| **saferemind** | inference (decode-time) | `saferemind/src/attack.py` | n/a (operates on an eval JSONL) | serve base model on vLLM, then `python saferemind/src/attack.py --dataset <jsonl> --target-url ... --target-model <name>` | decode-time wrapper (no checkpoint) |
| **psr** | inference (decode-time) | `psr/src/attack.py` | n/a (operates on an eval JSONL) | serve base model on vLLM, then `python psr/src/attack.py --dataset <jsonl> --target-url ... --target-model <name>` | decode-time wrapper (no checkpoint) |

> **Note on filenames.** The three inference entry points are named `attack*.py`
> for historical reasons. They are **DEFENSE** runners (decode-time safety
> wrappers), not attacks. The naming is kept only to match the original tree.

## Training methods

All six write their finetuned cell to `models/final/<METHOD>-<base>` (e.g.
`STAR1-Qwen3-8B`, `R1-ACT-Olmo-3-7B-Think`). The four primary base models are
Qwen3-8B, OLMo-3-7B-Think, Phi-4-reasoning, and GPT-OSS-20B; per-model launch
scripts select LoRA target modules / chat-template handling per family
(`base_model` tag `Qwen | Olmo | Phi4/Phi | GPTOSS/GPT_OSS`).

| Method | Paper / upstream | Derives from (vendored, **not** copied here) | Recipe |
|---|---|---|---|
| star1 | STAR-1 (`UCSC-VLAA/STAR-1`) | `defenses/star1_repo/` (n/a — HF dataset only) | full-param SFT, 5 epochs, ZeRO-3 |
| safekey | SafeKey (`eric-ai-lab/SafeKey`, arXiv 2505.16186) | upstream `eric-ai-lab/SafeKey` | SFT + `safety_head` + `key_sentence_prediction` aux losses |
| r1_act | R1-Act (arXiv 2508.00324) | original R1-Act unsloth code | LoRA SFT (r=16, α=16), 15 epochs |
| safechain | SafeChain (`uw-nsl/safechain`, `UWNSL/SafeChain`) | upstream `uw-nsl/safechain` (no public SFT script; STAR-1-style adaptation) | full-param SFT over the SafeChain dataset |
| stair | STAIR (thu-ml/STAIR, arXiv 2502.02384) | `defenses/stair_repo/` | SFT → per-round MCTS → DPO-pair extraction → DPO (LoRA) |
| rapo | RAPO (ICLR 2026 TrustAI workshop, OpenReview `smLgjabnLP`) | `defenses/rapo_repo/` | SFT (generated data) → GRPO RL with an LLM reward judge |

### Notes

- **Launch scripts are portable**: each `run_*.sh` resolves `SCRIPT_DIR` /
  `PROJECT_DIR` from its own location and picks a free `main_process_port`;
  paths/hyperparameters are env-overridable (`MODEL_PATH`, `OUTPUT_DIR`,
  `DATA_PATH`, `N_EPOCHS`, ...). The two `rapo/submit_*.sh` Slurm scripts assume
  the original research-tree layout (`$ROOT/.venv`, `$ROOT/models`,
  `$ROOT/defenses/rapo`); `ROOT` defaults to the original absolute path and is
  overridable via the `ROOT` env var.
- **`stair/run_full_pipeline.sh`** expects the upstream STAIR SFT data at
  `defenses/stair_repo/data/stair_sft.json` and auto-downloads
  `thu-ml/STAIR-Prompts`; supply the SFT JSON (from the vendored STAIR repo or
  HF) before running.
- **`common/`** holds the shared training harness: `lora_sft.py` (generic LoRA
  SFT used by SafeChain / STAR-1 / SafeKey / SafePath training variants,
  including the SafeKey aux losses) and `olmo3_fix.py`, plus a ZeRO-3 config.

## Inference methods

Decode-time wrappers — **no finetuning**. Serve the base (or any) model on a
local OpenAI-compatible vLLM endpoint, then run the wrapper against it over an
eval JSONL (rows carrying at least a `goal` field). Output is a JSONL of
wrapped generations consumable by `pipeline/eval/`. Each method directory also
ships a small `server*.py` proxy helper (tokenizer-only OpenAI-compatible shim)
and shares `defenses/_common/m4_sampling.py` (imported via `../../_common`).

| Method | Paper | Mechanism |
|---|---|---|
| safepath | SAFEPATH (NeurIPS 2025, arXiv 2505.14667) | zero-shot safety primer: prefix `<think>\nLet's think about safety first.` after the generation prompt, leaving `<think>` open |
| saferemind | Entropy-based safety reminder for LRMs | per-token entropy from top-K logprobs; on an entropy-drop trigger, truncate and inject a safety reminder, then continue |
| psr | Progressive / Periodic Self-Reflection | every `K` tokens run a reflection query on the partial output; backtrack to the last safe checkpoint and regenerate if flagged unsafe |
