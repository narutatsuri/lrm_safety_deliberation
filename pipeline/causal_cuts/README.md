# Causal cuts — thinking-trace oscillation + cut-replay pipeline

The from-scratch drivers that detect stance oscillations in reasoning traces, cut
the trace at each refuse↔comply transition, force-close `</think>`, replay K=100
fresh responses per cut, classify each with the 4-guardrail panel, and aggregate
into the per-cut cache that the paper's causal figures/tables consume.

These are the **original research drivers**, copied here verbatim (logic
unchanged) and lightly adapted to read their repo root from
`LRM_SAFETY_ARTIFACTS` (default: the original repo, found via
`Path(__file__).resolve().parents[3]`). They still import the original repo
modules through that root (`scripts.neurips_final.common` for the model registry,
`scripts.neurips_s16.common` for the guardrail vote, `experiments.causal_cuts.*`
and `experiments.cot_audit.*` for shared helpers). The shared `lrm_safety_deliberation`
library now also carries these pieces:

* model registry + decoding → `lrm_safety_deliberation.models`
* the 4 guardrail classifiers + majority/soft vote → `lrm_safety_deliberation.guardrails`
  (supersedes `scripts.neurips_s16.common.extract_guardrail_vote` /
  `GUARDRAIL_ORDER` / `OR_BENCH_REFUSAL_RUBRIC`)
* bootstrap / Fisher+Holm statistics → `lrm_safety_deliberation.stats`

> You do **not** need to run any of this to regenerate the paper. `make
> artifacts` downloads the intermediate caches and the `reproduce/` scripts run
> on CPU in minutes. This pipeline is for full from-scratch reproduction
> (GPU-weeks + API spend).

## What this feeds

The aggregated cache `m4_allcuts.jsonl` and the oscillation summaries feed:

* `camera_ready/reproduce/fig_thinking_causal.py` (the causal "budge" figure)
* `camera_ready/reproduce/tab_inference_defense_suppression.py` (inference-defense
  suppression table)

## Files

| File | Role |
|---|---|
| `generate_cuts.py` | Core cut detector (`salient_transitions`, `flip`/`segment` modes) + K-rollout generator. Imported by every other generator for its helpers. |
| `generate_cuts_repro.py` | K=100 cut-replay over the **base-model** majority-3 sample (`repro_majority3.jsonl`). GPU. |
| `generate_cuts_repro_inference.py` | K=100 cut-replay over the **inference-defense** majority-3 sample (re-prepends the SafePath-ZS primer). GPU. |
| `classify_segment_repro.py` | 4-guardrail classifier for the base-model cut rollouts. GPU. |
| `classify_cuts_inference.py` | 4-guardrail classifier for the inference-defense cut rollouts (thin wrapper over `classify_segment_repro`). GPU. |
| `sample_n2400.py` | Stratified n=2400 (300/cell × 8) base stance-audit sample → `sample_n2400.jsonl`. |
| `build_majority3.py` | Majority-of-3 chunk stance labels over the n=2400 sample (GPT-5.4 + Gemini-3-Pro + Opus chain). |
| `m4_build_majority3.py` | Majority-of-3 + cut detection for the **M=4 base** sample (`m4_sample.jsonl` → `m4_majority3.jsonl` + `m4_cuts.jsonl`). |
| `m4_inference_defense_sample.py` | Builds the M=4 inference-defense stance-audit manifest (chunks each defense trace). |
| `m4_inference_defense_build_majority3.py` | Majority-of-3 chunk labels for the M=4 inference-defense sample (GPT-5.4 + Gemini-3-Pro + Sonnet-4.6). |
| `build_m4_allcuts.py` | Reads the giant K=100 rollout/guardrail JSONLs once → per-cut cache `m4_allcuts.jsonl`. |
| `analyze_m4_inference_defenses.py` | Stage-6 oscillation analysis (oscillation/budge/sig fracs) for the inference-defense cells vs base. |
| `percut_base_vs_inference.py` | Per-cut base-vs-inference comparison (same unit as the cut-meaning waffle). |
| `label_defense_stances.py` | Bundled helper (`chunk_trace`, `extract_thinking`) reused by the samplers; original lives in `experiments/cot_audit/`. |

The original `experiments/causal_cuts/slurm/` submit wrappers (`submit_segment*.sh`,
`submit_classify*.sh`, `submit_segment_m4*.sh`, …) drive the GPU stages on the
cluster; they are not copied here — adapt the commands below to your scheduler.

## Run order

The two cut families (base-model and inference-defense) share the same shape:
sample → API stance judges → majority-3 → GPU K=100 cut-replay → GPU 4-guardrail
classify → aggregate → analyze.

```
# ---- 1. Sample the stance-audit pool ------------------------------------
python sample_n2400.py                       # base n=2400 sample (CPU)
python m4_inference_defense_sample.py        # inference-defense M=4 manifest (CPU)
#   (the M=4 base sample m4_sample.jsonl is produced upstream; see HISTORY.md)

# ---- 2. API stance judges (per annotator; see ../iaa for the runner shape)
#   Run the 3-judge panel over each sample to produce, per cell:
#     {gpt54,gemini, opus_chain|sonnet46}.jsonl
#   base panel        = GPT-5.4 + Gemini-3-Pro + Opus chain
#   inference-defense = GPT-5.4 + Gemini-3-Pro + Sonnet-4.6  (Sonnet replaces Opus)
#   Requires OPENAI_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, ANTHROPIC_API_KEY in .env.

# ---- 3. Majority-of-3 + cut detection -----------------------------------
python build_majority3.py                    # base n=2400 -> labeled_traces_majority3.jsonl
python m4_build_majority3.py                  # M=4 base    -> m4_majority3.jsonl + m4_cuts.jsonl
python m4_inference_defense_build_majority3.py  # M=4 defense -> m4_inference_defense_majority3.jsonl

# ---- 4. GPU K=100 cut-replay (one job per model x split / cell x split) --
python generate_cuts_repro.py --model <MODEL> --split <harmful|benign> --K 100
python generate_cuts_repro_inference.py --model <MODEL> --split <harmful|benign> --K 100
#   -> .../causal_cuts[/_inference]/<CELL>/<SPLIT>/rollouts_segment_repro.jsonl

# ---- 5. GPU 4-guardrail classify of every cut rollout -------------------
python classify_segment_repro.py --model <MODEL> --split <harmful|benign> --guardrail <GR>
python classify_cuts_inference.py --cell <CELL> --split <harmful|benign> --guardrail <GR>
#   GR in {wildguard, qwen3guard, granite_guardian, oss_safeguard}
#   -> .../guardrails_segment_repro__<GR>.jsonl

# ---- 6. Aggregate + analyze ---------------------------------------------
python build_m4_allcuts.py                   # -> m4_allcuts.jsonl (per-cut cache)
python analyze_m4_inference_defenses.py      # -> m4_inference_defense_oscillation.{json,md}
python percut_base_vs_inference.py           # per-cut base-vs-inference table
```

(`--model`/`--split`/`--guardrail`/`--cell` flags follow the original scripts'
`argparse`; run any driver with `-h` for the full set. Generators default
`--K 32`; the paper used `--K 100`.)

## API keys & judges

The stance-judge panel (Stage 2) and the inference-defense panel call paid APIs.
Put the keys in `$LRM_SAFETY_ARTIFACTS/.env`:

* `OPENAI_API_KEY` — GPT-5.4 (`gpt-5.4`)
* `GEMINI_API_KEY` / `GOOGLE_API_KEY` — Gemini-3-Pro (`gemini-3-pro-preview`)
* `ANTHROPIC_API_KEY` — Sonnet-4.6 (`claude-sonnet-4-6`) for the inference-defense
  panel; Opus 4.7 (`claude-opus-4-7`) for the original base panel. (Opus is the
  costliest annotator — prefer GPT-5.4 + Gemini unless a 3rd is required.)

## Inputs / outputs (sizes are illustrative)

* **Inputs**: the stance-audit pool (`audit_chunked_n3885.jsonl`) and the M=4
  defense generations under `results/full_splits_v2/m4/`.
* **Heavy intermediates** (multi-GB, *not* shipped — regenerate with the GPU
  stages): `rollouts_segment*.jsonl` (cut replays),
  `guardrails_segment_repro__*.jsonl`, `m4_inference_defense_sample.jsonl`
  (~210 MB), `m4_inference_defense_majority3.jsonl` (~100 MB).
* **Aggregated cache**: `m4_allcuts.jsonl` (~2 MB, one row per detected cut) — the
  small artifact the `reproduce/` scripts actually read.
* **Summaries**: `m4_inference_defense_oscillation.{json,md}`,
  `m4_percut_base_vs_inference.md`.
