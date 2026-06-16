# Pipeline · Representations

Hidden-state extraction + linear-probe / refusal-valley drivers. These are the
original research drivers, copied here verbatim (logic unchanged) and lightly
adapted to (a) import the bundled `common.py` next to them rather than the live
`scripts/neurips_final/` tree, and (b) resolve the artifact root from the
`LRM_SAFETY_ARTIFACTS` env var (falling back to the repo root). They produce the
representation tensors, valley caches, and K-hop probe JSONs that the
`reproduce/` CPU scripts consume.

> You do **not** need to run these to regenerate the paper's figures — `make
> artifacts` downloads the caches. This is the GPU-from-scratch path (extraction
> needs an H100-class GPU; the analysis drivers are CPU/GPU-light).

## What lives in the library vs. here

`common.py` is a bundled copy of `scripts/neurips_final/common.py`. It still
carries the `MODELS` registry, the 4-guardrail majority vote
(`majority_tie_negative`), the suite map, and the jsonl helpers so the drivers
run standalone. The **canonical** versions of those pieces now live once in the
shared `lrm_safety_deliberation` library and supersede the copies here:

* model registry + decoding specs → `lrm_safety_deliberation.models` (supersedes the
  `common.MODELS` dict / per-model `batch_size`, `channel`, `max_model_len`)
* the four guardrail classifiers + majority/soft vote → `lrm_safety_deliberation.guardrails`
  (supersedes `majority_tie_negative` and the inline `guardrail_votes`)
* trained representation probe + bootstrap / Fisher / κ stats → `lrm_safety_deliberation.stats`
* artifact paths → `lrm_safety_deliberation.paths` (set `LRM_SAFETY_ARTIFACTS` to relocate)

## Files

| File | Role |
|---|---|
| `common.py` | bundled MODELS registry + jsonl/vote/suite helpers the extract drivers import |
| `extract_representations.py` | GPU. last-layer prefill + thinking-hopping + thinking-pooling hidden states |
| `extract_kshop_representations.py` | GPU. K-hop (K=20) user-content-span prefill hidden states, 5-layer grid |
| `explore_valley_metrics.py` | refusal-valley separability metrics (logit-Fisher, Cohen's d, Mahalanobis, norm-Fisher, Bhattacharyya) on hopping reps |
| `compute_norm_fisher_pooling.py` | norm-Fisher valley on chunk-mean-pooled thinking reps |
| `compute_kshop_norm_fisher.py` | norm-Fisher per (layer, k anchor, pool) on the K-hop artifacts |
| `compute_kshop_bestpipe.py` | per-k probe AUROC/BAcc under the best pipeline (StandardScaler→PCA100→LogReg C=0.03) |
| `first_token_auroc_table.py` | State-A (prompt-only first-token) probe AUROC + 95% CI table (`tab:auroc_ci`) |
| `submit_kshop.sh` | SLURM (pli-c) submitter for `extract_kshop_representations.py` |

## Run order

```
extract_representations.py            ─┬─▶ explore_valley_metrics.py
                                       └─▶ compute_norm_fisher_pooling.py

extract_kshop_representations.py      ─┬─▶ compute_kshop_norm_fisher.py
                                       └─▶ compute_kshop_bestpipe.py

(State-A prefill reps, see note)       ──▶ first_token_auroc_table.py   # tab:auroc_ci
```

Models in the paper: `Qwen3-8B`, `Qwen3-32B`, `Olmo-3-7B-Think`,
`Phi-4-reasoning`, `GPT-OSS-20B` (the K-hop / norm-Fisher drivers loop over the
four thinking models internally).

## Commands

Set the artifact root once (defaults to the repo root if unset):

```bash
export LRM_SAFETY_ARTIFACTS=/path/to/artifacts   # holds eval_results/, experiments/
PY=.venv/bin/python                              # uv-managed venv; transformers + torch + sklearn
```

Extraction (one model at a time; **GPU**, H100-80G class):

```bash
# main-trace reps: prefill / thinking-hopping / thinking-pooling
$PY extract_representations.py --model Qwen3-8B --suite safety
$PY extract_representations.py --model Qwen3-8B --suite supplementary   # orfuzz, phtest

# K-hop user-content-span reps (K=20, 5-layer grid), both suites in one shot
$PY extract_kshop_representations.py --model Qwen3-8B --suites safety supplementary
# or on the cluster:
bash submit_kshop.sh Qwen3-8B 1          # <MODEL> [walltime_hours], partition pli-c
```

Valley / probe analysis (CPU or light GPU):

```bash
# refusal-valley separability metrics (hopping + pooling)
$PY explore_valley_metrics.py    --models Qwen3-8B Olmo-3-7B-Think Phi-4-reasoning GPT-OSS-20B
$PY compute_norm_fisher_pooling.py --models Qwen3-8B Olmo-3-7B-Think Phi-4-reasoning GPT-OSS-20B

# K-hop norm-Fisher + best-pipeline probe (loop over the four thinking models internally — no flags)
$PY compute_kshop_norm_fisher.py
$PY compute_kshop_bestpipe.py

# State-A first-token AUROC + 95% CI table for tab:auroc_ci (no flags; all five models)
$PY first_token_auroc_table.py
```

## Inputs

* `eval_results/neurips_final/<model>/classifications/base/{harmful,benign}_guardrails.jsonl`
  — per-item generations + 4-guardrail votes (consumed by `extract_*`).
* `eval_results/neurips_final_supplementary/<model>/classifications/...` — the
  `orfuzz` / `phtest` benign sub-sources.
* For `first_token_auroc_table.py`: State-A prefill reps under
  `eval_results/neurips_s16/<model>/representations/prefill_think_before/*.pt`
  and the supplementary `prefill_think_before/{orfuzz,phtest}.pt` (produced by
  the prefill-think-before extraction variant; labels joined by id from the
  hopping reps).

## Outputs

| Driver | Output |
|---|---|
| `extract_representations.py` | `eval_results/neurips_final/<m>/representations/{prefill,thinking_hopping,thinking_pooling}/{harmful,benign}.pt` |
| `extract_kshop_representations.py` | `experiments/refusal_cliff/artifacts/kshop_K20_user_content/<m>/<intent>.pt` |
| `explore_valley_metrics.py` | `eval_results/neurips_final/<m>/valley/explore_<intent>.pt` |
| `compute_norm_fisher_pooling.py` | `eval_results/neurips_final/<m>/valley/explore_pooling_{harmful,benign_combined}.pt` |
| `compute_kshop_norm_fisher.py` | `experiments/refusal_cliff/results/kshop_K20_user_content/norm_fisher_<m>.json` |
| `compute_kshop_bestpipe.py` | `experiments/refusal_cliff/results/kshop_K20_user_content/bestpipe_pca100_c003.json` |
| `first_token_auroc_table.py` | `neurips/first_token_auroc_stateA.json` (+ markdown/LaTeX to stdout) |

## Notes

* Plotting scripts (`plot_*.py`) are **not** here — they live under
  `camera_ready/reproduce/` and run on CPU against these outputs.
* The cluster submitters for the main-trace extraction + analysis
  (`run_reps_analyze.sh`, `submit_reps_analyze.sh`, `submit_kshop_analyze.sh`)
  were **not** copied: they chain non-canonical drivers (`gather_outputs.py`,
  `analyze_model.py`, `analyze_kshop.py`) outside this stage's scope. Only
  `submit_kshop.sh` (which calls exactly `extract_kshop_representations.py`) is
  included. Run the analysis drivers directly with the commands above.
