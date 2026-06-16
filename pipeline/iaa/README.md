# IAA — inter-annotator agreement panel (n=500 stance audit)

The from-scratch drivers behind the paper's chunk-level stance inter-annotator
agreement: sample 500 reasoning traces stratified across (group × base model ×
split), re-derive their chunk text with the production chunker, then have a panel
of LLM annotators label each chunk's stance (−1 refuse / 0 neutral / +1 comply),
and compute Fleiss/Cohen κ across them.

These are the **original research drivers**, copied here verbatim (logic
unchanged) and lightly adapted to read their repo root from
`LRM_SAFETY_ARTIFACTS` (default: the original repo via
`Path(__file__).resolve().parents[3]`). They still import the original
`experiments.cot_audit.prompts` (the shared stance template) and
`experiments.cot_audit.label_defense_stances` (`chunk_trace`, `extract_thinking`)
through that root.

> The paper's **final IAA table is produced by
> `camera_ready/reproduce/tab_iaa.py`**, which reads the annotator JSONLs below
> and computes κ with `lrm_safety_deliberation.stats` (`fleiss_kappa`, `cohen_kappa`,
> `cluster_bootstrap_ci`). The `iaa_n500_compare.py` / `compute_iaa_sonnet_table.py`
> drivers here are the original analyses; their κ helpers are superseded by
> `lrm_safety_deliberation.stats`.

## Files

| File | Role |
|---|---|
| `sample_traces_n500.py` | Stratified 500-trace sample (16 strata, seed 42) → `data/sample_n500_traces.jsonl`. CPU. |
| `prompts.py` | Canonical stance-label prompt template + system prompt (`PROMPT_TMPL_NEW`, `SYSTEM_PROMPT`). Pure module, no edits. |
| `iaa_n500_new_gpt.py` | GPT-5.4 annotator. API. |
| `iaa_n500_new_gemini.py` | Gemini-3-Pro annotator. API. |
| `iaa_n500_new_sonnet.py` | Sonnet-4.6 annotator (the downstream 3rd panelist). API. |
| `iaa_n500_new_opus.py` | Opus-4.7 annotator (4th reference panelist). API. |
| `iaa_n500_compare.py` | New-protocol IAA across annotators + old-vs-new per-chunk agreement. |
| `compute_iaa_sonnet_table.py` | Every cell of the Sonnet-as-3rd-annotator IAA table (with bootstrap CIs). |
| `label_defense_stances.py` | Bundled helper (`chunk_trace`, `extract_thinking`) reused by the sampler; original lives in `experiments/cot_audit/`. |

## Run order

```
# 1. Sample 500 traces (CPU)
python sample_traces_n500.py
#    -> experiments/cot_audit/data/sample_n500_traces.jsonl

# 2. Annotate with each panelist (API; resume-safe, append-only)
python iaa_n500_new_gpt.py        # -> outputs/iaa_n500_new/gpt54.jsonl
python iaa_n500_new_gemini.py     # -> outputs/iaa_n500_new/gemini.jsonl
python iaa_n500_new_sonnet.py     # -> outputs/iaa_n500_new/sonnet46.jsonl
python iaa_n500_new_opus.py       # -> outputs/iaa_n500_new/opus47.jsonl   (optional 4th)

# 3. Compute agreement
python iaa_n500_compare.py          # new-protocol IAA + old-vs-new agreement
python compute_iaa_sonnet_table.py  # full Sonnet-panel table + bootstrap CIs

# The paper table itself:
python ../../reproduce/tab_iaa.py   # lrm_safety_deliberation.stats kappa over the JSONLs above
```

Each annotator script takes `--sample`, `--out`, and a `--cost-cap-usd` cap
(`-h` for the full set). The Sonnet/Opus runners stop once the cap is hit.

## API providers, models & cost

Keys are loaded from `$LRM_SAFETY_ARTIFACTS/.env`.

| Annotator | Provider / key | Model id | $/M in · out |
|---|---|---|---|
| GPT-5.4 | OpenAI (`OPENAI_API_KEY`) | `gpt-5.4` | 1.25 · 10.0 |
| Gemini-3-Pro | Google (`GEMINI_API_KEY`/`GOOGLE_API_KEY`) | `gemini-3-pro-preview` | 1.25 · 10.0 |
| Sonnet-4.6 | Anthropic (`ANTHROPIC_API_KEY`) | `claude-sonnet-4-6` | 3.0 · 15.0 |
| Opus-4.7 | Anthropic (`ANTHROPIC_API_KEY`) | `claude-opus-4-7` | 15.0 · 75.0 |

Opus is by far the costliest annotator (default cap $40); the downstream panel is
GPT-5.4 + Gemini-3-Pro + Sonnet-4.6, so Opus is optional reference-only.

## Inputs / outputs

* **Input**: the GPT-5.4-labeled stance pool
  (`stance_audit_full/labeled_traces.jsonl` + `defenses_stance_v2/labeled_traces.jsonl`)
  plus the source `generations.jsonl` (for chunk re-derivation).
* **Sample**: `experiments/cot_audit/data/sample_n500_traces.jsonl`.
* **Annotator outputs**: `experiments/cot_audit/outputs/iaa_n500_new/*.jsonl`
  (`gpt54.jsonl`, `gemini.jsonl`, `sonnet46.jsonl`, `opus47.jsonl`) — these are
  the inputs to `reproduce/tab_iaa.py`.
* **Analysis summaries**: `outputs/iaa_n500_new/iaa_report.json` and the table
  printed by `compute_iaa_sonnet_table.py`.
