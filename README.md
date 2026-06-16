<!-- markdownlint-disable MD013 -->
# Do Thinking Tokens Help with Safety?

Reproduction code for **"Do Thinking Tokens Help with Safety?"** *(Accepted at the ICML 2026 AI4GOOD Workshop, Oral Presentation)*.

> **TL;DR.** In large reasoning models the safety decision is already encoded at
> the **first thinking token** (a linear probe reaches AUROC 0.84–0.95), then
> *diffuses* through the middle of the trace — class separability collapses into
> a **valley** — before partly re-crystallizing at the end. Extended thinking
> uniformly *lowers* refusal on harmful **and** benign prompts, the visible
> chain-of-thought is largely **performative** (cutting it rarely flips the
> decision), and both training- and inference-time defenses move models *along*
> the ASR–ORR trade-off rather than improving safety discrimination.

This repository is organized as a small **library** (`lrm_safety_deliberation/`) plus thin,
one-per-artifact **reproduction scripts** (`reproduce/`). Every figure and table
in the paper can be regenerated with a single command from a set of downloadable
intermediate artifacts; the full from-scratch pipeline (generation →
classification → representation extraction → training) lives under `pipeline/`.

---

## What's here

```
camera_ready/
├── lrm_safety_deliberation/        # the library: model registry, data/splits, guardrails,
│                      #   representations, probes, cuts, judges, stats, plotting
├── reproduce/         # one thin script per paper figure/table  (CPU, minutes)
├── pipeline/          # heavy GPU/training drivers + SLURM  (the from-scratch path)
├── configs/           # model / decoding / benchmark configuration
├── data/              # benchmark INPUT manifests  (shipped; 4,573 ASR + 3,135 ORR)
├── artifacts/         # intermediate artifacts  (downloaded; not in git)
├── figures/           # regenerated figures + tables land here  (not in git)
└── docs/              # ARTIFACT_MAP.md and per-family reproduction notes
```

The library is intentionally **stage-oriented**, not section-oriented — the same
primitives (one guardrail evaluator, one generation harness, one probe pipeline,
one plotting style) back every experiment:

| Module | Responsibility |
|---|---|
| `models` | Model registry: weights, decoding (Table `eval_decoding`), channel/streaming flags |
| `data` | Benchmark manifests + ASR/ORR split registry (`asr_full`, `orr_full`) |
| `guardrails` | The 4 classifiers (WildGuard, Qwen3Guard, Granite Guardian, OSS-Safeguard) + the majority/soft refusal vote |
| `generate` | vLLM harness: K-rollout think/no-think generation, force-close, channel models |
| `representations` | Hidden-state extraction (first token, hopping/pooling trace, K-hop prefill) |
| `probes` | First-token probe (`StandardScaler→PCA(100)→LogReg`) + AUROC/BAcc |
| `metrics` / `cuts` | Fisher / norm-Fisher valley, flip/Δ, salient cuts, Locked×Significant oscillation |
| `judges` | Multi-provider (OpenAI / Gemini / Anthropic) stance & faithfulness annotators |
| `stats` | Bootstrap CIs, Fisher-exact + Holm, Fleiss/Cohen κ |
| `plotting` | TeX Gyre Termes registration, model palette, paper rcParams |

---

## Quickstart

```bash
# 1. Install the core (CPU) environment — enough to regenerate every figure/table.
python -m venv .venv && source .venv/bin/activate
pip install -e .                     # or: make env

# 2. Download the intermediate artifacts (generations, classifications,
#    representations, analysis caches) from the HuggingFace Hub.
make artifacts                       # -> artifacts/

# 3. Regenerate everything (writes to figures/).
make all                             # or a single target, e.g. `make fig-refusal-valley`
```

The from-scratch path additionally needs the GPU extras and model weights:

```bash
pip install -e ".[gpu]"              # torch, vllm, transformers, accelerate, trl
pip install -e ".[judges]"           # openai, anthropic, google-genai (LLM judges)
# Place model weights under models/<name> (or set LRM_SAFETY_MODELS).
```

---

## Reproduce a figure or table

Each paper artifact maps to exactly one `reproduce/` script and one `make`
target. **Default tier** = CPU-only, regenerated from downloaded artifacts
(minutes). The "from scratch" column lists the `pipeline/` stages that produce
the intermediate artifacts (GPU; see `docs/ARTIFACT_MAP.md` for exact commands).

### Figures

| Paper | `make` target | Script | From-scratch pipeline |
|---|---|---|---|
| `fig:refusal_valley` | `fig-refusal-valley` | `reproduce/fig_refusal_valley.py` | generate → classify → extract-representations → valley-metrics |
| `fig:refusal_valley_supplementary` | `fig-refusal-valley` | `reproduce/fig_refusal_valley_supplementary.py` | (same, 5 supplementary models) |
| `fig:prefill_decision` | `fig-prefill-decision` | `reproduce/fig_prefill_decision.py` | extract-kshop → norm-Fisher + bestpipe probe |
| `fig:asr_orr_extended_thinking` | `fig-extended-thinking` | `reproduce/fig_extended_thinking.py` | K=32 generate (think) + nested-branching → classify → aggregate |
| `fig:no_think_to_think` | `fig-nothink-to-think` | `reproduce/fig_nothink_to_think.py` | K=32 generate (think+no-think) → classify |
| `fig:no_think_to_think_lenient` | `fig-nothink-to-think` | `reproduce/fig_nothink_to_think_lenient.py` | (same) |
| `fig:within_prefix_variance` | `fig-within-prefix` | `reproduce/fig_within_prefix_variance.py` | nested-branching (B=0..100) → classify → aggregate |
| `fig:thinking_causal` / `_bysplit` | `fig-thinking-causal` | `reproduce/fig_thinking_causal.py` | M=4 sample → stance judges → K=100 cut-replay → classify → oscillation |
| `fig:defense_pareto` | `fig-defense-pareto` | `reproduce/fig_defense_pareto.py` | full ASR/ORR eval of base + defense cells |
| `fig:training_slopegraph` | `fig-training-slopegraph` | `reproduce/fig_training_slopegraph.py` | first-token AUROC + oscillation per defense cell |

### Tables

| Paper | `make` target | Script | From-scratch pipeline |
|---|---|---|---|
| `tab:base_asr_orr` | `tab-base-asr-orr` | `reproduce/tab_base_asr_orr.py` | M=4 generate → 4-guardrail classify → aggregate |
| `tab:defenses_asr_orr` | `tab-defenses-asr-orr` | `reproduce/tab_defenses_asr_orr.py` | (same, for the 24 defense cells) |
| `tab:auroc_ci` | `tab-auroc-ci` | `reproduce/tab_auroc_ci.py` | extract first-token representations → CV probe |
| `tab:iaa` | `tab-iaa` | `reproduce/tab_iaa.py` | sample 500 traces → 3 LLM-judge annotators → Fleiss κ |
| `tab:inference_defense_suppression` | `tab-inference-suppression` | `reproduce/tab_inference_defense_suppression.py` | inference-defense M=4 → stance → K=100 cut-replay |
| `tab:eval_datasets` | — | `reproduce/tab_eval_datasets.py` | static (from `data/` manifests) |
| `tab:eval_decoding` | — | `reproduce/tab_eval_decoding.py` | static (from the model registry) |

> See **`docs/ARTIFACT_MAP.md`** for the complete lineage of every artifact —
> the exact scripts, inputs, outputs, sizes, and which steps need a GPU.

---

## Data & artifacts

* **Benchmark inputs** (`data/`) ship with the repo: 8 ASR benchmarks (advbench,
  harmbench, strongreject, sorrybench, jailbreakbench, wildjailbreak, fortress,
  hexphi; 4,573 prompts) and 4 ORR benchmarks (or_bench, falsereject, coconot,
  xstest_safe; 3,135 prompts).
* **Intermediate artifacts** (generations, guardrail classifications,
  representations, K=100 cut rollouts, analysis caches) are large and
  regenerable; `make artifacts` downloads them from the HuggingFace Hub into
  `artifacts/`. Point `LRM_SAFETY_ARTIFACTS` at an existing tree to reuse it.
* **Model weights** are not redistributed; the registry (`lrm_safety_deliberation/models.py`)
  records best-effort HuggingFace repo ids. Place weights under `models/<name>`
  or set `LRM_SAFETY_MODELS`.

Path roots are configurable via environment variables (`LRM_SAFETY_DATA`,
`LRM_SAFETY_MODELS`, `LRM_SAFETY_ARTIFACTS`, `LRM_SAFETY_FIGURES`); see
`lrm_safety_deliberation/paths.py`.

## Hardware

The reproduction (CPU) tier runs on a laptop. The from-scratch pipeline was run
on NVIDIA H100-80GB GPUs (single-GPU generation/classification/extraction;
2–8 GPUs for training). Full regeneration of all generations + K=100 cut
rollouts is on the order of GPU-weeks; the downloadable artifacts exist so that
the *analysis* is reproducible without that compute.

<!-- ## Citation

```bibtex
@inproceedings{refusalvalley,
  title     = {Do Thinking Tokens Help with Safety?},
  author    = {Narutatsu Ri and Abhishek Panigrahi and Sanjeev Arora},
  year      = {2026}
}
``` -->
