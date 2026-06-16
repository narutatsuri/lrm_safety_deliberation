# Artifact Map

The complete lineage of every paper figure and table: the reproduction script,
the from-scratch pipeline that produces its inputs, the intermediate artifacts a
"download-then-replot" run needs, and the gotchas. Six experiment families share
one pipeline:

```
data/ manifests
   │
   ▼
generate (vLLM, think / no-think, K rollouts, force-close, channel models)
   │
   ├─► guardrail classify (4 classifiers → majority/soft refusal vote)
   │        │
   │        ├─► behavioral metrics (ASR/ORR)            → tables/figures
   │        └─► flip / Δ (no-think ↔ think, B-sweep)    → figures
   │
   ├─► representation extraction (first token, hopping/pooling, K-hop)
   │        └─► probe (AUROC/BAcc) + norm-Fisher valley → figures/tables
   │
   └─► stance judges (LLM panel) ─► salient cuts ─► K=100 cut-replay
            └─► Locked × Significant oscillation         → figures/tables
```

Legend: **[GPU]** needs a GPU, **[API]** needs LLM-judge API keys, **[CPU]**
pure analysis/plotting (the default reproduction tier).

---

## Family 1 — Representation geometry

**Artifacts:** `fig:refusal_valley`, `fig:refusal_valley_supplementary`,
`fig:prefill_decision`, `tab:auroc_ci`.

**Pipeline**

1. **[GPU]** `extract_representations` — for each model/intent, dump last-layer
   hidden states at the first thinking token (`prefill`) and 100 interpolated
   positions over the trace (`thinking_hopping`, `thinking_pooling`).
   → `eval_results/neurips_final/<model>/representations/{prefill,thinking_hopping,thinking_pooling}/{harmful,benign}.pt` (~2–7 GB each).
2. **[CPU]** `valley_metrics` — per-position norm-Fisher (+ logit-Fisher, Cohen-d,
   Mahalanobis, Bhattacharyya). → `…/valley/explore_{harmful,benign_combined}.pt` (small).
3. **[GPU]** `extract_kshop` — hidden states at K=20 anchor positions over the
   user-content span × 5-layer grid. → `experiments/refusal_cliff/artifacts/kshop_K20_user_content/<model>/{harmful,benign,phtest,orfuzz}.pt` (~5–10 GB).
4. **[CPU]** `kshop_norm_fisher` → `…/results/kshop_K20_user_content/norm_fisher_<model>.json`.
5. **[CPU]** `kshop_bestpipe` — `StandardScaler→PCA(100)→LogReg(C=0.03)` probe,
   per-k AUROC/BAcc with per-benchmark Youden + nested bootstrap CI.
   → `…/results/kshop_K20_user_content/bestpipe_pca100_c003.json`.
6. **[GPU]** `first_token_auroc` — State-A (prompt-only) CV LogReg → `tab:auroc_ci` (AUROC 0.84–0.95).
7. **[CPU]** plot: `fig_refusal_valley(_supplementary)` from `explore_*.pt`;
   `fig_prefill_decision` from `norm_fisher_*.json` + `bestpipe_*.json`.

**Replot tier needs:** `valley/explore_*.pt` (tiny) and the two kshop `*.json`
caches. The multi-GB `*.pt` representation tensors are only needed to *recompute*
those caches.

**Gotchas:** KSHOP probe uses last layer only; benign curves are a macro over
{benign_main, phtest, orfuzz} surviving a `MIN_POS=20` filter; norm-Fisher is
per-dimension standardized; GPT-OSS uses harmony-channel thinking spans.

---

## Family 2 — Extended-thinking behavior (K=32)

**Artifacts:** `fig:asr_orr_extended_thinking`, `fig:no_think_to_think`,
`fig:no_think_to_think_lenient`, `fig:within_prefix_variance`.

**Pipeline**

1. **[GPU]** `k32_generate` — K=32 independent rollouts per prompt in `think` and
   `nothink` modes. → `eval_results/k32_full_pool/<model>/{think,nothink}/{asr,orr}/`.
2. **[GPU]** `nested_branching` — subsample M=8 think prefixes/prompt, truncate
   (or extend) thinking to budget B ∈ {0,20,…,200}%, resample N=8 continuations.
   → `eval_results/k32_nested_branching_full/<model>/<split>/cut{B}/`.
3. **[GPU]** `classify` (4 guardrails) over both pools.
4. **[CPU]** `aggregate_k32` — per-cell soft ASR/ORR + within-prefix variance +
   bootstrap CI. → `analysis/data/asr_orr_within_by_cut.json` (the replot input).
5. **[CPU]** plots: ASR/ORR-vs-thinking; majority-flip + delta-sign side-by-side;
   within-prefix variance with the B=0 baseline.

**Replot tier needs:** `asr_orr_within_by_cut.json` + the K=32 `classifications_*.jsonl`
(the flip/B=0 plots recompute per-prompt rates from votes).

**Gotchas:** flips are defined on the **per-prompt majority label** (>½ of K
rollouts majority-refuse); the lenient figure additionally reports sign-only Δ
and mean±std; B=0 = empty-prefix continuation variance, not "no thinking".

---

## Family 3 — Behavioral ASR/ORR battery

**Artifacts:** `tab:base_asr_orr`, `tab:defenses_asr_orr`, `fig:defense_pareto`.

**Pipeline**

1. **[GPU]** `full_eval` (`pipeline/eval`) — for each model/split: vLLM generate
   (M=4 rollouts, 16K thinking budget, force-close) → 4-guardrail classify →
   per-benchmark aggregate. → `eval_results/asr_orr_16k_K4/<split>/<model>/summary.json`
   (base) and `eval_results/defenses_eval/<cell>/__fullpipe__/{asr,orr}/…/summary.json` (defenses).
2. **[CPU]** `tab_base_asr_orr` — fractional 4-guardrail vote + 95% bootstrap CI.
3. **[CPU]** `tab_defenses_asr_orr` — aggregate 24 cells vs base.
4. **[CPU]** `fig_defense_pareto` — ASR–ORR plane (base + 6 training + 3 inference × 4 models).

**Replot tier needs:** the `summary.json` (base) and `results.jsonl` (for the
bootstrap CI) per cell — small per file.

**Gotchas:** majority = ≥3/4 with 2-2 ties → comply; ASR = comply-rate on harmful,
ORR = refusal-rate on benign; GPT-OSS uses harmony force-close; inference-defense
ASR/ORR are transcribed from the eval (no standalone json) — see the family-6 runners.

---

## Family 4 — CoT causal utility (causal cuts / oscillation)

**Artifacts:** `fig:thinking_causal`, `fig:thinking_causal_bysplit`,
`tab:inference_defense_suppression`.

**Pipeline**

1. **[CPU]** `m4_sample` — chunk M=4 natural traces. → `m4_sample.jsonl`.
2. **[API]** stance judges (GPT-5.4 + Gemini-3-Pro + Sonnet/Opus) → majority-of-3
   chunk stance labels. → `m4_majority3.jsonl`.
3. **[GPU]** `generate_cuts` — at each refuse↔comply transition, fork pre/post,
   close `</think>`, resample **K=100** final responses.
   → `experiments/causal_cuts/eval_results/causal_cuts/<model>/<split>/rollouts_segment_m4.jsonl` (multi-GB).
4. **[GPU]** `classify_cuts` (4 guardrails) over the K=100 rollouts.
5. **[CPU]** `build_m4_allcuts` — per-cut comply(pre/post) + Fisher p + probe score.
   → `m4_allcuts.jsonl` (the replot input, ~2 MB).
6. **[CPU]** Locked × Significant (Holm-corrected Fisher exact) → `fig_thinking_causal`.
7. **[CPU]** `analyze_inference_defenses` → `tab:inference_defense_suppression`.

**Replot tier needs:** `m4_allcuts.jsonl` + the inference-defense oscillation
JSON. The multi-GB K=100 rollout files are only needed to recompute those.

**Gotchas:** *Locked* = both pre/post comply saturated (≤0.05 or ≥0.95);
*Significant* = Holm-corrected Fisher exact p<0.05; *Performative* = not
significant. ~80–95% of cuts are performative. Phi-4 loop traces (>120 chunks)
are reported separately; GPT-OSS uses the harmony-channel forced-cut protocol.

---

## Family 5 — CoT faithfulness / inter-annotator agreement

**Artifact:** `tab:iaa`.

**Pipeline**

1. **[CPU]** `sample_traces_n500` — stratified 500-trace sample (16 strata). → `sample_n500_traces.jsonl`.
2. **[API]** three annotators (GPT-5.4 + Gemini-3-Pro + Sonnet-4.6; Opus-4.7 as
   reference) label every chunk's stance. → `outputs/iaa_n500_new/{gemini,sonnet46,opus47}.jsonl`.
3. **[CPU]** `compute_iaa` — Fleiss κ (trace-clustered bootstrap CI) + pairwise
   Cohen κ + concordance. → `tab:iaa` (Fleiss κ ≈ 0.63, substantial).

**Replot tier needs:** the four annotator `*.jsonl` (≈0.2 MB each) + the sample file.

**Gotchas:** the dedicated IAA study used the NEW (system-prompted) protocol;
downstream stance labeling used the OLD protocol — disclosed in the paper. Opus
refuses ~9% of harmful-content traces → kept reference-only.

---

## Family 6 — Defenses (training + inference) and the mechanism slopegraph

**Artifact:** `fig:training_slopegraph` (+ feeds `tab:defenses_asr_orr`,
`fig:defense_pareto`, `tab:inference_defense_suppression`).

**Training defenses** (`pipeline/defenses/`), one checkpoint per base model →
`models/final/<METHOD>-<base>`:

| Method | Recipe | Data |
|---|---|---|
| STAR-1 | SFT | `UCSC-VLAA/STAR-1` |
| SafeKey | SFT + safety-head / key-sentence aux | mixed SFT |
| R1-ACT | LoRA SFT (959 ex.) | in-repo |
| ThinkSafe | SFT (SafeChain) | `UWNSL/SafeChain` |
| STAIR | SFT → MCTS → DPO (×N) | `thu-ml/STAIR-SFT` |
| RAPO | SFT → on-policy GRPO | WildJailbreak + FalseReject |

**Inference defenses** (`pipeline/defenses/`, decode-time wrappers, no training):
SafePath-ZS (safety-primer prefix), SafeRemind (entropy-triggered reminder),
PSR (periodic self-reflection + backtrack).

**Slopegraph:** **[CPU]** `fig_training_slopegraph` plots, per defense cell,
ΔASR-coloured trajectories of {first-token AUROC, oscillation rate, meaningful
oscillation} from base → trained, with on-/off-policy banding.

**Replot tier needs:** `defense_first_token_auroc.json`, `defense_asr_orr_table_soft.json`,
and the trained/base stance `majority3.jsonl` (oscillation counts).

**Gotchas:** R1-ACT backfires (AUROC↓, oscillation↑); RAPO is silently on-policy
(SFT→GRPO); STAIR GPT-OSS needs native-channel extraction; first-token AUROC is
State-A (prompt-only, pre-decode).

---

*Source provenance (original research-tree paths for each script) is preserved in
the header docstring of the corresponding `reproduce/` and `pipeline/` module.*
