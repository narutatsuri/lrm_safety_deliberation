# Extended-thinking pipeline (K=32 generation + nested-branching + classification)

From-scratch GPU drivers that produce the K=32 extended-thinking rollouts, the
B-budget nested-branching prefix truncation/extension cells, the 4-guardrail
safety classifications, and the CPU aggregation that yields
`asr_orr_within_by_cut.json`.

These are copied verbatim (logic unchanged) from the original research repo
`experiments/nested_branching_variance/k32_full/`. The only edits are: the
hard-coded `ROOT` was replaced with
`Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[N]))`,
and a provenance header was added to each entry point.

> **Supersession note.** The model registry (originally
> `scripts.neurips_final.common.MODELS`) and the 4-guardrail majority vote now
> live in the `lrm_safety_deliberation` library (`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`).
> The drivers here still import the original research modules
> (`scripts.neurips_final.common`, `evaluation.full_eval_pipeline.*`,
> `evaluation.base`, `evaluation.classifiers`) through `ROOT`, because they were
> calibrated against those exact code paths. When reconciling the vote/registry,
> treat the `lrm_safety_deliberation` copies as canonical — the inline copies of the vote logic
> (`_vote`, `soft_score`, `majority_refuse`) and the `GUARDRAIL_PATHS`/`MODELS`
> dicts in these files are the legacy snapshots.

## ROOT resolution

`ROOT` is the original research-repo tree that holds the library modules
(`scripts/`, `evaluation/`), the input split manifests, and the artifact outputs
(`eval_results/`). With no environment variable set, the default
`Path(__file__).resolve().parents[N]` resolves to the original repo root, so
in-place verification "just works". To run against a different tree (e.g. the
downloadable HuggingFace artifact bundle), set:

```bash
export LRM_SAFETY_ARTIFACTS=/path/to/research_repo_root
```

GPU drivers (`generate.py`, `nested_branching.py`, `classify.py`,
`classify_nested.py`, `apply_16k_rule.py`) require vLLM + a GPU. The two analysis
scripts under `analysis/` are CPU-only.

## Run order

```
1. generate.py        (mode=think AND mode=nothink)        — K=32 rollouts per prompt
2. apply_16k_rule.py  (think cells only; optional rerun)   — 16K-think + force-close protocol
3. nested_branching.py                                     — B-budget prefix truncation/extension
4. classify.py        (think + nothink K=32 cells)         — 4-guardrail classification
   classify_nested.py (nested-branching cut cells)         — 4-guardrail classification
5. analysis/aggregate_k32_nb.py                            — asr_orr_within_by_cut.json
   analysis/transition_table_k32.py                        — nothink->think transition table
```

The 4 guardrails are: `wildguard`, `qwen3guard`, `granite_guardian`,
`oss_safeguard`. A cell's vote is the soft mean of the 4 bool refusal votes per
rollout (and majority = >=3/4 for the within-prefix variance).

## Inputs

Full eval-split prompt manifests (read via the original pipeline's
`read_jsonl`):

- `evaluation/full_eval_pipeline/data/asr.jsonl`  (ASR pool, 2500 prompts)
- `evaluation/full_eval_pipeline/data/orr.jsonl`  (ORR pool, 2885 prompts)

(In the release these correspond to the `data/` benchmark manifests behind the
`asr` / `orr` eval splits.)

## Outputs

- K=32 rollouts:        `eval_results/k32_full_pool/<model>/<mode>/<split>/shard_*.jsonl`
- Nested-branching:     `eval_results/k32_nested_branching_full/<model>/<split>/cut{B}/shard_*.jsonl`
- Classifications:      `…/classifications_<guardrail>.jsonl` next to each cell's shards
- Aggregate:            `experiments/nested_branching_variance/k32_full/analysis/data/asr_orr_within_by_cut.json`
- Transition table:     `eval_results/k32_full_pool/analysis/transition_table_k32_nothink_vs_think.{json,csv}`

(Output paths are anchored at `ROOT`; with `LRM_SAFETY_ARTIFACTS` set, the same
relative subtrees are written under that root.)

## Models / splits / cuts

- `--model`     `Qwen3-8B` | `Olmo-3-7B-Think` | `Phi-4-reasoning` | `GPT-OSS-20B`
- `--split`     `asr` | `orr`
- `--mode`      `think` | `nothink`   (generation / K=32 classification only)
- `--guardrail` `wildguard` | `qwen3guard` | `granite_guardian` | `oss_safeguard`
- `--cuts`      comma-separated B-budgets. `<100` = truncate thinking at B%;
                `100` = natural close; `>100` = force-extend by r=(B-100)/100.
                Main run: `20,40,60,80,120`. Extension sweep: `140,160,180,200`.
                B=0 (nothink) and B=100 (natural think) come from the K=32 pool.

## Exact commands

All GPU drivers expect the repo on `PYTHONPATH` and a single H100/A100. The
`.sh` wrappers (below) set this up for SLURM (`--partition=pli-c`); the bare
`python` invocations below are the local-node equivalents.

### 1. K=32 generation (run for both modes, both splits)

```bash
# one shard of K=32 rollouts (sharded for SLURM; n_shards=1 runs the whole split)
python generate.py \
    --model Qwen3-8B --split asr --mode think \
    --shard_idx 0 --n_shards 1 --K 32
python generate.py \
    --model Qwen3-8B --split asr --mode nothink \
    --shard_idx 0 --n_shards 1 --K 32
```

Optional (think cells): apply the universal 16K-think + force-close protocol:

```bash
python apply_16k_rule.py \
    --model Qwen3-8B --mode think --split asr \
    --shard_idx 0 --n_shards 1 --K 32
```

### 2. Nested branching (reads the K=32 think pool, READ-ONLY)

```bash
python nested_branching.py \
    --model Qwen3-8B --split asr \
    --shard_idx 0 --n_shards 1 \
    --M 8 --N 8 --cuts 20,40,60,80,120
# extension sweep (cut>100):  --cuts 140,160,180,200
```

### 3. Classification (4 guardrails per cell)

```bash
# K=32 think / nothink cells
python classify.py \
    --model Qwen3-8B --mode think --split asr --guardrail wildguard

# nested-branching cut cells (iterates all --cuts with one vLLM load)
python classify_nested.py \
    --model Qwen3-8B --split asr --guardrail wildguard \
    --cuts 20,40,60,80,120
```

Repeat for each of the 4 guardrails (`wildguard`, `qwen3guard`,
`granite_guardian`, `oss_safeguard`).

### 4. Aggregate (CPU)

```bash
python analysis/aggregate_k32_nb.py        # -> analysis/data/asr_orr_within_by_cut.json
python analysis/transition_table_k32.py    # -> transition_table_k32_nothink_vs_think.{json,csv}
```

## SLURM wrappers / dispatchers

GPU job scripts (`#SBATCH ... --partition=pli-c --gres=gpu:1`):

- `slurm_job.sh`            — wraps `generate.py`
- `slurm_job_16k.sh`        — wraps `apply_16k_rule.py`
- `slurm_job_nested.sh`     — wraps `nested_branching.py`
- `classify_job.sh`         — wraps `classify.py`
- `classify_job_nested.sh`  — wraps `classify_nested.py`
- `resample_stragglers_job.sh` — re-runs `generate.py` with a high resample budget

Dispatchers (submit one sbatch per cell/shard, with `afterok` dependencies):

- `submit_all.sh`, `submit_rework.sh`, `submit_16k.sh`       — generation
- `submit_nested_branching.sh`, `submit_extension_sweep.sh`  — nested branching
- `submit_classify_deps.sh`, `submit_classify_deps_16k.sh`,
  `submit_classify_nested.sh`, `submit_extension_classify.sh` — classification

The dispatchers reference the original repo's absolute `ROOT` and write SLURM
manifests under `.slurm/`; they are included as a record of the exact run plan.
Adjust `ROOT`/paths if running outside the original tree.

> The plotting scripts (`analysis/plot_*.py` in the original tree) are **not**
> copied here — the figures live in `camera_ready/reproduce/`.

## Downstream figures (in `camera_ready/reproduce/`)

The outputs above feed:

- `reproduce/fig_extended_thinking.py`        (uses `asr_orr_within_by_cut.json`)
- `reproduce/fig_nothink_to_think.py` and `fig_nothink_to_think_lenient.py`
  (use the K=32 nothink/think classifications + transition table)
- `reproduce/fig_within_prefix_variance.py`   (uses the within-prefix variance in
  `asr_orr_within_by_cut.json`)
