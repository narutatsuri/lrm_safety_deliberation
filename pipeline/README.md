# Pipeline — the from-scratch path

This directory holds the heavy GPU / training drivers that *produce* the
intermediate artifacts the `reproduce/` scripts consume. They are the original
research drivers, moved here and lightly adapted to import the shared
`lrm_safety_deliberation` library (model registry, 4-guardrail classifiers, data splits,
stats) instead of re-declaring those pieces. The logic is unchanged.

> You do **not** need to run any of this to regenerate the paper's figures and
> tables — `make artifacts` downloads the intermediate artifacts and the
> `reproduce/` scripts run on CPU in minutes. The pipeline is here for full
> from-scratch reproduction (GPU-weeks).

## Stages

| Stage | Dir | Produces | Hardware |
|---|---|---|---|
| ASR/ORR evaluation | `eval/` | `eval_results/asr_orr_16k_K4/`, `eval_results/defenses_eval/` | GPU |
| Representations | `representations/` | `eval_results/neurips_final/<m>/representations/`, `…/valley/`, kshop caches | GPU + CPU |
| Extended thinking (K=32) | `extended_thinking/` | `eval_results/k32_full_pool/`, `eval_results/k32_nested_branching_full/` | GPU |
| Causal cuts | `causal_cuts/` | `experiments/causal_cuts/…/rollouts_segment*.jsonl`, `m4_allcuts.jsonl` | GPU + API |
| IAA annotators | `iaa/` | `experiments/cot_audit/outputs/iaa_n500_new/` | API |
| Defenses | `defenses/` | `models/final/<METHOD>-<base>` (training) / decode-time wrappers (inference) | GPU |
| Cluster submission | `slurm/` | — | SLURM |

Each stage has its own `README.md` with the exact run order, commands, inputs,
and outputs. The recommended order to build everything from scratch:

```
1. eval/                 # base + defense ASR/ORR (also yields the generations reused below)
2. representations/      # extract hidden states, compute valley / probe caches
3. extended_thinking/    # K=32 think/no-think + nested-branching
4. defenses/             # train the 6 training defenses (inference defenses need no training)
5. causal_cuts/          # M=4 sample -> judges -> K=100 cut-replay -> oscillation cache
6. iaa/                  # 500-trace sample -> 3-annotator panel -> agreement
```

## What moved into the library

These were duplicated across the original drivers and now live once in
`lrm_safety_deliberation/`; the moved drivers import them:

* model registry + decoding (`lrm_safety_deliberation.models`)
* the four guardrail classifiers + majority/soft vote (`lrm_safety_deliberation.guardrails`)
* benchmark manifests + ASR/ORR splits (`lrm_safety_deliberation.data`)
* bootstrap / Fisher+Holm / κ statistics (`lrm_safety_deliberation.stats`)
* paths (`lrm_safety_deliberation.paths`; set `LRM_SAFETY_ARTIFACTS` to choose where outputs land)

See `docs/ARTIFACT_MAP.md` for the full per-artifact lineage.
