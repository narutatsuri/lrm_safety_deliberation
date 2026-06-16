#!/usr/bin/env python3
"""Majority-of-3 chunk labels for the M=4 inference-defense oscillation audit.

3 annotators = GPT-5.4 + Gemini-3-Pro + Sonnet-4.6 (Sonnet replaces Opus;
[[feedback-no-opus-default]]). Keyed on (cell, split, gid, rollout_idx).

Per chunk:
  - 2 or 3 annotators agree -> majority label.
  - all 3 disagree (1-1-1) -> None (unresolved); the cut/Δq pipeline drops any
    trace containing an unresolved chunk, identical to the base m4 audit.

Inputs:
  experiments/causal_cuts/rebuttal/m4_inference_defense_sample.jsonl     (chunks)
  experiments/cot_audit/outputs/m4_inference_defenses/{gpt54,gemini,sonnet46}.jsonl

Output:
  experiments/causal_cuts/rebuttal/m4_inference_defense_majority3.jsonl
"""
from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/m4_inference_defense_build_majority3.py
# The 3-way stance majority here is the judge panel; the 4-guardrail refusal vote
# + model registry now also live in the shared `lrm_safety_deliberation` library
# (lrm_safety_deliberation.guardrails / lrm_safety_deliberation.models). Set LRM_SAFETY_ARTIFACTS to
# relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
SAMPLE = ROOT / "experiments/causal_cuts/rebuttal/m4_inference_defense_sample.jsonl"
ODIR = ROOT / "experiments/cot_audit/outputs/m4_inference_defenses"
GPT = ODIR / "gpt54.jsonl"
GEM = ODIR / "gemini.jsonl"
SON = ODIR / "sonnet46.jsonl"
OUT = ROOT / "experiments/causal_cuts/rebuttal/m4_inference_defense_majority3.jsonl"


def load_labels(path):
    out = {}
    if not path.exists():
        return out
    for ln in open(path):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "labels" not in d or d["labels"] is None:
            continue
        k = (d["cell"], d["split"], int(d["global_id"]), int(d.get("rollout_idx", -1)))
        try:
            out[k] = [int(x) for x in d["labels"]]
        except Exception:
            continue
    return out


def main():
    sample = {}
    with open(SAMPLE) as f:
        for ln in f:
            r = json.loads(ln)
            k = (r["cell"], r["split"], int(r["global_id"]), int(r.get("rollout_idx", -1)))
            sample[k] = r
    print(f"[sample] {len(sample)} traces")

    gpt = load_labels(GPT)
    gem = load_labels(GEM)
    son = load_labels(SON)
    print(f"[labels] gpt={len(gpt)} gem={len(gem)} sonnet={len(son)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_full = n_2 = n_skip = 0
    n_unresolved_chunks = 0
    with open(OUT, "w") as fout:
        for k, row in sample.items():
            nc = int(row["n_chunks"])
            g, m, s = gpt.get(k), gem.get(k), son.get(k)
            # length-mismatch -> drop that annotator for this trace
            if g is not None and len(g) != nc: g = None
            if m is not None and len(m) != nc: m = None
            if s is not None and len(s) != nc: s = None
            available = [n for n, lab in [("gpt", g), ("gemini", m), ("sonnet", s)] if lab is not None]
            if not available:
                n_skip += 1
                continue

            maj, prov = [], []
            for i in range(nc):
                votes = [lab[i] for lab in (g, m, s) if lab is not None]
                cnt = Counter(votes)
                top, top_c = cnt.most_common(1)[0]
                if top_c >= 2:
                    maj.append(int(top)); prov.append("majority")
                else:
                    maj.append(None); prov.append("unresolved")
                    n_unresolved_chunks += 1

            if len(available) == 3:
                n_full += 1
            else:
                n_2 += 1

            fout.write(json.dumps({
                "cell": k[0], "split": k[1], "global_id": k[2], "rollout_idx": k[3],
                "model": row.get("base_model", row.get("cell")),
                "defense": row.get("defense", ""),
                "benchmark": row.get("benchmark", ""),
                "prompt": row.get("prompt", ""),
                "n_chunks": nc,
                "chunks": row["chunks"],
                "labels": maj,
                "labels_gpt54": g,
                "labels_gemini": m,
                "labels_sonnet46": s,
                "annotators_used": available,
                "chunk_provenance": prov,
                "force_closed": row.get("force_closed", False),
            }, ensure_ascii=False) + "\n")

    print(f"[write] {OUT.relative_to(ROOT)}")
    print(f"  3-annotator: {n_full}, 2-annotator: {n_2}, skipped(no labels): {n_skip}")
    print(f"  unresolved chunks (all-3-disagree): {n_unresolved_chunks}")


if __name__ == "__main__":
    main()
