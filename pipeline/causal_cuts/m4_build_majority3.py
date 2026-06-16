#!/usr/bin/env python3
"""Build majority-of-3 labels + detect oscillation cuts for the M=4 sample.

Adapts repro_build_majority3.py: Sonnet-4.6 takes Opus's slot (3rd annotator +
tie-breaker), keyed on (cell, split, gid, rollout_idx). Then runs
salient_transitions to detect oscillation cuts per trace and emits a per-trace
mode + the cut list for the K=100 generation stage.

Inputs:
  experiments/causal_cuts/rebuttal/m4_sample.jsonl
  experiments/cot_audit/outputs/m4/{gpt54,gemini,sonnet46}.jsonl
Outputs:
  experiments/causal_cuts/rebuttal/m4_majority3.jsonl   (per-trace majority labels)
  experiments/causal_cuts/rebuttal/m4_cuts.jsonl        (per detected cut, for K=100 gen)
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/m4_build_majority3.py
# salient_transitions is imported from the original generate_cuts.py via ROOT;
# the 4-guardrail vote + model registry now also live in the shared `lrm_safety_deliberation`
# library (lrm_safety_deliberation.guardrails / lrm_safety_deliberation.models). Set LRM_SAFETY_ARTIFACTS
# to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.causal_cuts.generate_cuts import salient_transitions

SAMPLE = ROOT / "experiments/causal_cuts/rebuttal/m4_sample.jsonl"
GPT = ROOT / "experiments/cot_audit/outputs/m4/gpt54.jsonl"
GEM = ROOT / "experiments/cot_audit/outputs/m4/gemini.jsonl"
SONNET = ROOT / "experiments/cot_audit/outputs/m4/sonnet46.jsonl"
OUT_MAJ = ROOT / "experiments/causal_cuts/rebuttal/m4_majority3.jsonl"
OUT_CUTS = ROOT / "experiments/causal_cuts/rebuttal/m4_cuts.jsonl"
L_PRE = L_MIN = 1


def load_labels(path):
    out = {}
    if not path.exists():
        return out
    for ln in open(path):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "labels" not in d:
            continue
        k = (d["cell"], d["split"], int(d["global_id"]), int(d.get("rollout_idx", -1)))
        out[k] = [int(x) for x in d["labels"]]
    return out


def main():
    sample = {}
    for ln in open(SAMPLE):
        r = json.loads(ln)
        k = (r["cell"], r["split"], int(r["global_id"]), int(r.get("rollout_idx", -1)))
        sample[k] = r
    print(f"[sample] {len(sample)} traces")

    gpt, gem, son = load_labels(GPT), load_labels(GEM), load_labels(SONNET)
    print(f"[labels] gpt={len(gpt)}, gemini={len(gem)}, sonnet={len(son)}")

    n_full = n_2 = n_1 = n_skip = 0
    maj_rows = []
    for k, row in sample.items():
        nc = int(row["n_chunks"])
        g, m, s = gpt.get(k), gem.get(k), son.get(k)
        # length-mismatch -> drop that annotator
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
            elif s is not None:          # tie -> Sonnet (Opus's old role)
                maj.append(int(s[i])); prov.append("tie_to_sonnet")
            else:
                maj.append(None); prov.append("unresolved")
        n_full += len(available) == 3
        n_2 += len(available) == 2
        n_1 += len(available) == 1
        maj_rows.append({
            "cell": k[0], "split": k[1], "global_id": k[2], "rollout_idx": k[3],
            "model": row.get("base_model", row.get("cell")),
            "benchmark": row.get("benchmark", ""), "prompt": row.get("prompt", ""),
            "n_chunks": nc, "chunks": row["chunks"], "labels": maj,
            "labels_gpt54": g, "labels_gemini": m, "labels_sonnet": s,
            "annotators_used": available, "chunk_provenance": prov,
        })
    OUT_MAJ.write_text("\n".join(json.dumps(r) for r in maj_rows) + "\n")
    print(f"[write] {OUT_MAJ.relative_to(ROOT)}: 3-annot={n_full}, 2-annot={n_2}, 1-annot={n_1}, skipped={n_skip}")

    # Cut detection on majority labels (None -> 0/neutral for transition logic)
    cut_rows = []
    n_with_cuts = 0
    for r in maj_rows:
        labels = [x if x is not None else 0 for x in r["labels"]]
        cuts = salient_transitions(labels, L_PRE, L_MIN)
        if cuts:
            n_with_cuts += 1
        for (t_idx, s_pre, pre_end, pre_len, s_post, post_end, post_len) in cuts:
            cut_rows.append({
                "cell": r["cell"], "split": r["split"], "global_id": r["global_id"],
                "rollout_idx": r["rollout_idx"], "model": r["model"],
                "benchmark": r["benchmark"], "prompt": r["prompt"],
                "n_chunks": r["n_chunks"], "chunks": r["chunks"],
                "transition_idx": t_idx,
                "s_pre_sign": s_pre, "s_pre_end_chunk": pre_end, "s_pre_len": pre_len,
                "s_post_sign": s_post, "s_post_end_chunk": post_end, "s_post_len": post_len,
                "pre_cut_chunk_idx": pre_end + 1, "post_cut_chunk_idx": post_end + 1,
            })
    OUT_CUTS.write_text("\n".join(json.dumps(r) for r in cut_rows) + "\n")
    print(f"[write] {OUT_CUTS.relative_to(ROOT)}: {len(cut_rows)} cuts across "
          f"{n_with_cuts} traces ({100*n_with_cuts/len(maj_rows):.1f}% of traces have >=1 cut)")


if __name__ == "__main__":
    main()
