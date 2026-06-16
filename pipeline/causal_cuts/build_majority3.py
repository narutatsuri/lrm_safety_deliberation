#!/usr/bin/env python3
"""Build chunk-level majority-of-3 stance labels.

For each trace in sample_n2400.jsonl with all 3 annotations available
(GPT-5.4 from stance_audit_full + Gemini-3-Pro + Opus chain), emit a row
with majority-of-3 labels. Tie-break: GPT-5.4.

For each trace, also flag any "label_drift" — i.e., chunks where majority
differs from GPT-5.4 — and report aggregate stats.

Output: experiments/causal_cuts/eval_results/stance_audit_full/labeled_traces_majority3.jsonl

For traces NOT in the n=2390 subset, the file does NOT include them — the
analysis script should fall back to GPT-5.4-only for those.
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/build_majority3.py
# The 3-way stance majority vote here is the stance-judge panel; the 4-guardrail
# refusal vote + model registry now also live in the shared `lrm_safety_deliberation` library
# (lrm_safety_deliberation.guardrails / lrm_safety_deliberation.models). Set LRM_SAFETY_ARTIFACTS to
# relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
# Defaults — can be overridden with CLI args.
SAMPLE = ROOT / "experiments/causal_cuts/rebuttal/sample_n2400.jsonl"
GEMINI = ROOT / "experiments/causal_cuts/eval_results/stance_audit_full/labeled_traces_gemini3pro.jsonl"
OPUS = ROOT / "experiments/causal_cuts/eval_results/stance_audit_full/labeled_traces_opus_chain.jsonl"
OUT = ROOT / "experiments/causal_cuts/eval_results/stance_audit_full/labeled_traces_majority3.jsonl"


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--gemini", default=str(GEMINI))
    ap.add_argument("--opus", default=str(OPUS))
    ap.add_argument("--out", default=str(OUT))
    return ap.parse_args()


def load_labels(path: Path, allow_partial=False) -> dict:
    """Return {(key, split, gid): labels} where key is registered under BOTH
    `cell` and `base_model` names (handles Olmo3-7B-Think vs Olmo-3-7B-Think
    naming drift between the audit's 'model' field and the canonical
    base_model used in sample files)."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if "labels" not in d:
                continue
            labels = [int(x) for x in d["labels"]]
            split = d["split"]
            gid = int(d["global_id"])
            # Register under both names so the merge can look up either way.
            for name in (d.get("cell"), d.get("base_model")):
                if name:
                    out[(name, split, gid)] = labels
    return out


def main():
    args = parse_args()
    sample_path = Path(args.sample)
    gemini_path = Path(args.gemini)
    opus_path = Path(args.opus)
    out_path = Path(args.out)

    # GPT-5.4 labels are embedded in the sample file as gpt54_labels
    sample = {}
    with open(sample_path) as f:
        for ln in f:
            d = json.loads(ln)
            key = (d["base_model"], d["split"], int(d["global_id"]))
            sample[key] = d
    print(f"[sample] {len(sample)} traces in n=2390 subset")

    # Gemini labels — keyed by (cell, split, gid). Note cell here can be the
    # display name (Olmo-3-7B-Think); base_model field aligns to sample.
    gem = load_labels(gemini_path)
    opus = load_labels(opus_path)
    print(f"[gemini] {len(gem)} labeled traces")
    print(f"[opus]   {len(opus)} labeled traces")

    # The Gemini/Opus outputs use cell == base_model (sample passes base_model
    # as both fields in our sampler). Verify and normalize.
    def get_lab(d_map, key):
        v = d_map.get(key)
        if v is not None:
            return v
        # try cell-only
        cell, split, gid = key
        return d_map.get((cell, split, gid))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_full = 0  # 3-annotator majority
    n_2of3 = 0  # 2-annotator (missing one)
    n_skip = 0
    label_agree = Counter()  # how each pair agrees with majority
    cell_agree = defaultdict(lambda: Counter())

    with open(out_path, "w") as fout:
        for key, sample_row in sample.items():
            gpt = sample_row["gpt54_labels"]
            n_chunks = sample_row["n_chunks"]
            if len(gpt) != n_chunks:
                # rare label-count mismatch flagged earlier; skip
                n_skip += 1
                continue
            gem_lab = get_lab(gem, key)
            opus_lab = get_lab(opus, key)
            cell = sample_row["base_model"]
            split = sample_row["split"]

            # If gem or opus label count mismatches n_chunks, fall back
            if gem_lab is not None and len(gem_lab) != n_chunks:
                gem_lab = None
            if opus_lab is not None and len(opus_lab) != n_chunks:
                opus_lab = None

            if gem_lab is None and opus_lab is None:
                n_skip += 1
                continue

            available = ["gpt54"]
            if gem_lab is not None:
                available.append("gemini")
            if opus_lab is not None:
                available.append("opus")

            # Per-chunk majority — pre-registered tie-break:
            #   1. If majority exists (>=2 agree), use it.
            #   2. If 3-way disagreement (only possible with 3 annotators), use Opus 4.7.
            #   3. If only 2 annotators and they disagree, mark chunk unresolved (label=None).
            # Never default to GPT-5.4 (least-central annotator under matched protocol).
            maj_labels = []
            chunk_provenance = []
            for i in range(n_chunks):
                votes = [gpt[i]]
                if gem_lab is not None:
                    votes.append(gem_lab[i])
                if opus_lab is not None:
                    votes.append(opus_lab[i])
                cnt = Counter(votes)
                top, top_count = cnt.most_common(1)[0]
                if top_count >= 2:
                    maj_labels.append(int(top))
                    chunk_provenance.append("majority")
                elif opus_lab is not None:
                    # 3-way disagreement → fall back to Opus 4.7
                    maj_labels.append(int(opus_lab[i]))
                    chunk_provenance.append("tie_to_opus")
                else:
                    # 2 annotators present (no Opus) and they disagree → unresolved.
                    maj_labels.append(None)
                    chunk_provenance.append("unresolved_2annotator")

            if len(available) == 3:
                n_full += 1
            else:
                n_2of3 += 1

            # Track agreement stats
            for i in range(n_chunks):
                if gpt[i] == maj_labels[i]:
                    label_agree["gpt54"] += 1
                    cell_agree[(cell, split)]["gpt54"] += 1
                if gem_lab is not None and gem_lab[i] == maj_labels[i]:
                    label_agree["gemini"] += 1
                    cell_agree[(cell, split)]["gemini"] += 1
                if opus_lab is not None and opus_lab[i] == maj_labels[i]:
                    label_agree["opus"] += 1
                    cell_agree[(cell, split)]["opus"] += 1
                cell_agree[(cell, split)]["_total_chunks"] += 1

            fout.write(json.dumps({
                "model": cell,
                "split": split,
                "global_id": key[2],
                "benchmark": sample_row.get("benchmark", ""),
                "prompt": sample_row.get("prompt", "")[:200],
                "n_chunks": n_chunks,
                "labels": maj_labels,
                "labels_gpt54": gpt,
                "labels_gemini": gem_lab,
                "labels_opus": opus_lab,
                "annotators_used": available,
                "chunk_provenance": chunk_provenance,
            }) + "\n")

    print()
    print(f"[write] {out_path.resolve().relative_to(ROOT)}")
    print(f"  with 3 annotators: {n_full}")
    print(f"  with 2 annotators: {n_2of3}")
    print(f"  skipped:           {n_skip}")
    print()
    total_chunks = sum(cell_agree[k]["_total_chunks"] for k in cell_agree)
    if total_chunks > 0:
        print(f"[chunk-level agreement with majority] over {total_chunks} chunks:")
        for ann in ["gpt54","gemini","opus"]:
            n_ag = label_agree[ann]
            n_with_ann = total_chunks if ann == "gpt54" else label_agree.get(ann, 0) and total_chunks
            # actually count traces with each annotator
        # Simpler: just report per-annotator vs majority
        for ann in ["gpt54","gemini","opus"]:
            n_ag = label_agree[ann]
            pct = 100.0 * n_ag / max(total_chunks, 1)
            print(f"  {ann:<8}: {n_ag}/{total_chunks} = {pct:.1f}% match-rate vs majority")
        print()
        print(f"[per-cell match-with-majority rates]")
        print(f"  {'cell':<24} {'split':<8} {'gpt54%':>8} {'gemini%':>8} {'opus%':>8} {'n_chunks':>8}")
        for (cell, split), c in sorted(cell_agree.items()):
            n = c["_total_chunks"]
            g = 100.0 * c.get("gpt54", 0) / max(n,1)
            gm = 100.0 * c.get("gemini", 0) / max(n,1)
            op = 100.0 * c.get("opus", 0) / max(n,1)
            print(f"  {cell:<24} {split:<8} {g:>7.1f}% {gm:>7.1f}% {op:>7.1f}% {n:>8}")


if __name__ == "__main__":
    main()
