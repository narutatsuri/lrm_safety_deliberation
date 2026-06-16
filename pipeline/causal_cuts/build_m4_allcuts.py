#!/usr/bin/env python3
"""Build the per-cut cache m4_allcuts.jsonl: ONE row per detected cut, across all
M=4 rollouts (rollout-0 from the base study + 3 new k32 rollouts) and all 4 base
models. Reads the giant K=100 rollout/guardrail JSONLs exactly ONCE so downstream
plots can iterate fast instead of re-parsing ~5M lines each.

Row schema (one per cut = a (model, split, gid, rollout, tidx) tuple):
  model, split, gid, rollout, tidx,
  comply_pre, comply_post,                         # majority-vote comply fractions over K=100
  n_comply_pre, n_pre, n_comply_post, n_post,      # raw counts (null for rollout-0 source)
  s_pre_len, s_post_len, s_post_sign, score,       # segment geometry (score = s_pre_len+s_post_len)
  fisher_p,                                         # uncorrected; rollout-0 uses precomputed value
  probe_score,                                      # first-token P(refuse) for the prompt (may be null)
  source                                            # "rollout0" | "new"

Holm vs uncorrected significance is left to downstream consumers (they have fisher_p).

Sources:
  rollout-0 cuts:  experiments/causal_cuts/rebuttal/sig_cuts_enriched.jsonl
  new cuts:        experiments/causal_cuts/eval_results/causal_cuts/<model>/<split>/
                       rollouts_segment_m4.jsonl + guardrails_segment_m4__<gr>.jsonl
  probe scores:    experiments/causal_cuts/rebuttal/osc_predictors_per_trace.jsonl
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import fisher_exact

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/rebuttal/build_m4_allcuts.py
# extract_guardrail_vote (imported from scripts.neurips_s16.common) is superseded
# by the shared `lrm_safety_deliberation` library (`lrm_safety_deliberation.guardrails.guardrail_vote`);
# this driver still imports the original repo module via ROOT so it runs
# standalone. Set LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from scripts.neurips_s16.common import extract_guardrail_vote

REB = ROOT / "experiments/causal_cuts/rebuttal"
CUTS_ROOT = ROOT / "experiments/causal_cuts/eval_results/causal_cuts"
OUT = REB / "m4_allcuts.jsonl"

MODELS = ["GPT-OSS-20B", "Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning"]
GUARDRAILS = ["wildguard", "qwen3guard", "granite_guardian", "oss_safeguard"]


def fisher_p(ncp, np_, ncq, nq):
    try:
        _, p = fisher_exact([[ncp, np_ - ncp], [ncq, nq - ncq]])
        return float(p)
    except Exception:
        return 1.0


def load_probe():
    probe = {}
    for ln in open(REB / "osc_predictors_per_trace.jsonl"):
        d = json.loads(ln)
        if d.get("probe_score") is not None:
            probe[(d["model"], d["split"], int(d["global_id"]))] = float(d["probe_score"])
    return probe


def rollout0_cuts(model):
    """All rollout-0 cuts for a model from sig_cuts_enriched (counts unavailable)."""
    out = []
    for ln in open(REB / "sig_cuts_enriched.jsonl"):
        d = json.loads(ln)
        if d["model"] != model:
            continue
        s_pre = d.get("s_pre_len_chunks", 0)
        s_post = d.get("s_post_len_chunks", 0)
        out.append(dict(
            model=model, split=d["split"], gid=int(d["global_id"]),
            rollout=None, tidx=int(d["transition_idx"]),
            comply_pre=d["comply_rate_pre"], comply_post=d["comply_rate_post"],
            n_comply_pre=None, n_pre=None, n_comply_post=None, n_post=None,
            s_pre_len=s_pre, s_post_len=s_post, score=s_pre + s_post,
            s_post_sign=d.get("s_post_sign", 0),
            fisher_p=float(d.get("fisher_p", 1.0)), source="rollout0",
        ))
    return out


def new_cuts(model, split):
    """All new-rollout cuts (one per gid/trace_rollout_idx/transition_idx) from the
    K=100 rollout + guardrail files. Returns None if data not present."""
    rp = CUTS_ROOT / model / split / "rollouts_segment_m4.jsonl"
    if not rp.exists():
        return None
    by_key = {}
    seg = {}  # (gid, tri, tidx) -> (s_pre_len, s_post_len, s_post_sign)
    for ln in open(rp):
        r = json.loads(ln)
        if r.get("skipped"):
            continue
        k = (r["global_id"], r["trace_rollout_idx"], r["transition_idx"], r["cut_kind"], r["rollout_idx"])
        by_key[k] = r
        tk = (r["global_id"], r["trace_rollout_idx"], r["transition_idx"])
        seg[tk] = (r.get("s_pre_len_chunks", 0), r.get("s_post_len_chunks", 0), r.get("s_post_sign", 0))
    for gr in GUARDRAILS:
        p = CUTS_ROOT / model / split / f"guardrails_segment_m4__{gr}.jsonl"
        if not p.exists():
            return None
        for ln in open(p):
            r = json.loads(ln)
            if r.get("skipped"):
                continue
            k = (r["global_id"], r["trace_rollout_idx"], r["transition_idx"], r["cut_kind"], r["rollout_idx"])
            if k in by_key:
                by_key[k].setdefault("g", {})[gr] = r.get(gr)
    # majority vote per K=100 rollout -> comply(1)/refuse(0)
    grouped = defaultdict(list)
    for k, r in by_key.items():
        votes = {}
        for gr in GUARDRAILS:
            pay = r.get("g", {}).get(gr)
            if isinstance(pay, dict):
                try:
                    votes[gr] = extract_guardrail_vote(gr, pay)
                except KeyError:
                    pass
        if not votes:
            continue
        is_ref = (sum(votes.values()) >= 2) if len(votes) == 4 else (sum(votes.values()) > len(votes) / 2)
        grouped[(r["global_id"], r["trace_rollout_idx"], r["transition_idx"], r["cut_kind"])].append(0 if is_ref else 1)
    # pair pre/post
    bun = defaultdict(dict)
    for (gid, tri, tidx, kind), labs in grouped.items():
        bun[(gid, tri, tidx)][kind] = (sum(labs), len(labs))
    out = []
    for (gid, tri, tidx), b in bun.items():
        if "pre" not in b or "post" not in b:
            continue
        cp, np_ = b["pre"]
        cq, nq = b["post"]
        s_pre, s_post, s_post_sign = seg.get((gid, tri, tidx), (0, 0, 0))
        out.append(dict(
            model=model, split=split, gid=int(gid),
            rollout=int(tri), tidx=int(tidx),
            comply_pre=cp / np_, comply_post=cq / nq,
            n_comply_pre=cp, n_pre=np_, n_comply_post=cq, n_post=nq,
            s_pre_len=s_pre, s_post_len=s_post, score=s_pre + s_post,
            s_post_sign=s_post_sign, fisher_p=fisher_p(cp, np_, cq, nq), source="new",
        ))
    return out


def main():
    probe = load_probe()
    rows = []
    summary = {}
    for m in MODELS:
        r0 = rollout0_cuts(m)
        n_new_total = 0
        new_ready = True
        per_split = {}
        for split in ("harmful", "benign"):
            nc = new_cuts(m, split)
            if nc is None:
                new_ready = False
                per_split[split] = None
                continue
            per_split[split] = len(nc)
            n_new_total += len(nc)
            rows.extend(nc)
        rows.extend(r0)
        summary[m] = dict(rollout0=len(r0), new=n_new_total if new_ready else None, per_split=per_split)
    # attach probe scores
    for r in rows:
        r["probe_score"] = probe.get((r["model"], r["split"], r["gid"]))
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    n_probe = sum(1 for r in rows if r["probe_score"] is not None)
    print(f"[write] {OUT.relative_to(ROOT)}: {len(rows)} cut rows  "
          f"({n_probe} with probe_score, {len(rows)-n_probe} null)")
    for m in MODELS:
        s = summary[m]
        print(f"  {m:<18} rollout0={s['rollout0']:>4}  new={s['new']}  per_split={s['per_split']}")


if __name__ == "__main__":
    main()
