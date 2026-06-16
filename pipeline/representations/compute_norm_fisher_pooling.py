"""Compute the whole-space normalized Fisher metric on chunk-mean-pooled
thinking representations (instead of token-hopping).

For each model, loads:
  representations/thinking_pooling/{harmful,benign}.pt
  ../neurips_final_supplementary/<M>/representations/thinking_pooling/{orfuzz,phtest}.pt

Concatenates harmful, and (benign + orfuzz + phtest) for benign_combined.
Computes norm_fisher (per-position z-score, then whole-space Fisher).

Saves:
  valley/explore_pooling_{harmful,benign_combined}.pt   (only norm_fisher + y + ids)

Provenance: copied (logic unchanged) from scripts/neurips_final/compute_norm_fisher_pooling.py.
The shared model registry / guardrail vote / probe + stats helpers now live in
the `lrm_safety_deliberation` library (`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`,
`lrm_safety_deliberation.stats`). Set LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
FINAL_DIR = ROOT / "eval_results" / "neurips_final"
SUPP_DIR = ROOT / "eval_results" / "neurips_final_supplementary"


def load_harmful(model: str):
    p = FINAL_DIR / model / "representations" / "thinking_pooling" / "harmful.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["X"], d["y"], list(d.get("ids") or [])


def load_benign_combined(model: str):
    paths = [
        FINAL_DIR / model / "representations" / "thinking_pooling" / "benign.pt",
        SUPP_DIR / model / "representations" / "thinking_pooling" / "orfuzz.pt",
        SUPP_DIR / model / "representations" / "thinking_pooling" / "phtest.pt",
    ]
    Xs, ys, ids = [], [], []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        Xs.append(d["X"])
        ys.append(d["y"])
        ids.extend(list(d.get("ids") or []))
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0), ids


def whole_space_fisher_normalized(X_cpu: torch.Tensor, y_cpu: torch.Tensor) -> np.ndarray:
    """Per-position standardize each dim (zero mean, unit variance across ALL samples),
    then whole-space Fisher. Same as in explore_valley_metrics.py.
    """
    X = X_cpu.to(torch.float32).numpy()
    y = y_cpu.numpy().astype(np.int64)
    N, T, D = X.shape
    fisher = np.zeros(T)
    for t in range(T):
        Xt = X[:, t, :]
        mu = Xt.mean(0)
        sd = Xt.std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        Xt = (Xt - mu) / sd
        pos = Xt[y == 1]
        neg = Xt[y == 0]
        mup = pos.mean(0)
        mun = neg.mean(0)
        between = ((mup - mun) ** 2).sum()
        vp = ((pos - mup) ** 2).sum(axis=1).mean()
        vn = ((neg - mun) ** 2).sum(axis=1).mean()
        denom = vp + vn
        fisher[t] = between / denom if denom > 1e-12 else 0.0
    return fisher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    for model in args.models:
        print(f"\n=== {model} ===")
        for intent, loader in [("harmful", load_harmful), ("benign_combined", load_benign_combined)]:
            t0 = time.time()
            X, y, ids = loader(model)
            print(f"  [{intent}] X={tuple(X.shape)}  pos={int(y.sum())}/{len(y)}")
            nf = whole_space_fisher_normalized(X, y)
            out = FINAL_DIR / model / "valley" / f"explore_pooling_{intent}.pt"
            torch.save(
                {
                    "norm_fisher": nf,
                    "y": y.numpy().astype(np.int64),
                    "ids": ids,
                    "intent": intent,
                    "method": "chunk_mean_pool",
                },
                out,
            )
            print(f"  [{intent}] wrote {out.relative_to(ROOT)}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
