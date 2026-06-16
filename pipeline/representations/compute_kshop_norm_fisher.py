#!/usr/bin/env python3
"""Compute the canonical norm_fisher (whole-space, per-position z-scored)
on the K-hop user-content artifacts, per (layer, k anchor, pool).

Same formula as scripts/defenses_eval/compute_valley_and_auroc.py:50:
  Per anchor: standardize each dim across all samples; then
    Fisher = ||μ+ − μ−||² / (tr(Σ+) + tr(Σ−))

Output: experiments/refusal_cliff/results/kshop_K20_user_content/norm_fisher_<MODEL>.json

Provenance: copied (logic unchanged) from scripts/neurips_final/compute_kshop_norm_fisher.py.
The shared model registry / guardrail vote now live in the `lrm_safety_deliberation` library
(`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`); the MODELS list below is a local
echo. Set LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
ART = ROOT / "experiments/refusal_cliff/artifacts/kshop_K20_user_content"
OUT = ROOT / "experiments/refusal_cliff/results/kshop_K20_user_content"
MODELS = ["Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning", "GPT-OSS-20B"]


def norm_fisher_one(X_slice: np.ndarray, y: np.ndarray) -> float:
    """X_slice: (N, D) at one (layer, k). y: (N,) {0, 1}."""
    if X_slice.shape[0] < 4:
        return float("nan")
    mu = X_slice.mean(0)
    sd = X_slice.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Xs = (X_slice - mu) / sd
    pos = Xs[y == 1]; neg = Xs[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    mup = pos.mean(0); mun = neg.mean(0)
    between = float(((mup - mun) ** 2).sum())
    vp = float(((pos - mup) ** 2).sum(axis=1).mean())
    vn = float(((neg - mun) ** 2).sum(axis=1).mean())
    denom = vp + vn
    return between / denom if denom > 1e-12 else 0.0


def process_model(model: str) -> dict:
    print(f"=== {model} ===", flush=True)
    out = {"model": model, "pools": {}}
    for intent, pool_label in (
        ("harmful", "harmful"),
        ("benign", "benign_main_only"),
        ("phtest", "phtest"),
        ("orfuzz", "orfuzz"),
    ):
        path = ART / model / f"{intent}.pt"
        if not path.exists():
            print(f"  [skip {pool_label}] artifact missing"); continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        X = d["X"].to(torch.float32).numpy()  # (N, K, L, D)
        y = d["y"].numpy().astype(np.int64)
        K_total = X.shape[1]; n_layers = X.shape[2]
        layer_indices = d["layer_indices"].tolist()
        if "n_layers_sampled" in d:
            pass
        N = X.shape[0]
        n_pos = int((y == 1).sum())
        print(f"  [{pool_label}] N={N} pos={n_pos} K={K_total} L={n_layers}", flush=True)
        t0 = time.time()
        fisher = np.full((n_layers, K_total), np.nan)
        for li in range(n_layers):
            for k in range(K_total):
                fisher[li, k] = norm_fisher_one(X[:, k, li, :], y)
        print(f"    norm_fisher computed in {time.time()-t0:.1f}s  "
              f"max={float(np.nanmax(fisher)):.4f}", flush=True)
        out["pools"][pool_label] = {
            "n": N, "n_pos": n_pos, "pos_rate": float(n_pos / N),
            "layer_indices": layer_indices,
            "K_total": K_total,
            "norm_fisher": fisher.tolist(),
        }
        del X
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        res = process_model(m)
        out_p = OUT / f"norm_fisher_{m}.json"
        out_p.write_text(json.dumps(res, indent=2))
        print(f"  wrote {out_p}")


if __name__ == "__main__":
    main()
