#!/usr/bin/env python3
"""Per-k (0..20) probe metrics under the BEST pipeline, final layer.

Pipeline: StandardScaler -> PCA(100) -> LogReg(C=0.03, class_weight=balanced),
5-fold OOF. BAcc uses a PER-GROUP held-out Youden threshold (each eval group's
threshold fit on its training folds, applied to its held-out fold). AUROC is
threshold-free.

Eval groups per (model, k):
  - harmful  : within-harmful refusal detection
  - benign   : MACRO over surviving benign sub-sources {benign_main, phtest,
               orfuzz} (drop any with < MINPOS refusals). nested bootstrap CI.
  - pooled   : combined harmful+benign eval (for the table's Pooled column).

K-hop tensors are loaded with mmap=True and sliced at [:, k, -1, :] so only the
final-layer, position-k bytes are read (avoids the multi-GB OOM).

Writes (incrementally per model):
  experiments/refusal_cliff/results/kshop_K20_user_content/bestpipe_pca100_c003.json

Provenance: copied (logic unchanged) from scripts/neurips_final/compute_kshop_bestpipe.py.
The shared model registry / guardrail vote / probe + stats helpers now live in
the `lrm_safety_deliberation` library (`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`,
`lrm_safety_deliberation.stats`); the MODELS list below is a local echo. Set
LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""
from __future__ import annotations
import os, json, sys
os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
ART = ROOT / "experiments/refusal_cliff/artifacts/kshop_K20_user_content"
OUT = ROOT / "experiments/refusal_cliff/results/kshop_K20_user_content/bestpipe_pca100_c003.json"
MODELS = ["Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning", "GPT-OSS-20B"]
INTENTS = ["harmful", "benign", "phtest", "orfuzz"]   # 'benign' = benign_main
BENIGN_SUBS = ["benign_main", "phtest", "orfuzz"]
K_TOTAL = 21
N_FOLDS, SEED, N_BOOT = 5, 42, 500
PCA_DIM, C, MINPOS = 100, 0.03, 20


def load_slice(model, it, k):
    d = torch.load(ART / model / f"{it}.pt", map_location="cpu",
                   weights_only=False, mmap=True)
    X = d["X"][:, k, -1, :].to(torch.float32).numpy()
    y = d["y"].numpy().astype(np.int64)
    del d
    return X, y


def youden(y, s):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def group_yhat(y, s, fid):
    """Per-fold held-out Youden predictions for one eval group."""
    yh = np.zeros(len(y), dtype=np.int64)
    for k in range(N_FOLDS):
        tr, te = fid != k, fid == k
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        t = youden(y[tr], s[tr])
        yh[te] = (s[te] >= t).astype(np.int64)
    return yh


def bacc(y, yh):
    pos, neg = y == 1, y == 0
    if not (pos.any() and neg.any()):
        return np.nan
    return 0.5 * ((yh[pos] == 1).mean() + (yh[neg] == 0).mean())


def ci(arr):
    a = np.asarray(arr, float); a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def group_metrics(y, s, fid, rng):
    """AUROC + per-group-Youden BAcc with bootstrap CI for a single group."""
    yh = group_yhat(y, s, fid)
    aus, bas = [], []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ys, ss, yhs = y[idx], s[idx], yh[idx]
        if len(np.unique(ys)) < 2:
            continue
        aus.append(roc_auc_score(ys, ss))
        b = bacc(ys, yhs)
        if not np.isnan(b):
            bas.append(b)
    au_m, au_lo, au_hi = ci(aus)
    ba_m, ba_lo, ba_hi = ci(bas)
    return {"auroc": au_m, "auroc_lo": au_lo, "auroc_hi": au_hi,
            "bacc": ba_m, "bacc_lo": ba_lo, "bacc_hi": ba_hi}


def macro_metrics(subs, rng):
    """subs: list of (y, s, yh). Nested bootstrap of the macro."""
    au_boot, ba_boot = np.full(N_BOOT, np.nan), np.full(N_BOOT, np.nan)
    for bi in range(N_BOOT):
        a_, b_ = [], []
        for y, s, yh in subs:
            n = len(y); idx = rng.integers(0, n, n)
            ys, ss, yhs = y[idx], s[idx], yh[idx]
            if len(np.unique(ys)) < 2:
                continue
            a_.append(roc_auc_score(ys, ss))
            bb = bacc(ys, yhs)
            if not np.isnan(bb):
                b_.append(bb)
        if a_: au_boot[bi] = np.mean(a_)
        if b_: ba_boot[bi] = np.mean(b_)
    au_m, au_lo, au_hi = ci(au_boot)
    ba_m, ba_lo, ba_hi = ci(ba_boot)
    return {"auroc": au_m, "auroc_lo": au_lo, "auroc_hi": au_hi,
            "bacc": ba_m, "bacc_lo": ba_lo, "bacc_hi": ba_hi}


def run_cell(model, k):
    Xs, ys, ss = [], [], []
    for it in INTENTS:
        X, y = load_slice(model, it, k)
        Xs.append(X); ys.append(y)
        tag = "harmful" if it == "harmful" else ("benign_main" if it == "benign" else it)
        ss += [tag] * len(y)
    X = np.concatenate(Xs); y = np.concatenate(ys); ss = np.array(ss, object)
    intent = np.array(["harmful" if t == "harmful" else "benign" for t in ss], object)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y)); fid = np.full(len(y), -1)
    for kk, (tr, te) in enumerate(skf.split(X, y)):
        clf = make_pipeline(StandardScaler(), PCA(PCA_DIM),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]; fid[te] = kk

    rng = np.random.default_rng(SEED)
    hm = intent == "harmful"
    out = {"harmful": group_metrics(y[hm], oof[hm], fid[hm], rng)}
    # benign macro
    subs = []
    for sub in BENIGN_SUBS:
        m = ss == sub
        if m.sum() == 0 or round(y[m].sum()) < MINPOS or len(np.unique(y[m])) < 2:
            continue
        subs.append((y[m], oof[m], group_yhat(y[m], oof[m], fid[m])))
    out["benign"] = macro_metrics(subs, rng)
    out["benign_kept"] = [sub for sub in BENIGN_SUBS
                          if (ss == sub).sum() and round(y[ss == sub].sum()) >= MINPOS]
    # pooled
    out["pooled"] = group_metrics(y, oof, fid, rng)
    return out


def main():
    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text())
    for m in MODELS:
        if m in results and len(results[m]) == K_TOTAL:
            print(f"[skip] {m} already done", flush=True); continue
        results.setdefault(m, {})
        for k in range(K_TOTAL):
            if str(k) in results[m]:
                continue
            cell = run_cell(m, k)
            results[m][str(k)] = cell
            print(f"{m} k={k}: H AUROC={cell['harmful']['auroc']:.3f} BAcc={cell['harmful']['bacc']:.3f}"
                  f" | B AUROC={cell['benign']['auroc']:.3f} BAcc={cell['benign']['bacc']:.3f}"
                  f" | P AUROC={cell['pooled']['auroc']:.3f} BAcc={cell['pooled']['bacc']:.3f}",
                  flush=True)
            OUT.write_text(json.dumps(results, indent=1))  # checkpoint each cell
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
