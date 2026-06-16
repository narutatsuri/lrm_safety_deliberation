"""Compute State A first-token AUROC table for 5 target models.

State A = prompt-only last-token hidden state (pre-decode, no thinking token processed).
Source:
  harmful      -> eval_results/neurips_s16/<M>/representations/prefill_think_before/harmful.pt
  benign core  -> eval_results/neurips_s16/<M>/representations/prefill_think_before/benign.pt
  orfuzz       -> eval_results/neurips_final_supplementary/<M>/representations/prefill_think_before/orfuzz.pt
  phtest       -> eval_results/neurips_final_supplementary/<M>/representations/prefill_think_before/phtest.pt

Labels joined by id from the corresponding hopping reps (4-guardrail majority on final response).

For each (model, intent): 5-fold CV LogReg on GPU, 500-bootstrap 95% CI. Prints markdown + LaTeX.

Provenance: copied (logic unchanged) from scripts/neurips_final/first_token_auroc_table.py.
The shared model registry / guardrail vote / probe + stats helpers now live in
the `lrm_safety_deliberation` library (`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`,
`lrm_safety_deliberation.stats`); the MODELS list below is a local echo. Set
LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))

MODELS = [
    ("Qwen3-8B", "Qwen3-8B"),
    ("Qwen3-32B", "Qwen3-32B"),
    ("Olmo-3-7B-Think", "OLMo-7B"),
    ("Phi-4-reasoning", "Phi-4"),
    ("GPT-OSS-20B", "GPT-OSS-20B"),
]

N_BOOT = 500


def load_state_a_harmful(m: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    p = ROOT / "eval_results" / "neurips_s16" / m / "representations" / "prefill_think_before" / "harmful.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["X"], d["ids"].numpy().astype(np.int64), None  # labels joined later


def load_state_a_benign_combined(m: str) -> tuple[torch.Tensor, np.ndarray]:
    parts = []
    sources = [
        ROOT / "eval_results" / "neurips_s16" / m / "representations" / "prefill_think_before" / "benign.pt",
        ROOT / "eval_results" / "neurips_final_supplementary" / m / "representations" / "prefill_think_before" / "orfuzz.pt",
        ROOT / "eval_results" / "neurips_final_supplementary" / m / "representations" / "prefill_think_before" / "phtest.pt",
    ]
    Xs, ids_list, tag_list = [], [], []
    tag_names = ["benign_core", "orfuzz", "phtest"]
    for p, tag in zip(sources, tag_names):
        if not p.exists():
            print(f"  MISSING: {p}")
            return None, None, None
        d = torch.load(p, map_location="cpu", weights_only=False)
        Xs.append(d["X"])
        ids_list.append(d["ids"].numpy().astype(np.int64))
        tag_list.append(np.full(len(d["ids"]), tag, dtype=object))
    X = torch.cat(Xs, dim=0)
    ids = np.concatenate(ids_list)
    tags = np.concatenate(tag_list)
    return X, ids, tags


def labels_from_hopping(m: str, which: str, source: str = "final") -> dict[int, int]:
    """Map id -> y from the corresponding hopping reps file."""
    if source == "final":
        p = ROOT / "eval_results" / "neurips_final" / m / "representations" / "thinking_hopping" / f"{which}.pt"
    else:
        p = ROOT / "eval_results" / "neurips_final_supplementary" / m / "representations" / "thinking_hopping" / f"{which}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    ids = [int(x) for x in d["ids"]]
    ys = d["y"].numpy().astype(np.int64).tolist()
    return dict(zip(ids, ys))


def gpu_cv_auroc(X_cpu: torch.Tensor, y: np.ndarray, device: str = "cuda",
                  C: float = 1.0, max_iter: int = 200, n_splits: int = 5) -> tuple[float, np.ndarray]:
    N, D = X_cpu.shape
    y_dev = torch.tensor(y, dtype=torch.float32, device=device)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(N, dtype=float)
    X_dev = X_cpu.to(torch.float32).to(device)
    for tr_idx, te_idx in cv.split(np.zeros(N), y):
        tr = torch.from_numpy(tr_idx).long().to(device)
        te = torch.from_numpy(te_idx).long().to(device)
        Xtr = X_dev[tr]
        Xte = X_dev[te]
        mu = Xtr.mean(0)
        sd = Xtr.std(0).clamp(min=1e-6)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        w = torch.zeros(D, requires_grad=True, device=device)
        b = torch.zeros(1, requires_grad=True, device=device)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter,
                                 tolerance_grad=1e-5, tolerance_change=1e-7,
                                 history_size=20)
        ytr = y_dev[tr]
        n_tr = Xtr.shape[0]
        reg = 1.0 / (C * n_tr)

        def closure():
            opt.zero_grad()
            logits = Xtr @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr)
            loss = loss + 0.5 * reg * (w * w).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            oof[te_idx] = torch.sigmoid(Xte @ w + b).cpu().numpy()
    auroc = float(roc_auc_score(y, oof))
    return auroc, oof


def bootstrap_ci(y: np.ndarray, oof: np.ndarray, n_boot: int = N_BOOT,
                  seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 5 or len(neg) < 5:
        return (float("nan"), float("nan"))
    aurocs = np.zeros(n_boot)
    for b in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            aurocs[b] = roc_auc_score(y[idx], oof[idx])
        except ValueError:
            aurocs[b] = 0.5
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    rows = []
    for model, display in MODELS:
        print(f"\n=== {model} ===")
        results = {"model": display, "model_key": model}

        # Harmful
        Xh, ids_h, _ = load_state_a_harmful(model)
        lbl_map_h = labels_from_hopping(model, "harmful", source="final")
        mask = np.array([i in lbl_map_h for i in ids_h])
        y_h = np.array([lbl_map_h[int(i)] for i in ids_h[mask]], dtype=np.int64)
        X_h = Xh[mask]
        print(f"  harmful: N={len(y_h)}  pos={int(y_h.sum())}")
        if y_h.sum() < 5 or (len(y_h) - y_h.sum()) < 5:
            results["harmful"] = (float("nan"), float("nan"), float("nan"))
        else:
            t0 = time.time()
            auroc, oof = gpu_cv_auroc(X_h, y_h, device=device)
            lo, hi = bootstrap_ci(y_h, oof)
            print(f"  harmful AUROC={auroc:.3f} [{lo:.3f}, {hi:.3f}]  ({time.time()-t0:.1f}s)")
            results["harmful"] = (auroc, lo, hi, int(len(y_h)), int(y_h.sum()))

        # Benign combined
        Xb, ids_b, tags_b = load_state_a_benign_combined(model)
        if Xb is None:
            print(f"  benign combined: INCOMPLETE (Qwen3-32B ORFuzz/PHTest missing)")
            results["benign_combined"] = None
        else:
            lbl_core = labels_from_hopping(model, "benign", source="final")
            lbl_orfuzz = labels_from_hopping(model, "orfuzz", source="supp")
            lbl_phtest = labels_from_hopping(model, "phtest", source="supp")
            ys = []
            keep = []
            for idx, (i, tag) in enumerate(zip(ids_b, tags_b)):
                if tag == "benign_core":
                    m = lbl_core
                elif tag == "orfuzz":
                    m = lbl_orfuzz
                else:
                    m = lbl_phtest
                if int(i) in m:
                    ys.append(m[int(i)])
                    keep.append(idx)
            keep = np.array(keep, dtype=np.int64)
            y_b = np.array(ys, dtype=np.int64)
            X_b = Xb[keep]
            print(f"  benign combined: N={len(y_b)}  pos={int(y_b.sum())}")
            if y_b.sum() < 5 or (len(y_b) - y_b.sum()) < 5:
                results["benign_combined"] = (float("nan"), float("nan"), float("nan"))
            else:
                t0 = time.time()
                auroc, oof = gpu_cv_auroc(X_b, y_b, device=device)
                lo, hi = bootstrap_ci(y_b, oof)
                print(f"  benign  AUROC={auroc:.3f} [{lo:.3f}, {hi:.3f}]  ({time.time()-t0:.1f}s)")
                results["benign_combined"] = (auroc, lo, hi, int(len(y_b)), int(y_b.sum()))
        rows.append(results)

    # Markdown
    print("\n\n# First-token (State A) AUROC — combined eval pool\n")
    print("| Model | Harmful AUROC (N / pos) | Benign-combined AUROC (N / pos) |")
    print("|---|---|---|")
    for r in rows:
        h = r.get("harmful")
        b = r.get("benign_combined")
        h_str = f"{h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}] ({h[3]}/{h[4]})" if h and not np.isnan(h[0]) else "—"
        b_str = (f"{b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] ({b[3]}/{b[4]})"
                 if b and isinstance(b, tuple) and not np.isnan(b[0]) else "pending" if b is None else "—")
        print(f"| {r['model']} | {h_str} | {b_str} |")

    # LaTeX
    print("\n\n% LaTeX\n")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Model & Harmful & Benign (combined) \\")
    print(r"\midrule")
    for r in rows:
        h = r.get("harmful")
        b = r.get("benign_combined")
        h_str = f"{h[0]:.3f} ({h[1]:.3f}, {h[2]:.3f})" if h and not np.isnan(h[0]) else "---"
        b_str = (f"{b[0]:.3f} ({b[1]:.3f}, {b[2]:.3f})"
                 if b and isinstance(b, tuple) and not np.isnan(b[0]) else "pending" if b is None else "---")
        print(f"{r['model']} & {h_str} & {b_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Save JSON
    out = ROOT / "neurips" / "first_token_auroc_stateA.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in rows:
        s = {"model": r["model"], "model_key": r["model_key"]}
        for k in ("harmful", "benign_combined"):
            v = r.get(k)
            if v is None:
                s[k] = None
            elif isinstance(v, tuple) and not np.isnan(v[0]):
                s[k] = {"auroc": v[0], "ci_lo": v[1], "ci_hi": v[2], "n": v[3], "n_pos": v[4]}
            else:
                s[k] = None
        serializable.append(s)
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
