"""Explore which separability metric gives the cleanest refusal-valley visual.

Metrics computed per-position on hopping reps (benign combined + harmful), all OOF-5-fold:

  A) logit_fisher  = (mu_z+ - mu_z-)^2 / (var_z+ + var_z-)       where z = LogReg logit (pre-sigmoid)
  B) cohen_d_logit = (mu_z+ - mu_z-) / sqrt((var_z+ + var_z-)/2)
  C) mahalanobis   = sqrt((mu+ - mu-)^T (Sigma + lambda I)^{-1} (mu+ - mu-))    (shrinkage LDA)
  D) norm_fisher   = ||mu+ - mu-||^2 / (tr(Sigma+) + tr(Sigma-))  AFTER per-position z-score standardization
  E) bhattacharyya = 1/4 (mu_z+ - mu_z-)^2 / ((var_z+ + var_z-)/2) + 1/2 ln(((var_z+ + var_z-)/2) / sqrt(var_z+ var_z-))

LogReg direction (A, B, E) is the conceptually cleanest: discriminatively trained L2-regularized
linear probe. Fisher/Cohen's d on the pre-sigmoid logit is an unbounded scalar summary of
class separation in the probe direction. Bhattacharyya adds variance mismatch but is noisier.

Mahalanobis with shrinkage (C) is closed-form LDA accounting for covariance. Needs lambda tuning.

Normalized whole-space Fisher (D) is a diagnostic: if the valley persists after per-dim
standardization, then the original valley was NOT driven by magnitude drift.

Saves eval_results/neurips_final/<M>/valley/explore_<intent>.pt.

Provenance: copied (logic unchanged) from scripts/neurips_final/explore_valley_metrics.py.
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
FINAL_DIR = ROOT / "eval_results" / "neurips_final"
SUPP_DIR = ROOT / "eval_results" / "neurips_final_supplementary"


def load_harmful(model: str):
    p = FINAL_DIR / model / "representations" / "thinking_hopping" / "harmful.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["X"], d["y"], list(d.get("ids") or [])


def load_benign_combined(model: str):
    paths = [
        FINAL_DIR / model / "representations" / "thinking_hopping" / "benign.pt",
        SUPP_DIR / model / "representations" / "thinking_hopping" / "orfuzz.pt",
        SUPP_DIR / model / "representations" / "thinking_hopping" / "phtest.pt",
    ]
    Xs, ys, ids = [], [], []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        Xs.append(d["X"]); ys.append(d["y"])
        ids.extend(list(d.get("ids") or []))
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0), ids


def gpu_logreg_logit_oof(X_cpu: torch.Tensor, y_cpu: torch.Tensor, C: float = 1.0,
                           max_iter: int = 200, n_splits: int = 5, device: str = "cuda"):
    """Per-position LogReg; returns (logit_fisher[T], cohen_d[T], bhattacharyya[T], oof_logits[N,T])."""
    N, T, D = X_cpu.shape
    y_np = y_cpu.numpy().astype(np.int64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(N), y_np))
    y_dev = y_cpu.to(torch.float32).to(device)

    logit_fisher = np.zeros(T)
    cohen_d = np.zeros(T)
    bhattacharyya = np.zeros(T)
    oof_logit = np.zeros((N, T))

    for t in range(T):
        Xt = X_cpu[:, t, :].to(torch.float32).to(device)
        oof_t = torch.zeros(N, device=device)
        for tr_idx, te_idx in splits:
            tr = torch.from_numpy(tr_idx).long().to(device)
            te = torch.from_numpy(te_idx).long().to(device)
            Xtr = Xt[tr]; Xte = Xt[te]
            mu = Xtr.mean(0); sd = Xtr.std(0).clamp(min=1e-6)
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            w = torch.zeros(D, requires_grad=True, device=device)
            b = torch.zeros(1, requires_grad=True, device=device)
            opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter,
                                     tolerance_grad=1e-5, tolerance_change=1e-7, history_size=20)
            ytr = y_dev[tr]
            n_tr = Xtr.shape[0]; reg = 1.0 / (C * n_tr)
            def closure():
                opt.zero_grad()
                logits = Xtr @ w + b
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr)
                loss = loss + 0.5 * reg * (w * w).sum()
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                oof_t[te] = Xte @ w + b  # PRE-sigmoid logit
        logits_np = oof_t.cpu().numpy()
        pos = logits_np[y_np == 1]; neg = logits_np[y_np == 0]
        if len(pos) < 2 or len(neg) < 2:
            logit_fisher[t] = cohen_d[t] = bhattacharyya[t] = np.nan
        else:
            mu_p, mu_n = pos.mean(), neg.mean()
            var_p, var_n = pos.var(ddof=1), neg.var(ddof=1)
            denom_f = var_p + var_n
            logit_fisher[t] = (mu_p - mu_n) ** 2 / denom_f if denom_f > 1e-12 else 0.0
            pooled_sd = np.sqrt((var_p + var_n) / 2.0)
            cohen_d[t] = (mu_p - mu_n) / pooled_sd if pooled_sd > 1e-12 else 0.0
            avg_var = (var_p + var_n) / 2.0
            if var_p > 1e-12 and var_n > 1e-12 and avg_var > 1e-12:
                bhattacharyya[t] = (
                    0.25 * (mu_p - mu_n) ** 2 / avg_var
                    + 0.5 * np.log(avg_var / np.sqrt(var_p * var_n))
                )
            else:
                bhattacharyya[t] = 0.0
        oof_logit[:, t] = logits_np
        del Xt, oof_t
        torch.cuda.empty_cache()
        if t % 20 == 0:
            print(f"    t={t:3d}  Fisher={logit_fisher[t]:.3f}  d={cohen_d[t]:.3f}  B={bhattacharyya[t]:.3f}")
    return logit_fisher, cohen_d, bhattacharyya, oof_logit


def gpu_mahalanobis_oof(X_cpu: torch.Tensor, y_cpu: torch.Tensor, lam: float = 1.0,
                         n_splits: int = 5, device: str = "cuda"):
    """Cross-validated Mahalanobis distance between class means using shrinkage covariance.

    Uses train-fold (mu+, mu-, Sigma+lam*I) and projects held-out data onto
    w = (Sigma+lam*I)^{-1} (mu+ - mu-). Returns per-position Mahalanobis distance
    (proper unit: sqrt of squared distance).
    """
    N, T, D = X_cpu.shape
    y_np = y_cpu.numpy().astype(np.int64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(N), y_np))
    y_dev = y_cpu.to(torch.float32).to(device)

    mahal = np.zeros(T)
    for t in range(T):
        Xt = X_cpu[:, t, :].to(torch.float32).to(device)
        d_folds = []
        for tr_idx, _ in splits:
            tr = torch.from_numpy(tr_idx).long().to(device)
            Xtr = Xt[tr]; ytr = y_dev[tr]
            mu_pos = Xtr[ytr > 0.5].mean(0)
            mu_neg = Xtr[ytr < 0.5].mean(0)
            delta = mu_pos - mu_neg
            # Shrinkage covariance: (1-a) S + a * (tr(S)/D) * I; use a such that Sigma+lam*I behaves
            Xc = Xtr - Xtr.mean(0)
            # Sigma = (Xc^T Xc) / (n-1)  -> D x D; too big for D=5120. Use Woodbury trick:
            # (Sigma + lam I)^{-1} delta = (lam I + (Xc^T Xc)/(n-1))^{-1} delta
            # Let A = Xc / sqrt(n-1), so Sigma = A^T A. Woodbury:
            # (lam I + A^T A)^{-1} = (1/lam) I - (1/lam^2) A^T (I + (1/lam) A A^T)^{-1} A
            n = Xc.shape[0]
            A = Xc / np.sqrt(n - 1)
            # A A^T is n x n, tractable
            AAt = A @ A.T
            inner = torch.eye(n, device=device) + (1.0 / lam) * AAt
            sol = torch.linalg.solve(inner, A @ delta)  # [n]
            sig_inv_delta = (1.0 / lam) * delta - (1.0 / (lam ** 2)) * (A.T @ sol)
            dist_sq = delta @ sig_inv_delta
            d_folds.append(float(torch.sqrt(dist_sq.clamp(min=0.0)).cpu()))
        mahal[t] = float(np.mean(d_folds))
        del Xt
        torch.cuda.empty_cache()
        if t % 20 == 0:
            print(f"    t={t:3d}  Mahal={mahal[t]:.3f}")
    return mahal


def whole_space_fisher_normalized(X_cpu: torch.Tensor, y_cpu: torch.Tensor) -> np.ndarray:
    """Per-position standardize each dim (zero mean, unit variance across ALL samples),
    then compute whole-space Fisher. Isolates the "signal in safety direction" from raw
    activation magnitude.
    """
    X = X_cpu.to(torch.float32).numpy()
    y = y_cpu.numpy().astype(np.int64)
    N, T, D = X.shape
    fisher = np.zeros(T)
    for t in range(T):
        Xt = X[:, t, :]
        mu = Xt.mean(0); sd = Xt.std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        Xt = (Xt - mu) / sd
        pos = Xt[y == 1]; neg = Xt[y == 0]
        mup = pos.mean(0); mun = neg.mean(0)
        between = ((mup - mun) ** 2).sum()
        vp = ((pos - mup) ** 2).sum(axis=1).mean()
        vn = ((neg - mun) ** 2).sum(axis=1).mean()
        denom = vp + vn
        fisher[t] = between / denom if denom > 1e-12 else 0.0
    return fisher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--mahal_lam", type=float, default=1.0)
    ap.add_argument("--skip_mahal", action="store_true",
                    help="Skip Mahalanobis (Woodbury solve can be slow for large n)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    for model in args.models:
        print(f"\n=== {model} ===")
        for intent, loader in [("harmful", load_harmful), ("benign_combined", load_benign_combined)]:
            t0 = time.time()
            X, y, ids = loader(model)
            print(f"  [{intent}] X={tuple(X.shape)}  pos={int(y.sum())}/{len(y)}")
            lf, cd, bc, oof_logit = gpu_logreg_logit_oof(X, y, device=device)
            print(f"  [{intent}] LogReg done in {time.time()-t0:.1f}s")
            if args.skip_mahal:
                mh = None
            else:
                tm = time.time()
                mh = gpu_mahalanobis_oof(X, y, lam=args.mahal_lam, device=device)
                print(f"  [{intent}] Mahal done in {time.time()-tm:.1f}s")
            tn = time.time()
            nf = whole_space_fisher_normalized(X, y)
            print(f"  [{intent}] NormFisher done in {time.time()-tn:.1f}s")
            out = FINAL_DIR / model / "valley" / f"explore_{intent}.pt"
            torch.save({
                "logit_fisher": lf,
                "cohen_d_logit": cd,
                "bhattacharyya": bc,
                "mahalanobis": mh,
                "norm_fisher": nf,
                "oof_logit": oof_logit,
                "y": y.numpy().astype(np.int64),
                "ids": ids,
                "intent": intent,
                "method": "logit_probe_oof",
                "mahal_lambda": args.mahal_lam,
            }, out)
            print(f"  [{intent}] wrote {out.relative_to(ROOT)}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
