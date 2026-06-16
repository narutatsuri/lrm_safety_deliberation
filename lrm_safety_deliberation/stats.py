"""Statistics shared across analyses: agreement, bootstrap CIs, multiple testing.

These were scattered as one-off helpers throughout the research scripts. They
are gathered here so every figure/table uses the same (tested) implementation.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "fleiss_kappa",
    "cohen_kappa",
    "bootstrap_mean_ci",
    "cluster_bootstrap_ci",
    "holm_significant",
    "fisher_exact_2x2",
]

_LABELS = (-1, 0, 1)


# --------------------------------------------------------------------------- #
# Inter-annotator agreement
# --------------------------------------------------------------------------- #
def fleiss_kappa(items: Iterable[Sequence[int]], labels: Sequence[int] = _LABELS) -> float:
    """Fleiss' kappa for ``n`` raters over many items.

    ``items`` is an iterable of equal-length label tuples (one per item, one
    entry per rater).  Labels default to the stance set ``{-1, 0, 1}``.
    """
    rows = [list(r) for r in items]
    if not rows:
        return float("nan")
    N = len(rows)
    n = len(rows[0])
    if n < 2:
        return float("nan")

    cat_total: Counter = Counter()
    p_i = []
    for row in rows:
        cnt = Counter(row)
        p_i.append(sum(cnt[c] * (cnt[c] - 1) for c in labels) / (n * (n - 1)))
        for c in labels:
            cat_total[c] += cnt[c]
    p_bar = sum(p_i) / N
    p_j = {c: cat_total[c] / (N * n) for c in labels}
    p_e = sum(p_j[c] ** 2 for c in labels)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def cohen_kappa(pairs: Iterable[Sequence[int]], labels: Sequence[int] = _LABELS) -> float:
    """Cohen's kappa for two raters over ``pairs`` of ``(label_a, label_b)``."""
    items = [tuple(p) for p in pairs]
    if not items:
        return float("nan")
    N = len(items)
    obs = sum(1 for a, b in items if a == b) / N
    a_count = Counter(a for a, _ in items)
    b_count = Counter(b for _, b in items)
    pe = sum((a_count[c] / N) * (b_count[c] / N) for c in labels)
    if pe >= 1.0:
        return 1.0
    return (obs - pe) / (1 - pe)


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean.

    Returns ``(point_estimate, lo, hi)`` where the interval is the central
    ``1 - alpha`` percentile band over ``n_boot`` resamples.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence],
    statistic: Callable[[list], float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster (block) bootstrap CI for an arbitrary statistic.

    ``clusters`` is a list of groups (e.g. all chunks of one trace).  Each
    bootstrap iteration resamples whole clusters with replacement, flattens
    their members, and evaluates ``statistic`` on the pooled members.  Use this
    for trace-clustered kappa CIs (members are per-chunk rater tuples).
    """
    groups = [list(c) for c in clusters]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    point = statistic([m for g in groups for m in g])
    rng = np.random.default_rng(seed)
    n = len(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        pooled = [m for j in pick for m in groups[j]]
        boots.append(statistic(pooled))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Multiple-testing correction & contingency tests
# --------------------------------------------------------------------------- #
def holm_significant(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm–Bonferroni step-down rejection mask (in the input order).

    Sorts p-values ascending and rejects while ``p_(k) <= alpha / (n - k)``,
    stopping at the first failure (the step-down family used for the per-model
    causal-cut significance test).
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    sig = np.zeros(n, dtype=bool)
    if n == 0:
        return sig
    order = np.argsort(p)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (n - rank):
            sig[idx] = True
        else:
            break
    return sig


def fisher_exact_2x2(n_comply_pre: int, n_pre: int, n_comply_post: int, n_post: int) -> float:
    """Two-sided Fisher's exact p-value for a pre/post comply-vs-refuse 2x2 table.

    Table = ``[[comply_pre, refuse_pre], [comply_post, refuse_post]]``.
    """
    from scipy.stats import fisher_exact
    table = [
        [n_comply_pre, n_pre - n_comply_pre],
        [n_comply_post, n_post - n_comply_post],
    ]
    _, p = fisher_exact(table, alternative="two-sided")
    return float(p)
