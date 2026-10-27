"""Bootstrap confidence intervals and paired significance over queries."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Mean difference (a - b), its CI, and a two-sided bootstrap p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)), float(min(p, 1.0))
