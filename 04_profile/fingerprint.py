"""A3.5 Fingerprint of a configuration (illustrative).

Standardized contrast of the posterior-weighted mean of each interpretable summary feature
against the corpus mean, with bootstrap intervals; plus the clips with the highest membership.
Interpretation and adjudication are done by people (no code, no language model).
"""
import numpy as np


def fingerprint(F, tau_k, names, top=10, n_clips=5, n_boot=2000, seed=0):
    """F: clips x features (interpretable summaries); tau_k: membership probability of one configuration."""
    F, tau_k = np.asarray(F, float), np.asarray(tau_k, float)
    sd = F.std(0, ddof=1) + 1e-12

    def contrast(idx):
        w = tau_k[idx] / tau_k[idx].sum()
        return (w @ F[idx] - F[idx].mean(0)) / sd

    est = contrast(np.arange(len(F)))
    rng = np.random.default_rng(seed)
    boot = np.array([contrast(rng.integers(0, len(F), len(F))) for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    order = np.argsort(-np.abs(est))[:top]
    rows = [(names[j], float(est[j]), float(lo[j]), float(hi[j])) for j in order]
    return rows, np.argsort(-tau_k)[:n_clips]
