"""Metrics for the validity gate (appendix A2.4).  numpy / scipy only."""
import numpy as np
from scipy import stats


def pearson_r(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.corrcoef(y, p)[0, 1])


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def binary_accuracy(y, p, center=0.0):
    """Sign agreement around `center` (labels must be centered for this to mean anything)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((y > center) == (p > center)))


def fisher_ci(r, n, level=0.95):
    """Confidence interval for a correlation by the Fisher z transformation (not a bootstrap)."""
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    q = stats.norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def bootstrap_ci(stat, arrays, n_boot=2000, level=0.95, seed=0):
    """Percentile bootstrap over cases.  `stat(*arrays)` is evaluated on each resample."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = stat(*[np.asarray(a)[i] for a in arrays])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def steiger_z(r_jk, r_jh, r_kh, n):
    """Steiger (1980) test of H0: rho_jk = rho_jh for two dependent correlations sharing
    variable j (Z-bar form with the pooled-correlation variance).  Two-sided p-value.

    j = estimate T_hat, k = trustworthiness rating, h = a rival rating, r_kh = corr(k, h).
    """
    rbar = (r_jk + r_jh) / 2.0
    cov = (r_kh * (1 - 2 * rbar ** 2) - 0.5 * rbar ** 2 * (1 - 2 * rbar ** 2 - r_kh ** 2)) / (1 - rbar ** 2) ** 2
    z = (np.arctanh(r_jk) - np.arctanh(r_jh)) * np.sqrt((n - 3) / (2 - 2 * cov))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def icc2k(ratings):
    """ICC(2,k): reliability of the mean of k raters (two-way random effects, absolute agreement).

    `ratings` is an (n_clips, k_raters) array without missing values.
    """
    x = np.asarray(ratings, float)
    n, k = x.shape
    gm = x.mean()
    ms_r = k * np.sum((x.mean(axis=1) - gm) ** 2) / (n - 1)          # between clips
    ms_c = n * np.sum((x.mean(axis=0) - gm) ** 2) / (k - 1)          # between raters
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm
    ms_e = np.sum(resid ** 2) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n))
