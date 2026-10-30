"""A6.2 Decision rules (illustrative).

Each contrast (beta_3k, and each beta_2k) is classified against a bound delta = 0.20 SD of Y with a
(1 - 2 alpha_k) interval, alpha_k the Holm-adjusted level (E19):
  interval inside (-delta, delta)        -> equivalent to zero
  interval wholly outside [-delta, delta] -> meaningful
  otherwise                               -> inconclusive
The Benjamini-Hochberg adjustment is added as a sensitivity column. Estimates, standard errors and
degrees of freedom come from the registered mixed model (07_reopen/reopen_model.py).
"""
import numpy as np
from scipy import stats


def delta_from_sd(y, multiple=0.20):
    return multiple * np.std(y, ddof=1)


def fit_jointly(k_prime, M):
    """A6.1: contrasts are fitted jointly unless K' - 1 exceeds M / 10 (then each against the pooled remainder)."""
    return (k_prime - 1) <= M / 10.0


def tost_p(est, se, df, delta):
    """Larger of the two one-sided p values for |beta| >= delta; < alpha_k exactly when the interval is inside."""
    return max(stats.t.sf((est + delta) / se, df), stats.t.cdf((est - delta) / se, df))


def holm_levels(p, alpha=0.05):
    """Step-down levels alpha / (m - rank), rank 0 for the smallest p, returned in the original order."""
    p = np.asarray(p, float)
    lev = np.empty(len(p))
    for rank, i in enumerate(np.argsort(p)):
        lev[i] = alpha / (len(p) - rank)
    return lev


def bh_adjust(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def classify(est, se, df, delta, alpha_k):
    half = stats.t.ppf(1 - alpha_k, df) * se
    lo, hi = est - half, est + half
    if -delta < lo and hi < delta:
        label = "equivalent"
    elif lo > delta or hi < -delta:
        label = "meaningful"
    else:
        label = "inconclusive"
    return label, (float(lo), float(hi))


def classify_family(contrasts, delta, alpha=0.05):
    """contrasts: list of dicts with name, est, se, df and (for reporting, A6.3) stability."""
    p = np.array([tost_p(c["est"], c["se"], c["df"], delta) for c in contrasts])
    lev, q = holm_levels(p, alpha), bh_adjust(p)
    rows = []
    for c, a_k, p_i, q_i in zip(contrasts, lev, p, q):
        label, ci = classify(c["est"], c["se"], c["df"], delta, a_k)
        rows.append({**c, "alpha_k": float(a_k), "interval": ci, "class": label, "p_equiv": float(p_i), "q_bh": float(q_i)})
    return rows
