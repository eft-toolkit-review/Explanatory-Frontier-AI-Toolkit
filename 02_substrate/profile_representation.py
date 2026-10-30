"""A1.3 A label-free representation for profiling (illustrative).

Builds one summary vector per clip from four blocks, balances the blocks, and projects on the
leading principal components (E2).  Nothing here takes a label or a rating as input: the
representation is built without access to the focal construct.

"""
import numpy as np


def mean_pool(embeddings):
    """Block (i): mean-pooled language embedding. (tokens, d) -> (d,)"""
    return np.asarray(embeddings, float).mean(axis=0)


def mean_std(x):
    """Block (ii) for one clip: mean and standard deviation of each acoustic feature over time."""
    x = np.asarray(x, float)
    return np.concatenate([x.mean(axis=0), x.std(axis=0)])


def mean_std_absdelta(x):
    """Block (iii) for one clip: mean, standard deviation and mean absolute change of each visual feature."""
    x = np.asarray(x, float)
    return np.concatenate([x.mean(axis=0), x.std(axis=0), np.abs(np.diff(x, axis=0)).mean(axis=0)])


def lagged_peak_corr(a, v, rate_hz=10.0, max_lag_s=2.0):
    """Peak correlation between an acoustic and a visual channel within +/- max_lag_s seconds,
    and the lag (seconds) at which it occurs.  Positive lag: the visual channel follows the acoustic one.
    The peak is the correlation of largest absolute value."""
    a, v = np.asarray(a, float), np.asarray(v, float)
    L = int(round(max_lag_s * rate_hz))
    best, best_lag = 0.0, 0
    for lag in range(-L, L + 1):
        x, y = (a[: len(a) - lag], v[lag:]) if lag >= 0 else (a[-lag:], v[: len(v) + lag])
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if abs(c) > abs(best):
            best, best_lag = c, lag
    return best, best_lag / rate_hz


def coordination_block(acoustic, visual, rate_hz=10.0, max_lag_s=2.0):
    """Block (iv): 4 acoustic x 4 visual channels (resampled to 10 Hz) -> 16 peak correlations and
    16 lags = 32 features.  acoustic, visual: (steps, 4)."""
    acoustic, visual = np.asarray(acoustic, float), np.asarray(visual, float)
    feats = []
    for i in range(acoustic.shape[1]):
        for j in range(visual.shape[1]):
            c, lag = lagged_peak_corr(acoustic[:, i], visual[:, j], rate_hz, max_lag_s)
            feats += [c, lag]
    return np.array(feats)


def fit_profile(blocks, var_retained=0.90, max_components=50):
    """z = W' D (s - mu)  (E2).

    blocks: {name: array (clips, p_block)}.  D scales every block by 1 / sqrt(its number of
    features), so that no block dominates.  W holds the leading principal directions: the fewest
    that retain `var_retained` of the variance, at most `max_components`.  No labels are used.
    """
    names = list(blocks)
    mats = [np.asarray(blocks[n], float) for n in names]
    mu = np.concatenate([m.mean(axis=0) for m in mats])
    d = np.concatenate([np.full(m.shape[1], 1.0 / np.sqrt(m.shape[1])) for m in mats])
    s = (np.hstack(mats) - mu) * d
    _, sv, vt = np.linalg.svd(s, full_matrices=False)
    ratio = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    k = int(min(np.searchsorted(ratio, var_retained) + 1, max_components, len(sv)))
    return {"names": names, "mu": mu, "d": d, "W": vt[:k].T, "k": k, "retained": float(ratio[k - 1])}


def project(blocks, profile):
    s = (np.hstack([np.asarray(blocks[n], float) for n in profile["names"]]) - profile["mu"]) * profile["d"]
    return s @ profile["W"]
