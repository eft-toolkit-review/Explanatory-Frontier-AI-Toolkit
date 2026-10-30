"""A3.4 Stability by maximum Jaccard matching (illustrative; Hennig, 2007).

B bootstrap samples of the clips; refit the K*-component mixture; assign ALL clips with the
bootstrap model; for each reference configuration take the highest Jaccard with any bootstrap component.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def jaccard(a, b):
    u = np.sum(a | b)
    return np.sum(a & b) / u if u else 0.0


def bootstrap_stability(P, ref_labels, k, B=500, reg=1e-3, n_init=20, theta=0.75, dissolved=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(P)
    J = np.zeros((B, k))
    for b in range(B):
        i = rng.integers(0, n, n)
        gm = GaussianMixture(k, covariance_type="full", reg_covar=reg, n_init=n_init,
                             random_state=int(rng.integers(1 << 31))).fit(P[i])
        lab = gm.predict(P)
        for c in range(k):
            ref = ref_labels == c
            J[b, c] = max(jaccard(ref, lab == m) for m in range(k))
    gamma = J.mean(0)                                          # S_k in E11
    return {"gamma": gamma, "dissolved_share": (J <= dissolved).mean(0), "retained": gamma >= theta, "J": J}


def pool_remainder(labels, retained):
    """Configurations below tau are pooled into one remainder category (label -1)."""
    keep = {c for c, r in enumerate(retained) if r}
    return np.array([c if c in keep else -1 for c in labels])


def baseline_configuration(retained, expected_size):
    """The retained configuration with the largest expected size is the baseline in A6."""
    cand = [c for c, r in enumerate(retained) if r]
    return max(cand, key=lambda c: expected_size[c]) if cand else None
