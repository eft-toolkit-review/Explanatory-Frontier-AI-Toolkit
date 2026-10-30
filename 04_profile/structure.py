"""A3.3 Structure discovery (illustrative).

PCA to m components (10 suggested), HDBSCAN as a diagnostic only, then a Gaussian mixture
(full covariance, regularization 1e-3, 20 starts). K is chosen by the integrated completed
likelihood; any K with a component whose expected size is below 30 is rejected.Again, only illustrative"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold


def reduce(Z_res, m=10):
    return PCA(m, random_state=0).fit_transform(Z_res)


def hdbscan_diagnostic(P, min_cluster_size=30, min_samples=10):
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(P)
    return {"labels": lab, "n_clusters": int(lab.max() + 1), "noise_share": float(np.mean(lab == -1))}


def fit_gmm(P, k, reg=1e-3, n_init=20, seed=0):
    return GaussianMixture(k, covariance_type="full", reg_covar=reg, n_init=n_init, random_state=seed).fit(P)


def icl(gm, P):
    """ICL = BIC + 2 * entropy of the posteriors (lower is better). Overlapping components are penalized."""
    tau = gm.predict_proba(P)
    return gm.bic(P) - 2.0 * np.sum(tau * np.log(tau + 1e-300))


def select_k(P, k_max=8, min_size=30, reg=1e-3, n_init=20, seed=0, folds=5):
    rows, models = [], {}
    for k in range(1, k_max + 1):
        gm = fit_gmm(P, k, reg, n_init, seed)
        size = gm.predict_proba(P).sum(0)
        ho = np.mean([fit_gmm(P[tr], k, reg, n_init, seed).score(P[te])
                      for tr, te in KFold(folds, shuffle=True, random_state=seed).split(P)])
        rows.append({"K": k, "BIC": gm.bic(P), "ICL": icl(gm, P), "heldout_ll": float(ho),
                     "min_expected_size": float(size.min()), "admissible": bool(size.min() >= min_size)})
        models[k] = gm
    ok = [r for r in rows if r["admissible"]]
    k_star = min(ok, key=lambda r: r["ICL"])["K"] if ok else 1
    return k_star, rows, models[k_star]


def hdbscan_vs_mixture_ari(hdb_labels, gm, P):
    """Noise points are kept as their own label."""
    return adjusted_rand_score(hdb_labels, gm.predict(P))
