"""A3.2 Leakage audit and iterative nullspace projection (illustrative).

Probe: gradient boosting (depth 3, 200 trees, learning rate 0.05), trained on the development set D
to predict T_hat from the residual profile, and evaluated once on the untouched evaluation set E (E6).
The residual is accepted if the 95th percentile of the bootstrap distribution of R2_oos over CEOs in E
is below epsilon (E7). epsilon = 0.05, or the 95th percentile of R2_oos from 500 refits of the probe with
T_hat (in D) permuted, if that is larger.

If the audit fails: up to 3 rounds of nullspace projection, using D only (E8).
A round is dropped if it would keep less than 70% of the variance left after the first residualization.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


def make_probe(seed=0):
    return GradientBoostingRegressor(max_depth=3, n_estimators=200, learning_rate=0.05, random_state=seed)


def r2_oos(t_e, pred_e, t_bar_d):
    """E6: 1 - SSE / SST, with the mean of T_hat over D as the benchmark."""
    return 1.0 - np.sum((t_e - pred_e) ** 2) / np.sum((t_e - t_bar_d) ** 2)


def leakage_bound(Zres_d, t_d, Zres_e, t_e, n_perm=500, floor=0.05, seed=0):
    """epsilon: 0.05, or the 95th percentile of R2_oos under permuted T_hat if larger."""
    rng = np.random.default_rng(seed)
    t_bar = t_d.mean()
    null = np.array([r2_oos(t_e, make_probe(seed).fit(Zres_d, rng.permutation(t_d)).predict(Zres_e), t_bar)
                     for _ in range(n_perm)])
    return max(floor, float(np.quantile(null, 0.95))), null


def leakage_audit(Zres_d, t_d, Zres_e, t_e, eps=None, n_boot=2000, n_perm=500, seed=0):
    """Fit on D, evaluate once on E. Run it again with the criterion label T as t_d, t_e for the
    sensitivity analysis. (For a binary target the benchmark is an AUC of 0.5; not sketched here.)"""
    t_bar = t_d.mean()
    pred = make_probe(seed).fit(Zres_d, t_d).predict(Zres_e)
    r2 = r2_oos(t_e, pred, t_bar)
    rng = np.random.default_rng(seed)
    n = len(t_e)
    boot = np.empty(n_boot)
    for b in range(n_boot):                                    # resample CEOs in E
        i = rng.integers(0, n, n)
        boot[b] = r2_oos(t_e[i], pred[i], t_bar)
    q95 = float(np.quantile(boot, 0.95))
    if eps is None:
        eps, _ = leakage_bound(Zres_d, t_d, Zres_e, t_e, n_perm, seed=seed)
    return {"r2_oos": float(r2), "q95": q95, "epsilon": eps, "passed": q95 < eps}


def nullspace_projection(Z_d, t_d, Z_apply, rounds=3, min_retained=0.70, alpha=1.0):
    """E8: z <- (I - u u') z, where u is the unit direction of a linear probe fitted on D only.
    Retained variance is measured on D, relative to the variance before the first round."""
    P = np.eye(Z_d.shape[1])
    v0 = Z_d.var(0).sum()
    kept = []
    for _ in range(rounds):
        u = Ridge(alpha=alpha).fit(Z_d @ P, t_d).coef_
        u = u / np.linalg.norm(u)
        P_new = P @ (np.eye(len(u)) - np.outer(u, u))
        v = (Z_d @ P_new).var(0).sum() / v0
        if v < min_retained:                                   # round abandoned
            break
        P = P_new
        kept.append(float(v))
    return Z_apply @ P, kept
