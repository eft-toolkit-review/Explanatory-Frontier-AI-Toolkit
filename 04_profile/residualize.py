"""A3.1 Cross-fitted residualization (illustrative).

Z_res[i] = Z[i] - f_k(T_hat[i]) for every clip i in held-out fold k, where the
map f_k is fitted on the other K-1 folds only.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def natural_spline_basis(x, knots):
    """Natural cubic spline basis (Hastie, Tibshirani, Friedman, sec. 5.2.1), intercept dropped.
    K knots give K - 1 columns, so 5 knots = 4 df."""
    x = np.asarray(x, float)
    K = len(knots)

    def d(k):
        return (np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3) / (knots[-1] - knots[k])

    cols = [x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.column_stack(cols)


def _fit_map(t_train, Z_train, df=4, alphas=np.logspace(-3, 3, 13)):
    knots = np.quantile(t_train, np.linspace(0, 1, df + 1))     # df + 1 knots -> df columns
    B = natural_spline_basis(t_train, knots)
    mu, sd = B.mean(0), B.std(0) + 1e-12
    model = RidgeCV(alphas=alphas, cv=5).fit((B - mu) / sd, Z_train)

    def predict(t_new):
        Bn = natural_spline_basis(np.clip(t_new, knots[0], knots[-1]), knots)
        return model.predict((Bn - mu) / sd)

    return predict


def cross_fit_residual(Z, t_hat, k=5, df=4, seed=0):
    """Return (Z_res, Z_hat, fold). Every row of Z_hat comes from a map that never saw that row.
    t_hat must be out of sample for every clip (A3.1 requirement; Mode K out-of-fold estimates)."""
    Z, t_hat = np.asarray(Z, float), np.asarray(t_hat, float)
    fold = np.empty(len(Z), int)
    Z_hat = np.empty_like(Z)
    for j, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(Z)):
        fold[te] = j
        Z_hat[te] = _fit_map(t_hat[tr], Z[tr], df)(t_hat[te])
    return Z - Z_hat, Z_hat, fold
