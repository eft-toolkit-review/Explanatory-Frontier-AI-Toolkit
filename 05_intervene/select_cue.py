"""A4.2 Selection rule (illustrative).

For each catalog element e: shuffle that element's features across CEOs in E, re-predict with the fixed
trained models(E13). 
"""
import numpy as np


def r2_rows(y, pred):
    """R2 for each row of pred (rows = permutations) or for a single prediction vector."""
    pred = np.atleast_2d(pred)
    sse = ((y - pred) ** 2).sum(1)
    return 1.0 - sse / ((y - y.mean()) ** 2).sum()


def permuted_predictions(predict, X, columns, n_perm=200, seed=0):
    """X: array (clips, ..., features) for E. columns: indices (last axis) that measure the element.
    Returns (n_perm, clips) predictions after shuffling those features across CEOs."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, len(X)))
    for p in range(n_perm):
        Xp = X.copy()
        Xp[..., columns] = X[rng.permutation(len(X))][..., columns]
        out[p] = predict(Xp)
    return out


def element_loss(y, pred, pred_perm, n_boot=2000, seed=0):
    """E13: Delta_e = R2_T - R2_{T, pi_e}; L_e = 2.5th percentile of its bootstrap distribution over CEOs."""
    rng = np.random.default_rng(seed)
    n = len(y)
    delta = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        delta[b] = r2_rows(y[i], pred[i])[0] - r2_rows(y[i], pred_perm[:, i]).mean()
    full = r2_rows(y, pred)[0] - r2_rows(y, pred_perm).mean()
    return {"delta": float(full), "L": float(np.quantile(delta, 0.025))}


def select_element(L, passes_fidelity):
    """L: {element: L_e}. passes_fidelity: {element: bool}. Returns e*, or None (no confirmatory experiment)
    and the ranking, which is reported either way (A12)."""
    ranking = sorted(L, key=L.get, reverse=True)
    ok = [e for e in ranking if L[e] > 0 and passes_fidelity.get(e, False)]
    return (ok[0] if ok else None), ranking
