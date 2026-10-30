"""Evaluation splits: Mode S (single split) and Mode K (untouched set E + K-fold cross-fitting).

Stratification is on quintiles of the trustworthiness rating T.  All randomness comes from
a seed derived from the registry master seed and the name of the calling script.
"""
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

MASTER_SEED = 20261101  # registry: master_seed


def derive_seed(script_name, master_seed=MASTER_SEED):
    h = hashlib.sha256(f"{master_seed}:{script_name}".encode()).hexdigest()
    return int(h[:8], 16)


def quintile_strata(t, n=5):
    t = np.asarray(t, dtype=float)
    ranks = t.argsort(kind="mergesort").argsort(kind="mergesort")
    return np.minimum((ranks * n) // len(t), n - 1)


def mode_s(t, sizes=(609, 204, 203), seed=0):
    """Single split train / validation / test with the given sizes, stratified on T quintile."""
    t = np.asarray(t)
    n = len(t)
    if sum(sizes) != n:
        raise ValueError(f"sizes {sizes} sum to {sum(sizes)}, but there are {n} cases")
    idx = np.arange(n)
    strata = quintile_strata(t)
    rest, test = train_test_split(idx, test_size=sizes[2], stratify=strata, random_state=seed)
    train, val = train_test_split(rest, test_size=sizes[1], stratify=strata[rest], random_state=seed + 1)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def mode_k(t, e_share=0.20, k=5, inner_val_share=0.20, seed=0, e_index=None):
    """Untouched set E (never used for fitting or tuning) + k outer folds on the rest D.

    If `e_index` is given it is used as E instead of a random draw.
    Returns {"E": idx, "folds": [{"train": idx, "inner_val": idx, "outer_test": idx}, ...]}.
    Each fold fits on `train`, tunes on `inner_val`, and predicts `outer_test`; together the
    outer-test parts cover D exactly once (out-of-sample estimate for every case in D).
    """
    t = np.asarray(t)
    n = len(t)
    idx = np.arange(n)
    strata = quintile_strata(t)
    if e_index is not None:
        # use a given set as E, e.g. the pilot's test clips, so Mode S and Mode K are comparable
        e = np.asarray(e_index)
        d = np.setdiff1d(idx, e)
    else:
        n_e = int(round(e_share * n))
        d, e = train_test_split(idx, test_size=n_e, stratify=strata, random_state=seed)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + 2)
    folds = []
    for j, (tr, te) in enumerate(skf.split(d, strata[d])):
        tr_idx, te_idx = d[tr], d[te]
        fit, inner = train_test_split(tr_idx, test_size=inner_val_share,
                                      stratify=strata[tr_idx], random_state=seed + 10 + j)
        folds.append({"train": np.sort(fit), "inner_val": np.sort(inner), "outer_test": np.sort(te_idx)})
    return {"E": np.sort(e), "D": np.sort(d), "folds": folds}
