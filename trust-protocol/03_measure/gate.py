"""Validity gate (appendix A2.4).  Decision rule only; thresholds come from the registry.

passes  <=>  (i) r >= 0.5 * r_max   and   (ii) lower bound of the interval for
Delta R^2 (GMFN minus best unimodal LSTM, on the same evaluation cases) exceeds zero.
"""
import math
import numpy as np
from metrics import r2, pearson_r, bootstrap_ci


def r_max_from_icc(icc_k):
    """Noise ceiling for an out-of-sample correlation: sqrt of the reliability of the mean rating."""
    if not 0 < icc_k <= 1:
        raise ValueError("ICC(2,k) must lie in (0, 1]")
    return math.sqrt(icc_k)


def delta_r2(y, p_gmfn, p_unimodal):
    return r2(y, p_gmfn) - r2(y, p_unimodal)


def validity_gate(y, p_gmfn, p_best_unimodal, icc_k, r_over_rmax_min=0.5,
                  n_boot=2000, level=0.95, seed=0):
    y, pg, pu = (np.asarray(a, float) for a in (y, p_gmfn, p_best_unimodal))
    r = pearson_r(y, pg)
    rmax = r_max_from_icc(icc_k)
    lo, hi = bootstrap_ci(delta_r2, (y, pg, pu), n_boot=n_boot, level=level, seed=seed)
    ok_corr = r >= r_over_rmax_min * rmax
    ok_fusion = lo > 0
    return {
        "r": r, "r_max": rmax, "r_threshold": r_over_rmax_min * rmax,
        "delta_r2": delta_r2(y, pg, pu), "delta_r2_ci": (lo, hi),
        "pass_correlation": bool(ok_corr), "pass_fusion": bool(ok_fusion),
        "passes": bool(ok_corr and ok_fusion),
    }
