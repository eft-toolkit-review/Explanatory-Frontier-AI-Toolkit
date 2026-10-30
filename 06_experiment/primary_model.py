"""A5.3 Primary model (E16), illustrative.

    Y_pj = b0 + b1 * D_pj + u_p + w_j + v_j * D_pj + e_pj

Y: rating of trustworthiness (1-7). D: 0 = low version, 1 = high version. u_p: participant, w_j: clip,
v_j: clip-specific treatment effect. Estimated by REML with statsmodels MixedLM, participants and clips as
crossed variance components. The clip intercept and the clip slope are independent here (the model has no
intercept-slope correlation). The test of b1 is a t test with (clips - fixed effects) degrees of freedom,
because the treatment effect varies by clip and the clips are the effective sample (E17). Two-sided, alpha = .05.

If the slope variance is estimated at zero or the fit does not converge, the slope variance is dropped,
never on the basis of the test of b1.
"""
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

COMPONENTS = {
    "slope":    {"participant": "0 + C(participant)", "clip": "0 + C(clip)", "clip_slope": "0 + C(clip):D"},
    "no_slope": {"participant": "0 + C(participant)", "clip": "0 + C(clip)"},
}


def fit_mixed(fixed, d, n_fixed):
    """Fit 'Y ~ fixed' with crossed participant and clip components. Returns (structure, result, df)."""
    d = d.assign(all_=1)
    for structure, vc in COMPONENTS.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = smf.mixedlm(f"Y ~ {fixed}", d, groups="all_", re_formula="0", vc_formula=vc).fit(reml=True)
        names = res.model.exog_vc.names
        at_zero = any(v / d["Y"].var() < 1e-4 for n, v in zip(names, res.vcomp) if n == "clip_slope")
        if res.converged and not caught and not at_zero:
            break
    df = d["clip"].nunique() - n_fixed                       # clips minus fixed effects
    return structure, res, df


def coefficient(res, name, df):
    est, se = res.params[name], res.bse[name]
    t = est / se
    return {"est": float(est), "se": float(se), "df": int(df), "t": float(t), "p": float(2 * stats.t.sf(abs(t), df))}


def fit_primary(d, outcome="Y"):
    """d: one row per rating with participant, clip, D and the outcome."""
    d = d.assign(Y=d[outcome])
    structure, res, df = fit_mixed("D", d, n_fixed=2)
    return {"structure": structure, "b1": coefficient(res, "D", df), "fit": res}


RIVALS = ["authenticity", "competence", "warmth", "charisma", "confidence"]   # specificity analysis: fit_primary(d, outcome=r)
