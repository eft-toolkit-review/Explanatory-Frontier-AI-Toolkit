"""A6.1 Reopening model (E18), illustrative. Run only after the lock record exists (A7.2).

    Y_pj = b0 + b1 D_pj + sum_k b2k P_j(C_k) + sum_k b3k D_pj P_j(C_k) + u_p + w_j + v_j D_pj + e_pj

The baseline configuration is omitted, because the membership probabilities sum to one. P_j(C_k) is read from
the frozen, hashed profile files. If K' - 1 exceeds M / 10, each contrast is fitted separately
(decision_rules.fit_jointly). Same estimator, same fallback and same degrees of freedom rule as
06_experiment/primary_model.py. The estimates go to decision_rules.py for classification (E19).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "06_experiment"))
from primary_model import coefficient, fit_mixed  # noqa: E402


def fit_reopen(d, members, joint=True):
    """d: one row per rating with participant, clip, D, Y and one column per non-baseline configuration.
    members: names of those columns. Returns one row per configuration."""
    groups = [list(members)] if joint else [[m] for m in members]
    rows = []
    for g in groups:
        fixed = " + ".join(["D"] + g + [f"D:{k}" for k in g])
        structure, res, df = fit_mixed(fixed, d, n_fixed=2 + 2 * len(g))
        for k in g:
            b2, b3 = coefficient(res, k, df), coefficient(res, f"D:{k}", df)
            rows.append({"configuration": k, "structure": structure,
                         "b2": b2["est"], "se2": b2["se"], "df2": df, "b3": b3["est"], "se3": b3["se"], "df3": df})
    return pd.DataFrame(rows)
