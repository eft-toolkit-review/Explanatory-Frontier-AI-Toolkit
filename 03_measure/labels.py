"""A2.1 Criterion labels (illustrative).

Turns annotator ratings into the label used by the models, and reports how reliable it is.
Input: a (clips x raters) array of 1-7 ratings, one array per construct.
"""
import numpy as np
from metrics import icc2k

MIDPOINT = 4.0  # midpoint of the 1-7 scale


def criterion_label(raters):
    """Label T_i = mean rating of clip i, plus the reliability of that mean (ICC(2,k)).

    r_max = sqrt(ICC(2,k)) is the noise ceiling for any out-of-sample correlation with T.
    """
    raters = np.asarray(raters, float)
    t = raters.mean(axis=1)
    icc = icc2k(raters)
    return {"T": t, "icc": icc, "r_max": float(np.sqrt(max(icc, 0.0)))}


def center(t, midpoint=MIDPOINT):
    """Rescale 1-7 ratings to -3..+3 (the pilot loader subtracts 4). Binary accuracy is then
    measured as agreement in sign."""
    return np.asarray(t, float) - midpoint


def uncenter(t, midpoint=MIDPOINT):
    return np.asarray(t, float) + midpoint


def rating_report(raters_by_construct):
    """Reliability of each construct's mean rating and the correlations among the mean ratings.

    raters_by_construct: {"trustworthiness": array, "authenticity": array, ...}, same clips.
    The correlation matrix is reported next to the discriminant checks (A2.4), because ratings of
    adjacent constructs tend to move together.
    """
    names = list(raters_by_construct)
    labels = {n: criterion_label(raters_by_construct[n]) for n in names}
    means = np.column_stack([labels[n]["T"] for n in names])
    return {
        "reliability": {n: {"icc": labels[n]["icc"], "r_max": labels[n]["r_max"]} for n in names},
        "names": names,
        "correlations": np.corrcoef(means.T),
    }
