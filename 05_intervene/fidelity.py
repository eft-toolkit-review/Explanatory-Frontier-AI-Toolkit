"""A4.5 Fidelity battery and A4.6 failure rules: the acceptance rules of Table A4.2 as functions (illustrative).
These functions only apply the registered thresholds, which are
suggested parameters (A13).
"""
import numpy as np
from scipy import stats


def identity_ok(sim_edit, sim_ceiling, share=0.95):
    """Median ArcFace similarity of edited vs source frames >= 95% of the ceiling (source vs its own re-encode)."""
    return bool(np.median(sim_edit) >= share * np.median(sim_ceiling))


def content_ok(wer, max_abs=0.02):
    """Whisper transcript of the edited audio vs the source transcript: word error rate at most 2% absolute."""
    return bool(wer <= max_abs)


def target_ok(realized_change, nominal_dose, low, high, band=(0.8, 1.2)):
    """Realized change within 80-120% of the nominal dose, and the low version below the high one."""
    ratio = realized_change / nominal_dose
    return bool(band[0] <= ratio <= band[1] and low < high)


def equivalence_ok(diff, within_sd, margin=0.2, alpha=0.05):
    """Non-target invariance: paired differences (high - low) of one other catalog measure, in units of its
    within-clip SD, must lie within +/- 0.2 SD (two one-sided t tests, alpha = .05)."""
    d = np.asarray(diff, float) / within_sd
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((d.mean() + margin) / se, n - 1)       # H0: mean <= -margin
    p_upper = stats.t.cdf((d.mean() - margin) / se, n - 1)      # H0: mean >= +margin
    return bool(max(p_lower, p_upper) < alpha)


def sync_ok(sync_edit, sync_source, share=0.95):
    """Lip-sync score of edited clips at least 95% of the source clips. (The blind artifact review is done by people.)"""
    return bool(np.mean(sync_edit) >= share * np.mean(sync_source))


def detectability_ok(n_correct, n_total, ceiling=0.60, level=0.95):
    """Two-alternative forced choice: upper one-sided 95% (Clopper-Pearson) bound of pooled accuracy <= 0.60."""
    upper = 1.0 if n_correct == n_total else stats.beta.ppf(level, n_correct + 1, n_total - n_correct)
    return bool(upper <= ceiling), float(upper)


def pilot_decision(n_pass, n_pilot=20, need=16):
    """A4.6: of 20 pilot clips at least 16 must pass the battery, otherwise e* is dropped and the next-ranked is tried."""
    return bool(n_pass >= need)


def main_decision(n_failed, n_eligible, max_share=0.20):
    """A4.6: if more than 20% of eligible clips fail, the element is declared non-manipulable with the current tools."""
    return "non-manipulable" if n_failed / n_eligible > max_share else "go"
