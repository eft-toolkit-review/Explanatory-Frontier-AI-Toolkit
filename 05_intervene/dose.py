"""A4.3 Dose and A4.4 pitch recipe, as arithmetic on the measured quantity (illustrative).

Dose (E14)，Pitch (E15). This file only transforms an F0 contour. The resynthesis with Praat TD-PSOLA
(Parselmouth) is not included in this release.
"""
import numpy as np


def segment_values(series, rate_hz, stat=np.mean, seg_s=15.0):
    """Value of the element in each 15-second segment of one clip (e.g. the mean of the smile composite)."""
    n = int(seg_s * rate_hz)
    return np.array([stat(series[i:i + n]) for i in range(0, len(series) - n + 1, n)])


def element_iqr(segment_values_by_clip):
    """Pooled IQR of the segment values after subtracting each clip's mean."""
    pooled = np.concatenate([v - v.mean() for v in segment_values_by_clip if len(v) > 0])
    q1, q3 = np.quantile(pooled, [0.25, 0.75])
    return float(q3 - q1)


def doses(x_e, iqr_e):
    """E14: (low, high) target values of the element for one clip."""
    return x_e - iqr_e / 2.0, x_e + iqr_e / 2.0


def semitones(f0_hz, ref_hz=100.0):
    return 12.0 * np.log2(np.asarray(f0_hz, float) / ref_hz)


def shift_level(l, delta):
    """E15, level. Unvoiced frames (NaN) stay unvoiced."""
    return l + delta


def scale_variability(l, gamma):
    """E15, variability: scale deviations around the clip mean."""
    m = np.nanmean(l)
    return m + gamma * (l - m)


def gamma_for_target_sd(l, target_sd):
    """Scale that moves the standard deviation of l to a target value."""
    return target_sd / np.nanstd(l)
