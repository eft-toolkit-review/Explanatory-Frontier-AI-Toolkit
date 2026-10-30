"""A5.4 Precision of the treatment effect (E17), illustrative.

Var(b1) ~ (1/M) * (sigma_v^2 + 4 * (sigma_u^2 + sigma_e^2) / n_c)

M source clips, n_c ratings per clip (half of them per version). sigma_v^2: variance of the clip-specific
treatment effect, sigma_u^2: participants, sigma_e^2: residual. The variance components come from the pilot.
The minimum detectable effect is obtained by simulation ; it is not computed here.
"""
import numpy as np


def ratings_per_clip(participants=360, clips_per_participant=8, clips=48):
    return participants * clips_per_participant / clips            # 360 * 8 / 48 = 60


def var_b1(M, s2_v, s2_u, s2_e, n_c):
    return (s2_v + 4.0 * (s2_u + s2_e) / n_c) / M


def se_b1(*args, **kwargs):
    return float(np.sqrt(var_b1(*args, **kwargs)))
