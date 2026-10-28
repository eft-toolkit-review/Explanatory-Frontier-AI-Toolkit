"""Synthetic stand-in for the CEO corpus.  No real person or recording is involved.

Clip-level features for three modalities, a trustworthiness rating T on a 1-7 scale (mean of
k raters), and ratings of rival constructs that correlate with T.  Used only to exercise the
protocol code; the numbers it produces say nothing about the real corpus.
"""
import numpy as np


def generate(n=1016, k_raters=5, dims=(64, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)                                   # latent trustworthiness
    T_true = np.clip(4 + 1.0 * z, 1, 7)
    feats = []
    for j, d in enumerate(dims):
        w = rng.normal(size=d) / np.sqrt(d)
        other = rng.normal(size=(n, d))
        strength = (0.20, 0.30, 0.30)[j]                        # each modality carries part of the signal
        feats.append(strength * np.outer(z, w) * np.sqrt(d) + other)
    rater_noise = rng.normal(scale=0.8, size=(n, k_raters))
    rater_bias = rng.normal(scale=0.2, size=(1, k_raters))
    raters = np.clip(T_true[:, None] + rater_noise + rater_bias, 1, 7)
    T = raters.mean(axis=1)
    rivals = {}
    for name, rho in (("authenticity", 0.5), ("competence", 0.4), ("warmth", 0.45),
                      ("charisma", 0.35), ("confidence", 0.4)):
        r = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
        rivals[name] = np.clip(4 + r + rng.normal(scale=0.3, size=n), 1, 7)
    return {"L": feats[0], "A": feats[1], "V": feats[2], "T": T, "raters": raters, "rivals": rivals}
