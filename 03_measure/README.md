# 03_measure: construct measurement (Step 2a)

This folder is the measurement step: it turns the three behavior streams (language, voice, face)
into one estimate of perceived trustworthiness with a **Graph Memory Fusion Network (GMFN)**, and
decides whether that estimate is good enough to be used further (the validity gate, appendix A2.4).

| Path | What it is | Status |
|---|---|---|
| `gmfn_pilot_code/` | **The GMFN code from the pilot**: full GMFN, its ablations, and four benchmark models (EF-LSTM, TFN, LMF, MFN). Model files are unchanged apart from anonymized names. See its README | Provided, not yet run here |
| `gmfn.py` | A cleaner re-write of the same GMFN, written so that it runs on CPU and can be tested. Its equivalence with the pilot model is still to be checked weight by weight | Provided, not yet run here |
| `dfg_structure.py` | The wiring of the fusion graph (which vertex feeds which; 19 efficacies = 6 + 6 + 7), with no deep-learning dependency | Verified here |
| `train.py` | Training loop and the random hyperparameter search (space of appendix Table A2.1; every configuration is logged) | Provided, not yet run here |
| `splits.py` | Seeded data splits: Mode S (609 / 204 / 203) and Mode K (untouched set + 5-fold cross-fitting) | Verified here |
| `metrics.py` | Correlation, MAE, R2, Fisher interval, bootstrap interval, Steiger test, reliability (ICC) | Verified here |
| `gate.py` | The validity gate: correlation at least half the noise ceiling, and fusion better than the best single modality | Verified here |

"Verified here" means checked by the unit tests in `tests/` on simulated data. Nothing in this folder
has been run on the real corpus; the pilot result in the appendix cannot be reproduced from here.
