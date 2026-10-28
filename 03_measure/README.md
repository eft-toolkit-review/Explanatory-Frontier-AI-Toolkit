# 03_measure: construct measurement (Step 2a)

This folder is the measurement step: it turns the three behavior streams (language, voice, face)
into one estimate of perceived trustworthiness with a **Graph Memory Fusion Network (GMFN)**, and
decides whether that estimate is good enough to be used further (the validity gate, appendix A2.4).

| Path | What it is |
|---|---|---|
| `gmfn_pilot_code/` | **The GMFN code from the pilot**: full GMFN, its ablations, and four benchmark models (EF-LSTM, TFN, LMF, MFN). Model files are unchanged apart from anonymized names. See its README |
| `gmfn.py` | A cleaner re-write of the same GMFN, written so that it runs on CPU and can be tested. Its equivalence with the pilot model is still to be checked weight by weight |
| `dfg_structure.py` | The wiring of the fusion graph (which vertex feeds which; 19 efficacies = 6 + 6 + 7), with no deep-learning dependency |
| `train.py` | Training loop and the random hyperparameter search (space of appendix Table A2.1; every configuration is logged) |
| `splits.py` | Seeded data splits: Mode S (609 / 204 / 203) and Mode K (untouched set + 5-fold cross-fitting) |
| `metrics.py` | Correlation, MAE, R2, Fisher interval, bootstrap interval, Steiger test, reliability (ICC) |
| `gate.py` | The validity gate: correlation at least half the noise ceiling, and fusion better than the best single modality |

Feature extraction:
Language	bert-base-uncased (Devlin et al., 2019)	768-d contextual embedding of each token, computed over the full transcript	per token 
Audio	COVAREP (Degottex et al., 2014)	74 voice-quality and spectral features, including F0 and voicing	10 ms frames
Visual	OpenFace 3.0 (Hu et al., 2025)	Action-unit intensities and left- and right-eye gaze yaw and pitch from the pinned release; landmarks used for quality control only	per video frame
