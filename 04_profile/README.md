# 04_profile: residualized profiling (Step 2b, A3)

**Purpose.** Remove what the construct estimate explains from the behavioral representation, check that it is gone, and look for reproducible structure in what remains. No outcome information is used.

## Files

residualize.py : corresponding to Technical Appendix A3.1 Cross-fitted residualization
(illustrative code). 

leakage_audit.py: corresponding to Technical Appendix A3.2 (illustrative code).
A gradient-boosted probe is trained on the development set D to recover the construct estimate from
the residual, and is evaluated once on the untouched evaluation set E (E6). Acceptance rule is only hypothetical, to be tested.

structure.py: corresponding to Technical Appendix A3.3 (illustrative code).


stability.py: corresponding to Technical Appendix A3.4 (illustrative code). Stability test

fingerprint.py : corresponding to Technical Appendix A3.5 (illustrative code) 


## Not in this folder

A3.5 adjudication (two researchers, a third to resolve, Cohen's kappa) and A3.6 robustness (the
GMFN hidden state with Procrustes alignment, the Amazon transcripts) are described in the appendix
only. They need the real residual profile.

## Suggested values used as defaults (A13, to be fixed before the step they belong to)

| Item | Value |
|---|---|
| Residual map | ridge on a natural cubic spline of the construct estimate, 4 df, lambda by 5-fold CV |
| Leakage probe | gradient boosting, depth 3, 200 trees, learning rate 0.05 |
| Leakage bound | 0.05, or the 95th percentile of 500 permutation refits if larger |
| Projection rounds | at most 3; a round is dropped if retained variance falls below 70% |
| Structure space | 10 principal components |
| HDBSCAN | minimum cluster size 30, minimum samples 10 |
| Mixture | full covariance, regularization 1e-3, K up to 8, 20 starts, smallest expected size 30 |
| Stability | 500 bootstrap samples; retained at 0.75; dissolved at 0.5 |


