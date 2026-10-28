# Demo report (synthetic data, mode K, seed 0)

Untouched set E: 203; D: 813; folds: 5

ICC(2,k) = 0.869; noise ceiling r_max = 0.932

| model | R2 | r | MAE |
|---|---|---|---|
| L | 0.843 | 0.918 | 0.309 |
| A | 0.740 | 0.861 | 0.393 |
| V | 0.650 | 0.811 | 0.466 |
| LAV | 0.845 | 0.921 | 0.308 |

Best unimodal: L.  Validity gate: **FAIL**
- r = 0.921 vs threshold 0.466
- Delta R2 = 0.002, 95% bootstrap interval (-0.005, 0.011)

Rival constructs (Steiger test, estimate vs. rating):

| rival | r(est, T) | r(est, rival) | r(T, rival) | z | p |
|---|---|---|---|---|---|
| authenticity | 0.921 | 0.416 | 0.373 | 12.51 | 0 |
| competence | 0.921 | 0.307 | 0.251 | 13.24 | 0 |
| warmth | 0.921 | 0.470 | 0.466 | 12.30 | 0 |
| charisma | 0.921 | 0.374 | 0.364 | 13.08 | 0 |
| confidence | 0.921 | 0.400 | 0.370 | 12.74 | 0 |
