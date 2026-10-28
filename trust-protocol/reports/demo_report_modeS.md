# Demo report (synthetic data, mode S, seed 2)

Split sizes: train 609, val 204, test 203

ICC(2,k) = 0.889; noise ceiling r_max = 0.943

| model | R2 | r | MAE |
|---|---|---|---|
| L | 0.647 | 0.805 | 0.499 |
| A | 0.615 | 0.786 | 0.521 |
| V | 0.588 | 0.770 | 0.544 |
| LAV | 0.779 | 0.884 | 0.404 |

Best unimodal: L.  Validity gate: **PASS**
- r = 0.884 vs threshold 0.472
- Delta R2 = 0.132, 95% bootstrap interval (0.081, 0.193)

Rival constructs (Steiger test, estimate vs. rating):

| rival | r(est, T) | r(est, rival) | r(T, rival) | z | p |
|---|---|---|---|---|---|
| authenticity | 0.884 | 0.512 | 0.508 | 9.67 | 0 |
| competence | 0.884 | 0.393 | 0.388 | 10.84 | 0 |
| warmth | 0.884 | 0.435 | 0.421 | 10.39 | 0 |
| charisma | 0.884 | 0.386 | 0.345 | 10.67 | 0 |
| confidence | 0.884 | 0.387 | 0.373 | 10.84 | 0 |
