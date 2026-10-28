# Third-party notices

The code in this repository is released under the MIT License (see `LICENSE`). This file records
the origin and terms of third-party material. Components that carry other terms are **not
bundled**: they are named so that a user can check the terms before installing them.

## Architecture lineage (ideas and public reference code, not copied files)

| Source | Used for | License |
|---|---|---|
| Memory Fusion Network (Zadeh et al., 2018), public reference implementation | Gated multi-view memory in `03_measure/gmfn.py` | MIT (as shown on the code host; confirm in the upstream LICENSE file) |
| Dynamic Fusion Graph, CMU-MultimodalSDK (Bagher Zadeh et al., 2018) | Graph wiring in `03_measure/dfg_structure.py` and `gmfn.py` | TO CONFIRM against the upstream LICENSE file before release |

`gmfn.py` is a re-implementation written from the published description and the public code; no
upstream file is copied. If any upstream file is later copied in, keep its copyright header here.

## Dependencies installed by the user (not redistributed here)

numpy, scipy, pandas, scikit-learn (BSD-3-Clause); PyYAML (MIT); PyTorch (BSD-style).

## Planned components with restrictive terms (not included in this release)

Listed in appendix Table A10.2. They will be installed by the user, never vendored: COVAREP
(MATLAB required), OpenFace 3.0 (research use), Parselmouth/Praat (GPL-3.0 or later),
InsightFace models (non-commercial research use). The non-commercial terms restrict commercial
reuse of those components, not the MIT-licensed code in this repository.
