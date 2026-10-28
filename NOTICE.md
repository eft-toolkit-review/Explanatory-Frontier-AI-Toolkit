# Notice: where parts of this project come from

This repository's own code is released under the MIT licence (see `LICENSE`). This file says which
parts build on other people's public work, and which tools we deliberately do not include. It is
information for users; it does not change the licence of anything.

## 1. Public work this project builds on

| Project | What we use it for | Where | Its licence |
|---|---|---|---|
| Memory Fusion Network (public reference code) | The idea and layout of the gated memory | `03_measure/gmfn.py`, `03_measure/gmfn_pilot_code/models/` | MIT (shown on the code host); to be confirmed against the upstream LICENSE file |
| CMU-MultimodalSDK (Dynamic Fusion Graph) | The graph wiring | same | to be confirmed against the upstream LICENSE file |
| Tensor Fusion Network reference code | Benchmark model A12 | `03_measure/gmfn_pilot_code/models/A12_TFN.py` | to be confirmed |
| Low-rank Multimodal Fusion reference code | Benchmark model A13 | `03_measure/gmfn_pilot_code/models/A13_LMF.py` | to be confirmed |

`03_measure/gmfn.py` is written from the published description and is not a copy of any upstream file.
The files in `gmfn_pilot_code/models/` keep the source references in their headers.

## 2. Tools installed by the user, not included here

numpy, scipy, pandas, scikit-learn, PyYAML, PyTorch, easydict and tqdm are installed with pip and keep
their own licences.

## 3. Planned components with restrictive licences (not included)

Some tools in the full pipeline (appendix Table A10.2) are free for research but restrict commercial
use or are copyleft: COVAREP (needs MATLAB), OpenFace 3.0 (research use), Parselmouth/Praat (GPL-3.0 or
later) and InsightFace models (non-commercial research use). They are not part of this repository. A
user installs them separately and should read their terms first. Their terms limit commercial reuse of
those tools; they do not limit the MIT-licensed code here.
