# Notice: where parts of this project come from

This repository's own code is released under the MIT licence (see `LICENSE`). This file says which parts
build on other people's public work, which tools of the full pipeline are not included, and which material
is withheld. It is information for users; it does not change the licence of anything.


## 1. Tools of the full pipeline that are not included

The tools below are named in the Technical Appendix (Table A10.2) and in `05_intervene`. None of them is part
of this repository, and no code here imports them. A user installs them separately and should read their terms
first. The restrictions below concern those tools; they do not limit the MIT-licensed code in this repository.

| Tool | Used for | Licence or access |
|---|---|---|
| Whisper large-v3 | Transcription | MIT |
| bert-base-uncased | Language features | Apache-2.0 |
| COVAREP | Acoustic features | Needs MATLAB; see its own terms |
| OpenFace 3.0 | Visual features | Released for research use |
| LivePortrait | Visual edits | Code and weights MIT |
| InsightFace buffalo_l models (loaded by LivePortrait; also ArcFace identity check) | Face detection, identity check | Non-commercial research use only |
| Parselmouth and Praat | Vocal edits | GPL-3.0 or later |

## 2. Software installed by the user

numpy, scipy, pandas, scikit-learn, statsmodels, PyYAML, PyTorch, easydict and tqdm are installed with pip
and keep their own licences.

## 3. Material that is not in this repository

Raw videos, transcripts, features, ratings, model weights, and participant data are not in this repository.
Transcripts reproduce copyrighted speech, and the link from identifiers to named people would publish a score
for each person. These items are held by the authors (Technical Appendix A10.1).
