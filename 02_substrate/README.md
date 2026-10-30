# 02_substrate: shared behavioral substrate (Step 1, A1.2 and A1.3)

**Purpose.** Turn each video into a time-indexed record of language, voice and face that is built
without access to the focal construct or to any experimental outcome. It is fixed before the
construct-specific analyses and reused in all later steps.

## Files

`schema.yaml` : the data interface (A1.2). Which streams, dimensions and labels a feature file holds.
Any corpus that follows this layout can be passed through `03_measure`.

`align.py` : corresponding to Technical Appendix A1.2 Features and alignment (illustrative code).
Summarizes the audio and visual frames inside each token's time interval so that every modality has
one vector per token (E1), and standardizes features with training data only.

`profile_representation.py` : corresponding to Technical Appendix A1.3 A label-free representation
for profiling (illustrative code). Builds the four summary blocks of a clip (language, acoustic,
visual, cross-modal coordination), balances the blocks, and projects them on the leading principal
components (E2). No function takes a label or a rating.

## Substrate features (Table A1.1)

| Stream | Tool (pinned in the registry) | Features | Rate |
|---|---|---|---|
| Language | bert-base-uncased | 768-d contextual embedding of each token | per token |
| Audio | COVAREP | 74 voice-quality and spectral features, including F0 and voicing | 10 ms frames |
| Visual | OpenFace 3.0 | action-unit intensities and gaze yaw and pitch | per video frame |

## Status

The pilot features were extracted before the tool versions were pinned; a re-run with pinned
versions is planned. The extraction scripts are not part of this release. The two Python files need
numpy only, and have been checked on simulated data only.
