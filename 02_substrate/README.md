# 02_substrate: Shared substrate (Step 1, A1.2 and A1.3)

**Purpose.** Break each video into a label-free representation for profiling. It is fixed before the
construct-specific analyses and reused in all later steps.

## Files

schema.yaml : the data interface (A1.2). Which streams, dimensions and labels a feature file holds.

align.py : corresponding to Technical Appendix A1.2 Features and alignment (illustrative code).
Summarizes the audio and visual frames inside each token's time interval so that every modality has
one vector per token (E1), and at the mean time, standardizes features with training data only.

profile_representation.py : corresponding to Technical Appendix A1.3 (illustrative code). Builds the four summary blocks of a clip (text, image, audio , cross-modal coordination), balances the blocks, and projects them on the leading principal
components (E2). 

## Substrate features (Table A1.1)

| Stream | Tool (pinned in the registry) | Features | Rate |
|---|---|---|---|
| Language | bert-base-uncased | 768-d contextual embedding of each token | per token |
| Audio | COVAREP | 74 voice-quality and spectral features, including F0 and voicing | 10 ms frames |
| Visual | OpenFace 3.0 | action-unit intensities and gaze yaw and pitch | per video frame |

