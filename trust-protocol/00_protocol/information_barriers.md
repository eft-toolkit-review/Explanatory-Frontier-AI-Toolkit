# Information barriers (appendix Table A7.1)

No object below may be built with the experimental outcome Y until the confirmatory result is locked.

| Object | Built from | Frozen | Used by | Outcome Y allowed? |
|---|---|---|---|---|
| Whisper transcripts | Audio | Before profile | Step 1 | No |
| z_i | Step 1 features, no labels | Before profile | A3 | No |
| T_hat_i | Features and labels in D; features only in E | Before profile | A3, A4 | No |
| z_i^res, C_k, P_i(C_k) | z_i and T_hat_i | Before registration | A6 | No |
| Selected cue e* | Fold models, E, pilot battery | Before registration | A4, A5 | No |
| Stimuli | Source clips, recipe | Before registration | A5 | No |
| Y_pj | Participants | After data collection | A5 first, then A6 | Yes, after the lock record |

Status in this release: the barrier is enforced in code only for the last row, by
`07_reopen/run_reopen.py`, which refuses to run without a verifying lock record. The other rows
are procedures for the team; their code is planned (`04_profile`, `05_intervene`, `06_experiment`).
