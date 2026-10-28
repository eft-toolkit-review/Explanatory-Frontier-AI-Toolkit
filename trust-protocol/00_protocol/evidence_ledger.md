# Evidence ledger (appendix Table A8.1)

What this repository demonstrates, claim by claim. "Planned" is not "demonstrated".

| Claim | Status in this repository | Where |
|---|---|---|
| Parameter registry is machine-readable and validated | Demonstrated (CI) | `00_protocol/registry.yaml`, `tools/registry.py` |
| Seeded splits for Mode S (609/204/203) and Mode K (E=203, D=813, 5 folds, 20% inner validation) | Demonstrated (unit tests) | `03_measure/splits.py` |
| Validity gate, Fisher interval, Steiger test, ICC(2,k) | Demonstrated on simulated data (unit tests, including type-I error of the Steiger test) | `03_measure/metrics.py`, `gate.py` |
| DFG wiring: 19 efficacies = 6 + 6 + 7 | Demonstrated (unit tests, no torch needed) | `03_measure/dfg_structure.py` |
| GMFN forward pass, shapes, efficacy output, toy training | Implemented; tests run in CI with torch | `03_measure/gmfn.py`, `tests/test_gmfn_torch.py` |
| Hash lock record and reopening guard | Demonstrated (unit tests, CI) | `tools/lock.py`, `07_reopen/run_reopen.py` |
| End-to-end flow on synthetic data (ridge stand-in for GMFN) | Demonstrated | `demo/run_demo.py`, `reports/` |
| GMFN on the real corpus (pilot: r = 0.83, MAE = 0.46, n = 203) | Reported in the appendix; **not reproducible from this repository** (real data not released) | appendix A2.4 |
| Residualization, leakage audit (A3) | Planned | `04_profile/` |
| Structure discovery, stability (A3) | Planned | `04_profile/` |
| Cue ranking, stimulus generation, fidelity battery (A4) | Planned | `05_intervene/` |
| Experiment analysis (A5), reopening (A6) | Planned | `06_experiment/`, `07_reopen/` |
