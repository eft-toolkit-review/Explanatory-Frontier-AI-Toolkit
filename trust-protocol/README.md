# Trust measurement protocol: research artifact (Stage 1)

A protocol-first prototype for measuring observer-perceived trustworthiness from multimodal
behavior (language, voice, face) and for keeping every analytic choice fixed before it is used.
It accompanies the technical appendix of the proposal. **This is a proof of concept**: the
measurement step and the protocol machinery are implemented; later steps are marked *planned*.

## What is here, and what is not

| Folder | Contents | Status |
|---|---|---|
| `00_protocol/` | Parameter registry (appendix Table A13) in machine-readable form, lock sequence, information barriers, evidence ledger, deviations log, preregistration template | Implemented |
| `02_substrate/` | Data schema (`schema.yaml`) | Implemented |
| `03_measure/` | Seeded splits (Mode S, Mode K), GMFN and Dynamic Fusion Graph, training and search, metrics, validity gate | Implemented |
| `tools/` | Registry validator, SHA-256 lock record, anonymity scan | Implemented |
| `07_reopen/` | Guard that refuses to run without a verifying lock record | Guard implemented; analysis planned |
| `demo/` | Synthetic data and an end-to-end run (ridge stand-in for GMFN, no torch needed) | Implemented |
| `01_ingest/`, `04_profile/`, `05_intervene/`, `06_experiment/` | Collection, residualization and leakage audit, structure discovery, stimulus generation, experiment analysis | **Planned** |

`00_protocol/evidence_ledger.md` lists claim by claim what is demonstrated here and what is not.

## Data

No real recordings, features or ratings are included, and none are needed to run this
repository. `demo/make_synthetic.py` generates synthetic data in the layout of
`02_substrate/schema.yaml`. Numbers produced on synthetic data show that the code runs and that
the decision rules behave as intended; they say nothing about the real corpus. The pilot result
reported in the appendix cannot be reproduced from this repository.

## Quick start

```bash
pip install -r requirements.txt            # numpy, scipy, pandas, scikit-learn, pyyaml
python -m unittest discover -s tests -v    # unit tests (GMFN tests skip without torch)
python tools/registry.py                   # validate the parameter registry
python demo/run_demo.py --mode S           # synthetic corpus -> splits -> baselines -> gate
python demo/run_demo.py --mode K
```

GMFN itself needs PyTorch: `pip install -r requirements-torch.txt`, then
`python -m unittest tests.test_gmfn_torch -v`.

## Protocol in brief

1. Parameters are written down in `00_protocol/registry.yaml` with the stage before which each
   must be fixed. Everything is `suggested` until the team locks it.
2. `python tools/lock.py create --stage <name> --files <paths>` writes SHA-256 hashes;
   `python tools/lock.py verify` detects any later change.
3. The validity gate (`03_measure/gate.py`) passes only if the correlation reaches half the noise
   ceiling and fusion beats the best unimodal model with a positive lower interval bound.
4. `07_reopen/run_reopen.py` refuses to run until a lock record verifies.

Randomness: every script derives its seed from the registry master seed and its own name
(`03_measure/splits.py: derive_seed`).

## Licence and third-party material

MIT (see `LICENSE`). Origins and terms of third-party material are in `NOTICE.md`.
