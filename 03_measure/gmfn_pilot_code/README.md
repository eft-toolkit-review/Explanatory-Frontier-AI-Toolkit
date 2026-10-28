# gmfn_pilot_code: the GMFN and benchmark models used in the pilot

The GMFN and the benchmark models as they were implemented for the pilot, kept for provenance
and for the baseline comparison in appendix Table A2.2.

| Path | Contents |
|---|---|
| `models/A7_Graph_MFN.py` | Full GMFN (dynamic fusion graph + multi-view gated memory) |
| `models/A1` to `A6`, `A8` to `A10` | GMFN with one or two modalities (A1 to A6) and ablations: no memory (A8), no graph (A9), no weighting of the LSTM transform (A10) |
| `models/A11_EF_LSTM.py`, `A12_TFN.py`, `A13_LMF.py`, `A14_MFN.py` | Benchmarks: early-fusion LSTM, Tensor Fusion, Low-rank Fusion, Memory Fusion Network |
| `config/config_pilot.json` | Dimensions and hyperparameters per model |
| `data_loader.py`, `utils/`, `module_test.py` | Loading of a feature file, metrics (Acc-2, F1, MAE, correlation), evaluation loop |
| `run_evaluation.py` | One script to evaluate any model on a feature file |

Changes relative to the original code: the dataset key was renamed to `Corpus`, a comment on the
label coding was made generic, the `language` entry was removed from the config, and the 14
per-model evaluation scripts were replaced by `run_evaluation.py`. The model files are unchanged.

Not included: trained weights and the training script. The evaluation script needs weights from
the team (planned). Known limitation: `A7_Graph_MFN.py` calls `.cuda()` in the graph, so it needs a
GPU; the refactored `../gmfn.py` has no such restriction and is the version used by the tests.
`gmfn.py` follows the same wiring, but its equivalence with `A7_Graph_MFN.py` has not yet been
checked weight by weight (planned).

Data conventions in the loader: labels are stored on the 1 to 7 scale and centred by subtracting
4; with `need_normalized` set, acoustic and visual features are averaged over time.
