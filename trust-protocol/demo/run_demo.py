"""End-to-end protocol demo on synthetic data (no torch needed).

    python demo/run_demo.py [--mode S|K] [--seed 0]

Steps: seeded split -> ridge baselines (L, A, V, fused) -> metrics -> validity gate -> rival check.
The "fused" model here is a ridge regression on concatenated features, a stand-in that lets the
protocol logic run anywhere.  The GMFN itself is in 03_measure/gmfn.py (needs torch).
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_measure"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate, metrics, splits  # noqa: E402
from make_synthetic import generate  # noqa: E402


def ridge_fit_predict(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    m = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(sc.transform(X[tr]), y[tr])
    return m.predict(sc.transform(X[te]))


def run(mode="S", seed=0, n=1016):
    d = generate(n=n, seed=seed)
    T = d["T"]
    X = {"L": d["L"], "A": d["A"], "V": d["V"], "LAV": np.hstack([d["L"], d["A"], d["V"]])}
    s = splits.derive_seed("demo/run_demo.py") + seed
    lines = [f"# Demo report (synthetic data, mode {mode}, seed {seed})", ""]
    if mode == "S":
        sp = splits.mode_s(T, seed=s)
        e_idx = sp["test"]
        preds = {k: ridge_fit_predict(v, T, np.r_[sp["train"], sp["val"]], e_idx) for k, v in X.items()}
        lines.append(f"Split sizes: train {len(sp['train'])}, val {len(sp['val'])}, test {len(e_idx)}")
    else:
        sp = splits.mode_k(T, seed=s)
        e_idx = sp["E"]
        preds = {k: ridge_fit_predict(v, T, sp["D"], e_idx) for k, v in X.items()}
        lines.append(f"Untouched set E: {len(e_idx)}; D: {len(sp['D'])}; folds: {len(sp['folds'])}")
    y = T[e_idx]
    icc = metrics.icc2k(d["raters"])
    lines += ["", f"ICC(2,k) = {icc:.3f}; noise ceiling r_max = {np.sqrt(icc):.3f}", "",
              "| model | R2 | r | MAE |", "|---|---|---|---|"]
    for k in ("L", "A", "V", "LAV"):
        p = preds[k]
        lines.append(f"| {k} | {metrics.r2(y, p):.3f} | {metrics.pearson_r(y, p):.3f} | {metrics.mae(y, p):.3f} |")
    best_uni = max(("L", "A", "V"), key=lambda k: metrics.r2(y, preds[k]))
    g = gate.validity_gate(y, preds["LAV"], preds[best_uni], icc_k=icc, seed=s)
    lines += ["", f"Best unimodal: {best_uni}.  Validity gate: **{'PASS' if g['passes'] else 'FAIL'}**",
              f"- r = {g['r']:.3f} vs threshold {g['r_threshold']:.3f}",
              f"- Delta R2 = {g['delta_r2']:.3f}, 95% bootstrap interval ({g['delta_r2_ci'][0]:.3f}, {g['delta_r2_ci'][1]:.3f})",
              "", "Rival constructs (Steiger test, estimate vs. rating):", "",
              "| rival | r(est, T) | r(est, rival) | r(T, rival) | z | p |", "|---|---|---|---|---|---|"]
    pe = preds["LAV"]
    for name, rv in d["rivals"].items():
        r_t, r_r, r_tr = (metrics.pearson_r(pe, y), metrics.pearson_r(pe, rv[e_idx]),
                          metrics.pearson_r(y, rv[e_idx]))
        z, p = metrics.steiger_z(r_t, r_r, r_tr, len(y))
        lines.append(f"| {name} | {r_t:.3f} | {r_r:.3f} | {r_tr:.3f} | {z:.2f} | {p:.3g} |")
    return "\n".join(lines), g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["S", "K"], default="S")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    report, _ = run(a.mode, a.seed)
    out = os.path.join(ROOT, "reports", f"demo_report_mode{a.mode}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(report + "\n")
    print(report)
