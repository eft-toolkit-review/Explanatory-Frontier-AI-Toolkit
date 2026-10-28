"""Training loop and random hyperparameter search for GMFN (torch).

Protocol hooks: every configuration tried and its validation score are written to the search
log (registry: search_log); the seed comes from splits.derive_seed (registry: master_seed).
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

from gmfn import GMFN  # noqa: F401  (used by build_model)

SEARCH_SPACE = {  # appendix Table A2.1
    "hidden_dims": [(128, 16, 128), (64, 16, 64), (128, 32, 128), (256, 32, 256), (64, 32, 64)],
    "memsize": [64, 128, 256, 300, 400],
    "inner_node_dim": [20, 32, 64, 128],
    "hidden_layer": [32, 64, 128, 256],   # drawn separately for NN, gamma1, gamma2, out
    "dropout": [0.0, 0.2, 0.5, 0.7],      # drawn separately for the same four components
    "batch_size": [32, 64],
    "lr": [0.0005, 0.001, 0.002, 0.005, 0.01],
    "weight_decay": [0.0, 0.001, 0.005, 0.01],
}
COMPONENTS = ("nn", "gamma1", "gamma2", "out")


def sample_config(rng):
    pick = lambda k: SEARCH_SPACE[k][int(rng.integers(len(SEARCH_SPACE[k])))]
    cfg = {k: pick(k) for k in ("hidden_dims", "memsize", "inner_node_dim", "batch_size", "lr", "weight_decay")}
    for c in COMPONENTS:
        cfg[c] = (pick("hidden_layer"), pick("dropout"))
    return cfg


def build_model(cfg, feature_dims=(768, 74, 35)):
    return GMFN(feature_dims=feature_dims, hidden_dims=cfg["hidden_dims"], memsize=cfg["memsize"],
                inner_node_dim=cfg["inner_node_dim"], nn_cfg=cfg["nn"], gamma1_cfg=cfg["gamma1"],
                gamma2_cfg=cfg["gamma2"], out_cfg=cfg["out"])


def fit(model, data, idx_train, idx_val, lr=5e-4, batch_size=64, max_epochs=100, patience=10,
        weight_decay=0.0, device="cpu", seed=0):
    """data = dict(text, audio, video, y) as tensors; returns best validation MAE and the model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
    best, best_state, bad = float("inf"), None, 0

    def batches(idx, shuffle):
        idx = np.array(idx)
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), batch_size):
            b = torch.as_tensor(idx[s:s + batch_size])
            yield [data[k][b].to(device) for k in ("text", "audio", "video", "y")]

    for epoch in range(max_epochs):
        model.train()
        for t, a, v, y in batches(idx_train, True):
            opt.zero_grad()
            loss = loss_fn(model(t, a, v).squeeze(1), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = [torch.abs(model(t, a, v).squeeze(1) - y).cpu() for t, a, v, y in batches(idx_val, False)]
        val = float(torch.cat(errs).mean())
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, model


def predict(model, data, idx, device="cpu", batch_size=128):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = torch.as_tensor(np.array(idx)[s:s + batch_size])
            out.append(model(*[data[k][b].to(device) for k in ("text", "audio", "video")]).squeeze(1).cpu())
    return torch.cat(out).numpy()


def random_search(data, idx_train, idx_val, n_configs=30, seed=0, log_path="search_log.jsonl", device="cpu"):
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, None)
    with open(log_path, "a") as log:
        for i in range(n_configs):
            cfg = sample_config(rng)
            model = build_model(cfg)
            score, model = fit(model, data, idx_train, idx_val, lr=cfg["lr"], batch_size=cfg["batch_size"],
                               weight_decay=cfg["weight_decay"], device=device, seed=seed + i)
            log.write(json.dumps({"i": i, "config": {k: list(v) if isinstance(v, tuple) else v
                                                    for k, v in cfg.items()},
                                  "val_mae": score, "time": time.time()}, default=str) + "\n")
            if score < best[0]:
                best = (score, cfg, model)
    return best
