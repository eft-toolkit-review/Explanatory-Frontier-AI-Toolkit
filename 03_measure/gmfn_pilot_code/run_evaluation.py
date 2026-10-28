"""Evaluate one model on a feature file.  Replaces the 14 per-model scripts of the original code.

    python run_evaluation.py --model A7_Graph_MFN --features path/to/features.pkl --weights path/to/model.pth

The model names are the keys of config/config_pilot.json (A1_Graph_MFN_L ... A14_MFN).  Trained
weights are not part of this repository.  The feature file is a pickle with arrays 'text',
'audio', 'vision' of shape (clips, steps, dim), labels on a 1-7 scale ('labels') and an 'info' list.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_seed  # noqa: E402
from config.config import get_config_regression  # noqa: E402
from data_loader import MMDataLoader  # noqa: E402
from models import AMIO  # noqa: E402
from module_test import do_test  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A7_Graph_MFN")
    ap.add_argument("--features", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_pilot.json")
    args = get_config_regression(a.model, "Corpus", cfg)
    args["featurePath"] = a.features
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    loader = MMDataLoader(args, 1)
    model = AMIO(args).to(args["device"])
    model.load_state_dict(torch.load(a.weights, map_location=args["device"]), strict=False)
    do_test(args, a.model, "Corpus", model, loader)
