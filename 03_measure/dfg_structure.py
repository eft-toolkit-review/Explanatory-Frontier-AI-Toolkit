"""Wiring of the Dynamic Fusion Graph (DFG) used by GMFN.  Pure Python, no dependencies.

Vertices: three unimodal (L language, A acoustic, V visual), three bimodal (LA, LV, AV),
one trimodal (LAV) and the output vertex T.  Every directed edge carries one efficacy,
a learned scalar per time step.  The torch module in gmfn.py is built from this table,
so a wiring error shows up in tests/test_dfg_structure.py without needing torch.
"""
from itertools import combinations

MODALITIES = ("L", "A", "V")

# Vertices in the order in which they are computed and fed to T.
VERTICES = ("L", "A", "V", "LA", "LV", "AV", "LAV")

# Which lower vertices feed each internal vertex.  A bimodal vertex receives only its two
# unimodal parents; LAV receives all six lower vertices (as in the public MFN-graph code).
PARENTS = {
    "LA": ("L", "A"),
    "LV": ("L", "V"),
    "AV": ("A", "V"),
    "LAV": ("L", "A", "V", "LA", "LV", "AV"),
}

# Order in which the seven vertex outputs are concatenated as the input of T.
T_INPUT_ORDER = VERTICES


def edges():
    """All directed edges (source, target), grouped as bimodal, trimodal, then into T."""
    out = []
    for tgt in ("LA", "LV", "AV", "LAV"):
        for src in PARENTS[tgt]:
            out.append((src, tgt))
    for src in T_INPUT_ORDER:
        out.append((src, "T"))
    return out


def n_efficacies():
    return len(edges())


def efficacy_slices():
    """Index range of each target's efficacies inside the 19-vector, in `edges()` order."""
    res, i = {}, 0
    for tgt in ("LA", "LV", "AV", "LAV", "T"):
        k = sum(1 for _, t in edges() if t == tgt)
        res[tgt] = (i, i + k)
        i += k
    return res


def is_unimodal(v):
    return len(v) == 1


def modalities_of(v):
    return tuple(c for c in v if c in MODALITIES)


def check_powerset():
    """The vertex set equals the non-empty subsets of {L,A,V}, ordered by size."""
    subsets = []
    for r in (1, 2, 3):
        subsets += ["".join(c) for c in combinations(MODALITIES, r)]
    return tuple(subsets) == VERTICES
