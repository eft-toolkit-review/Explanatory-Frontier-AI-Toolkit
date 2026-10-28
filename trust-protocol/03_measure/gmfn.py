"""GMFN: Graph Memory Fusion Network (torch).

Re-implementation of the architecture used in the pilot, written from the published
description and the public Graph-MFN code (Zadeh et al., 2018; Bagher Zadeh et al., 2018).
The graph wiring is taken from dfg_structure.py so that it can be tested without torch.

Per time step t:
  1. one LSTMCell per modality (L, A, V);
  2. singleton input of each modality = ReLU(Linear([h_prev, h_new]));
  3. efficacies = EfficacyNet(concat of the three singleton inputs)  -> 19 signed, unbounded scalars;
  4. Dynamic Fusion Graph: 3 bimodal vertices, 1 trimodal vertex, output vertex T (tau);
  5. gated memory: mem = g1 * mem + g2 * tanh(MLP(tau)),  g = sigmoid(MLP([tau, mem])).
Output: MLP([last h_L, last h_A, last h_V, last mem]) -> one trustworthiness value.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dfg_structure import PARENTS, T_INPUT_ORDER, efficacy_slices, edges, n_efficacies


def _two_linear(d_in, d_out, width=100):
    """Vertex network of the DFG: Linear -> Linear, no nonlinearity (as in the public code)."""
    return nn.Sequential(nn.Linear(d_in, width), nn.Linear(width, d_out))


class DynamicFusionGraph(nn.Module):
    def __init__(self, singleton_dims, inner_dim):
        super().__init__()
        self.inner = inner_dim
        # output size of each vertex: unimodal vertices pass their singleton input through
        self.dim = {"L": singleton_dims[0], "A": singleton_dims[1], "V": singleton_dims[2]}
        for v in ("LA", "LV", "AV", "LAV"):
            self.dim[v] = inner_dim
        self.nets = nn.ModuleDict()
        for v, parents in PARENTS.items():
            self.nets[v] = _two_linear(sum(self.dim[p] for p in parents), inner_dim)
        self.t_net = _two_linear(sum(self.dim[v] for v in T_INPUT_ORDER), inner_dim)
        self.eff_net = nn.Sequential(
            nn.Linear(sum(singleton_dims), 100),
            nn.Linear(100, inner_dim),
            nn.Linear(inner_dim, n_efficacies()),
        )
        self.slices = efficacy_slices()

    def forward(self, singletons):
        out = dict(zip(("L", "A", "V"), singletons))
        eff = self.eff_net(torch.cat(singletons, dim=1))  # (batch, 19), signed and unbounded
        for v in ("LA", "LV", "AV", "LAV"):
            a, b = self.slices[v]
            parts = [out[p] * eff[:, a + i].unsqueeze(1) for i, p in enumerate(PARENTS[v])]
            out[v] = self.nets[v](torch.cat(parts, dim=1))
        a, b = self.slices["T"]
        parts = [out[v] * eff[:, a + i].unsqueeze(1) for i, v in enumerate(T_INPUT_ORDER)]
        tau = self.t_net(torch.cat(parts, dim=1))
        return tau, out, eff


class GMFN(nn.Module):
    """Defaults mirror the final pilot configuration (appendix Table A2.1)."""

    def __init__(self, feature_dims=(768, 74, 35), hidden_dims=(128, 16, 128), memsize=400,
                 inner_node_dim=64, nn_cfg=(32, 0.0), gamma1_cfg=(32, 0.0), gamma2_cfg=(64, 0.7),
                 out_cfg=(128, 0.0)):
        super().__init__()
        d_l, d_a, d_v = feature_dims
        h_l, h_a, h_v = hidden_dims
        self.h_dims, self.mem_dim = hidden_dims, memsize
        self.lstm = nn.ModuleList([nn.LSTMCell(d_l, h_l), nn.LSTMCell(d_a, h_a), nn.LSTMCell(d_v, h_v)])
        self.single = nn.ModuleList([nn.Linear(2 * h, h) for h in hidden_dims])
        self.dfg = DynamicFusionGraph(hidden_dims, inner_node_dim)
        # memory
        self.att_fc1, self.att_fc2 = nn.Linear(inner_node_dim, nn_cfg[0]), nn.Linear(nn_cfg[0], memsize)
        self.att_drop = nn.Dropout(nn_cfg[1])
        gin = inner_node_dim + memsize
        self.g1_fc1, self.g1_fc2 = nn.Linear(gin, gamma1_cfg[0]), nn.Linear(gamma1_cfg[0], memsize)
        self.g2_fc1, self.g2_fc2 = nn.Linear(gin, gamma2_cfg[0]), nn.Linear(gamma2_cfg[0], memsize)
        self.g1_drop, self.g2_drop = nn.Dropout(gamma1_cfg[1]), nn.Dropout(gamma2_cfg[1])
        # output
        self.out_fc1 = nn.Linear(sum(hidden_dims) + memsize, out_cfg[0])
        self.out_fc2 = nn.Linear(out_cfg[0], 1)
        self.out_drop = nn.Dropout(out_cfg[1])

    def _memory(self, tau, mem):
        c_hat = torch.tanh(self.att_fc2(self.att_drop(F.relu(self.att_fc1(tau)))))
        both = torch.cat([tau, mem], dim=1)
        g1 = torch.sigmoid(self.g1_fc2(self.g1_drop(F.relu(self.g1_fc1(both)))))
        g2 = torch.sigmoid(self.g2_fc2(self.g2_drop(F.relu(self.g2_fc1(both)))))
        return g1 * mem + g2 * c_hat

    def forward(self, text, audio, video, return_efficacies=False):
        """Inputs are (batch, time, feature).  Returns (batch, 1) [and a list of per-step efficacies]."""
        xs = [text.transpose(0, 1), audio.transpose(0, 1), video.transpose(0, 1)]
        n, steps = xs[0].size(1), xs[0].size(0)
        dev = xs[0].device
        h = [torch.zeros(n, d, device=dev) for d in self.h_dims]
        c = [torch.zeros(n, d, device=dev) for d in self.h_dims]
        mem = torch.zeros(n, self.mem_dim, device=dev)
        effs = []
        for t in range(steps):
            new_h, new_c, singles = [], [], []
            for m in range(3):
                hh, cc = self.lstm[m](xs[m][t], (h[m], c[m]))
                singles.append(F.relu(self.single[m](torch.cat([h[m], hh], dim=1))))
                new_h.append(hh)
                new_c.append(cc)
            tau, _, eff = self.dfg(singles)
            mem = self._memory(tau, mem)
            h, c = new_h, new_c
            if return_efficacies:
                effs.append(eff.detach())
        last = torch.cat(h + [mem], dim=1)
        y = self.out_fc2(self.out_drop(F.relu(self.out_fc1(last))))
        return (y, effs) if return_efficacies else y
