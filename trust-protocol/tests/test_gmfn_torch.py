"""Needs torch; skipped when it is not installed (CI installs it)."""
import unittest
import _paths  # noqa: F401

try:
    import torch
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch not installed")
class TestGMFN(unittest.TestCase):
    def test_shapes_and_efficacies(self):
        from gmfn import GMFN
        m = GMFN(feature_dims=(12, 6, 5), hidden_dims=(8, 4, 8), memsize=16, inner_node_dim=10,
                 nn_cfg=(8, 0.0), gamma1_cfg=(8, 0.0), gamma2_cfg=(8, 0.0), out_cfg=(8, 0.0))
        b, t = 3, 7
        y, effs = m(torch.randn(b, t, 12), torch.randn(b, t, 6), torch.randn(b, t, 5), return_efficacies=True)
        self.assertEqual(tuple(y.shape), (b, 1))
        self.assertEqual(len(effs), t)
        self.assertEqual(tuple(effs[0].shape), (b, 19))

    def test_gradients_reach_every_vertex_network(self):
        from gmfn import GMFN
        m = GMFN(feature_dims=(12, 6, 5), hidden_dims=(8, 4, 8), memsize=16, inner_node_dim=10,
                 nn_cfg=(8, 0.0), gamma1_cfg=(8, 0.0), gamma2_cfg=(8, 0.0), out_cfg=(8, 0.0))
        m(torch.randn(2, 4, 12), torch.randn(2, 4, 6), torch.randn(2, 4, 5)).sum().backward()
        for name, p in m.dfg.named_parameters():
            self.assertIsNotNone(p.grad, name)

    def test_learns_toy_signal(self):
        from gmfn import GMFN
        from train import fit
        torch.manual_seed(0)
        n, t = 160, 5
        text, audio, video = torch.randn(n, t, 12), torch.randn(n, t, 6), torch.randn(n, t, 5)
        y = text[:, :, 0].mean(1) + audio[:, :, 0].mean(1)
        data = {"text": text, "audio": audio, "video": video, "y": y}
        m = GMFN(feature_dims=(12, 6, 5), hidden_dims=(8, 4, 8), memsize=16, inner_node_dim=10,
                 nn_cfg=(8, 0.0), gamma1_cfg=(8, 0.0), gamma2_cfg=(8, 0.0), out_cfg=(8, 0.0))
        base = float(y[120:].abs().mean())
        best, _ = fit(m, data, list(range(120)), list(range(120, n)), lr=5e-3, batch_size=32, max_epochs=60)
        self.assertLess(best, base)


if __name__ == "__main__":
    unittest.main()
