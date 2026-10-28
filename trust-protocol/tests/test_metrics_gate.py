import unittest
import numpy as np
import _paths  # noqa: F401
import gate
import metrics


class TestMetrics(unittest.TestCase):
    def test_basic(self):
        y = np.array([1., 2, 3, 4]); p = np.array([1., 2, 3, 5])
        self.assertAlmostEqual(metrics.mae(y, p), 0.25)
        self.assertAlmostEqual(metrics.r2(y, y), 1.0)
        self.assertGreater(metrics.pearson_r(y, p), 0.9)

    def test_fisher_ci_matches_pilot_report(self):
        lo, hi = metrics.fisher_ci(0.83, 203)
        self.assertAlmostEqual(lo, 0.78, delta=0.01)
        self.assertAlmostEqual(hi, 0.87, delta=0.01)

    def test_steiger_type1_error(self):
        """Under H0 (equal population correlations) the test rejects about 5% of the time."""
        rng = np.random.default_rng(0)
        n, rej, reps = 200, 0, 1500
        cov = np.array([[1, .5, .5], [.5, 1, .4], [.5, .4, 1]])   # corr(j,k)=corr(j,h)=.5
        for _ in range(reps):
            x = rng.multivariate_normal(np.zeros(3), cov, size=n)
            c = np.corrcoef(x.T)
            _, p = metrics.steiger_z(c[0, 1], c[0, 2], c[1, 2], n)
            rej += p < 0.05
        self.assertAlmostEqual(rej / reps, 0.05, delta=0.02)

    def test_steiger_power(self):
        rng = np.random.default_rng(1)
        n, rej, reps = 200, 0, 300
        cov = np.array([[1, .7, .4], [.7, 1, .4], [.4, .4, 1]])
        for _ in range(reps):
            x = rng.multivariate_normal(np.zeros(3), cov, size=n)
            c = np.corrcoef(x.T)
            rej += metrics.steiger_z(c[0, 1], c[0, 2], c[1, 2], n)[1] < 0.05
        self.assertGreater(rej / reps, 0.9)

    def test_icc2k_recovers_reliability(self):
        rng = np.random.default_rng(2)
        n, k, noise = 2000, 5, 1.0
        true = rng.normal(size=(n, 1))
        x = true + rng.normal(scale=noise, size=(n, k))
        expected = 1.0 / (1.0 + noise ** 2 / k)       # reliability of a k-rater mean
        self.assertAlmostEqual(metrics.icc2k(x), expected, delta=0.02)


class TestGate(unittest.TestCase):
    def test_rmax(self):
        self.assertAlmostEqual(gate.r_max_from_icc(0.81), 0.9)
        with self.assertRaises(ValueError):
            gate.r_max_from_icc(0)

    def test_pass_and_fail(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=300)
        good = y + rng.normal(scale=0.5, size=300)
        weak = y + rng.normal(scale=1.5, size=300)
        g = gate.validity_gate(y, good, weak, icc_k=0.8, n_boot=300)
        self.assertTrue(g["passes"])
        g2 = gate.validity_gate(y, weak, good, icc_k=0.8, n_boot=300)   # fusion worse than unimodal
        self.assertFalse(g2["pass_fusion"])
        noise = rng.normal(size=300)
        g3 = gate.validity_gate(y, noise, noise * 0.9, icc_k=0.8, n_boot=300)
        self.assertFalse(g3["pass_correlation"])


if __name__ == "__main__":
    unittest.main()
