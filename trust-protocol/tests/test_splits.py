import unittest
import numpy as np
import _paths  # noqa: F401
import splits


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.t = np.random.default_rng(0).normal(4, 1, 1016)

    def test_mode_s_sizes_and_disjoint(self):
        s = splits.mode_s(self.t, seed=1)
        self.assertEqual([len(s[k]) for k in ("train", "val", "test")], [609, 204, 203])
        allidx = np.concatenate([s["train"], s["val"], s["test"]])
        self.assertEqual(len(set(allidx)), 1016)

    def test_mode_s_bad_sizes(self):
        with self.assertRaises(ValueError):
            splits.mode_s(self.t[:100], sizes=(609, 204, 203))

    def test_mode_k_structure(self):
        k = splits.mode_k(self.t, seed=3)
        self.assertEqual((len(k["E"]), len(k["D"])), (203, 813))
        self.assertEqual(len(k["folds"]), 5)
        self.assertFalse(set(k["E"]) & set(k["D"]))
        outer = np.concatenate([f["outer_test"] for f in k["folds"]])
        self.assertEqual(sorted(outer), sorted(k["D"]))   # every D case predicted exactly once
        for f in k["folds"]:
            used = set(f["train"]) | set(f["inner_val"]) | set(f["outer_test"])
            self.assertFalse(used & set(k["E"]))           # E never touched
            self.assertFalse(set(f["train"]) & set(f["outer_test"]))
            self.assertFalse(set(f["inner_val"]) & set(f["outer_test"]))
            self.assertAlmostEqual(len(f["inner_val"]) / (len(f["train"]) + len(f["inner_val"])), 0.2, delta=0.01)

    def test_stratified(self):
        k = splits.mode_k(self.t, seed=3)
        q = splits.quintile_strata(self.t)
        share = np.bincount(q[k["E"]], minlength=5) / len(k["E"])
        self.assertTrue(np.all(np.abs(share - 0.2) < 0.03))

    def test_reproducible_and_seed_derivation(self):
        a, b = splits.mode_s(self.t, seed=5), splits.mode_s(self.t, seed=5)
        self.assertTrue(np.array_equal(a["test"], b["test"]))
        self.assertEqual(splits.derive_seed("x.py"), splits.derive_seed("x.py"))
        self.assertNotEqual(splits.derive_seed("x.py"), splits.derive_seed("y.py"))


if __name__ == "__main__":
    unittest.main()
