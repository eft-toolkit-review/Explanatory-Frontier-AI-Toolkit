import unittest
import _paths  # noqa: F401
import dfg_structure as d


class TestDFG(unittest.TestCase):
    def test_vertex_set_is_powerset(self):
        self.assertTrue(d.check_powerset())

    def test_nineteen_efficacies(self):
        self.assertEqual(d.n_efficacies(), 19)
        sl = d.efficacy_slices()
        self.assertEqual({k: v[1] - v[0] for k, v in sl.items()}, {"LA": 2, "LV": 2, "AV": 2, "LAV": 6, "T": 7})

    def test_bimodal_parents(self):
        self.assertEqual(d.PARENTS["LV"], ("L", "V"))   # LV is fed only by L and V
        self.assertEqual(d.PARENTS["LA"], ("L", "A"))
        self.assertEqual(d.PARENTS["AV"], ("A", "V"))

    def test_trimodal_receives_six_lower_vertices(self):
        self.assertEqual(set(d.PARENTS["LAV"]), {"L", "A", "V", "LA", "LV", "AV"})
        self.assertEqual(len(d.PARENTS["LAV"]), 6)

    def test_t_input_order_and_size(self):
        self.assertEqual(d.T_INPUT_ORDER, ("L", "A", "V", "LA", "LV", "AV", "LAV"))
        self.assertEqual(sum(1 for s, t in d.edges() if t == "T"), 7)

    def test_no_edge_skips_into_bimodal(self):
        for s, t in d.edges():
            if t in ("LA", "LV", "AV"):
                self.assertEqual(len(s), 1)


if __name__ == "__main__":
    unittest.main()
