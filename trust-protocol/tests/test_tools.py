import json
import os
import tempfile
import unittest
import _paths  # noqa: F401
import anonymity_scan
import lock
import registry


class TestRegistry(unittest.TestCase):
    def test_valid(self):
        doc = registry.load()
        self.assertEqual(registry.validate(doc), [])
        self.assertEqual(len(doc["entries"]), 33)
        self.assertEqual(doc["machine"]["master_seed"], 20261101)
        self.assertTrue(all(e["status"] == "suggested" for e in doc["entries"]))

    def test_detects_problems(self):
        doc = registry.load()
        doc["entries"].append(dict(doc["entries"][0]))
        doc["entries"][1]["status"] = "maybe"
        errs = registry.validate(doc)
        self.assertTrue(any("duplicate" in e for e in errs))
        self.assertTrue(any("bad status" in e for e in errs))


class TestLock(unittest.TestCase):
    def test_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, "lock.json")
            f = "00_protocol/registry.yaml"
            self.assertFalse(lock.verify(rec)[0])                  # nothing locked yet
            lock.create("test", [f], path=rec)
            self.assertTrue(lock.verify(rec)[0])
            with open(rec) as fh:
                data = json.load(fh)
            data["files"][f] = "0" * 64                            # tamper with a hash
            with open(rec, "w") as fh:
                json.dump(data, fh)
            ok, msgs = lock.verify(rec)
            self.assertFalse(ok)
            self.assertTrue(any("changed" in m or "does not match" in m for m in msgs))


class TestAnonymity(unittest.TestCase):
    def test_repo_is_clean(self):
        hits = anonymity_scan.scan([])
        self.assertEqual(hits, [], hits)

    def test_denylist_catches(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.md")
            fixture = "written by Jane Q. Author, " + "jane" + "@" + "uni.edu, " + "/Use" + "rs/jane/work\n"
            with open(p, "w") as fh:
                fh.write(fixture)
            old = anonymity_scan.ROOT
            anonymity_scan.ROOT = d
            try:
                kinds = {h[2] for h in anonymity_scan.scan(["jane q. author"])}
            finally:
                anonymity_scan.ROOT = old
            self.assertTrue({"denylist", "email", "home path"} <= kinds)


if __name__ == "__main__":
    unittest.main()


class TestReopenGuard(unittest.TestCase):
    def test_refuses_without_lock(self):
        import subprocess
        import sys
        script = os.path.join(_paths.ROOT, "07_reopen", "run_reopen.py")
        if os.path.exists(lock.LOCK):
            self.skipTest("a lock record exists in this checkout")
        r = subprocess.run([sys.executable, script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        self.assertIn("REFUSED", r.stdout)
