import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("03_measure", "tools", "demo"):
    sys.path.insert(0, os.path.join(ROOT, p))
