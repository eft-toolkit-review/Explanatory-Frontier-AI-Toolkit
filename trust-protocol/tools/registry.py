"""Load and validate 00_protocol/registry.yaml."""
import os
import sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "00_protocol", "registry.yaml")
REQUIRED = ("id", "parameter", "role", "value", "fixed_before", "status")
STATUSES = ("suggested", "frozen")


def load(path=PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def validate(doc):
    errs, seen = [], set()
    for e in doc["entries"]:
        for k in REQUIRED:
            if k not in e or e[k] in (None, ""):
                errs.append(f"{e.get('id', '?')}: missing {k}")
        if e.get("id") in seen:
            errs.append(f"duplicate id {e['id']}")
        seen.add(e.get("id"))
        if e.get("status") not in STATUSES:
            errs.append(f"{e.get('id')}: bad status {e.get('status')}")
    if "master_seed" not in doc.get("machine", {}):
        errs.append("machine.master_seed missing")
    return errs


if __name__ == "__main__":
    doc = load()
    errs = validate(doc)
    for e in errs:
        print("ERROR:", e)
    n = len(doc["entries"])
    frozen = sum(1 for e in doc["entries"] if e["status"] == "frozen")
    print(f"{n} entries, {frozen} frozen, {n - frozen} suggested")
    sys.exit(1 if errs else 0)
