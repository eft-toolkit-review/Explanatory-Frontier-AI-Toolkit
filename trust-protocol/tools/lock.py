"""Lock record: fix the registry (and chosen files) before a protocol stage, verify later.

    python tools/lock.py create --stage training --files 00_protocol/registry.yaml
    python tools/lock.py verify

`create` writes 00_protocol/lock_record.json with SHA-256 hashes of the listed files and of the
registry content.  Archive that hash outside the repository (preregistration, e.g. AsPredicted)
so a later change to a locked file can be detected by anyone.  Nothing is locked in the Stage-1
artifact: lock_record.json does not exist until the team creates it.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "00_protocol", "lock_record.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create(stage, files, path=LOCK):
    rec = {"stage": stage,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "files": {f: sha256_file(os.path.join(ROOT, f)) for f in sorted(files)}}
    payload = json.dumps(rec["files"], sort_keys=True).encode()
    rec["record_sha256"] = hashlib.sha256(payload).hexdigest()
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


def verify(path=LOCK):
    """Return (ok, messages).  No lock record means nothing is locked: ok is False."""
    if not os.path.exists(path):
        return False, ["no lock record: nothing has been locked"]
    with open(path) as f:
        rec = json.load(f)
    msgs = []
    for rel, digest in rec["files"].items():
        full = os.path.join(ROOT, rel)
        if not os.path.exists(full):
            msgs.append(f"missing: {rel}")
        elif sha256_file(full) != digest:
            msgs.append(f"changed since lock: {rel}")
    payload = json.dumps(rec["files"], sort_keys=True).encode()
    if hashlib.sha256(payload).hexdigest() != rec.get("record_sha256"):
        msgs.append("record hash does not match its file list")
    return (not msgs), msgs or [f"ok: stage '{rec['stage']}', {len(rec['files'])} files, "
                                f"record {rec['record_sha256'][:16]}"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--stage", required=True)
    c.add_argument("--files", nargs="+", required=True)
    sub.add_parser("verify")
    a = ap.parse_args()
    if a.cmd == "create":
        r = create(a.stage, a.files)
        print("locked", r["record_sha256"])
    else:
        ok, msgs = verify()
        print("\n".join(msgs))
        sys.exit(0 if ok else 1)
