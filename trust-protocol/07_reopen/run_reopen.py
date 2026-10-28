"""Reopening of the frozen profile (appendix A6).  PLANNED: not implemented in this release.

By protocol this script may run only after the confirmatory lock record exists and verifies.
Exit codes: 3 = refused (no valid lock record), 4 = lock record valid but analysis not implemented yet.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lock  # noqa: E402


def main():
    ok, msgs = lock.verify()
    if not ok:
        print("REFUSED: the confirmatory lock record is missing or does not verify.")
        print("\n".join(msgs))
        return 3
    print("Lock record verified:", *msgs)
    print("The reopening analysis is planned and not implemented in this release.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
