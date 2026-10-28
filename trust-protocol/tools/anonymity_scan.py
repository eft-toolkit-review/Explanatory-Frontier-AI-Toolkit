"""Scan the working tree (and git history, if present) for identifying strings.

    python tools/anonymity_scan.py [--denylist path/to/denylist.txt]

Built-in checks: e-mail addresses, absolute home paths, ORCID ids, links to hosts other than
a short allow-list, non-neutral git author names/e-mails.  Put names and affiliations you must
never publish (one per line, case-insensitive) in a LOCAL denylist file that is NOT committed
(.anon_denylist is git-ignored); the file itself would otherwise reveal them.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
TEXT_EXT = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".cfg", ".toml", ".csv", ""}
ALLOWED_HOSTS = {"github.com", "doi.org", "arxiv.org", "aspredicted.org", "pypi.org", "python.org",
                 "pytorch.org", "download.pytorch.org", "opensource.org", "www.python.org", "creativecommons.org"}
PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "home path": re.compile(r"(/Users/|/home/[a-z]|C:\\Users\\)"),
    "orcid": re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b"),
    "url": re.compile(r"https?://([\w.-]+)[^\s)>\"']*"),
}


def iter_files():
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in TEXT_EXT and f != ".anon_denylist":
                yield os.path.join(d, f)


def scan(denylist=()):
    hits = []
    deny = [w.lower() for w in denylist if w.strip()]
    for path in iter_files():
        rel = os.path.relpath(path, ROOT)
        if rel.endswith("anonymity_scan.py"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for w in deny:
                if w in low:
                    hits.append((rel, i, "denylist", w))
            for name, rx in PATTERNS.items():
                for m in rx.finditer(line):
                    if name == "url":
                        if m.group(1).lower() in ALLOWED_HOSTS:
                            continue
                        hits.append((rel, i, "url", m.group(0)))
                    elif name == "email" and m.group(0).endswith("@users.noreply.github.com"):
                        continue
                    else:
                        hits.append((rel, i, name, m.group(0)))
    return hits


def git_authors():
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        return []
    out = subprocess.run(["git", "-C", ROOT, "log", "--format=%an <%ae>|%cn <%ce>"],
                         capture_output=True, text=True).stdout.splitlines()
    return sorted({p for line in out for p in line.split("|")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--denylist", default=os.path.join(ROOT, ".anon_denylist"))
    a = ap.parse_args()
    words = []
    if os.path.exists(a.denylist):
        with open(a.denylist) as fh:
            words = fh.read().splitlines()
    hits = scan(words)
    for h in hits:
        print("%s:%d  [%s]  %s" % h)
    authors = git_authors()
    if authors:
        print("git identities in history:", *authors, sep="\n  ")
    if not words:
        print("note: no denylist given; only built-in patterns were checked")
    sys.exit(1 if hits else 0)
