#!/usr/bin/env python3
"""Check a local download against the Hugging Face tree at a pinned revision: every file present with the listed size,
and the sha256 of every LFS file equal to its LFS oid (read with O_DIRECT where the filesystem allows it, so ~200 GB of
weights do not flush the page cache). stdlib only.

    hf_verify.py REPO REVISION LOCAL_DIR [FILE ...]     (no FILE: every file of the revision)

Prints one line a mismatch and a summary; exit status 0 only when nothing is missing, short or different.
"""
import hashlib, json, mmap, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

repo, rev, local = sys.argv[1], sys.argv[2], sys.argv[3]
only = set(sys.argv[4:])
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
tree = [e for e in json.load(urllib.request.urlopen(url, timeout=60)) if e["type"] == "file"]
if only:
    tree = [e for e in tree if e["path"] in only]
    missing_names = only - {e["path"] for e in tree}
    if missing_names:
        sys.exit(f"not in {repo}@{rev}: {sorted(missing_names)}")

BLOCK = 16 << 20


def sha256(path):
    h = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        buf = mmap.mmap(-1, BLOCK)  # page-aligned, as O_DIRECT needs
        with os.fdopen(fd, "rb", buffering=0) as f:
            while n := f.readinto(buf):
                h.update(memoryview(buf)[:n])
    except OSError:  # no O_DIRECT on this filesystem: plain reads
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while b := f.read(BLOCK):
                h.update(b)
    return h.hexdigest()


bad = 0
lfs = []
for e in tree:
    p = os.path.join(local, e["path"])
    if not os.path.isfile(p):
        print("MISSING", e["path"]); bad += 1; continue
    if os.path.getsize(p) != e["size"]:
        print("SIZE", e["path"], os.path.getsize(p), "!=", e["size"]); bad += 1; continue
    if e.get("lfs"):
        lfs.append(e)
with ThreadPoolExecutor(8) as ex:
    for e, got in zip(lfs, ex.map(lambda e: sha256(os.path.join(local, e["path"])), lfs)):
        if got != e["lfs"]["oid"]:
            print("SHA256", e["path"], got, "!=", e["lfs"]["oid"]); bad += 1
total = sum(e["size"] for e in tree)
print(f"{repo}@{rev[:8]}: files={len(tree)} bytes={total} lfs-checked={len(lfs)} bad={bad}")
sys.exit(1 if bad else 0)
