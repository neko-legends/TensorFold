#!/usr/bin/env python3
"""Compare `tensorfold vision` dumps (ZIG_DIR) with vision_ref.py's (REF_DIR): cosine, relative L2, max |diff| per stage.
  vision_cmp.py REF_DIR ZIG_DIR [--min-cos 0.999]"""
import sys, os, numpy as np

def load(p):
    return (np.fromfile(p, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)

ref, zig = sys.argv[1], sys.argv[2]
min_cos = float(sys.argv[sys.argv.index("--min-cos") + 1]) if "--min-cos" in sys.argv else 0.999
stages = [("embed", "embed"), ("block00", "block00"), ("block15", "block15"), ("block31", "block31"), ("feat_ln", "feat_ln"), ("out", "zig_out")]
bad = 0
for r, z in stages:
    pr, pz = os.path.join(ref, r + ".bin"), os.path.join(zig, z + ".bin")
    if not (os.path.exists(pr) and os.path.exists(pz)):
        print(f"{r:8s} missing"); continue
    a, b = load(pr), load(pz)
    if a.shape != b.shape:
        print(f"{r:8s} SHAPE {a.shape} vs {b.shape}"); bad += 1; continue
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    mx = float(np.abs(a - b).max())
    ok = cos >= min_cos
    bad += not ok
    print(f"{r:8s} n={a.size:>10d} cos={cos:.6f} rel_l2={rel:.4f} max_abs={mx:.3f} {'ok' if ok else 'FAIL'}")
sys.exit(1 if bad else 0)
