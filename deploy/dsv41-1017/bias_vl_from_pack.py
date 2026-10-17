#!/usr/bin/env python3
"""The 43 `*.ffn.gate.bias_vl` (F32[384] trunk, F32[128] MTP) of a local safetensors checkpoint, written in the format of the engine's
/cache/dsv41-bias-vl/bias_vl.safetensors (sorted keys, metadata source/count/sha256 per tensor). stdlib only.
    bias_vl_from_pack.py PACK_DIR SOURCE_LABEL OUT"""
import hashlib, json, struct, sys
from pathlib import Path
pack, label, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
wm = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
keys = sorted(k for k in wm if k.endswith("ffn.gate.bias_vl"))
data = {}
for k in keys:
    with open(pack / wm[k], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); m = h[k]
        assert m["dtype"] == "F32" and len(m["shape"]) == 1, (k, m)   # [384] trunk, [128] MTP
        s, e = m["data_offsets"]; f.seek(8 + n + s); data[k] = (f.read(e - s), m["shape"]); assert len(data[k][0]) == 4 * m["shape"][0]
meta = {"source": label, "count": str(len(keys))}
hdr, off = {}, 0
for k in keys:
    b, shape = data[k]
    meta["sha256:" + k] = hashlib.sha256(b).hexdigest()
    hdr[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(b)]}; off += len(b)
hb = json.dumps({"__metadata__": meta, **hdr}, separators=(",", ":")).encode(); hb += b" " * (-len(hb) % 8)
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb)
    for k in keys: f.write(data[k][0])
print(out, len(keys), "tensors")
