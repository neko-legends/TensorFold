#!/usr/bin/env python3
"""The server's image request against the checkpoint's own HF processor: prompt ids (our virtual ids read as the
`<image>` id) and each image's normalized pixels (ours bf16 [3, H, W]; HF float32 rounded to bf16 the same way).

  TF_NEMOTRON_VISION_DUMP=DIR tensorfold-native serve ...   then one chat request with the image(s) and PROMPT
  vision_e2e_ref.py MODEL_DIR DIR PROMPT IMAGE [IMAGE...] [--thinking]
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

args = [a for a in sys.argv[1:] if not a.startswith("--")]
thinking = "--thinking" in sys.argv
model, dump, prompt, images = args[0], args[1], args[2], args[3:]
proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
tok = proc.tokenizer
content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
imgs = [Image.open(p).convert("RGB") for p in images]
out = proc(text=text, images=imgs, return_tensors="pt")
ref_ids = out["input_ids"][0].tolist()
ours = np.fromfile(os.path.join(dump, "ids.bin"), dtype=np.uint32).astype(np.int64)
image_id = proc.image_token_id
mapped = [image_id if v >= (1 << 24) else int(v) for v in ours]
same = mapped == ref_ids
prefix = not same and len(mapped) < len(ref_ids) and ref_ids[: len(mapped)] == mapped
print(f"ids: ours {len(mapped)} ref {len(ref_ids)} {'IDENTICAL' if same else 'a PREFIX of the reference' if prefix else 'DIFFER'}")
if prefix:
    # the server dumps the rendered history; the generation prompt follows it in the served prompt
    print(f"  the reference's last {len(ref_ids) - len(mapped)} ids: {tok.decode(ref_ids[len(mapped):])!r}")
elif not same:
    for i, (a, b) in enumerate(zip(mapped, ref_ids)):
        if a != b:
            print(f"  first difference at {i}: ours {a} ({tok.decode([a])!r}) ref {b} ({tok.decode([b])!r})")
            break
    print("  ours:", tok.decode([t for t in mapped if t != image_id])[:400])
    print("  ref: ", tok.decode([t for t in ref_ids if t != image_id])[:400])
pv = out["pixel_values"]
pvs = [pv[i] for i in range(pv.shape[0])] if isinstance(pv, torch.Tensor) else pv
for k, ref in enumerate(pvs):
    meta = json.load(open(os.path.join(dump, f"img{k}.json")))
    h, w = meta["rows"] * 16, meta["cols"] * 16
    mine = (np.fromfile(os.path.join(dump, f"img{k}.bin"), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(3, h, w)
    r = ref.to(torch.bfloat16).float().numpy()
    if r.shape != mine.shape:
        print(f"image {k}: grid ours {mine.shape} ref {r.shape} DIFFER")
        continue
    d = np.abs(r - mine)
    exact = float((d == 0).mean())
    print(f"image {k}: grid {meta['rows']}x{meta['cols']} patches ({meta['vids']} ids), pixels bf16-exact {exact*100:.3f}%, max abs {d.max():.4g}, mean abs {d.mean():.3g}")
