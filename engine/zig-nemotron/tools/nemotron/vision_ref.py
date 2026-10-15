#!/usr/bin/env python3
"""Golden reference for Nemotron 3.5 Super VL's vision path (M3): the checkpoint's image processor + RADIO tower +
vision_final_layernorm + pixel shuffle + mlp1, run in bf16 on CPU, line for line the remote code
(modeling_radio.py / modeling_nemotron_h_omni.py / image_processing_nemotron_h_omni.py, transformers 5.8.1).

  vision_ref.py MODEL_DIR IMAGE OUT_DIR [--max-patches N] [--save-blocks 0,15,31]

OUT_DIR gets: meta.json (grid, tokens), pixels_f32.bin (the processor's normalized [3,H,W] float32), pixels_bf16.bin
(what the tower eats), embed.bin (patch embed + pos + prefix, bf16 [n+10,1280]), blockNN.bin (hidden after block NN),
feat_ln.bin (final LN of the features), out.bin (the projected embeddings, bf16 [tokens, 4096]).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

PATCH, DS = 16, 2


def compute_target_patches(w, h, avail, min_patches=1024):
    closest_h = round(h / PATCH + 0.5)
    closest_w = round(w / PATCH + 0.5)
    patches = closest_h * closest_w
    factor = min(math.sqrt(avail / patches), 1.0)
    th = math.floor(factor * closest_h)
    tw = math.floor(factor * closest_w)
    if avail > min_patches and th * tw < min_patches:
        up = math.sqrt(min_patches / (th * tw))
        th = math.ceil(up * th)
        tw = math.ceil(up * tw)
    rem = th % DS
    if rem:
        if (th + DS - rem) * tw <= avail:
            th += DS - rem
        else:
            th = max(DS, th - rem)
    rem = tw % DS
    if rem:
        if th * (tw + DS - rem) <= avail:
            tw += DS - rem
        else:
            tw = max(DS, tw - rem)
    return tw, th


def bf(t, path):
    t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("image"); ap.add_argument("out")
    ap.add_argument("--max-patches", type=int, default=13312)
    ap.add_argument("--save-blocks", default="0,15,31")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    pp = json.load(open(os.path.join(a.model, "preprocessor_config.json")))
    files = {}

    def W(name):
        f = idx[name]
        if f not in files:
            files[f] = safe_open(os.path.join(a.model, f), "pt")
        return files[f].get_tensor(name)

    # ---- image processor (_preprocess, image path)
    img = Image.open(a.image).convert("RGB")
    budget = max(min((16384 - 4) * DS * DS, a.max_patches), 1024)
    wp, hp = compute_target_patches(img.width, img.height, max(budget, 1024))
    tw, th = wp * PATCH, hp * PATCH
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    if t.shape[-2] != th or t.shape[-1] != tw:
        t = F.interpolate(t, size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(pp["norm_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(pp["norm_std"]).view(1, 3, 1, 1)
    t = (t / 255.0 - mean) / std
    px = t.squeeze(0)  # [3,H,W] f32
    px.contiguous().numpy().tofile(os.path.join(a.out, "pixels_f32.bin"))
    pxb = px.to(torch.bfloat16)
    bf(pxb, os.path.join(a.out, "pixels_bf16.bin"))
    tokens = wp * hp // (DS * DS)
    json.dump({"src_w": img.width, "src_h": img.height, "wp": wp, "hp": hp, "tokens": tokens}, open(os.path.join(a.out, "meta.json"), "w"))
    print(f"image {img.width}x{img.height} -> {tw}x{th} px = {hp}x{wp} patches -> {tokens} tokens", flush=True)

    # ---- RADIO tower
    R = "vision_model.radio_model.model."
    x4 = pxb.unsqueeze(0)
    rows, cols = hp, wp
    patches = x4.reshape(1, 3, rows, PATCH, cols, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(1, rows * cols, 3 * PATCH * PATCH)
    h = F.linear(patches, W(R + "patch_generator.embedder.weight").to(torch.bfloat16))
    pos = W(R + "patch_generator.pos_embed").to(torch.bfloat16)  # (1, 16384, 1280)
    pos = pos.reshape(1, 128, 128, -1).permute(0, 3, 1, 2)
    md = max(rows, cols)
    pos = F.interpolate(pos.float(), size=(md, md), mode="bilinear", align_corners=False).to(torch.bfloat16)
    pos = pos[..., :rows, :cols]
    pos = pos.flatten(2).permute(0, 2, 1)
    h = h + pos
    prefix = W(R + "patch_generator.cls_token.token").to(torch.bfloat16).unsqueeze(0)
    x = torch.cat([prefix, h], dim=1)
    bf(x[0], os.path.join(a.out, "embed.bin"))
    save = {int(v) for v in a.save_blocks.split(",") if v != ""}
    H, nh = 1280, 16
    t0 = time.time()
    for i in range(32):
        p = R + f"blocks.{i}."
        g = lambda n: W(p + n).to(torch.bfloat16)
        hn = F.layer_norm(x, (H,), g("norm1.weight"), g("norm1.bias"), 1e-6)
        qkv = F.linear(hn, g("attn.qkv.weight"), g("attn.qkv.bias"))
        q, k, v = qkv.chunk(3, dim=-1)
        sh = lambda z: z.reshape(1, -1, nh, H // nh).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v))
        o = o.transpose(1, 2).reshape(1, -1, H)
        x = F.linear(o, g("attn.proj.weight"), g("attn.proj.bias")) + x
        hn = F.layer_norm(x, (H,), g("norm2.weight"), g("norm2.bias"), 1e-6)
        m = F.linear(F.gelu(F.linear(hn, g("mlp.fc1.weight"), g("mlp.fc1.bias"))), g("mlp.fc2.weight"), g("mlp.fc2.bias"))
        x = m + x
        if i in save:
            bf(x[0], os.path.join(a.out, f"block{i:02d}.bin"))
        if i % 4 == 3:
            print(f"block {i} done {time.time() - t0:.0f}s", flush=True)
    feats = x[:, 10:]
    feats = F.layer_norm(feats, (H,), W("vision_projector.vision_final_layernorm.weight").to(torch.bfloat16),
                         W("vision_projector.vision_final_layernorm.bias").to(torch.bfloat16), 1e-6)
    bf(feats[0], os.path.join(a.out, "feat_ln.bin"))
    # ---- projector
    v = feats.reshape(1, hp, wp, -1)
    n, w_, h_, c = v.size()
    v = v.view(n, w_, int(h_ * 0.5), int(c / 0.5))
    v = v.permute(0, 2, 1, 3).contiguous()
    v = v.view(n, int(h_ * 0.5), int(w_ * 0.5), int(c / 0.25))
    v = v.permute(0, 2, 1, 3).contiguous()
    v = v.reshape(1, -1, v.shape[-1])
    nw = W("mlp1.0.weight").to(torch.float32)
    vf = v.to(torch.float32)
    vf = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = (nw * vf).to(torch.bfloat16)
    v = F.linear(v, W("mlp1.1.weight").to(torch.bfloat16))
    v = torch.square(F.relu(v))
    v = F.linear(v, W("mlp1.3.weight").to(torch.bfloat16))
    assert v.shape[1] == tokens, (v.shape, tokens)
    bf(v[0], os.path.join(a.out, "out.bin"))
    print("out", tuple(v.shape), "absmean", v.float().abs().mean().item(), f"{time.time() - t0:.0f}s")


main()
