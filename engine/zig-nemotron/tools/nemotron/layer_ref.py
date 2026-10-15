#!/usr/bin/env python3
"""Per-layer numerics gate for Nemotron 3.5 Super VL: the checkpoint's own HF block (transformers 5.8.1, NemotronH) against
tensorfold's per-block hidden dumps.

Weights are dequantized to bf16 the way ModelOpt defines them (FP8: e4m3 * weight_scale; NVFP4: e2m1 * e4m3 block scale
(groups of 16) * weight_scale_2). Each block gets tensorfold's own residual input h_i (from `tensorfold run --eager --dump`:
NNN_h.bin is the residual entering block NNN, bf16 [rows, hidden]) so errors do not compound, and its output is compared
to tensorfold's h_{i+1}.

  layer_ref.py MODEL_DIR DUMP_DIR --rows N [--layers 0,1,7] [--tf-first-layer 0]
"""
import argparse, json, math, os, struct, sys
import torch
from safetensors import safe_open
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock

LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class Ckpt:
    def __init__(self, d, prefix="language_model."):
        self.d, self.prefix = d, prefix
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.open = {}

    def raw(self, name):
        full = self.prefix + name
        f = self.map[full]
        if f not in self.open:
            self.open[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.open[f].get_tensor(full)

    def has(self, name):
        return self.prefix + name in self.map

    def dequant(self, pre):
        """`pre`.weight as bf16 [n, k], whatever its stored format."""
        w = self.raw(pre + ".weight")
        if w.dtype == torch.float8_e4m3fn:
            return (w.float() * self.raw(pre + ".weight_scale").float()).to(torch.bfloat16)
        if w.dtype == torch.uint8:
            n, kb = w.shape
            lo, hi = (w & 0xF).long(), (w >> 4).long()
            codes = torch.stack([LUT[lo], LUT[hi]], dim=-1).reshape(n, kb * 2)
            bs = self.raw(pre + ".weight_scale").float()  # [n, k/16]
            s2 = self.raw(pre + ".weight_scale_2").float()
            return (codes * bs.repeat_interleave(16, dim=1) * s2).to(torch.bfloat16)
        return w.to(torch.bfloat16)


def load_block(ck, cfg, i):
    blk = NemotronHBlock(cfg, i).to(torch.bfloat16)
    p = f"backbone.layers.{i}."
    sd = {"norm.weight": ck.raw(p + "norm.weight")}
    kind = cfg.layers_block_type[i]
    m = p + "mixer."
    if kind == "mamba":
        for n in ("A_log", "D", "dt_bias", "conv1d.weight", "conv1d.bias", "norm.weight"):
            sd["mixer." + n] = ck.raw(m + n)
        for n in ("in_proj", "out_proj"):
            sd[f"mixer.{n}.weight"] = ck.dequant(m + n)
    elif kind == "attention":
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"mixer.{n}.weight"] = ck.dequant(m + n)
    else:  # moe
        sd["mixer.gate.weight"] = ck.raw(m + "gate.weight")
        sd["mixer.gate.e_score_correction_bias"] = ck.raw(m + "gate.e_score_correction_bias")
        for n in ("fc1_latent_proj", "fc2_latent_proj"):
            sd[f"mixer.{n}.weight"] = ck.raw(m + n + ".weight")
        for n in ("up_proj", "down_proj"):
            sd[f"mixer.shared_experts.{n}.weight"] = ck.dequant(m + "shared_experts." + n)
            sd[f"mixer.experts.{n}"] = torch.stack([ck.dequant(m + f"experts.{e}.{n}") for e in range(cfg.n_routed_experts)])
    missing, unexpected = blk.load_state_dict({k: v.to(torch.bfloat16) if v.dtype != torch.float32 else v for k, v in sd.items()}, strict=False)
    missing = [k for k in missing if k != "mixer.gate.e_score_correction_bias"]
    assert not missing and not unexpected, (missing, unexpected)
    if kind == "moe":
        # the router's bias stays fp32 (it is ~13.5 in layer 3: bf16 would round it to 1/16 and change the picks)
        blk.mixer.gate._buffers["e_score_correction_bias"] = sd["mixer.gate.e_score_correction_bias"].float()
    return blk.eval()


def bf16_file(path, rows, hidden):
    b = open(path, "rb").read()
    return torch.frombuffer(bytearray(b), dtype=torch.bfloat16).reshape(rows, hidden).clone()


def stats(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)
    rel = ((a - b).norm() / (b.norm() + 1e-30)).item()
    return cos.item(), rel, (a - b).abs().max().item(), b.abs().max().item()


def raw_file(path, dtype, shape):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dtype).reshape(shape).clone()


def moe_detail(blk, cfg, dump, i, rows):
    """A latent MoE layer's parts: the router's picks, the routed path alone, the shared expert alone."""
    need = [f"{dump}/{i:03d}_{n}.bin" for n in ("y", "pick", "wts", "routed", "shared")]
    if not all(os.path.exists(p) for p in need):
        return
    y = bf16_file(need[0], rows, cfg.hidden_size)
    k = cfg.num_experts_per_tok
    pick = raw_file(need[1], torch.int32, (rows, k)).long()
    wts = raw_file(need[2], torch.float32, (rows, k))
    routed_tf = bf16_file(need[3], rows, cfg.hidden_size)
    shared_tf = bf16_file(need[4], rows, cfg.hidden_size)
    mx = blk.mixer
    with torch.no_grad():
        logits = mx.gate(y)
        idx, w = mx.route_tokens_to_experts(logits)
        same = sum(set(idx[r].tolist()) == set(pick[r].tolist()) for r in range(rows))
        common = sum(len(set(idx[r].tolist()) & set(pick[r].tolist())) for r in range(rows))
        # our picks, sorted by expert id, against HF's by id: the weights of the experts both chose
        wd = 0.0
        for r in range(rows):
            ours = dict(zip(pick[r].tolist(), wts[r].tolist()))
            hf = dict(zip(idx[r].tolist(), w[r].tolist()))
            for e in set(ours) & set(hf):
                wd = max(wd, abs(ours[e] - hf[e]) / max(abs(hf[e]), 1e-9))
        print(f"   router: {same}/{rows} rows pick the identical expert set, {common}/{rows * k} picks shared, max rel weight diff {wd:.2e}")
        shared_ref = mx.shared_experts(y)
        c, r, m, _ = stats(shared_tf, shared_ref)
        print(f"   shared expert alone: cos {c:.5f} rel_l2 {r:.4f}")
        lat = mx.fc1_latent_proj(y)
        routed_ref = mx.fc2_latent_proj(mx.experts(lat, idx, w))
        c, r, m, _ = stats(routed_tf, routed_ref)
        print(f"   routed path vs HF's own picks: cos {c:.5f} rel_l2 {r:.4f}")
        routed_own = mx.fc2_latent_proj(mx.experts(lat, pick, wts))
        c, r, m, _ = stats(routed_tf, routed_own)
        print(f"   routed path vs HF experts on OUR picks: cos {c:.5f} rel_l2 {r:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--layers", default="0,1,7")
    a = ap.parse_args()
    cj = json.load(open(os.path.join(a.model, "config.json")))["llm_config"]
    cj = {k: v for k, v in cj.items() if k not in ("architectures", "auto_map")}
    cj["time_step_limit"] = [float(x["__float__"]) if isinstance(x, dict) else x for x in cj["time_step_limit"]]  # {"__float__": "Infinity"}
    cfg = NemotronHConfig(**cj)
    cfg._attn_implementation = "eager"
    ck = Ckpt(a.model)
    ok = True
    for i in [int(x) for x in a.layers.split(",")]:
        h_in = bf16_file(f"{a.dump}/{i:03d}_h.bin", a.rows, cfg.hidden_size)
        h_out = bf16_file(f"{a.dump}/{i + 1:03d}_h.bin", a.rows, cfg.hidden_size)
        blk = load_block(ck, cfg, i)
        with torch.no_grad():
            kwargs = {}
            if cfg.layers_block_type[i] == "attention":
                m = torch.full((1, 1, a.rows, a.rows), float("-inf")).triu(1)
                kwargs["attention_mask"] = m.to(torch.bfloat16)
            ref = blk(h_in.unsqueeze(0), **kwargs)[0]
        d_ref, d_tf = ref.float() - h_in.float(), h_out.float() - h_in.float()
        cos, rel, mx, ref_mx = stats(d_tf, d_ref)
        cos_h, rel_h, _, _ = stats(h_out, ref)
        print(f"layer {i:2d} {cfg.layers_block_type[i]:9s} delta: cos {cos:.5f} rel_l2 {rel:.4f} max_abs {mx:.4f} (ref max {ref_mx:.3f}) | h: cos {cos_h:.6f} rel {rel_h:.5f}", flush=True)
        ok &= cos > 0.99
        if cfg.layers_block_type[i] == "moe":
            moe_detail(blk, cfg, a.dump, i, a.rows)
        del blk
    sys.exit(0 if ok else 1)


main()
