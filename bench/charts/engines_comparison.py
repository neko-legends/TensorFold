#!/usr/bin/env python3
"""Chart 2: every engine DeepSeek V4.1 has run on four DGX Sparks (TP=4) — Python TensorFold, Zig (four configs),
vLLM, SGLang.

Three subcommands:

  build   read the raw artifacts from the local mirror and write the data file (every value -> its source file, the
          best / median / n, the harness). Re-run it whenever a source changes; never hand-edit a value.
  render  draw the data file to PNG + SVG (and fail loudly on the layout self-check, as prefill_pipe_comparison does).
  table   print the data file as the markdown table docs/ZIG-RESULTS.md carries.

  python bench/charts/engines_comparison.py build
  uv run --with matplotlib python bench/charts/engines_comparison.py render
  python bench/charts/engines_comparison.py table

The chart mixes harnesses, so it encodes three tiers, in chart 1's language:
  same-window controlled   Zig pipelined prefill off -> on in one window: a solid ink box round the pair, an ink tab.
  same harness, other day  Python vs Zig (bench/zig_vs_python.py): solid bars, a grey dashed pill with the % vs
                           Python over RoCE (the previous engine, live until 2026-10-11).
  different harness        vLLM and SGLang (v41bench.py / prefillprobe.py, other prompts, other dates, another
                           checkpoint precision): hollow outlined bars, italic values, no % — indicative only. They
                           get a value only where the metric is defined the same way; elsewhere the slot says why.

Statistic: the best of the repetitions without a failed stream (TTFT: the lowest), median and n beside it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefill_pipe_comparison
from prefill_pipe_comparison import (
    CONTROL_STYLE,
    CROSS_STYLE_TIGHT,
    GRID,
    INK,
    INK2,
    MUTED,
    SURFACE,
    check_layout,
    pct,
    signed,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/charts/data/engines-dsv41-20261011.json"
OUT = ROOT / "docs/img/engines-dsv41-comparison-20261011"
MIRROR = Path("/home/jun/nest/tmp/zvp-20261011")
# eight bars a group: two labels 1.5 pt apart already read as one number ("4,0844,084"), so this chart asks for 3 pt
prefill_pipe_comparison.MIN_GAP_PT = 3.0

TF_HARNESS = "bench/zig_vs_python.py (Jay's bench_http cells: 384-token replies, distinct prompt a stream)"
V41_HARNESS = "v41bench.py prompt set v1 (MiaAI-Lab / Tony kit; <=200-token replies) + prefillprobe.py"

# Cells: the TensorFold harness's definitions. vLLM / SGLang fill only the cells v41_series maps; the rest get NO_LIKE.
CELLS = {
    "prefill-32k": {"file": "long.json", "work": "long-32768", "temp": 0.0, "streams": 1, "field": "prompt_tok_s",
                        "better": "higher", "label": "32K prompt", "unit": "prompt tok/s"},
    "prefill-131k": {"file": "long.json", "work": "long-131072", "temp": 0.0, "streams": 1, "field": "prompt_tok_s",
                         "better": "higher", "label": "131K prompt", "unit": "prompt tok/s"},
    "ttft-154k": {"file": "long.json", "work": "long-131072", "temp": 0.0, "streams": 1, "field": "ttft_s",
                      "better": "lower", "label": "154K-token\nprompt", "unit": "s"},
    "long-decode-131k": {"file": "long.json", "work": "long-131072", "temp": 0.0, "streams": 1, "field": "mean_tok_s",
                             "better": "higher", "label": "decode after\nthe 154K prompt", "unit": "tok/s"},
    "mixwin": {"file": "mixwin.json", "work": "mixwin-131072", "temp": 0.0, "streams": 4, "field": "aggregate_tok_s",
                   "better": "higher", "label": "mixed windows\naggregate", "unit": "tok/s"},
    "code-x1": {"file": "warm.json", "work": "code", "temp": 0.0, "streams": 1, "field": "rate",
                    "better": "higher", "label": "code ×1", "unit": "tok/s"},
    "code-x4": {"file": "warm.json", "work": "code", "temp": 0.0, "streams": 4, "field": "rate",
                    "better": "higher", "label": "code ×4\naggregate", "unit": "tok/s"},
    "code-t07-x4": {"file": "warm.json", "work": "code", "temp": 0.7, "streams": 4, "field": "rate",
                        "better": "higher", "label": "code T0.7 ×4\naggregate", "unit": "tok/s"},
    "prose-x1": {"file": "warm.json", "work": "prose", "temp": 0.0, "streams": 1, "field": "rate",
                     "better": "higher", "label": "prose ×1", "unit": "tok/s"},
    "prose-x4": {"file": "warm.json", "work": "prose", "temp": 0.0, "streams": 4, "field": "rate",
                     "better": "higher", "label": "prose ×4\naggregate", "unit": "tok/s"},
}
PANELS = [
    ("Prefill throughput", "prompt tokens / s  (higher is better ↑)", ["prefill-32k", "prefill-131k"]),
    ("Time to first token", "seconds  (lower is better ↓)", ["ttft-154k"]),
    ("Long-context decode", "tokens / s  (higher is better ↑)", ["long-decode-131k", "mixwin"]),
    ("Short-prompt decode (T0 unless noted)", "tokens / s  (higher is better ↑)",
     ["code-x1", "code-x4", "code-t07-x4", "prose-x1", "prose-x4"]),
]

# Why a vLLM / SGLang slot is empty: the reason printed in the slot (short) and in the footnotes (long).
NO_LIKE = {
    "ttft-154k": "their longest cold prompt is 145.5-145.9K tokens, not 154K: TTFT scales with length",
    "long-decode-131k": "no decode-after-a-long-prompt cell in their harness",
    "mixwin": "no mixed-window cell in their harness",
    "code-x4": "their C4 is one prompt x4, aggregate over wall clock incl. TTFT; ours: 4 prompts, live span",
    "code-t07-x4": "their bench is T0 only",
    "prose-x4": "their C4 is one prompt x4, aggregate over wall clock incl. TTFT; ours: 4 prompts, live span",
}

# ---------------------------------------------------------------------------------------------------- build


def tf_rate(c: dict) -> float | None:
    return c.get("mean_tok_s") if c["streams"] == 1 else (c.get("aggregate_tok_s") or c.get("mean_tok_s"))


def summarize(v: list[float], better: str) -> dict:
    return {"value": min(v) if better == "lower" else max(v), "median": round(statistics.median(v), 3), "n": len(v)}


def tf_series(rel: str, **meta) -> dict:
    run = MIRROR / rel
    values, missing = {}, []
    for cid, d in CELLS.items():
        f = run / d["file"]
        if not f.exists():  # a trimmed run: the cell was not measured — never a zero
            missing.append(cid)
            continue
        j = json.loads(f.read_text())
        meta.setdefault("salt_tag", j.get("salt_tag"))
        rows = [c for c in j["cells"] if c["work"] == d["work"] and c["temp"] == d["temp"]
                and c["streams"] == d["streams"] and c.get("rep") != "cold" and not c.get("failed_streams")]
        v = [x for x in (tf_rate(c) if d["field"] == "rate" else c.get(d["field"]) for c in rows) if x is not None]
        if not v:
            missing.append(cid)
            continue
        cell = summarize(v, d["better"]) | {"source": f"{rel}/{d['file']}", "harness": "zig_vs_python"}
        if d["file"] == "long.json":
            cell["prompt_tokens"] = sorted({c["prompt_tokens"] for c in rows})
        values[cid] = cell
    s = {**meta, "tier": "tensorfold", "harness": TF_HARNESS, "run_dir": rel, "values": values,
         "not_measured": missing}
    # the w8 runs: their cold pass (one rep, the first requests after boot) is all there is for the short cells.
    # Kept here for the record; it is not plotted and not compared with anyone's warm best-of-3.
    if missing and (run / "cold.json").exists():
        j = json.loads((run / "cold.json").read_text())
        s["cold_pass_only"] = {cid: {"value": tf_rate(c), "n": 1, "source": f"{rel}/cold.json"}
                               for cid in missing for d in [CELLS[cid]] if d["file"] == "warm.json"
                               for c in j["cells"] if c["work"] == d["work"] and c["temp"] == d["temp"]
                               and c["streams"] == d["streams"]}
    return s


def v41_series(bench_files: list[str], probe_files: list[str], **meta) -> dict:
    """vLLM / SGLang: v41bench.py JSONs (C1 per-category decode, cold prefill rows) and prefillprobe.py JSONs."""
    def load(rel):
        return json.loads((MIRROR / rel).read_text())
    benches = [(rel, load(rel)) for rel in bench_files]
    probes = [(rel, load(rel)) for rel in probe_files]
    values = {}
    # decode: C1, the category named like ours; per-request rate = (tokens - 1) / (last - first token), as ours
    for cid, cat in (("code-x1", "coding"), ("prose-x1", "prose")):
        rows = [(rel, b) for rel, j in benches for b in j["batches"] if b["c"] == 1 and b["category"] == cat]
        v = [b["per_stream_tok_s"] for _, b in rows]
        values[cid] = summarize(v, "higher") | {
            "source": sorted({rel for rel, _ in rows}), "harness": "v41bench",
            "completion_tokens": sorted({r["completion_tokens"] for _, b in rows for r in b["requests"]}),
            "prompt_tokens": sorted({r["prompt_tokens"] for _, b in rows for r in b["requests"]})}
    # cold prefill: every cold row (bench + probe) whose prompt is in the cell's length band
    for cid, lo, hi in (("prefill-32k", 30_000, 50_000), ("prefill-131k", 140_000, 160_000)):
        rows = [(rel, r) for rel, j in benches for r in j.get("prefill", []) if lo <= r["prompt_tokens"] <= hi]
        rows += [(rel, r) for rel, j in probes for r in j["rows"] if lo <= r["prompt_tokens"] <= hi]
        if rows:
            values[cid] = summarize([r["prefill_tok_s"] for _, r in rows], "higher") | {
                "source": sorted({rel for rel, _ in rows}), "harness": "v41bench cold prefill / prefillprobe",
                "prompt_tokens": sorted({r["prompt_tokens"] for _, r in rows}),
                "ttft_s": sorted({r["ttft_s"] for _, r in rows})}
    return {**meta, "tier": "other-harness", "harness": V41_HARNESS, "values": values,
            "no_like_metric": {c: NO_LIKE[c] for c in CELLS if c not in values}}


def cmd_build(a: argparse.Namespace) -> int:
    tp4 = "TP=4"
    exl3 = "EXL3 2.9 bpw"
    series = [
        tf_series("python-tp4-nccl", id="python-nccl", label="Python TensorFold", engine="python",
                  transport="NCCL", tp=tp4, quant=exl3, date="2026-10-10", window="2026-10-10"),
        tf_series("python-tp4-roce-live", id="python-roce", label="Python TensorFold (previous engine)", engine="python",
                  transport="RoCE", tp=tp4, quant=exl3, date="2026-10-10", window="2026-10-10"),
        tf_series("w7/zig-tp4-nccl-control", id="zig-nccl-control", label="Zig", engine="zig",
                  transport="NCCL", tp=tp4, quant=exl3, date="2026-10-11", window="w7"),
        tf_series("w7/zig-tp4-nccl-pipe", id="zig-nccl-pipe", label="Zig + pipelined prefill", engine="zig-pipe",
                  transport="NCCL", tp=tp4, quant=exl3, date="2026-10-11", window="w7"),
        tf_series("w8/zig-tp4-roce-control", id="zig-roce-control", label="Zig", engine="zig",
                  transport="RoCE", tp=tp4, quant=exl3, date="2026-10-11", window="w8"),
        tf_series("w8/zig-tp4-roce-pipe", id="zig-roce-pipe", label="Zig + pipelined prefill", engine="zig-pipe",
                  transport="RoCE", tp=tp4, quant=exl3, date="2026-10-11", window="w8",
                  review="live since 2026-10-11"),
        v41_series(["vllm/dsv41-vllm/phase4/final-champion/bench-final-champion.json"],
                   ["vllm/dsv41-vllm/phase4/final-champion/prefill-probe.json"],
                   id="vllm", label="vLLM champion", engine="vllm", transport="NCCL", tp=tp4,
                   quant="FP8 + MXFP4", date="2026-09-11", window="2026-09-11",
                   config="vllm/dsv41-vllm/CHAMPION.env: greedy DSpark draft k=5, MAX_BATCHED=16384, MAXLEN=430080 "
                          "(the phase-4 champion: B3 greedy draft and B7 batched-16K accepted, every other arm "
                          "rejected; STATUS.md 'Final (Phase 4b)'); run phase4/final-champion",
                   gates="STATUS.md: 'Gates on final world: ALL PASS incl NIAH 32k x3 + 400k'; the gate log itself "
                         "(phase4/gates-final-champion.log) is not in the mirror"),
        v41_series([f"sglang/trial-20260914/bench/rep{i}/bench-sglang-r{i}.json" for i in (1, 2, 3)],
                   ["sglang/trial-20260914/bench/sglang-prefill/prefill-probe.json"],
                   id="sglang", label="SGLang + MiaAI kit, DSpark", engine="sglang", transport="NCCL", tp="TP=4 EP=4",
                   quant="FP8 + MXFP4", date="2026-09-14", window="2026-09-14",
                   config="sglang/trial-20260914/WORLD.md: lmsysorg/sglang:dev-dsv41 (2026-09-11) + MiaAI-Lab TP4 kit, "
                          "image 4e2c1ff3, TP=4 EP=4, DSpark on (block 5), native checkpoint",
                   gates="sglang/trial-20260914/gates/: G1 30/30 clean, G2 TOOL_ROUNDTRIP 3/3, G3 REASONING_PARSER PASS, "
                         "G4 needle 32K/400K/1M pass; WORLD.md: G0 (tool-result continuation repro) 3/3 clean. "
                         "The corruption Jun remembers is the 2026-09-10 DSpark-on-GB10 bug of the earlier image, "
                         "fixed by the 2026-09-11 dev-dsv41 image (WORLD.md)"),
    ]
    data = json.loads(a.data.read_text()) if a.data.exists() else {}
    data.update({
        "mirror": str(MIRROR),
        "artifact_root": "forge:~/zigtp4-run/out/zvp (TensorFold runs); forge:~/dsv41-vllm, "
                         "forge:~/dsv41-sglang-trial-20260914, forge:~/tf-claude-20261004 (vLLM, SGLang)",
        "statistic": "best of the repetitions without a failed stream (TTFT: the lowest); median and n beside it",
        "baseline": "python-roce",
        "controlled": [{"a": "zig-nccl-control", "b": "zig-nccl-pipe", "note": "window 7, NCCL, salt zvp-20261011w7"},
                       {"a": "zig-roce-control", "b": "zig-roce-pipe", "note": "window 8, RoCE, salt zvp-20261011w8"}],
        "cells": {k: {x: v[x] for x in ("label", "unit", "better")} | {"tensorfold_def": f"{v['file']} {v['work']} "
                  f"T{v['temp']} x{v['streams']} {v['field']}"} for k, v in CELLS.items()},
        "series": series,
    })
    a.data.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for s in series:
        print(f"{s['id']}: {len(s['values'])} cells" + (f", not measured {s['not_measured']}" if s.get("not_measured")
                                                          else ""))
    return 0

# ---------------------------------------------------------------------------------------------------- render


# Colour = engine. Python / Zig / Zig + pipeline keep chart 1's slots (2, 3, 1 of the dataviz reference palette);
# vLLM takes slot 7 (violet). SGLang is drawn in the neutral ink-grey on purpose: the different-harness tier is
# hollow, and no sixth hue clears the all-pairs floors beside orange (magenta 12.9, red 7.1 normal-vision ΔE).
# validate_palette.js "#eb6834,#1baf7a,#2a78d6,#4a3aa7,#52514e" --mode light --pairs all: CVD worst 9.2, normal-vision
# worst 16.3 (PASS); chroma floor FAIL on #52514e only (it is meant to read grey). Texture: TensorFold over its own
# RoCE all-gather = white /// on the fill; vLLM / SGLang = hollow, an outline and a sparse hatch in their colour.
ENGINE_COLOR = {"python": "#eb6834", "zig": "#1baf7a", "zig-pipe": "#2a78d6", "vllm": "#4a3aa7", "sglang": "#52514e"}
OTHER_HATCH = {"vllm": "..", "sglang": "xx"}


def tint(hex_: str, k: float) -> str:
    c = [int(hex_[i:i + 2], 16) for i in (1, 3, 5)]
    s = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(s[i] + (c[i] - s[i]) * k):02x}" for i in range(3))


def legend_label(s: dict) -> str:
    extra = {"zig-nccl-control": "pipeline off", "zig-roce-control": "pipeline off"}.get(s["id"])
    head = s["label"] + (f", {extra}" if extra else "")
    tail = f"{s['transport']} · {s['tp']} · {s['quant']} · {s['date']}" + (
        f" ({s['window']})" if s["window"].startswith("w") else "")
    if s.get("review"):
        tail += f" · {s['review']}"
    if s["tier"] == "other-harness":
        tail += " · other harness"
    return f"{head}\n{tail}"


def cmd_render(a: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    data = json.loads(a.data.read_text())
    series = data["series"]
    n = len(series)
    base = next(s for s in series if s["id"] == data["baseline"])
    plt.rcParams.update({
        "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 12, "svg.fonttype": "path",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
        "text.color": INK, "hatch.linewidth": 1.4,
    })
    fig = plt.figure(figsize=(20, 17.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1.0, 2.0], height_ratios=[1, 1],
                          left=0.05, right=0.99, top=0.675, bottom=0.222, hspace=0.42, wspace=0.13)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, :])]
    idx = {s["id"]: i for i, s in enumerate(series)}
    zig = [s["id"] for s in series if s["engine"].startswith("zig")]

    for ax, (title, ylabel, cells) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        group_w = 0.92
        w = group_w / n
        vmax = max(s["values"][c]["value"] for s in series for c in cells if c in s["values"])
        val_fs, pct_fs = 10.5, 9.5
        pill_at = 3 + val_fs * 1.45 + 4
        pill_h = pct_fs * 1.75 + 3
        pill_rows = 2  # neighbouring Zig bars alternate rows, so their pills never touch
        stack_pts = pill_at + pill_h * pill_rows + 3
        box_pts = stack_pts + 4
        tab_pts = 10.5 * 1.3 + 2 * 0.32 * 10.5 + 3
        controlled = [(idx[c["a"]], idx[c["b"]]) for c in data["controlled"]
                      if any(c2 in series[idx[c["a"]]]["values"] and c2 in series[idx[c["b"]]]["values"]
                             for c2 in cells)] if len(cells) <= 2 else []
        h_pts = ax.get_position().height * fig.get_figheight() * 72
        need = (box_pts + tab_pts + 6) if controlled else (stack_pts + 6)
        ymax = vmax / (1 - need / h_pts)
        ax.set_ylim(0, ymax)
        per_pt = ymax / h_pts
        for i, s in enumerate(series):
            col = ENGINE_COLOR[s["engine"]]
            other = s["tier"] == "other-harness"
            for g, cid in enumerate(cells):
                x = g - group_w / 2 + w * (i + 0.5)
                cell = s["values"].get(cid)
                if cell is None:  # say why the slot is empty, in the slot
                    why = "not measured" if cid in s.get("not_measured", []) else "no like metric"
                    ax.annotate(why, (x, 0), xytext=(0, 4), textcoords="offset points", rotation=90, ha="center",
                                va="bottom", fontsize=9.5, color=MUTED, fontstyle="italic", zorder=4)
                    continue
                v = cell["value"]
                if other:
                    ax.bar(x, v, width=w - 0.03, color=tint(col, 0.10), edgecolor=col, linewidth=1.8,
                           hatch=OTHER_HATCH[s["engine"]], hatchcolor=tint(col, 0.55), zorder=2)
                else:
                    ax.bar(x, v, width=w - 0.02, color=col, edgecolor=SURFACE, linewidth=0,
                           hatch="///" if s["transport"] == "RoCE" else None, hatchcolor=SURFACE, zorder=2)
                fmt = f"{v:,.1f}" if CELLS[cid]["unit"] == "s" else f"{v:,.0f}"
                ax.annotate(fmt, (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=val_fs if v < 1000 else 9.0, fontweight="normal" if other else "bold",
                            fontstyle="italic" if other else "normal", color=INK2 if other else INK, zorder=4)
                if s["id"] in zig and cid in base["values"]:
                    p = pct(v, base["values"][cid]["value"])
                    # both pills of a controlled pair stand on the taller bar, one row each, so they never collide
                    mate = next((c["a"] if c["b"] == s["id"] else c["b"] for c in data["controlled"]
                                 if s["id"] in (c["a"], c["b"])), s["id"])
                    top = max(v, series[idx[mate]]["values"].get(cid, {}).get("value", v))
                    ax.annotate(signed(p), (x, top), xytext=(0, pill_at + (zig.index(s["id"]) % pill_rows) * pill_h),
                                textcoords="offset points", ha="center", va="bottom", fontsize=pct_fs, zorder=4,
                                **CROSS_STYLE_TIGHT)
        for ia, ib in controlled:
            for g, cid in enumerate(cells):
                va, vb = series[ia]["values"].get(cid), series[ib]["values"].get(cid)
                if not va or not vb:
                    continue
                x0 = g - group_w / 2 + w * min(ia, ib) + 0.006
                x1 = g - group_w / 2 + w * (max(ia, ib) + 1) - 0.006
                y = max(va["value"], vb["value"]) + box_pts * per_pt
                ax.add_patch(Rectangle((x0, 0), x1 - x0, y, fill=False, ec=INK, lw=1.5, zorder=3, clip_on=False))
                ax.annotate(f"● {signed(pct(vb['value'], va['value']))}", ((x0 + x1) / 2, y),
                            xytext=(0, 0.32 * 10.5 + 1), textcoords="offset points", ha="center", va="bottom",
                            fontsize=10.5, zorder=5, **CONTROL_STYLE)
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_xticks(range(len(cells)), [CELLS[c]["label"] for c in cells], fontsize=13, color=INK)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0, labelsize=11.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.set_title(title, loc="left", fontsize=15.5, fontweight="bold", color=INK, pad=37)
        ax.annotate(ylabel, (0, 1), xycoords="axes fraction", xytext=(0, 12), textcoords="offset points",
                    fontsize=12, color=INK2, va="bottom")

    fig.text(0.05, 0.968, data["title"], fontsize=19, fontweight="bold", color=INK, va="bottom")
    fig.text(0.05, 0.958, data["subtitle"], fontsize=13, color=INK2, va="top")
    handles = []
    for s in series:
        col = ENGINE_COLOR[s["engine"]]
        if s["tier"] == "other-harness":
            handles.append(Patch(facecolor=tint(col, 0.10), edgecolor=col, linewidth=1.8,
                                 hatch=OTHER_HATCH[s["engine"]], hatchcolor=tint(col, 0.55), label=legend_label(s)))
        else:
            handles.append(Patch(facecolor=col, hatch="///" if s["transport"] == "RoCE" else None, edgecolor=SURFACE,
                                 hatchcolor=SURFACE, linewidth=0, label=legend_label(s)))
    # two columns, read down: the TensorFold series left (Python, w7), right (w8, then the other harnesses)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.937), ncol=2, frameon=False,
               fontsize=12, handlelength=2.4, handleheight=1.6, columnspacing=2.0, labelspacing=0.75,
               labelcolor=INK)
    # the key to the three tiers, each sentence behind a sample of its mark
    y = 0.789
    samples = [("● +n%", dict(fontsize=10.5, **CONTROL_STYLE)), ("+n%", dict(fontsize=10.5, **CROSS_STYLE_TIGHT)),
               ("123", {"fontsize": 11, "fontstyle": "italic", "color": INK2,
                            "bbox": {"boxstyle": "square,pad=0.25", "fc": tint(ENGINE_COLOR["vllm"], 0.10),
                                  "ec": ENGINE_COLOR["vllm"], "lw": 1.6}})]
    for (sample, style), sentence in zip(samples, data["tier_key"]):
        fig.text(0.052, y, sample, va="top", **style)
        fig.text(0.052 + 0.04, y, sentence, fontsize=12.5, color=INK2, va="top")
        y -= 0.021
    notes = "\n".join(textwrap.fill(f, 232, break_on_hyphens=False) for f in data["footnotes"])
    fig.text(0.05, 0.008, notes, fontsize=11.5, color=INK2, va="bottom", linespacing=1.38)

    for line in check_layout(fig, f"{a.out}.FAILED.png"):
        print(line)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{a.out}.png", dpi=a.dpi, facecolor=SURFACE)
    if not a.png_only:
        fig.savefig(f"{a.out}.svg", facecolor=SURFACE, metadata={"Date": None})
    print(f"wrote {a.out}.png" + ("" if a.png_only else f", {a.out}.svg"))
    return 0

# ---------------------------------------------------------------------------------------------------- table


def cmd_table(a: argparse.Namespace) -> int:
    data = json.loads(a.data.read_text())
    series = data["series"]
    base = next(s for s in series if s["id"] == data["baseline"])
    short = {"python-nccl": "Python NCCL", "python-roce": "Python RoCE (previous)", "zig-nccl-control": "Zig NCCL",
             "zig-nccl-pipe": "Zig + pipe NCCL", "zig-roce-control": "Zig RoCE", "zig-roce-pipe": "Zig + pipe RoCE (live)",
             "vllm": "vLLM † (2026-09-11)", "sglang": "SGLang † (2026-09-14)"}
    print("| Cell | " + " | ".join(short[s["id"]] for s in series) + " |")
    print("| --- |" + " ---: |" * len(series))
    for cid, d in CELLS.items():
        row = []
        for s in series:
            c = s["values"].get(cid)
            if c is None:
                row.append("not measured" if cid in s.get("not_measured", []) else "—")
                continue
            v = f"{c['value']:,.2f}" if d["unit"] == "s" else f"{c['value']:,.1f}"
            if s["tier"] == "other-harness":
                v = f"*{v}*"
            elif s["id"] != base["id"] and cid in base["values"]:
                p = pct(c["value"], base["values"][cid]["value"])
                v += f" ({signed(p) if round(p) else '±0%'})"
            row.append(v)
        label = d["label"].replace("\n", " ")
        print(f"| {label} ({d['unit']}) | " + " | ".join(row) + " |")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("table")
    r = sub.add_parser("render")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--dpi", type=float, default=150, help="1000 px wide (phone check): --dpi 50 --png-only")
    r.add_argument("--png-only", action="store_true")
    a = p.parse_args()
    return {"build": cmd_build, "render": cmd_render, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
