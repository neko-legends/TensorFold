#!/usr/bin/env python3
"""TP=4 comparison chart: Python TensorFold vs Zig vs Zig + pipelined prefill.

Two subcommands:

  add     read one bench run directory (bench/zig_vs_python.py's warm/long/mix/mixwin JSONs) and write
          its cells into the data file as a series (replaces a series with the same id).
  render  draw the data file to PNG + SVG.

The data file is the chart's only input; every value in it carries the artifact it came from. To add the
window-8 RoCE runs later:

  python bench/charts/prefill_pipe_comparison.py add --run-dir <zvp>/w8/zig-tp4-roce-pipe \
      --id zig-roce-pipe --engine zig-pipe --transport RoCE --window "w8 · 2026-10-11" \
      --label "Zig + pipelined prefill"
  uv run --with matplotlib python bench/charts/prefill_pipe_comparison.py render

The statistic is bench/zig_vs_python.py's headline: best of the repetitions without a failed stream
(time to first token: the lowest), with the median and the repetition count beside it in the data file.

The engine sets the colour; the transport sets the texture (RoCE hatched). Percentages on Zig bars are
against the Python series over the same transport (else the data file's `baseline`). The title, subtitle,
percent note and footnotes are plain text in the data file: edit them when a series is added. The `controlled` pairs are same-window, one-variable-changed
comparisons and get their own annotation; everything against Python is cross-session.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/charts/data/prefill-pipe-20261011.json"
OUT = ROOT / "docs/img/prefill-pipe-comparison-20261011"

# Cell definitions: file in the run directory, workload, temperature, streams, the field read, and which
# direction is better. `rate` is zig_vs_python.rate(): one stream's decode tok/s, or the x4 aggregate.
CELLS = {
    "prefill-32k": {"file": "long.json", "work": "long-32768", "temp": 0.0, "streams": 1, "field": "prompt_tok_s",
        "better": "higher", "label": "32K prompt", "unit": "prompt tok/s"},
    "prefill-131k": {"file": "long.json", "work": "long-131072", "temp": 0.0, "streams": 1, "field": "prompt_tok_s",
        "better": "higher", "label": "131K prompt", "unit": "prompt tok/s"},
    "ttft-154k": {"file": "long.json", "work": "long-131072", "temp": 0.0, "streams": 1, "field": "ttft_s",
        "better": "lower", "label": "154K-token prompt", "unit": "s"},
    "code-x1": {"file": "warm.json", "work": "code", "temp": 0.0, "streams": 1, "field": "rate",
        "better": "higher", "label": "code ×1", "unit": "tok/s"},
    "code-x4": {"file": "warm.json", "work": "code", "temp": 0.0, "streams": 4, "field": "rate",
        "better": "higher", "label": "code ×4", "unit": "tok/s"},
    "prose-x4": {"file": "warm.json", "work": "prose", "temp": 0.0, "streams": 4, "field": "rate",
        "better": "higher", "label": "prose ×4", "unit": "tok/s"},
    "long-decode-32k": {"file": "long.json", "work": "long-32768", "temp": 0.0, "streams": 1, "field": "mean_tok_s",
        "better": "higher", "label": "decode after\n32K prompt", "unit": "tok/s"},
    "mix-long": {"file": "mix.json", "work": "mix-long-131072", "temp": 0.0, "streams": 1, "field": "rate",
        "better": "higher", "label": "mix: 131K\ndecode", "unit": "tok/s"},
    "mix-code": {"file": "mix.json", "work": "mix-code", "temp": 0.0, "streams": 4, "field": "rate",
        "better": "higher", "label": "mix: code ×4\nper stream", "unit": "tok/s"},
    "mixwin": {"file": "mixwin.json", "work": "mixwin-131072", "temp": 0.0, "streams": 4, "field": "aggregate_tok_s",
        "better": "higher", "label": "mixed windows\naggregate", "unit": "tok/s"},
}
PANELS = [
    ("Prefill throughput", "prompt tokens / s  (higher is better ↑)", ["prefill-32k", "prefill-131k"]),
    ("Time to first token", "seconds  (lower is better ↓)", ["ttft-154k"]),
    ("Decode throughput", "tokens / s  (higher is better ↑)",
     ["code-x1", "code-x4", "prose-x4", "long-decode-32k", "mix-long", "mix-code", "mixwin"]),
]


def rate(c: dict) -> float | None:
    return c.get("mean_tok_s") if c["streams"] == 1 else (c.get("aggregate_tok_s") or c.get("mean_tok_s"))


def extract(run_dir: Path, source_prefix: str) -> tuple[dict, dict]:
    values, meta = {}, {}
    for cid, d in CELLS.items():
        f = run_dir / d["file"]
        if not f.exists():  # a partial run: its missing cells are left out, not zero
            continue
        j = json.loads(f.read_text())
        meta.setdefault("salt_tag", j.get("salt_tag"))
        meta.setdefault("started", j.get("started"))
        rows = [c for c in j["cells"] if c["work"] == d["work"] and c["temp"] == d["temp"]
                and c["streams"] == d["streams"] and c.get("rep") != "cold" and not c.get("failed_streams")]
        v = [rate(c) if d["field"] == "rate" else c.get(d["field"]) for c in rows]
        v = [x for x in v if x is not None]
        if not v:
            continue
        best = min(v) if d["better"] == "lower" else max(v)
        values[cid] = {"value": best, "median": round(statistics.median(v), 3), "n": len(v),
                       "source": f"{source_prefix}/{d['file']}"}
    return values, meta


def cmd_add(a: argparse.Namespace) -> int:
    data = json.loads(a.data.read_text())
    run_dir = a.run_dir.resolve()
    prefix = a.source_prefix or f"{data['artifact_root']}/{run_dir.relative_to(Path(a.mirror).resolve())}"
    values, meta = extract(run_dir, prefix)
    s = {"id": a.id, "label": a.label, "engine": a.engine, "transport": a.transport, "window": a.window,
         "salt_tag": meta["salt_tag"], "started": meta["started"], "run_dir": prefix, "values": values}
    data["series"] = [x for x in data["series"] if x["id"] != a.id] + [s]
    a.data.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"{a.id}: {len(values)} cells from {prefix}")
    return 0


# Colour = engine (validated categorical slots 2, 3, 1 of the dataviz reference palette, light surface,
# all-pairs CVD ΔE >= 9.2). Texture = transport.
ENGINE_COLOR = {"python": "#eb6834", "zig": "#1baf7a", "zig-pipe": "#2a78d6"}
ENGINE_ORDER = ["python", "zig", "zig-pipe"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"


def pct(v: float, b: float) -> float:
    return (v / b - 1.0) * 100.0


def signed(p: float) -> str:
    return f"{p:+.0f}%".replace("-", "\u2212")


def cmd_render(a: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = json.loads(a.data.read_text())
    series = sorted(data["series"], key=lambda s: (ENGINE_ORDER.index(s["engine"]), s["transport"] != "NCCL"))
    # % base: the Python series over the same transport, else the data file's baseline
    fallback = next(s for s in series if s["id"] == data["baseline"])
    base_of = {s["id"]: next((b for b in series if b["engine"] == "python" and b["transport"] == s["transport"]),
                             fallback) for s in series}
    plt.rcParams.update({
        "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 11, "svg.fonttype": "path",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
        "text.color": INK, "hatch.linewidth": 1.6, "hatch.color": SURFACE,
    })
    fig = plt.figure(figsize=(13, 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1], height_ratios=[1, 1.05],
                          left=0.065, right=0.985, top=0.775, bottom=0.175, hspace=0.42, wspace=0.16)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

    n = len(series)
    for ax, (title, ylabel, cells) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        group_w = 0.84
        w = group_w / n
        top = 0.0
        vmax = max(s["values"][c]["value"] for s in series for c in cells if c in s["values"])
        small = len(cells) > 2
        for i, s in enumerate(series):
            for g, cid in enumerate(cells):
                cell = s["values"].get(cid)
                if cell is None:
                    continue
                x = g - group_w / 2 + w * (i + 0.5)
                v = cell["value"]
                top = max(top, v)
                ax.bar(x, v, width=w - 0.025, color=ENGINE_COLOR[s["engine"]], edgecolor=SURFACE, linewidth=0,
                       hatch="///" if s["transport"] == "RoCE" else None, zorder=2)
                # whole numbers on the busy decode panel (the doc's table keeps the decimals)
                fmt = f"{v:,.1f}" if CELLS[cid]["unit"] == "s" else f"{v:,.0f}"
                ax.annotate(fmt, (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=(10.5 if n <= 4 else 9.5) if small else 12, fontweight="bold", color=INK, zorder=3)
                base = base_of[s["id"]]
                if s["engine"] != "python" and cid in base["values"]:
                    p = pct(v, base["values"][cid]["value"])
                    ax.annotate(signed(p), (x, v), xytext=(0, 3 + (15 if small else 17)),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=(9.5 if n <= 4 else 8.5) if small else 10.5, color=INK2, zorder=3)
        # same-window, one-variable-changed comparisons: a bracket over the pair
        for c in data.get("controlled", []):
            ia = next((i for i, s in enumerate(series) if s["id"] == c["a"]), None)
            ib = next((i for i, s in enumerate(series) if s["id"] == c["b"]), None)
            if ia is None or ib is None or small:
                continue
            for g, cid in enumerate(cells):
                va, vb = series[ia]["values"].get(cid), series[ib]["values"].get(cid)
                if not va or not vb:
                    continue
                xa = g - group_w / 2 + w * (ia + 0.5)
                xb = g - group_w / 2 + w * (ib + 0.5)
                y = max(va["value"], vb["value"]) + 0.3 * vmax  # clear of both bars' two label lines
                tick = 0.025 * vmax
                ax.plot([xa, xa, xb, xb], [y - tick, y, y, y - tick], color=INK, lw=1.2, zorder=3)
                p = pct(vb["value"], va["value"])
                ax.annotate(f"{signed(p)} · same window ●", ((xa + xb) / 2, y), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=10.5,
                            fontweight="bold", color=INK, zorder=3)
                top = max(top, y + 0.08 * vmax)
        ax.set_ylim(0, max(top, vmax * 1.18))
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_xticks(range(len(cells)), [CELLS[c]["label"] for c in cells], fontsize=12, color=INK)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=24)
        ax.text(0, 1.015, ylabel, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")

    fig.text(0.065, 0.955, data["title"], fontsize=19, fontweight="bold", color=INK)
    fig.text(0.065, 0.928, data["subtitle"], fontsize=11.5, color=INK2)
    handles = [Patch(facecolor=ENGINE_COLOR[s["engine"]], hatch="///" if s["transport"] == "RoCE" else None,
                     edgecolor=SURFACE, linewidth=0, label=f"{s['label']} — {s['transport']}, {s['window']}")
               for s in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.915), ncol=2, frameon=False,
               fontsize=10.5, handlelength=2.2, handleheight=1.3, columnspacing=2.0, labelcolor=INK)
    rows = (len(handles) + 1) // 2
    fig.text(0.065, 0.915 - 0.0215 * rows - 0.012, data["percent_note"], fontsize=10.5, color=INK2, va="top")
    notes = "\n".join(textwrap.fill(f, 172, break_on_hyphens=False) for f in data["footnotes"])
    fig.text(0.065, 0.012, notes, fontsize=9.5, color=INK2, va="bottom", linespacing=1.45)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{a.out}.png", dpi=a.dpi, facecolor=SURFACE)
    fig.savefig(f"{a.out}.svg", facecolor=SURFACE, metadata={"Date": None})
    print(f"wrote {a.out}.png, {a.out}.svg")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--dpi", type=int, default=160)
    ad = sub.add_parser("add")
    ad.add_argument("--run-dir", type=Path, required=True)
    ad.add_argument("--mirror", default="/home/jun/nest/tmp/zvp-20261011",
                    help="local copy of the artifact root; the source path is recorded relative to it")
    ad.add_argument("--source-prefix", help="record this source path instead of artifact_root/<relative>")
    ad.add_argument("--id", required=True)
    ad.add_argument("--label", required=True)
    ad.add_argument("--engine", choices=ENGINE_ORDER, required=True)
    ad.add_argument("--transport", choices=["NCCL", "RoCE"], required=True)
    ad.add_argument("--window", required=True)
    a = p.parse_args()
    return cmd_render(a) if a.cmd == "render" else cmd_add(a)


if __name__ == "__main__":
    sys.exit(main())
