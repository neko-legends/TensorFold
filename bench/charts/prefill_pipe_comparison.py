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
comparisons and get their own annotation; everything against Python is cross-session. The two never look alike:
controlled = a solid ink box round the pair with an inverted ink tab; cross-session = a grey italic dashed pill.

`render` fails (and keeps <out>.FAILED.png) if any text artist leaves the figure, or two of them overlap or sit
closer than MIN_GAP_PT. Phone-width check: `render --dpi 76.95 --png-only --out /tmp/x` gives a 1,000 px PNG.
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


# Colour = engine (validated categorical slots 2, 3, 1 of the dataviz reference palette, light surface:
# dataviz validate_palette, all-pairs worst CVD separation 9.2 in OKLab ΔE x100 under the Machado 2009 severity-1.0
# deutan simulation; CIEDE2000 under colorspacious full-severity CVD, worst pair 14.6). Texture = transport.
ENGINE_COLOR = {"python": "#eb6834", "zig": "#1baf7a", "zig-pipe": "#2a78d6"}
ENGINE_ORDER = ["python", "zig", "zig-pipe"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"


def pct(v: float, b: float) -> float:
    return (v / b - 1.0) * 100.0


def signed(p: float) -> str:
    return f"{p:+.0f}%".replace("-", "\u2212")


# The two annotation classes must never look alike. Cross-session (vs Python, a different day): grey italic in a
# dashed outline pill on each Zig bar. Controlled (same window, one variable): a solid ink box round the pair of
# bars with an inverted ink tab carrying the delta.
CROSS_STYLE = {"color": INK2, "fontstyle": "italic",
               "bbox": {"boxstyle": "round,pad=0.18,rounding_size=0.5", "fc": SURFACE, "ec": MUTED, "lw": 0.9,
                        "ls": (0, (2.2, 1.6))}}
CROSS_STYLE_TIGHT = {**CROSS_STYLE, "bbox": {**CROSS_STYLE["bbox"], "boxstyle": "round,pad=0.1,rounding_size=0.4"}}
CONTROL_STYLE = {"color": SURFACE, "fontweight": "bold",
                 "bbox": {"boxstyle": "square,pad=0.32", "fc": INK, "ec": INK, "lw": 1.2}}


MIN_GAP_PT = 1.5  # two labels closer than this read as one


def check_layout(fig, failed_png: str) -> list[str]:
    """Every visible text artist (with its bbox patch) inside the figure, and no two of them overlapping or closer
    than MIN_GAP_PT.

    Returns a report; raises SystemExit with the offenders otherwise."""
    import itertools

    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    fig.canvas.draw()  # legend and tick text get their final positions only at draw time
    r = fig.canvas.get_renderer()
    pt = 72.0 / fig.dpi
    W, H = fig.bbox.width, fig.bbox.height
    boxes = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        if t.get_bbox_patch() is not None:
            b = Bbox.union([b, t.get_bbox_patch().get_window_extent(r)])
        boxes.append((t.get_text().replace("\n", " ⏎ ")[:48], b))
    margins = sorted((min(b.x0, b.y0, W - b.x1, H - b.y1) * pt, name) for name, b in boxes)
    bad = [f"escapes the figure by {-m:.1f} pt: {name!r}" for m, name in margins if m < 0]
    pairs = []
    for (na, a), (nb, b) in itertools.combinations(boxes, 2):
        dx = max(b.x0 - a.x1, a.x0 - b.x1, 0.0)
        dy = max(b.y0 - a.y1, a.y0 - b.y1, 0.0)
        gap = max(dx, dy) * pt
        if a.overlaps(b):
            bad.append(f"overlap: {na!r} / {nb!r}")
        elif gap < MIN_GAP_PT:
            bad.append(f"{gap:.1f} pt apart (< {MIN_GAP_PT}): {na!r} / {nb!r}")
        pairs.append((gap, na, nb))
    if bad:
        fig.savefig(failed_png, facecolor=fig.get_facecolor())
        raise SystemExit(f"layout check FAILED (render kept at {failed_png}):\n  " + "\n  ".join(bad))
    gap, na, nb = min(pairs)
    return [(f"layout check: {len(boxes)} text artists, none outside the {W * pt:.0f}x{H * pt:.0f} pt figure, "
             f"no pair overlapping or under {MIN_GAP_PT} pt apart"),
            (f"  closest to the edge: {margins[0][1]!r} at {margins[0][0]:.1f} pt; next {margins[1][1]!r} at "
             f"{margins[1][0]:.1f} pt"),
            f"  tightest pair: {na!r} / {nb!r}, {gap:.1f} pt apart"]


def cmd_render(a: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    data = json.loads(a.data.read_text())
    series = sorted(data["series"], key=lambda s: (ENGINE_ORDER.index(s["engine"]), s["transport"] != "NCCL"))
    # % base: the Python series over the same transport, else the data file's baseline
    fallback = next(s for s in series if s["id"] == data["baseline"])
    base_of = {s["id"]: next((b for b in series if b["engine"] == "python" and b["transport"] == s["transport"]),
                             fallback) for s in series}
    plt.rcParams.update({
        "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 12, "svg.fonttype": "path",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
        "text.color": INK, "hatch.linewidth": 1.6, "hatch.color": SURFACE,
    })
    fig = plt.figure(figsize=(13, 14.8), facecolor=SURFACE)
    rows = (len(series) + 1) // 2  # legend rows; the panels move down for each row past two
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1], height_ratios=[1, 1.05],
                          left=0.07, right=0.985, top=0.755 - 0.0225 * max(rows - 2, 0), bottom=0.225, hspace=0.36, wspace=0.17)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

    n = len(series)
    for ax, (title, ylabel, cells) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        small = len(cells) > 2
        group_w = 0.95 if small else 0.84  # the decode panel's bars get the width its labels need
        w = group_w / n
        vmax = max(s["values"][c]["value"] for s in series for c in cells if c in s["values"])
        val_fs = (11 if n <= 4 else 10) if small else 13
        pct_fs = (10 if n <= 4 else 9) if small else 10.5
        # labels stack above a bar in points: value, then the cross-session pill; the controlled box clears both
        pill_at = 3 + val_fs * 1.45 + 3
        # with five or more series the pills of neighbouring Zig bars would touch: alternate ones go a row up
        zig = [s["id"] for s in series if s["engine"] != "python"]
        pill_rows = 2 if n >= 5 and small else 1
        # on the roomy panels, where a pill sits inside a controlled box, it names its baseline
        pill_lines = 1 if small or n >= 5 else 2
        pill_h = pct_fs * (1.75 + 1.25 * (pill_lines - 1))
        stack_pts = pill_at + pill_h * pill_rows + 4 * (pill_rows - 1)
        box_pts = stack_pts + 5
        tab_pts = 2 * 11 * 1.25 + 2 * 0.32 * 11 + 2
        controlled = [] if small else [
            (ia, ib) for c in data.get("controlled", [])
            for ia in [next((i for i, s in enumerate(series) if s["id"] == c["a"]), None)]
            for ib in [next((i for i, s in enumerate(series) if s["id"] == c["b"]), None)]
            if ia is not None and ib is not None]
        # the y limit that leaves the tallest stack (and the tab above a box) inside the axes
        h_pts = ax.get_position().height * fig.get_figheight() * 72
        need = (box_pts + tab_pts + 6) if controlled else (stack_pts + 6)
        ymax = vmax / (1 - need / h_pts)
        ax.set_ylim(0, ymax)
        per_pt = ymax / h_pts
        for i, s in enumerate(series):
            for g, cid in enumerate(cells):
                cell = s["values"].get(cid)
                if cell is None:
                    continue
                x = g - group_w / 2 + w * (i + 0.5)
                v = cell["value"]
                ax.bar(x, v, width=w - 0.025, color=ENGINE_COLOR[s["engine"]], edgecolor=SURFACE, linewidth=0,
                       hatch="///" if s["transport"] == "RoCE" else None, zorder=2)
                # whole numbers on the busy decode panel (the doc's table keeps the decimals)
                fmt = f"{v:,.1f}" if CELLS[cid]["unit"] == "s" else f"{v:,.0f}"
                ax.annotate(fmt, (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=val_fs, fontweight="bold", color=INK, zorder=4)
                base = base_of[s["id"]]
                if s["engine"] != "python" and cid in base["values"]:
                    p = pct(v, base["values"][cid]["value"])
                    ax.annotate(signed(p) + ("\nvs Python" if pill_lines == 2 else ""), (x, v),
                                xytext=(0, pill_at + (zig.index(s["id"]) % pill_rows) * (pill_h + 4)),
                                textcoords="offset points", ha="center", va="bottom", fontsize=pct_fs,
                                linespacing=1.1, multialignment="center", zorder=4,
                                **(CROSS_STYLE_TIGHT if small else CROSS_STYLE))
        # same-window, one-variable-changed comparisons: a box round the pair, its delta on an ink tab above
        for ia, ib in controlled:
            for g, cid in enumerate(cells):
                va, vb = series[ia]["values"].get(cid), series[ib]["values"].get(cid)
                if not va or not vb:
                    continue
                x0 = g - group_w / 2 + w * min(ia, ib) + 0.004
                x1 = g - group_w / 2 + w * (max(ia, ib) + 1) - 0.004
                y = max(va["value"], vb["value"]) + box_pts * per_pt
                ax.add_patch(Rectangle((x0, 0), x1 - x0, y, fill=False, ec=INK, lw=1.6, zorder=3, clip_on=False))
                p = pct(vb["value"], va["value"])
                ax.annotate(f"● {signed(p)}  pipeline off → on\nsame window, one variable", ((x0 + x1) / 2, y),
                            xytext=(0, 0.32 * 11 + 1), textcoords="offset points", ha="center", va="bottom",
                            fontsize=11, linespacing=1.25, multialignment="center", zorder=5, **CONTROL_STYLE)
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_xticks(range(len(cells)), [CELLS[c]["label"] for c in cells], fontsize=13, color=INK)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0, labelsize=11.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.set_title(title, loc="left", fontsize=15.5, fontweight="bold", color=INK, pad=36)
        ax.annotate(ylabel, (0, 1), xycoords="axes fraction", xytext=(0, 13), textcoords="offset points",
                    fontsize=12, color=INK2, va="bottom")

    fig.text(0.07, 0.962, data["title"], fontsize=17.5, fontweight="bold", color=INK, va="bottom")
    fig.text(0.07, 0.948, data["subtitle"], fontsize=12.5, color=INK2, va="top")
    handles = [Patch(facecolor=ENGINE_COLOR[s["engine"]], hatch="///" if s["transport"] == "RoCE" else None,
                     edgecolor=SURFACE, linewidth=0, label=f"{s['label']} · {s['transport']}   ({s['window']})")
               for s in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.063, 0.928), ncol=2, frameon=False,
               fontsize=12.5, handlelength=2.2, handleheight=1.3, columnspacing=1.6, labelcolor=INK)
    # the key to the two annotation classes: the percent note's sentences, each behind a sample of its mark
    y = 0.928 - 0.0225 * rows - 0.014
    note = [x.strip() for x in data["percent_note"].replace(". ", ".\n").splitlines() if x.strip()]
    samples = [("+n%", dict(fontsize=11.5, **CROSS_STYLE)), ("● +n%", dict(fontsize=11, **CONTROL_STYLE))]
    if len(note) == 2 and "Small" in note[0] and "Bracketed" in note[1]:
        for (sample, style), sentence in zip(samples, note):
            fig.text(0.073, y, sample, va="top", **style)
            fig.text(0.073 + 0.048, y, sentence, fontsize=12, color=INK2, va="top")
            y -= 0.025
    else:
        fig.text(0.07, y, data["percent_note"], fontsize=12, color=INK2, va="top")
    notes = "\n".join(textwrap.fill(f, 150, break_on_hyphens=False) for f in data["footnotes"])
    fig.text(0.07, 0.012, notes, fontsize=12, color=INK2, va="bottom", linespacing=1.4)

    for line in check_layout(fig, f"{a.out}.FAILED.png"):
        print(line)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{a.out}.png", dpi=a.dpi, facecolor=SURFACE)
    if not a.png_only:
        fig.savefig(f"{a.out}.svg", facecolor=SURFACE, metadata={"Date": None})
    print(f"wrote {a.out}.png" + ("" if a.png_only else f", {a.out}.svg"))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--dpi", type=float, default=160, help="1000 px wide (phone check): --dpi 76.95 --png-only")
    r.add_argument("--png-only", action="store_true")
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
