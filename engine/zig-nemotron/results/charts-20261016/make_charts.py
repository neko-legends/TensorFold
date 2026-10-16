#!/usr/bin/env python3
"""Nemotron 3.5 Super VL vs DeepSeek V4.1 Flash on four Sparks: the six result charts, light and dark.

    ./make-charts.sh                 # extract numbers.json from the result files, then render
    python3 make_charts.py extract   # numbers.json only (needs the spark-bench checkout for DeepSeek's qbench)
    python3 make_charts.py render    # charts only, from numbers.json

Every plotted number comes from numbers.json, and every entry there names the result file it came from.
The only typed-in values are DeepSeek's cold long-prompt times, which come from the spark-bench README
(2026-10-14 measurement). Style follows spark-bench's scripts/charts/make_lane_charts.py: Inter, the same
surfaces, blue for the model this work is about, gray for earlier configs.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent                                  # engine/zig-nemotron/results
REPO = RES.parents[2]
SPARK_BENCH = Path(os.environ.get("SPARK_BENCH", REPO.parent / "spark-bench"))
NUMBERS = HERE / "numbers.json"

DS, NE = "DeepSeek V4.1 Flash", "Nemotron 3.5 Super VL"


# ---------------------------------------------------------------- extract

def rel(p: Path) -> str:
    for root, name in ((REPO, "tensorfold"), (SPARK_BENCH, "spark-bench")):
        try:
            return f"{name}:{p.resolve().relative_to(root.resolve())}"
        except ValueError:
            pass
    return p.name


def load(p: Path) -> dict:
    return json.loads(p.read_text())


QUALITY = ["mmlu", "math", "coding", "if", "json", "tools", "longctx", "summary"]


def area_score(a: dict, k: str) -> tuple[float, int, int]:
    if k == "summary":
        return 100 * a["facts_kept_rate"], a["facts_kept"], a["facts_total"]
    return 100 * a["accuracy"], a["correct"], a["n"]


def vision_sums(rows: list[dict]) -> dict:
    rows = {r["bench"]: r for r in rows}
    t = [r for r in rows.values() if r["t_n_gt"]]
    s = [r for b, r in rows.items() if b != "k_f_counts" and r["s_n_gt"]]
    c = rows["k_f_counts"]

    def wmean(pairs):  # error averaged over every matched item, not per image
        n = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / n

    return {
        "text": {"exact": sum(r["t_exact"] for r in t), "of": sum(r["t_n_gt"] for r in t),
                 "err_px": wmean([(r["t_pos_err"], r["t_exact"]) for r in t])},
        "shapes": {"matched": sum(r["s_matched"] for r in s), "of": sum(r["s_n_gt"] for r in s),
                   "extra": sum(r["s_extra"] for r in s),
                   "err_px": wmean([(r["s_pos_err"], r["s_matched"]) for r in s])},
        "counts": {"matched": c["s_matched"], "of": c["s_n_gt"], "extra": c["s_extra"], "err_px": c["s_pos_err"]},
        "a3_shapes": {"matched": rows["a3_shapes"]["s_matched"], "of": rows["a3_shapes"]["s_n_gt"]},
    }


def extract() -> dict:
    src: dict[str, str] = {}
    out: dict = {"sources": src}

    # quality + speed: DeepSeek = spark-bench qbench re-run after the fabric fix (quality identical to the first run)
    ds_first = SPARK_BENCH / "results/qbench-dsv41-zig-20261015.json"
    ds_q = SPARK_BENCH / "results/qbench-dsv41-zig-rebench-20261015.json"
    da, da0 = load(ds_q)["areas"], load(ds_first)["areas"]
    for k in QUALITY:
        assert area_score(da[k], k) == area_score(da0[k], k), f"DeepSeek {k} differs between its two runs"
    nb = RES / "rebench-nemotron-tp4-20261015"
    na = {}
    for k in QUALITY + ["speed"]:
        na.update(load(nb / f"qbench-{k}.json")["areas"])
    src["quality.deepseek"] = rel(ds_q) + f" (same scores as {ds_first.name})"
    src["quality.nemotron"] = rel(nb) + "/qbench-*.json"
    out["quality"] = {k: {"deepseek": area_score(da[k], k), "nemotron": area_score(na[k], k)} for k in QUALITY}

    def speed(sp):
        return {"code": sp["c1"]["code"]["decode_tok_s"], "prose": sp["c1"]["prose"]["decode_tok_s"],
                "c4": sp["c4"]["aggregate_tok_s"],
                "ttft_code_ms": 1000 * sp["c1"]["code"]["ttft_s"], "ttft_prose_ms": 1000 * sp["c1"]["prose"]["ttft_s"]}

    # Nemotron speed: the live server (drafting on, build c3ede919), re-measured 2026-10-16 with the same speed area
    live = RES / "live-speed-20261016"
    nl = load(live / "qbench-speed-live.json")["areas"]["speed"]
    out["speed"] = {"deepseek": speed(da["speed"]), "nemotron": speed(nl)}
    src["speed"] = (f"DeepSeek: the speed area of {ds_q.name}; Nemotron: " + rel(live / "qbench-speed-live.json")
                    + " (server timing block on both engines)")

    # vision: the same harness on both, refine on
    dv = load(RES / "vision-dsv41-recheck.json")
    nv_rows = [r for b in ("a1_poster", "a2_dashboard", "a3_shapes", "a4_table_chart", "a5_big4096", "k_f_counts")
               for r in load(nb / f"vision-{b}.json")["rows"]]
    assert not any(r.get("error") for r in dv["rows"] + nv_rows)
    out["vision"] = {"deepseek": vision_sums(dv["rows"]), "nemotron": vision_sums(nv_rows),
                     "date": {"deepseek": dv["date"][:10], "nemotron": load(nb / "vision-a3_shapes.json")["date"][:10]}}
    src["vision.deepseek"] = rel(RES / "vision-dsv41-recheck.json")
    src["vision.nemotron"] = rel(nb) + "/vision-*.json"

    # cold prefill: Nemotron before / after (server first-token time); DeepSeek from the spark-bench README
    pre = {}
    for tag in ("before", "after"):
        p = RES / f"prefill-nemotron-20261015-{tag}.json"
        rows = load(p)["rows"]
        assert all(r["cached_tokens"] <= 16 for r in rows)
        pre[tag] = {str(t): {"s": st.mean(r["server_ttft_s"] for r in rows if r["target"] == t),
                             "tokens": round(st.mean(r["prompt_tokens"] for r in rows if r["target"] == t)),
                             "reps": sum(1 for r in rows if r["target"] == t)}
                    for t in (32768, 131072, 200000)}
        src[f"prefill.nemotron.{tag}"] = rel(p)
    pre["deepseek"] = {"200000": {"s": 36.8, "tokens": 199713, "reps": 1}}
    pre["deepseek_other"] = [{"tokens": 154196, "s": 30.0}, {"tokens": 332847, "s": 66.6}]
    src["prefill.deepseek"] = ("spark-bench README (cold 199,713 tokens 36.8 s and 332,847 tokens 66.6 s, 2026-10-14) "
                               "and artifacts/tensorfold-v41-zig-20261011/REPORT.md (154K 30.0 s)")
    out["prefill"] = pre

    # serving: 16 streams at 4 vs 16 slots; warm cache and disk tier
    sm = RES / "spare-mem-20261015"
    load16 = {}
    for tag, f in (("p4", "p4base-conc.json"), ("p16", "p16-conc.json")):
        lv = [lv for lv in load(sm / f)["levels"] if lv["streams"] == 16]
        load16[tag] = {k: {"median_s": st.mean(x["ttft_median_s"] for x in lv if x["kind"] == k),
                           "worst_s": max(x["ttft_max_s"] for x in lv if x["kind"] == k),
                           "agg_tok_s": st.mean(x["aggregate_tok_s"] for x in lv if x["kind"] == k),
                           "reps": sum(1 for x in lv if x["kind"] == k)}
                       for k in ("code", "prose", "mixed")}
        src[f"serving.{tag}"] = rel(sm / f)
    warm = {}
    for tag, f in (("cache16", "c16b-warm200x10000.json"), ("cache40", "c40-warm200x10000.json"),
                   ("restart_nodisk", "gA-warmfrom.json"), ("restart_disk", "gB-warmfrom.json")):
        s = load(sm / f)["summary"]
        warm[tag] = {"hits": s["hits"], "of": s["followups"], "median_s": s["ttft_median_s"]}
        src[f"warm.{tag}"] = rel(sm / f)
    out["serving"] = {"load16": load16, "warm": warm}

    # drafting: before = the 2026-10-15 live build (prompt cache off), after = the live server 2026-10-16; median of 3
    dr = RES / "drafting-20261015"
    b, a = load(dr / "final-before.json"), load(live / "draft-live.json")
    out["drafting"] = {k: {"before": b[k]["median_tok_s"], "after": a[k]["median_tok_s"],
                           "same_text": [r["token_sha"] for r in b[k]["runs"]] == [r["token_sha"] for r in a[k]["runs"]]}
                       for k in ("code", "prose", "story", "explain", "essay")}
    src["drafting"] = rel(dr / "final-before.json") + ", " + rel(live / "draft-live.json")

    NUMBERS.write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------- render

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e6e5e0", track="#f1f0ec",
                  ne="#2a78d6", ds="#eb6834", old="#b9b8b1"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e86", grid="#33332f", track="#232322",
                 ne="#3987e5", ds="#d95926", old="#5b5a55"),
}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "/inter/" in f.lower():
            font_manager.fontManager.addfont(f)
    return plt


class Chart:
    """One figure in the house style: title, subtitle, panels of horizontal bars, legend row, footnotes.

    All layout is in inches from the figure's bottom-left corner; bar thickness and gaps are fixed in inches
    too, so every chart has the same thin marks whatever its row count.
    """
    BAR, GAP = 0.2, 0.035          # bar thickness and the surface gap between bars of one group (inches)

    def __init__(self, plt, t, w, h, title, sub):
        self.plt, self.t, self.w, self.h = plt, t, w, h
        plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 12, "svg.fonttype": "path",
                             "svg.hashsalt": "nemotron-charts", "text.color": t["ink"]})
        self.fig = plt.figure(figsize=(w, h))
        self.fig.patch.set_facecolor(t["bg"])
        self.text(0.42, h - 0.42, title, size=19, weight="bold", va="top")
        self.text(0.42, h - 0.86, sub, size=12, color=t["ink2"], va="top")

    def text(self, x, y, s, size=12, color=None, weight="normal", va="center", ha="left", **kw):
        return self.fig.text(x / self.w, y / self.h, s, fontsize=size, color=color or self.t["ink"],
                             fontweight=weight, va=va, ha=ha, **kw)

    def panel(self, x0, x1, y_top, nrows, row_h, title, title_x, xmax, ticks, tickfmt):
        """A bar panel whose plot area spans x0..x1 and hangs from y_top; returns (axes, row centres)."""
        t = self.t
        y0 = y_top - nrows * row_h
        ax = self.fig.add_axes([x0 / self.w, y0 / self.h, (x1 - x0) / self.w, (y_top - y0) / self.h])
        ax.set_facecolor(t["bg"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, labelsize=11, colors=t["ink2"], pad=7)
        ax.set_xlim(0, xmax)
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tickfmt(v) for v in ticks])
        ax.set_yticks([])
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(0, color=t["grid"], linewidth=1.2, zorder=1)
        self.text(title_x, y_top + 0.3, title, size=13, weight="bold", va="bottom")
        ax.box = (x0, y0, x1, y_top)
        ax.row_h = row_h
        return ax, list(range(nrows))

    def slot(self, ax, i, n):
        """Centre offset (data units) of bar i of an n-bar group."""
        step = (self.BAR + self.GAP) / ax.row_h
        return (i - (n - 1) / 2) * step

    def bar(self, ax, y, v, color, label=None, label_color=None, note=None, label_x=None):
        """A bar from 0 to v, flat at the baseline and rounded (4 px) at the data end, then label + muted note."""
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
        x0, y0, x1, y1 = ax.box
        xmax = ax.get_xlim()[1]
        h = self.BAR / ax.row_h
        if v > 0:
            rx = min(4 / 150 / (x1 - x0) * xmax, v / 2)          # 4 px at 150 dpi
            ry = min(4 / 150 / ax.row_h, h / 2)
            top, bot, k = y - h / 2, y + h / 2, 0.5523
            verts = [(0, top), (v - rx, top), (v - rx + k * rx, top), (v, top + ry - k * ry), (v, top + ry),
                     (v, bot - ry), (v, bot - ry + k * ry), (v - rx + k * rx, bot), (v - rx, bot), (0, bot), (0, top)]
            codes = [MPath.MOVETO, MPath.LINETO] + [MPath.CURVE4] * 3 + [MPath.LINETO] + [MPath.CURVE4] * 3 + \
                    [MPath.LINETO, MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))
        if label is not None:
            lx = v if label_x is None else label_x
            tx = ax.annotate(label, xy=(lx, y), xytext=(7, 0), textcoords="offset points", va="center", ha="left",
                             fontsize=11.5, color=label_color or self.t["ink"], zorder=4, annotation_clip=False)
            if note:
                ax.annotate(note, xycoords=tx, xy=(1, 0.5), xytext=(7, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=10.5, color=self.t["muted"], zorder=4)

    def ylabels(self, ax, labels, x_right):
        """Category labels right-aligned left of the plot: main line in ink, optional second line muted."""
        x0, y0, x1, y1 = ax.box
        for i, lab in enumerate(labels):
            main, sub = (lab, None) if isinstance(lab, str) else lab
            yy = y1 - (i + 0.5) * ax.row_h
            if sub:
                self.text(x_right, yy + 0.1, main, size=12, ha="right")
                self.text(x_right, yy - 0.13, sub, size=10, color=self.t["muted"], ha="right")
            else:
                self.text(x_right, yy, main, size=12, ha="right")

    def legend(self, items, y=None, x=0.42):
        y = self.notes_top + 0.3 if y is None else y
        from matplotlib.patches import FancyBboxPatch
        r = self.fig.canvas.get_renderer()
        for name, color in items:
            self.fig.patches.append(FancyBboxPatch(
                (x / self.w, (y - 0.075) / self.h), 0.15 / self.w, 0.15 / self.h, transform=self.fig.transFigure,
                boxstyle=f"round,pad=0,rounding_size={0.03 / self.w}", facecolor=color, edgecolor="none"))
            tx = self.text(x + 0.23, y, name, size=11.5, color=self.t["ink2"])
            x = tx.get_window_extent(r).x1 / self.fig.dpi + 0.42

    def notes(self, lines, y=0.32, step=0.23):
        """Footnotes, wrapped to the figure width, bottom line at y."""
        import textwrap
        width = int((self.w - 0.84) * 14.6)
        wrapped = [w for s in lines for w in textwrap.wrap(s, width, break_on_hyphens=False)]
        for i, s in enumerate(reversed(wrapped)):
            self.text(0.42, y + i * step, s, size=10.2, color=self.t["muted"])
        self.notes_top = y + (len(wrapped) - 1) * step
        return self.notes_top

    def problems(self) -> list[str]:
        """Text that leaves the figure or overlaps other text or a legend swatch (bars are checked against labels too)."""
        from matplotlib.text import Text
        r = self.fig.canvas.get_renderer()
        W, H = self.fig.bbox.width, self.fig.bbox.height
        boxes = []
        for art in self.fig.findobj(Text):
            if art.get_text().strip() and art.get_visible():
                bb = art.get_window_extent(r)
                if bb.width > 0:
                    boxes.append((art.get_text(), bb))
        for pa in self.fig.patches:
            boxes.append(("<swatch>", pa.get_window_extent(r)))
        bars = [pa.get_window_extent(r) for ax in self.fig.axes for pa in ax.patches]
        out = []
        for i, (s, bb) in enumerate(boxes):
            if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W or bb.y1 > H:
                out.append(f"outside the figure: {s!r}")
            for s2, bb2 in boxes[i + 1:]:
                if bb.overlaps(bb2) and bb.intersection(bb, bb2).width > 1 and bb.intersection(bb, bb2).height > 1:
                    out.append(f"overlap: {s!r} / {s2!r}")
            if s != "<swatch>":
                for b2 in bars:
                    x = bb.intersection(bb, b2)
                    if x is not None and x.width > 1 and x.height > 1:
                        out.append(f"text on a bar: {s!r}")
        return out

    def save(self, out: Path, name: str, theme: str):
        self.fig.canvas.draw()
        bad = self.problems()
        if bad:
            raise SystemExit(f"{name}-{theme}: " + "; ".join(bad))
        self.fig.savefig(out / f"{name}-{theme}.svg", facecolor=self.t["bg"], metadata={"Date": None})
        self.fig.savefig(out / f"{name}-{theme}.png", facecolor=self.t["bg"], dpi=150)
        self.plt.close(self.fig)


def pct(v):
    return "100%" if round(v, 1) == 100 else f"{v:.1f}%"


PAIR = (("deepseek", "ds"), ("nemotron", "ne"))
TOP = 1.95   # plot areas start this far below the top edge (title, subtitle, panel title)


def chart_quality(c: Chart, d: dict):
    t = c.t
    labels = {"mmlu": ("Knowledge · MMLU", "200 questions"), "math": ("Math · GSM8K", "30 problems"),
              "coding": ("Coding", "5 tasks, hidden tests"), "if": ("Instruction following", "20 checked prompts"),
              "json": ("JSON schema output", "6 schemas"), "tools": ("Tool use", "8 calls + 2 chains"),
              "longctx": ("Long context", "passkey at 32K / 131K / 299K"),
              "summary": ("Summary fidelity", "20 facts in a long log")}
    q = d["quality"]
    ax, ys = c.panel(3.3, 9.9, c.h - TOP, len(q), 0.68, "Percent correct  (higher is better)", 0.42, 116,
                     [0, 25, 50, 75, 100], lambda v: f"{v}%")
    for y, k in zip(ys, q):
        for i, (who, col) in enumerate(PAIR):
            v, ok, n = q[k][who]
            c.bar(ax, y + c.slot(ax, i, 2), v, t[col], pct(v), note=f"{ok}/{n}")
    c.ylabels(ax, [labels[k] for k in q], 3.08)
    c.notes(["Same frozen prompts, item sets and scorers for both models · thinking off, greedy · DeepSeek: qbench "
             "2026-10-15 (a re-run the same day gave identical scores) · Nemotron: rebench of the current build, 2026-10-15"])
    c.legend([(DS, t["ds"]), (NE, t["ne"])])


def chart_speed(c: Chart, d: dict):
    t = c.t
    s = d["speed"]
    a, ys = c.panel(2.45, 6.55, c.h - TOP, 3, 0.78, "Writing speed  (tok/s, higher is better)", 0.42, 240,
                    [0, 50, 100, 150, 200], str)
    for y, k in zip(ys, ("code", "prose", "c4")):
        for i, (who, col) in enumerate(PAIR):
            c.bar(a, y + c.slot(a, i, 2), s[who][k], t[col], f"{s[who][k]:.0f}")
    c.ylabels(a, [("Code", "one stream"), ("Prose", "one stream"), ("4 streams", "total, code + prose")], 2.22)
    b, ys = c.panel(9.15, 11.3, c.h - TOP, 2, 0.78 * 1.5, "Time to first token  (ms, lower is better)", 7.55, 230,
                    [0, 50, 100, 150], str)
    for y, k in zip(ys, ("ttft_code_ms", "ttft_prose_ms")):
        for i, (who, col) in enumerate(PAIR):
            c.bar(b, y + c.slot(b, i, 2), s[who][k], t[col], f"{s[who][k]:.0f} ms")
    c.ylabels(b, [("Code prompt", "~50 tokens"), ("Prose prompt", "~50 tokens")], 8.92)
    c.notes(["Replies capped at 320 tokens · thinking off, greedy · times from each server's own timing block · one run per "
             "cell; single-stream code speed moves about ±7 % between runs of the same build · DeepSeek: 2026-10-15 · "
             "Nemotron: the live server with drafting on, 2026-10-16; its prompt cache may already hold these short prompts "
             "(the cold rebench on 2026-10-15 read 101 ms and 85 ms)"])
    c.legend([(DS, t["ds"]), (NE, t["ne"])])


def chart_prefill(c: Chart, d: dict):
    t = c.t
    p = d["prefill"]
    keys, names = ("32768", "131072", "200000"), ("32K", "131K", "200K")
    ax, ys = c.panel(2.45, 9.4, c.h - TOP, 3, 1.05, "Cold prompt, time to first token  (s, lower is better)", 0.42,
                     100, [0, 20, 40, 60, 80], lambda v: f"{v} s")
    for y, key in zip(ys, keys):
        b, a = p["before"][key]["s"], p["after"][key]["s"]
        c.bar(ax, y + c.slot(ax, 0, 3), b, t["old"], f"{b:.1f} s", label_color=t["ink2"])
        c.bar(ax, y + c.slot(ax, 1, 3), a, t["ne"], f"{a:.1f} s", note=f"{b / a:.2f}× faster than before")
        if key in p["deepseek"]:
            v = p["deepseek"][key]["s"]
            c.bar(ax, y + c.slot(ax, 2, 3), v, t["ds"], f"{v:.1f} s")
        else:
            ax.annotate("DeepSeek: no cold run at this length", xy=(0, y + c.slot(ax, 2, 3)), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=10.5, color=t["muted"], style="italic")
    c.ylabels(ax, [(f"{n}-token prompt", f"{round(p['after'][k]['tokens'], -2):,.0f} tokens") for k, n in zip(keys, names)],
              2.22)
    o = p["deepseek_other"]
    c.notes(["Nothing cached (Nemotron reuses only its 7-token chat template) · Nemotron: the server's first-token time, "
             "2026-10-15; current build = 2 runs averaged, first build = 1 run · DeepSeek: one "
             f"{p['deepseek']['200000']['tokens']:,}-token run on 2026-10-14, from earlier work; its other cold runs: "
             f"{o[0]['tokens'] // 1000}K tokens in {o[0]['s']:.1f} s, {o[1]['tokens'] // 1000}K tokens in {o[1]['s']:.1f} s"])
    c.legend([(f"{NE}, current build", t["ne"]), ("Nemotron, first build", t["old"]), (DS, t["ds"])])


def chart_serving(c: Chart, d: dict):
    t = c.t
    L, W = d["serving"]["load16"], d["serving"]["warm"]
    a, ys = c.panel(1.5, 5.2, c.h - TOP, 3, 0.95, "16 conversations at once: wait for the first token", 0.42, 21,
                    [0, 5, 10, 15, 20], lambda v: f"{v} s")
    for y, k in zip(ys, ("code", "prose", "mixed")):
        for i, (tag, col, lc) in enumerate((("p4", t["old"], t["ink2"]), ("p16", t["ne"], t["ink"]))):
            m, w = L[tag][k]["median_s"], L[tag][k]["worst_s"]
            yy = y + c.slot(a, i, 2)
            c.bar(a, yy, m, col)
            a.plot([m, w], [yy, yy], color=col, linewidth=1.5, solid_capstyle="butt", zorder=2)
            a.plot([w, w], [yy - 0.07, yy + 0.07], color=col, linewidth=1.5, zorder=2)
            c.bar(a, yy, 0, col, f"{m:.1f} s", label_color=lc, note=f"worst {w:.1f} s", label_x=w)
    c.ylabels(a, ["Code", "Prose", "Mixed"], 1.27)
    c.text(1.5, a.box[1] - 0.58, "bar = median · line out to the worst request · 2 runs each", size=10.2, color=t["muted"])
    b, ys = c.panel(8.55, 10.0, c.h - TOP, 2, 1.425, "Coming back to a long conversation", 6.55, 7, [0, 2, 4, 6],
                    lambda v: f"{v} s")
    for y, (old, new) in zip(ys, (("cache16", "cache40"), ("restart_nodisk", "restart_disk"))):
        for i, (tag, col, lc) in enumerate(((old, t["old"], t["ink2"]), (new, t["ne"], t["ink"]))):
            v = W[tag]["median_s"]
            c.bar(b, y + c.slot(b, i, 2), v, col, f"{v:.2f} s", label_color=lc,
                  note=f"{W[tag]['hits']}/{W[tag]['of']} hits")
    c.ylabels(b, [("200 agents", "10K-token history each"), ("8 agents", "30K tokens, server restarted")], 8.32)
    c.text(8.55, b.box[1] - 0.58, "median wait on the follow-up turn", size=10.2, color=t["muted"])
    agg = lambda tag: "–".join(f"{L[tag][k]['agg_tok_s']:.0f}" for k in ("prose", "code"))
    c.notes([f"{NE} only, 2026-10-15 · 320-token replies, thinking off · total writing speed at 16 streams: "
             f"{agg('p4')} tok/s before, {agg('p16')} tok/s now · replies are bit-identical at 4 and 16 slots"])
    c.legend([("Now: 16 slots, 40 GiB prompt cache, disk tier", t["ne"]),
                    ("Before: 4 slots, 16 GiB cache, no disk tier", t["old"])])


def chart_vision(c: Chart, d: dict):
    t = c.t
    v = d["vision"]
    a, ys = c.panel(2.5, 6.1, c.h - TOP, 3, 0.95, "Found and right  (percent, higher is better)", 0.42, 140,
                    [0, 25, 50, 75, 100], lambda x: f"{x}%")
    for y, k in zip(ys, ("text", "shapes", "counts")):
        for i, (who, col) in enumerate(PAIR):
            r = v[who][k]
            got = r["exact"] if k == "text" else r["matched"]
            note = f"{got}/{r['of']}" + (f" · {r['extra']} extra" if k != "text" and r["extra"] else "")
            c.bar(a, y + c.slot(a, i, 2), 100 * got / r["of"], t[col], pct(100 * got / r["of"]), note=note)
    c.ylabels(a, [("Text read exactly", "166 strings, 4 images"), ("Shapes matched", "30 shapes, 2 images"),
                  ("Objects counted", "23 objects, 1 image")], 2.27)
    b, ys = c.panel(9.2, 10.65, c.h - TOP, 3, 0.95, "Position error  (px, lower is better)", 7.8, 50,
                    [0, 20, 40], str)
    for y, k in zip(ys, ("text", "shapes", "counts")):
        for i, (who, col) in enumerate(PAIR):
            r = v[who][k]
            n = r["exact"] if k == "text" else r["matched"]
            c.bar(b, y + c.slot(b, i, 2), r["err_px"], t[col], f"{r['err_px']:.1f}", note=f"{n} matched")
    c.ylabels(b, ["Text boxes", "Shapes", "Counted objects"], 8.97)
    n3 = v["nemotron"]["a3_shapes"]
    c.notes(["Same six images, prompts and harness for both: tiled reads with a zoomed re-read, thinking off · position "
             "error = distance from the true centre, averaged over the items each model got right · extra = objects "
             f"reported that are not in the image · DeepSeek {v['date']['deepseek']}, Nemotron {v['date']['nemotron']}",
             f"Nemotron's shapes scene read {n3['matched']}/{n3['of']} with the current prefill setting (NCCL capped at 2 "
             "channels); with the cap off it read 19/20 at 26 px"])
    c.legend([(DS, t["ds"]), (NE, t["ne"])])


def chart_drafting(c: Chart, d: dict):
    t = c.t
    dr = d["drafting"]
    keys = ("code", "explain", "essay", "story", "prose")
    ax, ys = c.panel(1.75, 8.6, c.h - TOP, len(keys), 0.72, "Writing speed, one stream  (tok/s, higher is better)",
                     0.42, 220, [0, 50, 100, 150, 200], str)
    for y, k in zip(ys, keys):
        b, a = dr[k]["before"], dr[k]["after"]
        c.bar(ax, y + c.slot(ax, 0, 2), b, t["old"], f"{b:.1f}", label_color=t["ink2"])
        c.bar(ax, y + c.slot(ax, 1, 2), a, t["ne"], f"{a:.1f}", note=f"{100 * (a / b - 1):+.1f}%")
    c.ylabels(ax, [k.capitalize() for k in keys], 1.52)
    same = all(v["same_text"] for v in dr.values())
    c.notes([f"{NE} only · 320-token replies, greedy, median of 3 · before: 2026-10-15, 4 slots, prompt cache off · "
             "after: the live server 2026-10-16, 16 slots, prompt cache on"
             + (" · every reply is token-for-token identical before and after" if same else
                " · code and essay replies are token-for-token identical; under the live config the other three "
                "differ by a few words")])
    c.legend([("Drafts sized by confidence (live)", t["ne"]), ("Drafts sized by running hit rate (before)", t["old"])])


CHARTS = [
    ("nemotron-vs-deepseek-quality", 11.2, 9.0, "Quality: DeepSeek V4.1 Flash vs Nemotron 3.5 Super VL",
     "Four DGX Sparks · TensorFold Zig engine, TP=4 · DeepSeek 2.9-bit EXL3, Nemotron 120B-A12B NVFP4", chart_quality),
    ("nemotron-vs-deepseek-speed", 12, 6.0, "Speed: DeepSeek V4.1 Flash vs Nemotron 3.5 Super VL",
     "Four DGX Sparks · TensorFold Zig engine, TP=4 · short prompts; writing speed counts after the first token", chart_speed),
    ("nemotron-vs-deepseek-prefill", 11.6, 6.6, "Long prompts: time to the first token, nothing cached",
     "Four DGX Sparks · TensorFold Zig engine, TP=4 · Nemotron before and after the prefill work, DeepSeek for reference",
     chart_prefill),
    ("nemotron-serving", 12, 7.0, "Nemotron 3.5 Super VL: many conversations, and returning ones",
     "Four DGX Sparks · more parallel slots, a bigger prompt cache and a disk tier, in memory the model leaves free",
     chart_serving),
    ("nemotron-vs-deepseek-vision", 12, 6.9, "Vision: DeepSeek V4.1 Flash vs Nemotron 3.5 Super VL",
     "Four DGX Sparks · six synthetic test images with known text, shapes and positions", chart_vision),
    ("nemotron-drafting", 11, 7.1, "Nemotron 3.5 Super VL: drafts sized by confidence",
     "Four DGX Sparks · the model's own MTP head drafts ahead and stops when unsure · live since 2026-10-16",
     chart_drafting),
]


def render(out: Path):
    d = json.loads(NUMBERS.read_text())
    plt = mpl()
    out.mkdir(parents=True, exist_ok=True)
    for name, w, h, title, sub, fn in CHARTS:
        for theme, t in THEMES.items():
            c = Chart(plt, t, w, h, title, sub)
            fn(c, d)
            c.save(out, name, theme)
    print(f"wrote {len(CHARTS) * 2 * 2} files to {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("extract", "all"):
        extract()
        print(f"wrote {NUMBERS}")
    if cmd in ("render", "all"):
        render(Path(sys.argv[2]) if len(sys.argv) > 2 else HERE)
