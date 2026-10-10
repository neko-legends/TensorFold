#!/usr/bin/env python3
"""Like-for-like Zig vs Python serving bench: the cells of docs/ZIG-RESULTS.md, one script for any endpoint.

Run it against the Python TP=4 server and against the Zig TP=4 candidate (each freshly booted, nothing else on the
Sparks), with the same arguments, then compare the two output directories:

    python3 bench/zig_vs_python.py run --url http://forge:8000 --label python-tp4 --out results/zvp/python-tp4
    python3 bench/zig_vs_python.py run --url http://forge:8000 --label zig-tp4    --out results/zvp/zig-tp4
    python3 bench/zig_vs_python.py compare results/zvp/python-tp4 results/zvp/zig-tp4 > results/zvp/compare.md
    python3 bench/zig_vs_python.py smoke --url http://forge:8000      # one short request: the script works

The cells and the client are Jay's (engine/zig/tools/zig/dsv41_m2c2/bench_http.py, imported, not copied): code and
prose prompts, 384-token replies, thinking off, T0 (greedy) and T0.7 (top-p 0.95, seed 1234), 1 and 4 concurrent
streams (a distinct prompt a stream), decode tok/s = completion tokens after the first over first-to-last token,
x4 = the aggregate over the span all four are live. The method is ZIG-RESULTS.md's:

  1. cold: one repetition of every short cell, no warm-up (the first requests after a fresh boot);
  2. warm: --reps (3) repetitions; each repetition of a cell starts with an untimed 64-token warm-up of that cell;
  3. long: the 32K and 131K workloads (37,974 and 154,183 prompt tokens with this tokenizer), --long-passes (3)
     passes each; every pass is a FRESH prefill (a salt line before the document, the same bytes for both engines:
     pass p of a run uses salt "<--salt-tag> long pass p"), prompt tok/s = prompt tokens / time to first token, then
     the decode tok/s of a 128-token answer;
  4. mix: a 131K prompt with 4 code streams started 20 s after it (both must finish; decode of each side);
  5. mixwin: one 131K stream decoding in the same windows as 3 code streams (bench_http's mixwin cell);
  6. salted: --salted (2) more long passes with new salts (ZIG-RESULTS' "two salted long passes": reply hashes).

The headline is BEST of the warm repetitions, with the median beside it, and cold separately. T0 reply hashes are
kept so `compare` can check the two engines answer identically (same tokens at temperature 0 is the parity check).

Change --salt-tag for every new pair of runs: a repeated salt hits the server's prompt / session cache and the long
cells stop being cold. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "zig" / "tools" / "zig" / "dsv41_m2c2"))
import bench_http as bh  # noqa: E402  (Jay's client: the same requests and timing as his ZIG-RESULTS runs)

SHORT = [(w, t, s) for w in ("code", "prose") for t in (0.0, 0.7) for s in (1, 4)]


def rate(c: dict) -> float | None:
    """A cell's headline: one stream's decode tok/s, or the x4 aggregate."""
    return c.get("mean_tok_s") if c["streams"] == 1 else (c.get("aggregate_tok_s") or c.get("mean_tok_s"))


def models(url: str) -> list[str]:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def save(out: Path, name: str, cells: list[dict], meta: dict) -> None:
    for c in cells:
        c.pop("times", None)
    (out / f"{name}.json").write_text(json.dumps({**meta, "cells": cells}, indent=1))


def salted_long(url: str, model: str, n: int, salt: str, max_tokens: int = 128) -> dict:
    """bench_http.long_cell with a salt line before the document (a fresh prefill; the same bytes on both engines)."""
    r: dict = {}
    bh.one(url, model, f"{salt}\n\n{bh.document(n)}", 0.0, max_tokens, r)
    pt = r.get("prompt_tokens")
    return {"work": f"long-{n}", "temp": 0.0, "streams": 1, "salt": salt, "prompt_tokens": pt, "ttft_s": r.get("ttft_s"),
            "prompt_tok_s": round(pt / r["ttft_s"], 1) if pt and r.get("ttft_s") else None,
            "mean_tok_s": r.get("decode_tok_s"), "aggregate_tok_s": None, "tokens": [r.get("tokens")],
            "sha": [r.get("sha")], "tpr": [r.get("tpr")], "failed_streams": int(bool(r.get("error") or not r.get("first"))),
            "errors": [r["error"]] if r.get("error") else []}


def run(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = a.model or (models(a.url) or ["dsv41"])[0]
    meta = {"label": a.label, "url": a.url, "model": model, "salt_tag": a.salt_tag, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    log = open(out / "bench.log", "a")

    def note(s: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    parts = set(a.parts.split(","))
    note(f"{a.label}: {a.url} model {model}, parts {sorted(parts)}")
    if "cold" in parts:
        cells = []
        for w, t, s in SHORT:
            c = bh.cell(a.url, model, w, t, s, a.max_tokens)
            c["rep"] = "cold"
            cells.append(c)
            note(f"cold {w} T{t} x{s}: {rate(c)}")
        save(out, "cold", cells, meta)
    if "warm" in parts:
        cells = []
        for rep in range(a.reps):
            for w, t, s in SHORT:
                bh.cell(a.url, model, w, t, s, 64)                     # untimed warm-up of this cell
                c = bh.cell(a.url, model, w, t, s, a.max_tokens)
                c["rep"] = rep
                cells.append(c)
                note(f"warm {rep} {w} T{t} x{s}: {rate(c)} (failed streams {c['failed_streams']})")
        save(out, "warm", cells, meta)
    longs = [int(x) for x in a.long.split(",") if x]
    if "long" in parts:
        cells = []
        for p in range(a.long_passes):
            for n in longs:
                c = salted_long(a.url, model, n, f"{a.salt_tag} long pass {p}")
                c["rep"] = p
                cells.append(c)
                note(f"long {n} pass {p}: {c['prompt_tokens']} prompt tokens, prompt {c['prompt_tok_s']} tok/s, "
                     f"TTFT {c['ttft_s']} s, decode {c['mean_tok_s']}")
        save(out, "long", cells, meta)
    if "mix" in parts:
        cells = []
        for p in range(a.long_passes):
            box: dict = {}
            th = threading.Thread(target=lambda: box.update(c=salted_long(a.url, model, a.mix_long, f"{a.salt_tag} mix pass {p}", 512)))
            th.start()
            time.sleep(20)
            short = bh.cell(a.url, model, "code", 0.0, 4, a.max_tokens)
            th.join()
            lc = box["c"]
            lc["work"], lc["rep"], short["work"], short["rep"] = f"mix-long-{a.mix_long}", p, "mix-code", p
            cells += [lc, short]
            note(f"mix pass {p}: long decode {lc['mean_tok_s']} (failed {lc['failed_streams']}), "
                 f"code x4 {rate(short)} (failed {short['failed_streams']})")
        save(out, "mix", cells, meta)
    if "mixwin" in parts:
        cells = []
        for p in range(a.long_passes):
            bh.SALT = f"{a.salt_tag} mixwin pass {p}\n\n"               # the 3 code streams; the long one is the plain document
            c = bh.mixwin_cell(a.url, model, a.mix_long)
            bh.SALT = ""
            c["rep"] = p
            cells.append(c)
            note(f"mixwin pass {p}: aggregate {c['aggregate_tok_s']}, long live {c['long_live_tok_s']}")
        save(out, "mixwin", cells, meta)
    if "salted" in parts:
        cells = []
        for p in range(a.salted):
            for n in longs:
                c = salted_long(a.url, model, n, f"{a.salt_tag} salted {p}")
                c["rep"] = p
                cells.append(c)
                note(f"salted {n} {p}: sha {c['sha']}")
        save(out, "salted", cells, meta)
    note("done")
    return 0


def load(d: Path, name: str) -> list[dict]:
    try:
        return json.loads((d / f"{name}.json").read_text())["cells"]
    except (OSError, ValueError, KeyError):
        return []


def stats(cells: list[dict], key, val) -> dict:
    """key -> (best, median, n) over repetitions without a failed stream."""
    by: dict = {}
    for c in cells:
        v = val(c)
        if c.get("failed_streams") or v is None:
            continue
        by.setdefault(key(c), []).append(v)
    return {k: (max(v), statistics.median(v), len(v)) for k, v in by.items()}


def pct(a: float | None, b: float | None) -> str:
    return f"{100 * (b / a - 1):+.1f} %" if a and b else "-"


def compare(a: argparse.Namespace) -> int:
    pa, pb = Path(a.a), Path(a.b)
    la = json.loads(next(pa.glob("*.json")).read_text()).get("label", pa.name)
    lb = json.loads(next(pb.glob("*.json")).read_text()).get("label", pb.name)
    k3 = lambda c: (c["work"], c["temp"], c["streams"])  # noqa: E731
    print(f"# {lb} vs {la}\n")
    print(f"| cell | {la} best / median | {lb} best / median | best Δ | median Δ | cold: {la} / {lb} |")
    print("|---|---:|---:|---:|---:|---:|")
    wa, wb = stats(load(pa, "warm"), k3, rate), stats(load(pb, "warm"), k3, rate)
    ca, cb = stats(load(pa, "cold"), k3, rate), stats(load(pb, "cold"), k3, rate)
    for k in [(w, t, s) for w, t, s in SHORT]:
        x, y = wa.get(k), wb.get(k)
        cx, cy = (ca.get(k) or (None,))[0], (cb.get(k) or (None,))[0]
        name = f"{k[0]} T{k[1]:g} x{k[2]}"
        if not x or not y:
            print(f"| {name} | {x} | {y} | not measured on both | | |")
            continue
        print(f"| {name} | {x[0]:.1f} / {x[1]:.1f} | {y[0]:.1f} / {y[1]:.1f} | {pct(x[0], y[0])} | {pct(x[1], y[1])} | "
              f"{cx or '-'} / {cy or '-'} |")
    kl = lambda c: c["work"]  # noqa: E731
    pr = lambda c: c.get("prompt_tok_s")  # noqa: E731
    dec = lambda c: c.get("mean_tok_s")  # noqa: E731
    lpa, lpb = stats(load(pa, "long"), kl, pr), stats(load(pb, "long"), kl, pr)
    lda, ldb = stats(load(pa, "long"), kl, dec), stats(load(pb, "long"), kl, dec)
    if lpa or lpb:
        print(f"\n| long cell | {la} prompt tok/s best / median | {lb} | Δ | {la} decode best | {lb} | Δ |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for w in sorted(set(lpa) | set(lpb), key=lambda s: int(s.split("-")[-1])):
            x, y, dx, dy = lpa.get(w), lpb.get(w), lda.get(w), ldb.get(w)
            f = lambda t: f"{t[0]:.1f} / {t[1]:.1f}" if t else "-"  # noqa: E731
            print(f"| {w} | {f(x)} | {f(y)} | {pct(x and x[0], y and y[0])} | {dx and dx[0]} | {dy and dy[0]} | "
                  f"{pct(dx and dx[0], dy and dy[0])} |")
    for name, val in (("mix", lambda c: rate(c)), ("mixwin", lambda c: c.get("aggregate_tok_s"))):
        sa, sb = stats(load(pa, name), kl, val), stats(load(pb, name), kl, val)
        for w in sorted(set(sa) | set(sb)):
            x, y = sa.get(w), sb.get(w)
            print(f"\n{name} {w}: {la} best {x and x[0]}, {lb} best {y and y[0]} ({pct(x and x[0], y and y[0])})")
    # T0 parity: the same reply (hash) for the same prompt, cell by cell and repetition by repetition
    eq = tot = 0
    for f in ("cold", "warm", "long", "salted"):
        sa = {(k3(c), c.get("rep"), c.get("salt")): c.get("sha") for c in load(pa, f) if c["temp"] == 0}
        sb = {(k3(c), c.get("rep"), c.get("salt")): c.get("sha") for c in load(pb, f) if c["temp"] == 0}
        for k in sa.keys() & sb.keys():
            for x, y in zip(sa[k] or [], sb[k] or []):
                tot += 1
                eq += x == y
    print(f"\nT0 replies equal ({lb} vs {la}): {eq}/{tot}")
    parts = ("cold", "warm", "long", "mix", "mixwin", "salted")
    failed = [sum(c.get("failed_streams", 0) for f in parts for c in load(p, f)) for p in (pa, pb)]
    print(f"failed streams: {la} {failed[0]}, {lb} {failed[1]}")
    return 0


def smoke(a: argparse.Namespace) -> int:
    """One short greedy request through bench_http's own client (thinking off, 16 tokens)."""
    model = a.model or (models(a.url) or ["dsv41"])[0]
    r: dict = {}
    bh.one(a.url, model, "What is 17 * 23? Answer with the number only.", 0.0, 16, r)
    r.pop("times", None)
    print(json.dumps({"url": a.url, "model": model, **{k: r.get(k) for k in ("tokens", "ttft_s", "decode_tok_s", "sha", "prompt_tokens", "error")}}))
    return 0 if r.get("first") and not r.get("error") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="", help="served model id (default: the first of /v1/models)")
    r.add_argument("--parts", default="cold,warm,long,mix,mixwin,salted")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--max-tokens", type=int, default=384)
    r.add_argument("--long", default="32768,131072")
    r.add_argument("--long-passes", type=int, default=3)
    r.add_argument("--mix-long", type=int, default=131072)
    r.add_argument("--salted", type=int, default=2)
    r.add_argument("--salt-tag", default=os.environ.get("ZVP_SALT", "zvp-1"),
                   help="new for every pair of runs (the same for the Python and the Zig run of a pair)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    s = sub.add_parser("smoke")
    s.add_argument("--url", required=True)
    s.add_argument("--model", default="")
    a = ap.parse_args()
    return {"run": run, "compare": compare, "smoke": smoke}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
