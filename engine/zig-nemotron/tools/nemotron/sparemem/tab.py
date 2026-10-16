import json, sys, statistics as st
cfgs = sys.argv[1:]
D = {c: json.load(open(c + "-conc.json")) for c in cfgs}
print("| streams | kind | " + " | ".join(f"{c}: agg / per-stream / TTFT med (max)" for c in cfgs) + " |")
print("|---|---|" + "---|" * len(cfgs))
for n in (1, 4, 8, 16):
    for k in ("code", "prose", "mixed"):
        cells = []
        for c in cfgs:
            rs = [r for r in D[c]["levels"] if r["streams"] == n and r["kind"] == k]
            cells.append("%.0f / %.0f / %.2f (%.1f)" % (st.mean(r["aggregate_tok_s"] for r in rs), st.mean(r["per_stream_tok_s_median"] for r in rs),
                                                      st.mean(r["ttft_median_s"] for r in rs), max(r["ttft_max_s"] for r in rs)))
        print(f"| {n} | {k} | " + " | ".join(cells) + " |")
for c in cfgs:
    q = D[c]["qbench_c1"]
    print(c, "qbench c1 code", q["code"]["decode_tok_s"], "prose", q["prose"]["decode_tok_s"])
