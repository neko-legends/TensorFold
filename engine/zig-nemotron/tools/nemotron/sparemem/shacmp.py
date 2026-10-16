import json, sys
D = {t: json.load(open(t + "-conc.json")) for t in sys.argv[1:]}
for t, d in D.items(): print(t, {k: v["shas"] for k, v in d["qbench_c1"].items()})
for kind in ["code", "prose", "mixed"]:
    for i in range(16):
        s = {}
        for t, d in D.items():
            for r in d["levels"]:
                if r["kind"] == kind and i < r["streams"]:
                    s.setdefault(r["shas"][i], []).append("%s/c%d/%d" % (t, r["streams"], r["rep"]))
        print(kind, i, len(s), [(k, sorted(v)) for k, v in s.items()] if len(s) > 1 else "")
