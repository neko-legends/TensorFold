#!/usr/bin/env python3
"""Follow-up replies of two warm-prefix.py runs over the same conversations: identical or not, per agent and turn.
  samereplies.py A.json B.json"""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def replies(d):
    out = {}
    for c in d["convos"]:
        ms = c["messages"]
        for t in range(1, len(ms) // 2):
            out[(c["agent"], t)] = ms[2 * t + 1]["content"]
    return out
ra, rb = replies(a), replies(b)
keys = sorted(set(ra) & set(rb))
same = sum(ra[k] == rb[k] for k in keys)
print(f"replies identical: {same}/{len(keys)}")
for k in keys:
    if ra[k] != rb[k]:
        print("  differs", k, repr(ra[k][:80]), "|", repr(rb[k][:80]))
