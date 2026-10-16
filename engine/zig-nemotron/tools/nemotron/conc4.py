#!/usr/bin/env python3
"""Four greedy streams at once (code, prose, code, prose): aggregate tok/s, and each reply's tokens against the
single-stream replies of a draft-speed.py run (shared rounds must commit what a lone stream commits).
  conc4.py URL SPEED.json [REPS]"""

import concurrent.futures as cf, json, sys, time, urllib.request

url, ref = sys.argv[1].rstrip("/"), json.load(open(sys.argv[2]))
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TEXT = {"code": "Write a Python function that merges two sorted lists into one sorted list. Include a docstring and a short usage example. Do not explain outside the code.",
        "prose": "Write a calm 250-word description of a harbour at dawn, in plain prose. No lists, no headings."}


def one(kind):
    body = {"model": "x", "max_tokens": 320, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": TEXT[kind]}], "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return kind, d["usage"]["completion_tokens"], d["tensorfold"]["token_sha"]


bad = 0
for rep in range(reps):
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(one, ["code", "prose", "code", "prose"]))
    wall = time.time() - t0
    toks = sum(r[1] for r in rs)
    same = [r[2] in {x["token_sha"] for x in ref[r[0]]["runs"]} for r in rs]
    bad += not all(same)
    print(f"c4 rep {rep}: {toks} tokens in {wall:.2f} s = {toks / wall:.1f} tok/s aggregate; tokens as alone: {same}", flush=True)
sys.exit(1 if bad else 0)
