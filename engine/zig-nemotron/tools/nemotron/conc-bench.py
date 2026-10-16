#!/usr/bin/env python3
"""Concurrent decode: N streams at once (code, prose, or half of each), each a distinct prompt of qbench's speed shape
(320 tokens, temperature 0, thinking off), timed per chunk on the client. Per level: aggregate tok/s (all completion
tokens over the wall from the first send to the last token), each stream's decode tok/s after its first token, and
time to first token under that load, and each reply's text hash (stream i asks the same at every level, so a
batch that changed a stream's bits shows as a different hash). c1 also runs qbench's own two speed prompts (the single-stream reference).
  conc-bench.py HOST PORT MODEL OUT.json [LEVELS=1,4,8,16] [REPS=2] [KINDS=code,prose,mixed]"""

import concurrent.futures as cf, hashlib, http.client, json, statistics as st, sys, time

host, port, model, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
levels = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,4,8,16").split(",")]
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 2
kinds = (sys.argv[7] if len(sys.argv) > 7 else "code,prose,mixed").split(",")
TOKENS = 320
QB = {"code": "Write a Python function that merges two sorted lists into one sorted list. Include a docstring "
              "and a short usage example. Do not explain outside the code.",
      "prose": "Write a calm 250-word description of a harbour at dawn, in plain prose. No lists, no headings."}
CODE = ["merges two sorted lists into one sorted list", "returns the n-th Fibonacci number iteratively",
        "checks whether a string is a palindrome ignoring punctuation", "flattens an arbitrarily nested list",
        "counts word frequencies in a text and returns the top k", "parses an ISO 8601 date string into a tuple",
        "computes the longest common prefix of a list of strings", "rotates a square matrix by 90 degrees",
        "removes duplicates from a list while keeping the order", "converts a Roman numeral to an integer",
        "groups a list of words into anagram sets", "finds the median of two sorted arrays",
        "validates balanced brackets in an expression", "chunks a list into pieces of size n",
        "implements binary search over a sorted list", "computes a running average over a sliding window"]
PROSE = ["a harbour at dawn", "a mountain village in winter", "a desert market at noon", "a library late at night",
         "a forest after rain", "a train station at rush hour", "an orchard in autumn", "a lighthouse in a storm",
         "a quiet beach at dusk", "a city rooftop garden", "a river delta in spring", "a snowy railway crossing",
         "an old bookshop in the afternoon", "a farmers' field at harvest", "a fishing boat at sea", "a cathedral square"]


def prompt(kind, i):
    if kind == "mixed":
        kind = "code" if i % 2 == 0 else "prose"
    if kind == "code":
        return kind, f"Write a Python function that {CODE[i % 16]}. Include a docstring and a short usage example. Do not explain outside the code."
    return kind, f"Write a calm 250-word description of {PROSE[i % 16]}, in plain prose. No lists, no headings."


def one(text):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": TOKENS, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    c = http.client.HTTPConnection(host, port, timeout=1200)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    first = None
    usage = {}
    text = []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        d = json.loads(p)
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            if first is None and (dl.get("content") or dl.get("reasoning_content")):
                first = time.time()
            text.append(dl.get("content") or "")
    t1 = time.time()
    comp = usage.get("completion_tokens") or 0
    return {"t0": t0, "first": first, "t1": t1, "tokens": comp, "ttft_s": round(first - t0, 3),
            "decode_tok_s": round((comp - 1) / (t1 - first), 2) if comp > 1 else None,
            "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:12]}


def level(n, kind):
    ps = [prompt(kind, i)[1] for i in range(n)]
    with cf.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(one, ps))
    start, end = min(r["t0"] for r in rs), max(r["t1"] for r in rs)
    toks = sum(r["tokens"] for r in rs)
    dec = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
    tt = [r["ttft_s"] for r in rs]
    return {"streams": n, "kind": kind, "tokens": toks, "wall_s": round(end - start, 2), "aggregate_tok_s": round(toks / (end - start), 1),
            "per_stream_tok_s_median": round(st.median(dec), 1), "per_stream_tok_s_min": min(dec), "ttft_median_s": round(st.median(tt), 3),
            "ttft_max_s": max(tt), "short": sum(1 for r in rs if r["tokens"] < TOKENS), "shas": [r["sha"] for r in rs]}


res = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": f"{host}:{port}", "levels": [], "qbench_c1": {}}
for k in ("code", "prose"):  # qbench's own prompts, three times each: the single-stream reference
    rs = [one(QB[k]) for _ in range(3)]
    res["qbench_c1"][k] = {"decode_tok_s": [r["decode_tok_s"] for r in rs], "ttft_s": [r["ttft_s"] for r in rs], "shas": [r["sha"] for r in rs]}
    print("qbench c1", k, res["qbench_c1"][k], flush=True)
for n in levels:
    for kind in kinds:
        for rep in range(reps):
            row = level(n, kind)
            row["rep"] = rep
            res["levels"].append(row)
            print(json.dumps(row), flush=True)
json.dump(res, open(out, "w"), indent=1)
