#!/usr/bin/env python3
"""qbench speed shapes with real per-chunk stream timestamps: qbench reads the whole SSE body before parsing, so its
streamed TTFT is the wall time unless the server sends a `tensorfold` timing block (DeepSeek does, Nemotron does not).
  window-speed.py HOST PORT MODEL OUT.json"""

import os
import json, sys, time, http.client, concurrent.futures as cf
sys.path.insert(0, os.path.expanduser(os.environ.get("SPARK_BENCH", "~/git/spark-bench")) + "/bench"); import qbench
host, port, model, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def one(prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": qbench.SPEED_TOKENS, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    c = http.client.HTTPConnection(host, port, timeout=600); t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"}); r = c.getresponse()
    first = None; usage = {}; tf = {}
    while True:
        line = r.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data:"): continue
        p = line[5:].strip()
        if p == b"[DONE]": break
        d = json.loads(p)
        if d.get("usage"): usage = d["usage"]
        if d.get("tensorfold"): tf = d["tensorfold"]
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            if first is None and (dl.get("content") or dl.get("reasoning_content")): first = time.time()
    t1 = time.time(); comp = usage.get("completion_tokens"); ttft = first - t0
    return {"ttft_s": round(ttft, 3), "completion_tokens": comp, "wall_s": round(t1 - t0, 2), "decode_s": round(t1 - first, 3),
            "decode_tok_s": round((comp - 1) / (t1 - first), 2), "wall_tok_s": round(comp / (t1 - t0), 2), "server_tf": tf}
res = {"c1": {}, "c4": {}}
for rep in range(3):
    for k, p in qbench.SPEED_PROMPTS.items(): res["c1"].setdefault(k, []).append(one(p))
ps = [qbench.SPEED_PROMPTS[k] for k in ("code", "prose", "code", "prose")]
for rep in range(2):
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex: rs = list(ex.map(one, ps))
    w = time.time() - t0; toks = sum(r["completion_tokens"] for r in rs)
    res["c4"].setdefault("runs", []).append({"completion_tokens": toks, "wall_s": round(w, 2), "aggregate_tok_s": round(toks / w, 2),
        "per_stream_decode_tok_s": [r["decode_tok_s"] for r in rs], "ttft_s": [r["ttft_s"] for r in rs]})
json.dump(res, open(out, "w"), indent=1)
for k, v in res["c1"].items(): print(k, [(x["completion_tokens"], x["ttft_s"], x["decode_tok_s"], x["wall_tok_s"]) for x in v])
for x in res["c4"]["runs"]: print("c4", x["aggregate_tok_s"], x["per_stream_decode_tok_s"], x["ttft_s"])
