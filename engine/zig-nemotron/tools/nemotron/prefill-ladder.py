#!/usr/bin/env python3
"""Cold prefill ladder: one chat request per size, a unique random prefix (nothing cached), max_tokens 1.
Records the client's time to first token and the server's own clocks (the reply's `tensorfold` block).
  prefill-ladder.py URL MODEL OUT.json [SIZES=32768,131072,200000] [REPS=1] [LABEL]"""

import json, random, sys, time, uuid, urllib.request

url, model, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
sizes = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else "32768,131072,200000").split(",")]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 1
label = sys.argv[6] if len(sys.argv) > 6 else ""

rng = random.Random(20261015)  # the filler is the same every run; the nonce in front of it is not
WORDS = [w for w in """river stone lantern orchard copper meadow harbor violet thunder candle marble falcon willow
cedar ember glacier saffron compass quarry beacon tundra juniper anchor prism walnut citadel heron basalt nectar
ledger mosaic orbit pewter quiver ripple sorrel tablet umber vessel wicker yarrow zephyr almond bramble cobalt
dune fable garnet hollow ivory jasper kestrel lilac mantle nimbus onyx pollen quartz russet sable thistle""".split()]


def filler(n_words):
    out, line = [], []
    for i in range(n_words):
        line.append(rng.choice(WORDS))
        if len(line) == 12:
            out.append(" ".join(line) + ".")
            line = []
    return "\n".join(out)


def ask(text, timeout=1800):
    body = {"model": model, "max_tokens": 1, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": text + "\n\nReply with one word: what was the first word above?"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    return d, wall


# calibrate words per token on this model's tokenizer
probe = filler(4000)
d, _ = ask(f"[{uuid.uuid4().hex}] " + probe)
words_per_token = 4000 / d["usage"]["prompt_tokens"]
print(f"calibration: {d['usage']['prompt_tokens']} tokens for 4000 words ({words_per_token:.3f} words/token)", flush=True)

rows = []
for size in sizes:
    text_words = int(size * words_per_token)
    base = filler(text_words)
    for rep in range(reps):
        nonce = uuid.uuid4().hex
        d, wall = ask(f"[{nonce}] " + base)
        tf = d.get("tensorfold") or d.get("runtime") or {}
        n = d["usage"]["prompt_tokens"]
        pf = tf.get("prefill_s") or tf.get("prefill_seconds")
        ttft = tf.get("ttft_s") or tf.get("time_to_first_token")
        row = {"target": size, "rep": rep, "prompt_tokens": n, "cached_tokens": (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
               "wall_s": round(wall, 3), "server_ttft_s": ttft, "server_prefill_s": pf,
               "prompt_tok_s_wall": round(n / wall, 1), "prompt_tok_s_server": round(n / pf, 1) if pf else None,
               "reply": (d["choices"][0]["message"].get("content") or "")[:40], "tensorfold": tf}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "tensorfold"}), flush=True)

json.dump({"label": label, "url": url, "model": model, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, open(out, "w"), indent=1)
print("wrote", out)
