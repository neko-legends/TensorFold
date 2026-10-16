#!/usr/bin/env python3
"""Single-stream greedy decode speed and drafting stats by prompt kind, from the server's own timing block.
  draft-speed.py URL OUT.json [REPS] [BASELINE.json]
Per prompt: server tok/s (decode only), rounds, drafted, accepted, tokens a round, ms a round, and token_sha, so two
builds compare token for token (with a baseline: exit 1 if any reply's tokens differ)."""

import json, statistics, sys, urllib.request

url, out = sys.argv[1].rstrip("/"), sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
base = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None

PROMPTS = [  # qbench's speed shapes first (code, prose), then more prose and mixed replies
    ("code", "Write a Python function that merges two sorted lists into one sorted list. Include a docstring and a short usage example. Do not explain outside the code.", 320),
    ("prose", "Write a calm 250-word description of a harbour at dawn, in plain prose. No lists, no headings.", 320),
    ("story", "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.", 320),
    ("explain", "Explain how electrolysis of water works, step by step, for a high-school student.", 320),
    ("essay", "Write a persuasive paragraph arguing that cities should invest in public libraries.", 320),
]


def one(text, n):
    body = {"model": "x", "max_tokens": n, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    tf, sp = d.get("tensorfold", {}), d.get("speculative", {})
    toks = d["usage"]["completion_tokens"]
    rounds = sp.get("rounds") or 0
    return {"tokens": toks, "tok_s": round(tf.get("tokens_per_second", 0), 2), "decode_s": tf.get("decode_s"),
            "rounds": rounds, "drafted": sp.get("drafted"), "accepted": sp.get("accepted"),
            "tokens_per_round": round(toks / rounds, 3) if rounds else None,
            "ms_per_round": round(1000 * tf["decode_s"] / rounds, 2) if rounds else None,
            "token_sha": tf.get("token_sha"), "text": d["choices"][0]["message"].get("content", "")}


res = {}
for name, text, n in PROMPTS:
    runs = [one(text, n) for _ in range(reps)]
    med = statistics.median(r["tok_s"] for r in runs)
    res[name] = {"median_tok_s": med, "runs": runs}
    r0 = runs[0]
    acc = r0["accepted"] / r0["drafted"] if r0["drafted"] else 0
    print(f"{name:8s} {med:7.2f} tok/s  [{', '.join(str(r['tok_s']) for r in runs)}]  tokens {r0['tokens']} rounds {r0['rounds']} "
          f"drafted {r0['drafted']} accepted {r0['accepted']} ({acc:.0%}) tok/round {r0['tokens_per_round']} ms/round {r0['ms_per_round']} sha {r0['token_sha']}",
          flush=True)
json.dump(res, open(out, "w"), indent=1)
bad = 0
if base:
    for name in res:
        a = {r["token_sha"] for r in res[name]["runs"]}
        b = {r["token_sha"] for r in base[name]["runs"]} if name in base else set()
        same = a == b and len(a) == 1
        bad += not same
        print(f"{name:8s} {'IDENTICAL' if same else 'DIFFERS'} {sorted(a)} vs {sorted(b)}  {base[name]['median_tok_s'] if name in base else '-'} -> {res[name]['median_tok_s']} tok/s")
sys.exit(1 if bad else 0)
