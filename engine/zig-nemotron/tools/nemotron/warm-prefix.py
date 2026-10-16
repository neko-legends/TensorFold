#!/usr/bin/env python3
"""Warm-turn hit rate with several distinct long conversations: AGENTS agents each open with a unique ~PREFIX-token
document (turn 0, cold), then every agent asks follow-ups in round robin (turns 1..TURNS-1), so the cache must hold all
the agents' states at once. Per follow-up: prompt and cached tokens, and the server's time to first token. A hit is a
follow-up whose cached tokens reach the previous turn's prompt (the conversation so far was restored).
`--phase warm` skips turn 0 and replays the follow-ups from a saved run's transcripts (after a restart: what survived).
  warm-prefix.py URL MODEL OUT.json [AGENTS=8] [PREFIX=30000] [TURNS=3] [--phase all|cold|warm] [--from RUN.json]"""

import argparse, json, random, statistics as st, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("agents", nargs="?", type=int, default=8); ap.add_argument("prefix", nargs="?", type=int, default=30000)
ap.add_argument("turns", nargs="?", type=int, default=3)
ap.add_argument("--phase", default="all"); ap.add_argument("--from", dest="src")
a = ap.parse_args()
url = a.url.rstrip("/")
WORDS = """river stone lantern orchard copper meadow harbor violet thunder candle marble falcon willow cedar ember glacier
saffron compass quarry beacon tundra juniper anchor prism walnut citadel heron basalt nectar ledger mosaic orbit pewter
quiver ripple sorrel tablet umber vessel wicker yarrow zephyr almond bramble cobalt dune fable garnet hollow ivory jasper
kestrel lilac mantle nimbus onyx pollen quartz russet sable thistle""".split()
QUESTIONS = ["Which three words appear most often above? Answer in one line.",
             "Name the first and the last word of the document, nothing else.",
             "Write one sentence that uses four words from the document.",
             "Summarize our conversation so far in one sentence."]


def chat(messages, n=32):
    body = {"model": a.model, "max_tokens": n, "temperature": 0, "stream": False, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def doc(agent, words):
    rng = random.Random(1000 + agent)
    lines, line = [], []
    for _ in range(words):
        line.append(rng.choice(WORDS))
        if len(line) == 12:
            lines.append(" ".join(line) + ".")
            line = []
    return "\n".join(lines)


def turn(convo, q):
    convo["messages"].append({"role": "user", "content": q})
    d, wall = chat(convo["messages"])
    text = d["choices"][0]["message"].get("content") or ""
    convo["messages"].append({"role": "assistant", "content": text})
    u, tf = d["usage"], d.get("tensorfold") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    row = {"prompt_tokens": u["prompt_tokens"], "cached_tokens": cached, "wall_s": round(wall, 3), "ttft_s": tf.get("ttft_s"),
           "prefill_s": tf.get("prefill_s"), "prev_prompt": convo.get("prev", 0)}
    row["hit"] = cached >= convo.get("prev", 1 << 60) - 8  # the previous turn's prompt restored (a few template tokens aside)
    convo["prev"] = u["prompt_tokens"]
    return row


words = int(a.prefix * 0.579)  # 0.579 filler words a token on this tokenizer (prefill-ladder.py's calibration)
res = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "url": url, "agents": a.agents, "prefix": a.prefix, "turns": a.turns,
       "phase": a.phase, "cold": [], "warm": []}
if a.phase == "warm":
    convos = json.load(open(a.src))["convos"]
    for c in convos:  # the saved conversations end after their last turn: replay from turn 1's request
        c["messages"] = c["messages"][:2]
        c["prev"] = c["turn0_prompt"]
else:
    convos = []
    for ag in range(a.agents):
        c = {"agent": ag, "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex}]\n{doc(ag, words)}\n\nWhat is this document made of? One line."}]}
        d, wall = chat(c["messages"])
        text = d["choices"][0]["message"].get("content") or ""
        c["messages"].append({"role": "assistant", "content": text})
        c["prev"] = c["turn0_prompt"] = d["usage"]["prompt_tokens"]
        tf = d.get("tensorfold") or {}
        res["cold"].append({"agent": ag, "prompt_tokens": d["usage"]["prompt_tokens"], "wall_s": round(wall, 3), "ttft_s": tf.get("ttft_s")})
        print(json.dumps(res["cold"][-1]), flush=True)
        convos.append(c)
if a.phase != "cold":
    for t in range(1, a.turns):
        for c in convos:
            row = turn(c, QUESTIONS[(t - 1) % len(QUESTIONS)])
            row.update(agent=c["agent"], turn=t)
            res["warm"].append(row)
            print(json.dumps(row), flush=True)
    w = res["warm"]
    tt = [r["ttft_s"] or r["wall_s"] for r in w]
    res["summary"] = {"followups": len(w), "hits": sum(r["hit"] for r in w), "hit_rate": round(sum(r["hit"] for r in w) / len(w), 3),
                      "cached_share": round(sum(r["cached_tokens"] for r in w) / sum(r["prompt_tokens"] for r in w), 3),
                      "ttft_median_s": round(st.median(tt), 3), "ttft_mean_s": round(st.mean(tt), 3), "ttft_max_s": max(tt)}
    print("summary", json.dumps(res["summary"]), flush=True)
res["convos"] = [{"agent": c["agent"], "turn0_prompt": c["turn0_prompt"], "messages": c["messages"]} for c in convos]
json.dump(res, open(a.out, "w"), indent=1)
