#!/usr/bin/env python3
"""Prompt cache check: a long first turn, then follow-up turns that extend it; each turn's time to first token, the
server's cached tokens and the greedy reply. Run once with the cache on and once with it off (--prompt-cache-gib 0)
and compare the replies: a turn restored from a kept state must answer as the cold pass does.
  cache-check.py URL MODEL OUT.json [BASELINE.json] [DOC_CHARS=120000]"""

import glob, json, os, sys, time, urllib.request, uuid

url, model, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
base = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
doc_chars = int(sys.argv[5]) if len(sys.argv) > 5 else 120000
root = os.path.dirname(os.path.abspath(__file__)) + "/../.."

parts, n = [], 0
for p in sorted(glob.glob(root + "/docs/**/*.md", recursive=True)) + sorted(glob.glob(root + "/zig/src/families/nemotron/*.zig")):
    t = open(p, errors="replace").read()
    parts.append(f"=== {os.path.relpath(p, root)} ===\n{t}")
    n += len(t)
    if n >= doc_chars:
        break
doc = "\n".join(parts)[:doc_chars]
nonce = sys.argv[6] if len(sys.argv) > 6 else uuid.uuid4().hex  # the same nonce on and off: identical prompts

QUESTIONS = ["Which files above deal with tensor parallelism? Answer in three bullet points.",
             "Now name one function from the first of those files and say what it does, in two sentences.",
             "Summarize our conversation so far in one sentence."]


def chat(messages, n=80):
    body = {"model": model, "max_tokens": n, "temperature": 0, "stream": False, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


messages = [{"role": "user", "content": f"[{nonce}]\n{doc}\n\n{QUESTIONS[0]}"}]
rows = []
for turn in range(len(QUESTIONS)):
    if turn > 0:
        messages.append({"role": "user", "content": QUESTIONS[turn]})
    d, wall = chat(messages)
    text = d["choices"][0]["message"].get("content") or ""
    tf = d.get("tensorfold") or {}
    u = d["usage"]
    row = {"turn": turn, "prompt_tokens": u["prompt_tokens"], "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "wall_s": round(wall, 3), "ttft_s": tf.get("ttft_s"), "prefill_s": tf.get("prefill_s"), "text": text}
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    messages.append({"role": "assistant", "content": text})

json.dump({"url": url, "nonce": nonce, "doc_chars": doc_chars, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, open(out, "w"), indent=1)
bad = 0
if base:
    for r, b in zip(rows, base["rows"]):
        same = r["text"] == b["text"]
        bad += not same
        print(f"  turn {r['turn']}: {'identical' if same else 'DIFFERS'} (cached {r['cached_tokens']} vs {b['cached_tokens']}; ttft {r['ttft_s']} vs {b['ttft_s']})")
        if not same:
            print(f"    base {b['text'][:160]!r}\n    new  {r['text'][:160]!r}")
sys.exit(1 if bad else 0)
