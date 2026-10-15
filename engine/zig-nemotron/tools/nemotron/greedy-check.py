#!/usr/bin/env python3
"""Greedy outputs of a fixed prompt set (short chats and long real-text prompts), to compare two builds token for token.
  greedy-check.py URL MODEL OUT.json [BASELINE.json]
With a baseline: prints, per prompt, whether the text is identical or where it first differs; exit 1 on any difference."""

import glob, hashlib, json, os, sys, time, urllib.request

url, model, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
base = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
root = os.path.dirname(os.path.abspath(__file__)) + "/../.."


def corpus(limit_chars):
    """Real English and code text from the engine's own docs and sources, in a fixed order."""
    parts, n = [], 0
    for p in sorted(glob.glob(root + "/docs/**/*.md", recursive=True)) + sorted(glob.glob(root + "/zig/src/families/nemotron/*.zig")):
        t = open(p, errors="replace").read()
        parts.append(f"=== {os.path.relpath(p, root)} ===\n{t}")
        n += len(t)
        if n >= limit_chars:
            break
    return "\n".join(parts)[:limit_chars]


PROMPTS = [
    ("sky", "Why is the sky blue? Answer in three sentences.", 96),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", 128),
    ("water", "Explain how electrolysis of water works, step by step.", 96),
    ("doc20k", corpus(60000) + "\n\nSummarize the documents above in five bullet points.", 96),
    ("doc90k", corpus(280000) + "\n\nWhich source files above mention the Mamba scan? List them, then describe the scan in two sentences.", 96),
]

rows = []
for name, text, n in PROMPTS:
    body = {"model": model, "max_tokens": n, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    msg = d["choices"][0]["message"].get("content") or ""
    rows.append({"name": name, "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
                 "wall_s": round(wall, 2), "sha": hashlib.sha256(msg.encode()).hexdigest()[:12], "text": msg})
    print(f"{name}: {d['usage']['prompt_tokens']} prompt tokens, {wall:.1f} s, sha {rows[-1]['sha']}", flush=True)

json.dump({"url": url, "model": model, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}, open(out, "w"), indent=1)
bad = 0
if base:
    ref = {r["name"]: r for r in base["rows"]}
    for r in rows:
        b = ref.get(r["name"])
        if b is None:
            continue
        if b["text"] == r["text"]:
            print(f"  {r['name']}: identical ({len(r['text'])} chars)")
            continue
        bad += 1
        i = next((k for k in range(min(len(b["text"]), len(r["text"]))) if b["text"][k] != r["text"][k]), min(len(b["text"]), len(r["text"])))
        print(f"  {r['name']}: DIFFERS at char {i}: base ...{b['text'][max(0, i - 40):i + 40]!r}\n      new  ...{r['text'][max(0, i - 40):i + 40]!r}")
sys.exit(1 if bad else 0)
