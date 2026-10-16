#!/usr/bin/env python3
"""Vision serving under load: 1/4/8/16 concurrent image requests against one OpenAI-compatible endpoint, plus
single-stream decode speed on image prompts vs text prose and code.

  vision-serve.py URL OUT.json [LEVELS=1,4,8,16] [REQUESTS=16]          # cold: every image new to the server
  VISION_SERVE_WARM=1 vision-serve.py URL OUT.json ...                  # warm: the same images again (prompt cache hits)

Every level sends the same REQUESTS requests (the bench images below, cycled) through LEVELS closed-loop workers:
one fixed transcription + description prompt, greedy, thinking off, streamed. Cold: each request's image differs from
every other in one pixel's lowest red bit (with_nonce), so neither the image front end's cache nor the prompt cache
can serve it and every request pays the full image encode + prompt prefill (cached_tokens_max in each summary shows
it). Warm: the images as they are, each sent once first, so every request is a follow-up on an image the server has
seen. Per level: server prefill seconds, client TTFT median/worst, server decode tok/s per stream, aggregate tok/s,
images a minute, MTP acceptance (the server's speculative block), and a transcription score (ground-truth strings
found verbatim in the reply). Images come from $SPARK_VISION/bench (code-rendered, exact ground truth)."""

import base64, io, itertools, json, os, queue, re, statistics, sys, threading, time, urllib.request

url, out = sys.argv[1].rstrip("/"), sys.argv[2]
levels = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,8,16").split(",")]
n_req = int(sys.argv[4]) if len(sys.argv) > 4 else 16
bench = os.path.join(os.path.expanduser(os.environ.get("SPARK_VISION", "~/spark-vision")), "bench")
IMAGES = ["a1_poster", "a2_dashboard", "a4_table_chart", "k_a_spritesheet", "k_h1_lowcontrast", "k_h2_busy", "k_h3_wide",
          "a3_shapes"]
PROMPT = ("Transcribe every piece of text in this image exactly as written, one item per line, in reading order. "
          "Then describe the image in two sentences.")
MAX_TOKENS = 600
TEXT_PROMPTS = [  # draft-speed.py's qbench shapes
    ("code", "Write a Python function that merges two sorted lists into one sorted list. Include a docstring and a short usage example. Do not explain outside the code."),
    ("prose", "Write a calm 250-word description of a harbour at dawn, in plain prose. No lists, no headings."),
]
SINGLE_TOKENS = 320


def with_nonce(png: bytes, nonce: int) -> bytes:
    """Flip the low bit of one pixel's red channel (top row, position from the nonce): a new image to every cache
    (the server keys images by their pixels, so new PNG bytes alone still hit), visually the same picture."""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    x = nonce % im.width
    r, g, b = im.getpixel((x, 0))
    im.putpixel((x, 0), (r ^ 1, g, b))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def score(name: str, reply: str) -> dict:
    gt = json.load(open(os.path.join(bench, name + ".gt.json")))
    texts = [t["text"] for t in gt.get("texts") or []]
    r = norm(reply)
    return {"n_gt": len(texts), "exact": sum(norm(t) in r for t in texts)}


def stream(content, max_tokens):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, text, tf, sp, usage = None, [], {}, {}, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    first = first or time.time()
                    text.append(piece)
            tf = d.get("tensorfold", tf)
            sp = d.get("speculative", sp)
            usage = d.get("usage") or usage
    t1 = time.time()
    return {"t0": t0, "t1": t1, "ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 3),
            "prefill_s": tf.get("prefill_s"), "server_ttft_s": tf.get("ttft_s"), "decode_s": tf.get("decode_s"),
            "tok_s": round(tf.get("tokens_per_second", 0), 2), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "drafted": sp.get("drafted"), "accepted": sp.get("accepted"),
            "token_sha": tf.get("token_sha"), "text": "".join(text)}


NONCE = itertools.count(int(time.time()))


WARM = os.environ.get("VISION_SERVE_WARM") == "1"


def image_content(name):
    png = open(os.path.join(bench, name + ".png"), "rb").read()
    if not WARM:
        png = with_nonce(png, next(NONCE))
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": PROMPT}]


def run_level(n):
    jobs = queue.Queue()
    for i in range(n_req):
        jobs.put((i, IMAGES[i % len(IMAGES)]))
    rows, lock = [], threading.Lock()

    def worker():
        while True:
            try:
                i, name = jobs.get_nowait()
            except queue.Empty:
                return
            r = stream(image_content(name), MAX_TOKENS)
            r.update({"i": i, "image": name, **{"score_" + k: v for k, v in score(name, r["text"]).items()}})
            with lock:
                rows.append(r)

    t = time.time()
    ths = [threading.Thread(target=worker) for _ in range(n)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t
    rows.sort(key=lambda r: r["i"])
    ttft = [r["ttft_s"] for r in rows]
    toks = sum(r["completion_tokens"] for r in rows)
    dr, ac = sum(r["drafted"] or 0 for r in rows), sum(r["accepted"] or 0 for r in rows)
    s = {"concurrency": n, "requests": len(rows), "wall_s": round(wall, 2), "images_per_min": round(60 * len(rows) / wall, 2),
         "prefill_s_median": round(statistics.median(r["prefill_s"] for r in rows), 3),
         "prefill_s_worst": round(max(r["prefill_s"] for r in rows), 3),
         "ttft_s_median": round(statistics.median(ttft), 3), "ttft_s_worst": round(max(ttft), 3),
         "decode_tok_s_per_stream_median": round(statistics.median(r["tok_s"] for r in rows), 2),
         "aggregate_tok_s": round(toks / wall, 2), "completion_tokens": toks,
         "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in rows),
         "cached_tokens_max": max(r["cached_tokens"] or 0 for r in rows),
         "mtp_acceptance": round(ac / dr, 4) if dr else None,
         "text_exact": sum(r["score_exact"] for r in rows), "text_gt": sum(r["score_n_gt"] for r in rows)}
    print(json.dumps(s), flush=True)
    return {"summary": s, "rows": rows}


def single(kind, content, reps=3):
    runs = [stream(content() if callable(content) else content, SINGLE_TOKENS) for _ in range(reps)]
    dr, ac = sum(r["drafted"] or 0 for r in runs), sum(r["accepted"] or 0 for r in runs)
    s = {"kind": kind, "median_tok_s": statistics.median(r["tok_s"] for r in runs), "runs_tok_s": [r["tok_s"] for r in runs],
         "tokens": [r["completion_tokens"] for r in runs], "mtp_acceptance": round(ac / dr, 4) if dr else None}
    print(json.dumps(s), flush=True)
    return {"summary": s, "runs": runs}


res = {"url": "(the live server)", "date": time.strftime("%Y-%m-%d %H:%M %Z"), "images": IMAGES, "prompt": PROMPT,
       "max_tokens": MAX_TOKENS, "requests_per_level": n_req, "warm": WARM, "levels": []}
if WARM:
    for name in IMAGES:
        stream(image_content(name), 8)
for n in levels:
    res["levels"].append(run_level(n))
# single stream, same max tokens: every image once vs the text prompts (3 reps each)
img_runs = [stream(image_content(name), SINGLE_TOKENS) for name in IMAGES]
dr, ac = sum(r["drafted"] or 0 for r in img_runs), sum(r["accepted"] or 0 for r in img_runs)
res["single"] = {"image": {"summary": {"kind": "image", "median_tok_s": statistics.median(r["tok_s"] for r in img_runs),
                                       "runs_tok_s": [r["tok_s"] for r in img_runs], "tokens": [r["completion_tokens"] for r in img_runs],
                                       "mtp_acceptance": round(ac / dr, 4) if dr else None}, "runs": img_runs}}
print(json.dumps(res["single"]["image"]["summary"]), flush=True)
for kind, text in TEXT_PROMPTS:
    res["single"][kind] = single(kind, text)
for lv in res["levels"]:
    for r in lv["rows"]:
        r.pop("t0"), r.pop("t1")
for k in res["single"].values():
    for r in k["runs"]:
        r.pop("t0"), r.pop("t1")
json.dump(res, open(out, "w"), indent=1)
