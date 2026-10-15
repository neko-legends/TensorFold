#!/usr/bin/env python3
"""The vision cells of the comparison: spark-vision.py's structured `read` on code-rendered bench images (exact ground
truth), scored by the bench's own score.py, against any OpenAI-compatible endpoint. The same script runs against
DeepSeek and Nemotron (only the URL, model id and cache dir change; the images, modes, tile plans and prompts are
spark-vision.py's own), so the cells compare. Each row records the tile plan and the bbox frame the read used.

  window-vision.py URL MODEL OUT.json [IMAGES=a1_poster,a3_shapes,a4_table_chart,k_f_counts] [--cache-only] [--no-refine]

The cache dir is OUT's dir/vision-cache-MODEL unless SPARK_VISION_CACHE is set.
--cache-only re-scores from the run's cache dir (temp-0 replies keyed by image crop + prompt + model) without an
engine: a reply the cache lacks is an error cell. Use it after a harness fix (e.g. the bbox frame) to re-score a
window's replies; the read is the same code path, so a full cache hit is the same read. --no-refine skips
spark-vision's per-text zoom re-read (its crops follow the boxes, so a frame fix changes them and misses the cache).
"""
import json, os, sys, time
args = [a for a in sys.argv[1:] if a not in ("--cache-only", "--no-refine")]
cache_only = "--cache-only" in sys.argv
refine = "--no-refine" not in sys.argv
url, model, out = args[0], args[1], args[2]
images = (args[3] if len(args) > 3 else "a1_poster,a3_shapes,a4_table_chart,k_f_counts").split(",")
os.environ["SPARK_VISION_URL"] = url
os.environ["SPARK_VISION_MODEL"] = model
os.environ.setdefault("SPARK_VISION_CACHE", os.path.join(os.path.dirname(os.path.abspath(out)), "vision-cache-" + model.replace("/", "_")))
bench = "/home/jun/nest/tmp/spark-vision"
sys.path.insert(0, bench)
os.chdir(bench)
from score import sv, score_text, score_shapes  # noqa: E402  (score.py loads spark-vision.py with the env above)
if cache_only:
    import hashlib

    def cached_chat(png_bytes, prompt, max_tokens=4000, temperature=0.0, mime="image/png"):
        key = hashlib.sha1(png_bytes + prompt.encode() + str(max_tokens).encode() + sv.MODEL.encode()).hexdigest()
        cp = os.path.join(os.environ["SPARK_VISION_CACHE"], key + ".txt")
        if not os.path.exists(cp):
            raise sv.EngineDown(f"--cache-only: no cached reply {key}")
        sv.STATS["calls"] += 1
        return open(cp).read()
    sv.engine_check = lambda timeout=8: None
    sv.chat = cached_chat
rows = []
for b in images:
    gt = json.load(open(f"bench/{b}.gt.json"))
    for k in sv.STATS:
        sv.STATS[k] = 0
    t = time.time()
    try:
        res = sv.read(f"bench/{b}.png", refine=refine)
    except Exception as e:  # an engine failure is a cell, not a crash
        rows.append({"bench": b, "error": repr(e)})
        print(json.dumps(rows[-1]), flush=True)
        continue
    row = {"bench": b, "secs": round(time.time() - t, 1), "calls": res.get("stats", {}).get("calls"),
           "mode": res.get("mode"), "grid": res.get("grid"), "overlap": res.get("overlap"),
           "coords": res.get("engine", {}).get("coords"), "refine": refine}
    row.update({"t_" + k: v for k, v in score_text(gt, res).items()})
    if gt.get("shapes"):
        row.update({"s_" + k: v for k, v in score_shapes(gt, res).items()})
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump({"url": url, "model": model, "date": time.strftime("%Y-%m-%d %H:%M"), "cache_only": cache_only, "refine": refine, "rows": rows},
          open(out, "w"), indent=1)
