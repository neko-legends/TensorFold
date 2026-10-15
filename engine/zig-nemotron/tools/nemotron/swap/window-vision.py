#!/usr/bin/env python3
"""The vision cells of the comparison: spark-vision.py's structured `read` on code-rendered bench images (exact ground
truth), scored by the bench's own score.py, against any OpenAI-compatible endpoint. The same script runs against
DeepSeek and Nemotron (only the URL, model id and cache dir change), so the cells compare.

  window-vision.py URL MODEL OUT.json [IMAGES=a1_poster,a3_shapes,a4_table_chart,k_f_counts]
"""
import json, os, sys, time
url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
images = (sys.argv[4] if len(sys.argv) > 4 else "a1_poster,a3_shapes,a4_table_chart,k_f_counts").split(",")
os.environ["SPARK_VISION_URL"] = url
os.environ["SPARK_VISION_MODEL"] = model
os.environ["SPARK_VISION_CACHE"] = os.path.join(os.path.dirname(os.path.abspath(out)), "vision-cache-" + model.replace("/", "_"))
bench = "/home/jun/nest/tmp/spark-vision"
sys.path.insert(0, bench)
os.chdir(bench)
from score import sv, score_text, score_shapes  # noqa: E402  (score.py loads spark-vision.py with the env above)
rows = []
for b in images:
    gt = json.load(open(f"bench/{b}.gt.json"))
    for k in sv.STATS:
        sv.STATS[k] = 0
    t = time.time()
    try:
        res = sv.read(f"bench/{b}.png")
    except Exception as e:  # an engine failure is a cell, not a crash
        rows.append({"bench": b, "error": repr(e)})
        continue
    row = {"bench": b, "secs": round(time.time() - t, 1), "calls": res.get("stats", {}).get("calls")}
    row.update({"t_" + k: v for k, v in score_text(gt, res).items()})
    if gt.get("shapes"):
        row.update({"s_" + k: v for k, v in score_shapes(gt, res).items()})
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump({"url": url, "model": model, "date": time.strftime("%Y-%m-%d %H:%M"), "rows": rows}, open(out, "w"), indent=1)
