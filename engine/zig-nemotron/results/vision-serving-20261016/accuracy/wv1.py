# window-vision.py with spark-vision's per-read concurrency pinned to 1: one read = one stream
import os, runpy, sys
wv, url, model, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.environ["SPARK_VISION_URL"] = url
os.environ["SPARK_VISION_MODEL"] = model
root = os.path.expanduser(os.environ["SPARK_VISION"])
sys.path.insert(0, root); os.chdir(root)
import score
score.sv.MAX_CONCURRENCY = 1
sys.argv = [wv] + sys.argv[2:]
runpy.run_path(wv, run_name="__main__")
