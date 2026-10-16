# 16 private SPARK_VISION roots under $OUT: 4 images x 4 copies; each copy's pixels differ from the original and from
# each other in the low red bit on a sparse 48-px grid, so no copy can reuse another's (or a baseline's) cached crops
import os, sys
from PIL import Image
SV, OUT = os.path.expanduser(os.environ.get("SPARK_VISION", "~/spark-vision")), sys.argv[1]
for img in ["a1_poster", "a3_shapes", "a4_table_chart", "k_f_counts"]:
    for c in range(4):
        root = f"{OUT}/{img}-{c}"; os.makedirs(root + "/bench", exist_ok=True)
        if not os.path.exists(f"{root}/score.py"): os.symlink(f"{SV}/score.py", f"{root}/score.py")
        g = f"{root}/bench/{img}.gt.json"
        if not os.path.exists(g): os.symlink(f"{SV}/bench/{img}.gt.json", g)
        im = Image.open(f"{SV}/bench/{img}.png").convert("RGB"); px = im.load()
        off = 7 + 13 * c
        for y in range(off, im.height, 48):
            for x in range(off, im.width, 48):
                r, gg, b = px[x, y]; px[x, y] = (r ^ 1, gg, b)
        im.save(f"{root}/bench/{img}.png")
print("ok")
