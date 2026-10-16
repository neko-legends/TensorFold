#!/bin/bash
# The canonical Nemotron rebench against the persistent server (spark1:8100), one step a call, every step bracketed
# by the server's own counters so each reply is provably Nemotron's (results/rebench-nemotron-tp4-DATE/counters.jsonl).
#   rebench.sh fresh                 restart the server (nemotron-window.sh restart) and log the zero counters
#   rebench.sh qbench AREA           one qbench area (parity qbench: SPARK_BENCH, default ~/git/spark-bench-parity)
#   rebench.sh vision [IMAGES]       spark-vision cells (refine on, coords auto), a fresh reply cache (nothing reused)
#   rebench.sh speed                 window-speed.py (qbench's prompts, server timing, 1 and 4 streams)
set -u
MODEL=NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/sparks.sh"
URL=http://$R0:8100/v1
OUT=$(cd "$HERE/../../.." && pwd)/results/rebench-nemotron-tp4-${DATE:-20261015}
mkdir -p "$OUT"

counters() { # phase step: the server's counters, the replies rank 0 logged, and the binary that serves
  python3 - "$1" "$2" "$OUT/counters.jsonl" <<'EOF'
import json, os, subprocess, sys, time, urllib.request
R0 = os.environ["R0"]
phase, step, out = sys.argv[1:4]
m = urllib.request.urlopen(f"http://{R0}:8100/metrics", timeout=10).read().decode()
keep = ("tensorfold:request_latency_seconds_count", "tensorfold:prompt_tokens_total", "tensorfold:prompt_tokens_cached_total",
        "tensorfold:generation_tokens_total", "tensorfold:mtp_accepted_total", "tensorfold:mtp_drafted_total")
vals = {l.split()[0]: float(l.split()[1]) for l in m.splitlines() if l.split() and l.split()[0] in keep}
done = subprocess.run(["ssh", R0, "grep -a -c 'done chatcmpl' $HOME/nemotron-tf/runs/window-r0.log; "
                       "readlink -f /proc/$(pgrep -f 'tensorfold-native[.a-z0-9]* serve' | head -1)/exe | xargs sha256sum | cut -c1-12"],
                      capture_output=True, text=True).stdout.split()
model = json.loads(urllib.request.urlopen(f"http://{R0}:8100/v1/models", timeout=10).read())["data"][0]["id"]
row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "step": step, "phase": phase, "served_model": model,
       "rank0_done_replies": int(done[0]) if done else None, "binary_sha": done[1] if len(done) > 1 else None, **vals}
open(out, "a").write(json.dumps(row) + "\n")
print(json.dumps(row))
EOF
}

case "${1:-}" in
fresh)
  CONTEXT=307200 PARALLEL=4 "$HERE/nemotron-window.sh" restart && counters start fresh ;;
qbench)
  A=${2:?area}; counters before "qbench-$A"
  (cd "${SPARK_BENCH:-$HOME/git/spark-bench-parity}" && python3 bench/qbench.py --base-url $URL --model $MODEL --areas "$A" --workers 4 \
     --tag "rebench-nemotron-tp4-$A" --out "$OUT/qbench-$A.json") 2>&1 | tail -4
  counters after "qbench-$A" ;;
vision)
  counters before "vision-${2:-all}"
  SPARK_VISION_CACHE="$OUT/vision-cache" python3 "$HERE/window-vision.py" $URL $MODEL "$OUT/vision-${2:-all}.json" \
     "${2:-a1_poster,a2_dashboard,a3_shapes,a4_table_chart,a5_big4096,k_f_counts}" 2>&1 | tail -8
  counters after "vision-${2:-all}" ;;
speed)
  counters before speed
  python3 "$HERE/window-speed.py" "$R0" 8100 $MODEL "$OUT/speed.json" 2>&1 | tail -5
  counters after speed ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
