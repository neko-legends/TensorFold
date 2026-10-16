#!/bin/bash
# The comparison runs inside the swap window, against Nemotron on spark1:8100 (results stay in the worktree: the owner
# approves anything that goes to spark-bench). qbench is spark-bench's frozen suite (RERUN.md), thinking off.
#   window-bench.sh MODEL_ID [TAG]     (SPARK_BENCH: the spark-bench checkout whose qbench runs; default ~/git/spark-bench)
set -u
MODEL=${1:?served model id}; TAG=${2:-nemotron-tp4-$(date +%Y%m%d)}
OUT=$(cd "$(dirname "$0")/../../.." && pwd)/results; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)   # before the cd below
. "$HERE/sparks.sh"
URL=${URL:-http://$R0:8100/v1}
curl -sf -m 10 $URL/models >/dev/null || { echo "no server at $URL"; exit 1; }
cd "${SPARK_BENCH:-$HOME/git/spark-bench}" && python3 bench/qbench.py --base-url $URL --model "$MODEL" --areas all --workers 4 --tag "$TAG" --out "$OUT/qbench-$TAG.json"
python3 "$HERE/window-vision.py" $URL "$MODEL" "$OUT/vision-$TAG.json"
python3 "$HERE/window-speed.py" "$R0" 8100 "$MODEL" "$OUT/speed-$TAG.json"
echo "results: $OUT/qbench-$TAG.json $OUT/vision-$TAG.json"
