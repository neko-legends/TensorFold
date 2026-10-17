#!/usr/bin/env bash
# Pack the 1017 Engram tables (official checkpoint shards 47/48) into the per-rank TP=4 shards the engine reads,
# engram-l{1,14}-r<rank>of4.bin in ENGRAM_DIR on every node. Rank 0 is packed in place; ranks 1-3 are packed into a stage
# dir on rank 0, `--check`ed against the source (headers, sizes, sampled rows), copied to their node and sha256-checked
# there. CPU and disk only (~48 GB written a rank); the GPUs are not touched. Run on rank 0 after `download.sh engram`.
#
# The 1017 tables differ from the earlier V4.1's from the first row: do not reuse an earlier pack's shards.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SPARKS_ENV:-$here/sparks.env}"
SSH="ssh -o BatchMode=yes"
read -r -a W <<< "$WORKERS"
STAGE="${STAGE:-$ENGRAM_DIR.stage}"
pk() { # pk RANK OUTDIR --out|--check
    if [[ -n "$PY_IMAGE" ]]; then
        docker run --rm --name "dsv41-1017-engram-r$1" --entrypoint python3 -v "$REPO_DIR/scripts:/s:ro" \
            -v "$FP8_DIR:/src:ro" -v "$MODEL_DIR/config.json:/cfg/config.json:ro" -v "$2:/o" "$PY_IMAGE" \
            /s/pack_engram.py --src /src --config /cfg/config.json "$3" /o --world 4 --rank "$1"
    else
        python3 "$REPO_DIR/scripts/pack_engram.py" --src "$FP8_DIR" --config "$MODEL_DIR/config.json" "$3" "$2" \
            --world 4 --rank "$1"
    fi
}
mkdir -p "$ENGRAM_DIR" "$STAGE"
echo "[$(date -Is)] rank 0"; pk 0 "$ENGRAM_DIR" --out; pk 0 "$ENGRAM_DIR" --check
for r in 1 2 3; do
    h=${W[$((r-1))]}
    echo "[$(date -Is)] rank $r -> $h"
    find "$STAGE" -maxdepth 1 -name 'engram-*.bin' -delete
    pk "$r" "$STAGE" --out; pk "$r" "$STAGE" --check
    $SSH "$h" "mkdir -p '$ENGRAM_DIR'"
    rsync -a -e "$SSH" "$STAGE/" "$h:$ENGRAM_DIR/"
    (cd "$STAGE" && sha256sum engram-*.bin) | $SSH "$h" "cd '$ENGRAM_DIR' && sha256sum -c && du -sh ."
done
find "$STAGE" -maxdepth 1 -name 'engram-*.bin' -delete; rmdir "$STAGE"
echo "[$(date -Is)] engram pack DONE"
