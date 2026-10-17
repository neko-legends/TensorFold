#!/usr/bin/env bash
# Copy the checked EXL3 pack from rank 0 to the three workers (same path) and prove each copy: a sha256 manifest of every
# file is written on rank 0, then `sha256sum -c` runs on each worker. Run on rank 0 after `download.sh exl3` passed.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SPARKS_ENV:-$here/sparks.env}"
SSH="ssh -o BatchMode=yes"
MAN="$MODEL_DIR/.sha256-manifest"
( cd "$MODEL_DIR" && find . -type f -not -path './.cache/*' -not -name .sha256-manifest -print0 | sort -z \
    | xargs -0 -P8 sha256sum > "$MAN" )
echo "manifest: $(wc -l < "$MAN") files"
for h in $WORKERS; do    # in parallel: each worker on its own link pair
    ( t0=$(date +%s); $SSH "$h" "mkdir -p '$MODEL_DIR'"
      rsync -a -e "$SSH" --exclude .cache "$MODEL_DIR/" "$h:$MODEL_DIR/"
      echo "rsync -> $h: $(( $(date +%s) - t0 )) s" ) &
done; wait
fail=0
for h in $WORKERS; do
    bad=$($SSH "$h" "cd '$MODEL_DIR' && sha256sum -c .sha256-manifest 2>&1 | grep -cv ': OK\$'" || true)
    echo "verify $h: mismatches=$bad"; [[ "$bad" == 0 ]] || fail=1
done
exit $fail
