#!/bin/bash
# Swap a dev Nemotron server in and the live one back, exactly (run on forge INSIDE the sparks.lock).
#   drswap.sh save              record every rank's live env (TF_/NCCL_/GLM*/CUDA_) + cmdline into ~/nemotron-tf/drafting/live/
#   drswap.sh dev BIN [ARGS] [ENV]  save (unless a dev run is up), stop, start BIN with the saved env and the canonical
#                               serve flags (context 307200, parallel 4, segments 4) + ARGS/ENV
#   drswap.sh restore           stop, start every rank from the saved live state as it was, prove :8100 200 (retries)
set -u
HOSTS=(forge anvil ember flame); S=/home/jun/nemotron-tf/drafting/live; LOGD=/home/jun/nemotron-tf/runs
M=/home/jun/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
PAT='^/home/jun/nemotron-tf/[^ ]*tensorfold-native[^ ]* serve'
log() { echo "$(date -Is) $*"; }
stop() {
  for h in "${HOSTS[@]}"; do ssh "$h" "pkill -TERM -f '$PAT'; for i in \$(seq 1 30); do pgrep -f '$PAT' >/dev/null || break; sleep 1; done; pkill -KILL -f '$PAT'; true" & done; wait
  sleep 8; log "stopped (avail GB: $(for h in "${HOSTS[@]}"; do ssh $h "free -g | awk 'NR==2{print \$7}'"; done | tr '\n' ' '))"; }
start() { # $1 BIN ('' = saved cmdline), $2 extra args, $3 extra env, $4 log prefix
  for h in "${HOSTS[@]}"; do ssh $h "pgrep -af '$PAT'" && { log "a server is still up on $h: not starting"; return 1; }; done
  for r in 0 1 2 3; do h=${HOSTS[$r]}
    envs=$(cat $S/r$r.env | tr '\n' ' '); cmd=$(cat $S/r$r.cmd)
    [ -n "$1" ] && cmd="$1 serve $M --host 0.0.0.0 --port 8100 --no-update-check --context 307200 --parallel 4 --segments 4"
    ssh "$h" "cd /home/jun/nemotron-tf && (env $envs $3 setsid nohup $cmd $2 > $LOGD/$4-r$r.log 2>&1 < /dev/null &)"
  done; log "started: ${cmd%% serve*} ... ${cmd#*--no-update-check} $2 [$3]"
  for i in $(seq 1 120); do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:8100/v1/models)" = 200 ] && { log "serving: forge:8100 200"; return 0; }
    [ $i -gt 6 ] && ! pgrep -f "$PAT" >/dev/null && { log "rank 0 exited"; tail -3 $LOGD/$4-r0.log; return 1; }
    sleep 5
  done; log "not up in 10 min"; return 1; }
case "${1:-}" in
save) mkdir -p $S; for r in 0 1 2 3; do h=${HOSTS[$r]}
    ssh $h "pid=\$(pgrep -f '$PAT' | head -1); [ -n \"\$pid\" ] || exit 1; tr '\0' '\n' < /proc/\$pid/environ | grep -E '^(TF_|NCCL|GLM|CUDA_|PYTORCH)'" > $S/r$r.env.new || { log "rank $r not running: keeping the earlier save"; exit 1; }
    ssh $h "pid=\$(pgrep -f '$PAT' | head -1); tr '\0' ' ' < /proc/\$pid/cmdline; bin=\$(tr '\0' '\n' < /proc/\$pid/cmdline | head -1); echo; sha256sum \$bin | cut -c1-8 >&2" > $S/r$r.cmd.new 2>$S/r$r.sha.new
    ssh $h "cp $LOGD/window-r$r.log $LOGD/window-r$r.log.pre-dr-$(date +%H%M%S) 2>/dev/null"; done
    for r in 0 1 2 3; do for x in env cmd sha; do mv $S/r$r.$x.new $S/r$r.$x; done; done
    log "saved live: $(cat $S/r0.cmd) shas $(cat $S/r*.sha | tr '\n' ' ')" ;;
dev) if pgrep -af "$PAT" | grep -q -e /dev/tensorfold-native.dr; then log "a dr build is serving: keeping the earlier save"; else $0 save || exit 1; fi
     stop; start "$2" "${3:-}" "${4:-}" dr ;;
restore) for t in 1 2 3; do stop; start "" "" "" window && break; log "restore attempt $t failed, retrying"; done
    for r in 0 1 2 3; do ssh ${HOSTS[$r]} "pid=\$(pgrep -f '$PAT' | head -1); bin=\$(tr '\0' '\n' < /proc/\$pid/cmdline | head -1); sha256sum \$bin | cut -c1-8"; done | tr '\n' ' '; echo "(live shas now; saved: $(cat $S/r*.sha | tr '\n' ' '))" ;;
*) sed -n 2,7p $0 ;;
esac
