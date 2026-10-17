#!/usr/bin/env bash
# Start, stop or check DeepSeek-V4.1-Flash-1017 on four Sparks (Zig engine, TP=4). Run on rank 0's node.
#   deploy/dsv41-1017/serve-1017.sh start | stop | status
# Settings: deploy/dsv41-1017/sparks.env (WORKERS, REPO_DIR, ENV_FILE, PORT, ALIAS, READY_TIMEOUT, LOGDIR).
#
# start: waits for the GPUs to be free, removes any old dsv41-tf4-r* containers, launches the followers (ranks 3, 2, 1)
# over ssh with `CONFIG=$ENV_FILE scripts/run-zig.sh <rank>` detached on their nodes, then rank 0 here (the same
# run-zig.sh docker command, plus `--alias $ALIAS` when set), and waits for /health. The first start writes the
# prepared rank folders (405 s measured); later starts load them in ~35 s. Boot logs: $LOGDIR/boot-r<rank>.log.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SPARKS_ENV:-$here/sparks.env}"
read -r -a W <<< "$WORKERS"
NAME=dsv41-tf4       # = ZIG_NAME in zig-1017.env.example
log() { echo "[serve-1017] $(date +%T) $*"; }
on() { # on RANK CMD: run on that rank's node
    local r=$1; shift
    if [[ "$r" == 0 ]]; then bash -c "$*"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "${W[$((r-1))]}" "$*"; fi
}
stop_all() { local r; for r in 0 1 2 3; do on "$r" "docker rm -f $NAME-r$r >/dev/null 2>&1 || true" & done; wait; }
gpu_busy() { local r; for r in 0 1 2 3; do on "$r" "nvidia-smi --query-compute-apps=pid --format=csv,noheader" 2>/dev/null; done | grep -q '[0-9]'; }
running() { on "$1" "docker inspect -f '{{.State.Running}}' $NAME-r$1 2>/dev/null" || echo absent; }

cmd_start() {
    local r dt=0
    stop_all
    while gpu_busy; do   # a just-stopped engine takes a few seconds to release the GPU
        (( dt < 90 )) || { log "a CUDA process is still running on a node after 90 s; stop it first"; exit 1; }
        sleep 3; dt=$((dt+3))
    done
    for r in 0 1 2 3; do on "$r" "mkdir -p '$LOGDIR'"; done
    for r in 3 2 1; do
        on "$r" "cd '$REPO_DIR' && setsid nohup env CONFIG='$ENV_FILE' scripts/run-zig.sh $r </dev/null >'$LOGDIR/boot-r$r.log' 2>&1 &"
        log "rank $r launched on ${W[$((r-1))]}"
    done
    # rank 0: run-zig.sh's own command (--dry-run prints it) plus the alias; under env -i so no TF_DSV41_* of this
    # shell leaks into the container (run-zig.sh passes every TF_DSV41_* in its environment through to docker)
    local cmd
    cmd=$(cd "$REPO_DIR" && env -i HOME="$HOME" PATH="$PATH" CONFIG="$ENV_FILE" scripts/run-zig.sh 0 --dry-run)
    [[ -n "${ALIAS:-}" ]] && cmd+=" --alias $(printf '%q' "$ALIAS")"
    (cd "$REPO_DIR" && setsid nohup env -i HOME="$HOME" PATH="$PATH" bash -c "$cmd" </dev/null >"$LOGDIR/boot-r0.log" 2>&1 &)
    log "rank 0 launched; waiting for :$PORT/health (up to ${READY_TIMEOUT}s)"
    local t0; t0=$(date +%s)
    sleep 15   # let docker create the containers before the first inspect
    until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
        for r in 0 1 2 3; do
            [[ "$(running "$r")" == true ]] || { log "rank $r exited"; on "$r" "tail -60 '$LOGDIR/boot-r$r.log'"; exit 1; }
        done
        (( $(date +%s) - t0 < READY_TIMEOUT )) || { log "not ready after ${READY_TIMEOUT}s"; exit 1; }
        sleep 3
    done
    log "ready after $(( $(date +%s) - t0 )) s"
    grep -E 'serving|loaded in' "$LOGDIR/boot-r0.log" | tail -2 || true
}

case "${1:-}" in
    start) cmd_start ;;
    stop) stop_all; log "stopped" ;;
    status) for r in 0 1 2 3; do echo "rank $r: $(running "$r")"; done; curl -s -m 10 "http://127.0.0.1:$PORT/health" || true; echo ;;
    *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
