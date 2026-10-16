#!/bin/bash
# The Nemotron TP-4 swap window (needs Jun's explicit go): DeepSeek's live TP-4 stops, Nemotron serves TP-4 natively
# on the four Sparks (rank 0 spark1, HTTP :8100), then DeepSeek comes back. Follows $HOME/nemotron-rollback
# and AGENTS.md's window rule (maintenance flag first; the keeper stands down for 60 min after a touch, so `keep`
# re-touches it; restore with serve4.sh start and prove :8000/health).
#   nemotron-window.sh up      flag, stop dsv41-tf4-r0..r3 (exact names), start Nemotron, wait for :8100
#   nemotron-window.sh keep    re-touch the flag (run every <= 30 min while the window lasts)
#   nemotron-window.sh status  both engines' state
#   nemotron-window.sh down    stop Nemotron, serve4.sh start, prove spark1:8000/health, remove the flag
#   nemotron-window.sh restart stop Nemotron and start it again (a new BIN or env); DeepSeek is not touched
# env: CONTEXT (307200), PARALLEL (16), CACHE_GIB (40: the prompt cache), DISK (the prompt cache's disk tier,
#      TF_NEMOTRON_DISK: $HOME/nemotron-tf/prompt-disk; DISK= empty: none), EXTRA (more env for every rank),
#      ARGS (more serve flags), BIN (the server binary), STOP_WAIT (120 s for a clean stop before KILL),
#      SEGMENTS (4: staggered prompt segments), NCCL_CHANNELS (2: NCCL's default spreads a prompt all-reduce over 64
#      channel blocks of a 48-SM GB10 and starves the other segments;
#      NCCL_CHANNELS= empty: no cap)
set -u
HOSTS=(spark1 spark2 spark3 spark4)             # ranks 0..3, as DeepSeek's dsv41-tf4-r0..r3
MASTER=10.10.10.1
M=$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
BIN=${BIN:-$HOME/nemotron-tf/bin/tensorfold-native}
LOGD=$HOME/nemotron-tf/runs
log() { echo "$(date -Is) $*"; }

nemo_up() {
  NCCL_CHANNELS=${NCCL_CHANNELS-2} # set but empty: NCCL's own channel count
  DISK=${DISK-$HOME/nemotron-tf/prompt-disk} # set but empty: no disk tier
  for r in 0 1 2 3; do
    h=${HOSTS[$r]}
    ssh "$h" "mkdir -p $LOGD && cd $HOME/nemotron-tf && (env TF_TP_WORLD=4 TF_TP_RANK=$r TF_TP_MASTER=$MASTER TF_TP_PORT=29650 \
      TF_COMM_BACKEND=roce TF_TP_ROCE_HCA=roceP2p1s0f1 NCCL_IB_HCA=roceP2p1s0f1 GLM53_TF_ROCE_FAST=1 TF_TP_ROCE_FALLBACK=nccl \
      TF_NCCL_LIB=$HOME/nemotron-tf/lib/libnccl.so.2 NCCL_SOCKET_IFNAME=enP2p1s0f1np1 ${NCCL_CHANNELS:+NCCL_MAX_NCHANNELS=$NCCL_CHANNELS} \
      ${DISK:+TF_NEMOTRON_DISK=$DISK} ${EXTRA:-} \
      setsid nohup $BIN serve $M --host 0.0.0.0 --port 8100 --no-update-check --context ${CONTEXT:-307200} --parallel ${PARALLEL:-16} --segments ${SEGMENTS:-4} --prompt-cache-gib ${CACHE_GIB:-40} ${ARGS:-} \
      > $LOGD/window-r$r.log 2>&1 < /dev/null &)" && log "nemotron rank $r started on $h"
  done
  for i in $(seq 1 120); do
    curl -sf -m 5 http://spark1:8100/v1/models >/dev/null && { log "nemotron serving at spark1:8100"; return 0; }
    [ $i -gt 4 ] && ! ssh spark1 "pgrep -f 'tensorfold[-.a-z0-9]* serve'" >/dev/null && { log "rank 0 exited:"; ssh spark1 "tail -3 $LOGD/window-r0.log"; return 1; }
    sleep 5
  done
  log "nemotron did not come up in 10 min; logs: $LOGD/window-r*.log"; return 1
}

nemo_down() {
  # rank 0 first: its clean stop writes the prompt cache's disk tier (TF_NEMOTRON_DISK), each rank its share, then
  # stops the others; whatever still runs after STOP_WAIT s (120) is killed
  local pat="tensorfold[-.a-z0-9]* serve" w=${STOP_WAIT:-120}
  ssh spark1 "pkill -TERM -f '$pat'; for i in \$(seq $w); do pgrep -f '$pat' >/dev/null || break; sleep 1; done"
  for h in "${HOSTS[@]}"; do ssh "$h" "for i in \$(seq $w); do pgrep -f '$pat' >/dev/null || break; sleep 1; done; pkill -KILL -f '$pat'; true" & done; wait
  log "nemotron stopped"
}

case "${1:-}" in
up)
  ssh spark1 'touch $HOME/tf4/maintenance' && log "maintenance flag set (keeper stands down 60 min; run 'keep' every 30 min)"
  for r in 0 1 2 3; do ssh "${HOSTS[$r]}" "docker stop dsv41-tf4-r$r" & done; wait
  log "DeepSeek ranks stopped (--rm containers: serve4.sh start brings them back)"
  for h in "${HOSTS[@]}"; do ssh "$h" "free -g | sed -n 2p | awk '{print \"$h avail GB:\", \$7}'"; done
  nemo_up ;;
restart) nemo_down; sleep 2; nemo_up ;;
keep) ssh spark1 'touch $HOME/tf4/maintenance' && log "maintenance flag re-touched" ;;
status)
  for r in 0 1 2 3; do h=${HOSTS[$r]}; echo "$h: dsv41-tf4-r$r=$(ssh $h "docker inspect -f '{{.State.Running}}' dsv41-tf4-r$r 2>/dev/null || echo absent") nemotron=$(ssh $h "pgrep -f 'tensorfold[-.a-z0-9]* serve' >/dev/null && echo up || echo down")"; done
  curl -s -m 5 -o /dev/null -w "spark1:8000 %{http_code}\n" http://spark1:8000/health; curl -s -m 5 -o /dev/null -w "spark1:8100 %{http_code}\n" http://spark1:8100/v1/models ;;
down)
  nemo_down
  ssh spark1 'cd $HOME/tf4 && ./serve4.sh start' && log "serve4.sh start returned"
  for i in $(seq 1 90); do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://spark1:8000/health)" = 200 ] && { log "DeepSeek back: spark1:8000/health 200"; ssh spark1 'rm -f $HOME/tf4/maintenance'; log "maintenance flag removed"; exit 0; }
    sleep 10
  done
  log "DeepSeek NOT healthy after 15 min: see $HOME/nemotron-rollback/README.md (restore-deepseek.sh --run)"; exit 1 ;;
*) sed -n 2,12p "$0"; exit 2 ;;
esac
