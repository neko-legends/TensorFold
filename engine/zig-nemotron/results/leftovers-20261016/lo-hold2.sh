#!/bin/bash
# leftovers hold 2: a clean stop's disk-tier spill, observed (logs kept before the next start), then the live config again.
set -u
cd /home/jun/nemotron-tf
O=/home/jun/nemotron-tf/lo
W=/home/jun/nemotron-tf/sm/nemotron-window.sh
G=/home/jun/nemotron-tf/sm/tree/engine/zig-nemotron/tools/nemotron/greedy-check.py
U=http://localhost:8100
H=(forge anvil ember flame)
pat="tensorfold[-.a-z0-9]* serve"
step() { echo "== $(date -Is) $*"; }
lk() { for h in forge anvil; do ssh $h 'd=$(ls -td /home/jun/nemotron-tf/prompt-disk/*/ | head -1); flock -n ${d}lock true; echo "$(hostname) $1 flock -n ${d}lock exit=$? (1: held)"' _ "$1"; done; }
step "stop (as nemo_down: rank 0 first, TERM, 120 s)"
t0=$(date +%s)
ssh forge "pkill -TERM -f '$pat'; for i in \$(seq 120); do pgrep -f '$pat' >/dev/null || break; sleep 1; done"
for h in "${H[@]}"; do ssh "$h" "for i in \$(seq 120); do pgrep -f '$pat' >/dev/null || break; sleep 1; done; pkill -KILL -f '$pat'; true" & done; wait
echo "stopped in $(( $(date +%s) - t0 )) s"
for r in 0 1 2 3; do ssh ${H[$r]} "cat /home/jun/nemotron-tf/runs/window-r$r.log" > $O/s3-r$r.log; done
for h in "${H[@]}"; do ssh $h 'for d in /home/jun/nemotron-tf/prompt-disk/*/; do echo "$(hostname) $d $(ls $d | tr "\n" " ")"; done'; done
step "start (live config)"; $W restart || { $W restart; }
lk after-start
python3 $G $U x $O/r4-greedy.json $O/r3-greedy.json; echo "r4-greedy-vs-r3-exit=$?"
lk after-greedy
for r in 0 1 2 3; do ssh ${H[$r]} "cat /home/jun/nemotron-tf/runs/window-r$r.log" > $O/r4-r$r.log; done
curl -s -m 5 -o /dev/null -w "forge:8100 %{http_code}\n" $U/v1/models
echo BGDONE
