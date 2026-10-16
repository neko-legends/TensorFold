#!/bin/bash
# leftovers hold 3: c3ede919 (clean stop past an idle keep-alive) pinned vs b846 (cache 0), then live, stop spill, read-back.
set -u
cd $HOME/nemotron-tf
O=$HOME/nemotron-tf/lo
W=$HOME/nemotron-tf/sm/nemotron-window.sh
T=$HOME/nemotron-tf/sm/tree/engine/zig-nemotron/tools/nemotron
G=$T/greedy-check.py
S=$T/draft-speed.py
U=http://localhost:8100
NEW=$HOME/nemotron-tf/bin/tensorfold-native.lo-c3ede919
H=(spark1 spark2 spark3 spark4)
pat="tensorfold[-.a-z0-9]* serve"
step() { echo "== $(date -Is) $*"; }
logs() { for r in 0 1 2 3; do ssh ${H[$r]} "cat $HOME/nemotron-tf/runs/window-r$r.log" > $O/$1-r$r.log; done; }
up() { curl -s -m 5 -o /dev/null -w "spark1:8100 %{http_code}\n" $U/v1/models; }
lk() { for h in spark1 spark2; do ssh $h 'd=$(ls -td $HOME/nemotron-tf/prompt-disk/*/ | head -1); flock -n ${d}lock true; rc=$?; echo "$(hostname) flock -n ${d}lock exit=$rc (1: held by the server)"'; done; }
restore() { step "restore: live config, default BIN"; $W restart; echo "restore-exit=$?"; up; echo BGDONE; exit 1; }
stop() {
  t0=$(date +%s)
  ssh spark1 "pkill -TERM -f '$pat'; for i in \$(seq 120); do pgrep -f '$pat' >/dev/null || break; sleep 1; done"
  for h in "${H[@]}"; do ssh "$h" "for i in \$(seq 120); do pgrep -f '$pat' >/dev/null || break; sleep 1; done; pkill -KILL -f '$pat'; true" & done; wait
  echo "stopped in $(( $(date +%s) - t0 )) s"
}

step "R5a c3ede919 cache 0 P16 (pinned vs R0 b846)"; BIN=$NEW CACHE_GIB=0 $W restart || restore; up
python3 $G $U x $O/r5a-greedy.json $O/r0-greedy.json; echo "r5a-greedy-vs-r0-exit=$?"
python3 $S $U $O/r5a-speed.json 3 $O/r0-speed.json; echo "r5a-speed-vs-r0-exit=$?"

step "install c3ede919 as bin/tensorfold-native (copy to a new name, rename)"
for h in "${H[@]}"; do ssh $h 'cd $HOME/nemotron-tf/bin && cp -p tensorfold-native.lo-c3ede919 .tensorfold-native.new && mv -f .tensorfold-native.new tensorfold-native && echo "$(hostname) $(sha256sum tensorfold-native | cut -c1-16)"'; done

step "R5 live config"; $W restart || restore; up
python3 $G $U x $O/r5-greedy.json; echo "r5-greedy-exit=$?"
lk
step "stop R5 (the monitor's keep-alive is connected) and keep the logs"
stop; logs s5
grep -h -E "written at stop|disk tier" $O/s5-r0.log | tail -3
for h in "${H[@]}"; do ssh $h 'for d in $HOME/nemotron-tf/prompt-disk/*/; do echo "$(hostname) $d $(ls $d | tr "\n" " ")"; done'; done

step "R6 live config: read back from disk"; $W restart || restore; up
python3 $G $U x $O/r6-greedy.json $O/r5-greedy.json; echo "r6-greedy-vs-r5-exit=$?"
python3 $S $U $O/r6-speed.json 3; echo "r6-speed-exit=$?"
lk
logs r6
grep -h -E "disk tier at|learned tokens from disk|PeerDisk|written" $O/r6-r0.log | head
grep -h -iE "disk|follower" $O/r6-r1.log $O/r6-r2.log $O/r6-r3.log | head
for r in 0 1 2 3; do ssh ${H[$r]} 'echo "$(hostname) $(sha256sum $HOME/nemotron-tf/bin/tensorfold-native | cut -c1-16) $(tr "\0" " " < /proc/$(pgrep -f "tensorfold[-.a-z0-9]* serve")/cmdline | cut -c40-260)"'; done
ssh spark1 'tr "\0" "\n" < /proc/$(pgrep -f "tensorfold[-.a-z0-9]* serve")/environ | grep -E "NCCL_MAX|TF_NEMOTRON_DISK"'
up
echo BGDONE
