#!/bin/bash
# nemotron leftovers (2026-10-16): pinned A/B (cache 0, P16) of b846 vs 4803a901, then 4803a901 live, disk tier reap/lock/read-back.
set -u
cd /home/jun/nemotron-tf
O=/home/jun/nemotron-tf/lo
W=/home/jun/nemotron-tf/sm/nemotron-window.sh
T=/home/jun/nemotron-tf/sm/tree/engine/zig-nemotron/tools/nemotron
G=$T/greedy-check.py
S=$T/draft-speed.py
U=http://localhost:8100
NEW=/home/jun/nemotron-tf/bin/tensorfold-native.lo-4803a901
H=(forge anvil ember flame)
step() { echo "== $(date -Is) $*"; }
logs() { for r in 0 1 2 3; do ssh ${H[$r]} "cat /home/jun/nemotron-tf/runs/window-r$r.log" > $O/$1-r$r.log; done; }
restore() { step "restore: live config, default BIN"; $W restart; echo "restore-exit=$?"; curl -s -o /dev/null -w "forge:8100 %{http_code}\n" $U/v1/models; echo BGDONE; exit 1; }
up() { curl -s -m 5 -o /dev/null -w "forge:8100 %{http_code}\n" $U/v1/models; }

step "R0 b846 cache 0 P16"; CACHE_GIB=0 $W restart || restore; up
python3 $G $U x $O/r0-greedy.json; echo "r0-greedy-exit=$?"
python3 $S $U $O/r0-speed.json 3; echo "r0-speed-exit=$?"
logs r0

step "R1 4803a901 cache 0 P16"; BIN=$NEW CACHE_GIB=0 $W restart || restore; up
python3 $G $U x $O/r1-greedy.json $O/r0-greedy.json; echo "r1-greedy-vs-r0-exit=$?"
python3 $S $U $O/r1-speed.json 3 $O/r0-speed.json; echo "r1-speed-vs-r0-exit=$?"
logs r1

step "install 4803a901 as bin/tensorfold-native (copy to a new name, rename)"
for h in "${H[@]}"; do ssh $h 'cd /home/jun/nemotron-tf/bin && cp -p tensorfold-native.lo-4803a901 .tensorfold-native.new && mv -f .tensorfold-native.new tensorfold-native && echo "$(hostname) $(sha256sum tensorfold-native | cut -c1-16)"'; done

step "R2 live config (P16, 40 GiB, disk, segments 4, NCCL 2)"; $W restart || restore; up
python3 $G $U x $O/r2-greedy.json; echo "r2-greedy-exit=$?"
python3 $S $U $O/r2-speed.json 3; echo "r2-speed-exit=$?"
for h in "${H[@]}"; do ssh $h 'echo "$(hostname): $(ls -t /home/jun/nemotron-tf/prompt-disk | tr "\n" " ")"'; done
step "plant orphans (forge rank 0, anvil rank 1) and test the folder lock"
for h in forge anvil; do ssh $h 'cd /home/jun/nemotron-tf/prompt-disk && d=$(ls -td */ | head -1) && d=${d%/} && echo junk > $d/0000000000000001.bin && echo junk > $d/0000000000000002.bin.part && echo keep > $d/not-a-state.txt && echo "$(hostname) planted in $d: $(ls $d | tr "\n" " ")"; flock -n $d/lock true; echo "$(hostname) flock -n on the live folder exit=$? (1: held by the server)"'; done
logs r2

step "R3 live config again: disk read-back, reap"; $W restart || restore; up
python3 $G $U x $O/r3-greedy.json $O/r2-greedy.json; echo "r3-greedy-vs-r2-exit=$?"
python3 $G $U x $O/r3b-greedy.json $O/r3-greedy.json; echo "r3b-greedy-vs-r3-exit=$?"
for h in forge anvil; do ssh $h 'cd /home/jun/nemotron-tf/prompt-disk && d=$(ls -td */ | head -1) && echo "$(hostname) after reap $d: $(ls $d | tr "\n" " ")"'; done
logs r3
python3 $S $U $O/r3-speed.json 3; echo "r3-speed-exit=$?"
for r in 0 1 2 3; do ssh ${H[$r]} 'echo "$(hostname) $(sha256sum /home/jun/nemotron-tf/bin/tensorfold-native | cut -c1-16) $(tr "\0" " " < /proc/$(pgrep -f "tensorfold[-.a-z0-9]* serve")/cmdline | cut -c1-260)"'; done
up
echo BGDONE
