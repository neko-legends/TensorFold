#!/bin/bash
# one measuring hold: hold.sh TAG "STEPS" with env for the restart (PARALLEL, ARGS, BIN, SEGMENTS, ...); STEPS from
# restart greedy conc warm8 warm16 ladder; results in ~/nemotron-tf/sm/results/TAG-*.json
set -u
tag=$1; steps=$2; S=/home/jun/nemotron-tf/sm; R=$S/results; M=NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
T=$S/tree/engine/zig-nemotron/tools/nemotron
echo "== $(date -Is) hold $tag: $steps"; $S/mem.sh
for st in $steps; do
  echo "-- $(date -Is) $st"
  case $st in
    restart) CONTEXT=${CONTEXT:-307200} PARALLEL=${PARALLEL:-4} $S/nemotron-window.sh restart || exit 1
             grep -E "CUDA sm|prompt cache:.*GiB" /home/jun/nemotron-tf/runs/window-r0.log | head -3; sleep 5; $S/mem.sh ;;
    greedy)  python3 $T/greedy-check.py http://forge:8100 $M $R/$tag-greedy.json ${GBASE:-} | tail -8 ;;
    conc)    python3 $S/conc-bench.py forge 8100 $M $R/$tag-conc.json ${LEVELS:-1,4,8,16} ${REPS:-2} | grep -v '^{' ; python3 -c "
import json; d=json.load(open('$R/$tag-conc.json'))
for r in d['levels']: print(r['streams'], r['kind'], r['rep'], 'agg', r['aggregate_tok_s'], 'per', r['per_stream_tok_s_median'], 'min', r['per_stream_tok_s_min'], 'ttft', r['ttft_median_s'], r['ttft_max_s'], 'short', r['short'])" ;;
    warm8)   python3 $S/warm-prefix.py http://forge:8100 $M $R/$tag-warm8x30k.json 8 30000 3 | tail -1 ;;
    warm16)  python3 $S/warm-prefix.py http://forge:8100 $M $R/$tag-warm16x60k.json 16 60000 3 | tail -1 ;;
    warmfrom:*) src=${st#warmfrom:}; python3 $S/warm-prefix.py http://forge:8100 $M $R/$tag-warmfrom.json 0 0 2 --phase warm --from $R/$src ${ASK:+--ask $ASK} | tail -1
             python3 $S/samereplies.py $R/$src $R/$tag-warmfrom.json ;;
    warm:*)  IFS=: read -r _ ag pf tu <<< "$st"; python3 $S/warm-prefix.py http://forge:8100 $M $R/$tag-warm${ag}x${pf}.json $ag $pf $tu ${NONCE:+--nonce $NONCE} | tail -1 ;;
    same:*)  python3 $S/samereplies.py $R/${st#same:} $R/$tag-warm${LASTWARM:-}.json ;;
    ladder)  python3 $T/prefill-ladder.py http://forge:8100 $M $R/$tag-ladder.json ${SIZES:-32768,131072,200000} ${LREPS:-2} $tag | tail -8 ;;
    *) echo "unknown step $st" ;;
  esac
  grep "prompt cache:" /home/jun/nemotron-tf/runs/window-r0.log | tail -1
done
$S/mem.sh; echo "== $(date -Is) hold $tag done"
