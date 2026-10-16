#!/usr/bin/env bash
# Vision accuracy under load: window-vision's tiled read (no zoom re-read) of four bench images, one stream, then
# 16 reads at once (make-copies.py's 16 pixel-distinct copies), one stream each (wv1.py pins spark-vision to 1).
#   URL=http://<rank0>:8100/v1 WORK=/tmp/wv16 ./run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WV=$HERE/../../../tools/nemotron/swap/window-vision.py
O=$WORK/out; mkdir -p $O; M=NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4; U=$URL
PY=${PY:-python3}
$PY $HERE/make-copies.py $WORK
echo "single start $(date +%T)"
SPARK_VISION_CACHE=$WORK/cache-single $PY $HERE/wv1.py $WV $U $M $O/single.json a1_poster,a3_shapes,a4_table_chart,k_f_counts --no-refine > $O/single.log 2>&1
echo "single done $(date +%T)"
for d in $WORK/*-[0-3]; do n=$(basename $d); img=${n%-*}
  SPARK_VISION=$d SPARK_VISION_CACHE=$WORK/cache-$n $PY $HERE/wv1.py $WV $U $M $O/c16-$n.json $img --no-refine > $O/c16-$n.log 2>&1 &
done
wait; echo "c16 done $(date +%T)"
