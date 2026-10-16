#!/bin/bash
# bg.sh NAME 'commands': run under the Sparks lock, detached from the ssh session; log in sm/results/NAME.log
name=$1; shift
cd /home/jun/nemotron-tf/sm && setsid nohup flock -w 3600 /home/jun/nemotron-tf/sparks.lock bash -c "$*; echo BGDONE" > results/$name.log 2>&1 < /dev/null &
echo started $name
