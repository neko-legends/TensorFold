#!/bin/bash
# available / used GB on the four Sparks, one line
for h in spark1 spark2 spark3 spark4; do ssh $h "free -m | awk -v h=$h 'NR==2{printf \"%s avail %.1f used %.1f  \", h, \$7/1024, \$3/1024}'"; done; echo
