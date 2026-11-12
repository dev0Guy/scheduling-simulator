#!/bin/zsh
# Zoom + job tournament on machines 3,4 x jobs 10..50 (5000 episodes per cell). 1 and 2 machines need no zoom:
# their cells are the plain tournament results of scripts/tournament (native 1- and 2-machine models).
cd "$(dirname "$0")/../.."
mkdir -p /tmp/zt
for n in 10 20 30 40 50; do
  for m in 3 4; do
    [ -f /tmp/zt/m${m}_n${n}.npz ] && continue
    N_MACHINES=$m N_JOBS=$n JOB_USAGE_MODE=random .venv/bin/python scripts/machine_ae/zoom_tournament_eval.py 5000 8 /tmp/zt/m${m}_n${n}.npz 2>&1 | grep -v -e objc -e pygame > /tmp/zt/m${m}_n${n}.log
  done
done
