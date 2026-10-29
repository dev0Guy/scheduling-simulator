#!/bin/zsh
# Tournament evaluation of the 1-machine x 10-job model on the whole machines x jobs grid (5000 episodes per cell).
# Results: /tmp/tournament/m{M}_n{N}.npz and .log
cd "$(dirname "$0")/../.."
mkdir -p /tmp/tournament
for n in 10 20 30 40 50; do
  for m in 1 2 3 4; do
    [ -f /tmp/tournament/m${m}_n${n}.npz ] && continue
    N_MACHINES=$m N_JOBS=$n JOB_USAGE_MODE=random .venv/bin/python scripts/tournament/tournament_eval.py 5000 8 /tmp/tournament/m${m}_n${n}.npz 2>&1 | grep -v objc > /tmp/tournament/m${m}_n${n}.log
  done
done
