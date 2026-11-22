#!/bin/zsh
# Eval only: the 2-machine x 10-job model on 8 machines x N jobs, shown only 2 machines per slice (machine_select.py).
cd "$(dirname "$0")/.."/..
mkdir -p results/tournament_subset /tmp/subset_logs
for n in ${=NS:-10 20 30 40 50 60 70 80 90 100}; do
  [ -f results/tournament_subset/m2on8_n$n.npz ] && continue
  PYTHONPATH=src N_MACHINES=8 MODEL_MACHINES=2 N_JOBS=$n JOB_USAGE_MODE=random .venv/bin/python scripts/tournament/tournament_eval.py 5000 ${WORKERS:-6} results/tournament_subset/m2on8_n$n.npz 2>&1 | grep -v objc > /tmp/subset_logs/m2on8_n$n.log
done
echo ALL_DONE > /tmp/subset_logs/done
