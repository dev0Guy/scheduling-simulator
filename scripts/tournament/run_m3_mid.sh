#!/bin/zsh
# Old 3-machine x 10-job model (perlec9c, original workload), tournament over 80/90/100 jobs, 1000 episodes each.
cd "$(dirname "$0")/../.."
export PYTHONPATH=src JOB_USAGE_MODE=random
for n in ${=NS:-80 90 100}; do
  [ -f results/tournament_m3_mid/m3_n$n.npz ] && continue
  N_MACHINES=3 N_JOBS=$n .venv/bin/python scripts/tournament/tournament_eval.py ${EPISODES:-1000} ${WORKERS:-3} results/tournament_m3_mid/m3_n$n.npz 2>&1 | grep -v objc > /tmp/m3mid_logs/m3_n$n.log
done
echo ALL_DONE > /tmp/m3mid_logs/done_${NS// /_}
