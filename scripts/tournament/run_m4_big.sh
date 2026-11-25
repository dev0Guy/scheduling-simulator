#!/bin/zsh
# Old 4-machine x 10-job model (t3lhn7a9, original workload) in a tournament over 1000 jobs, 100 episodes.
# Results: results/tournament_m4_big/m4_n1000.npz
cd "$(dirname "$0")/../.."
mkdir -p results/tournament_m4_big /tmp/m4big_logs
export PYTHONPATH=src JOB_USAGE_MODE=random
for n in ${=NS:-1000}; do
  [ -f results/tournament_m4_big/m4_n$n.npz ] && continue
  N_MACHINES=4 N_JOBS=$n .venv/bin/python scripts/tournament/tournament_eval.py ${EPISODES:-100} ${WORKERS:-3} results/tournament_m4_big/m4_n$n.npz 2>&1 | grep -v objc > /tmp/m4big_logs/m4_n$n.log
done
echo ALL_DONE > /tmp/m4big_logs/done
