#!/bin/zsh
# Old 3-machine x 10-job model (perlec9c, original workload) in a tournament over 200..1000 jobs, 100 episodes per cell.
# Results: results/tournament_m3_big/m3_n{500,1000}.npz
cd "$(dirname "$0")/../.."
mkdir -p results/tournament_m3_big /tmp/m3big_logs
export PYTHONPATH=src JOB_USAGE_MODE=random
for n in ${=NS:-200 400 500 600 800 1000}; do
  [ -f results/tournament_m3_big/m3_n$n.npz ] && continue
  N_MACHINES=3 N_JOBS=$n .venv/bin/python scripts/tournament/tournament_eval.py ${EPISODES:-100} ${WORKERS:-3} results/tournament_m3_big/m3_n$n.npz 2>&1 | grep -v objc > /tmp/m3big_logs/m3_n$n.log
done
echo ALL_DONE > /tmp/m3big_logs/done2
