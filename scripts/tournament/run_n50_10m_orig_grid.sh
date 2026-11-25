#!/bin/zsh
# Eval only: the DeepRM-workload model (q1w0791o (2 machines x 10 slots, trained on 2..10 jobs), deterministic machine selection: 2 machines x 20 jobs, demand x2) on M machines x N jobs, tournament over the
# jobs (slices of 10) and only 2 machines shown per slice (machine_select.py). Results: results/tournament_n50_10m_orig/m{M}_n{N}.npz
cd "$(dirname "$0")/.."/..
mkdir -p results/tournament_n50_10m_orig /tmp/n50_10m_orig_logs
export PYTHONPATH=src JOB_USAGE_MODE=random N_TIME=15
export MODEL=models/q1w0791o/final_model.zip MODEL_MACHINES=2
for m in ${=MS:-2 3 4 5 6 7 8}; do
  for n in ${=NS:-10 20 30 40 50 60 70 80 90 100}; do
    [ -f results/tournament_n50_10m_orig/m${m}_n$n.npz ] && continue
    N_MACHINES=$m N_JOBS=$n .venv/bin/python scripts/tournament/tournament_eval.py ${EPISODES:-1000} ${WORKERS:-6} results/tournament_n50_10m_orig/m${m}_n$n.npz 2>&1 | grep -v objc > /tmp/n50_10m_orig_logs/m${m}_n$n.log
  done
done
echo ALL_DONE > /tmp/n50_10m_orig_logs/done
