#!/bin/zsh
# Eval only: the DeepRM-workload model (va02clzj (2 machines x 10 slots, trained on 2..10 jobs), deterministic machine selection: 2 machines x 20 jobs, demand x2) on M machines x N jobs, tournament over the
# jobs (slices of 10) and only 2 machines shown per slice (machine_select.py). Results: results/tournament_deeprm_n10_5m/m{M}_n{N}.npz
cd "$(dirname "$0")/.."/..
mkdir -p results/tournament_deeprm_n10_5m /tmp/deeprm_n10_5m_logs
export PYTHONPATH=src JOB_USAGE_MODE=random JOB_GEN_MODE=deeprm DEEPRM_LOAD=10 DEEPRM_DEMAND_SCALE=2 N_TIME=15
export MODEL=models/va02clzj/final_model.zip MODEL_MACHINES=2
for m in ${=MS:-2 3 4 5 6 7 8}; do
  for n in ${=NS:-10 20 30 40 50 60 70 80 90 100}; do
    [ -f results/tournament_deeprm_n10_5m/m${m}_n$n.npz ] && continue
    N_MACHINES=$m N_JOBS=$n .venv/bin/python scripts/tournament/tournament_eval.py 5000 ${WORKERS:-6} results/tournament_deeprm_n10_5m/m${m}_n$n.npz 2>&1 | grep -v objc > /tmp/deeprm_n10_5m_logs/m${m}_n$n.log
  done
done
echo ALL_DONE > /tmp/deeprm_n10_5m_logs/done
