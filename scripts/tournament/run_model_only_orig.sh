#!/bin/zsh
# Model-only evaluation of q1w0791o (freshly downloaded weights) on the original workload, 2 machines.
cd "$(dirname "$0")/.."/..
export PYTHONPATH=src JOB_USAGE_MODE=random N_TIME=15 MODEL=models/q1w0791o_fresh/final_model.zip MODEL_MACHINES=2 N_MACHINES=2 MODEL_ONLY=1
for n in ${=NS:-10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500}; do
  [ -f results/model_only_orig/m2_n$n.npz ] && continue
  eps=300; [ $n -le 100 ] && eps=1000
  N_JOBS=$n .venv/bin/python scripts/tournament/tournament_eval_model.py $eps ${WORKERS:-6} results/model_only_orig/m2_n$n.npz 2>&1 | grep -v objc > /tmp/mo_logs/n$n.log
done
echo x > /tmp/mo_logs/done
