#!/usr/bin/env bash
# Trains the regular masked PPO for the extended grid: 5-8 machines x 10..100 jobs and 1-4 machines x 60..100 jobs.
# The 10-job models of 5-8 machines already exist (tournament); they are not retrained.
cd "$(dirname "$0")/.."
PARALLEL=${PARALLEL:-4}
mkdir -p /tmp/scale2
{
  for n in 20 30 40 50 60 70 80 90 100; do for m in 5 6 7 8; do echo "$m $n"; done; done
  for n in 60 70 80 90 100; do for m in 1 2 3 4; do echo "$m $n"; done; done
} | xargs -P "$PARALLEL" -L 1 bash -c '
  m=$0; n=$1
  echo "[$(date +%H:%M)] start m$m n$n"
  N_MACHINES=$m N_JOBS=$n JOB_USAGE_MODE=random TRAIN_SEED=3 WANDB_TAGS="scale2,m$m,n$n" \
    .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_scale.py > /tmp/scale2/m${m}_n${n}.log 2>&1
  echo "[$(date +%H:%M)] done m$m n$n (exit $?)"
'
