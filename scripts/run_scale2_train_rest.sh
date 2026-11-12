#!/usr/bin/env bash
# Remaining models of the extended grid, started only while fewer than MAX_RUNNING trainings are active
# (four at once shared one GPU and the progress per run dropped to a few hundred steps per minute).
cd "$(dirname "$0")/.."
MAX_RUNNING=${MAX_RUNNING:-3}  # evaluations count as running too, so 3 = 2 evaluations + 1 training
running() { pgrep -f "Python src/scheduling_simulator/expiremnt/example_masked_scale" | wc -l | tr -d ' '; }
{
  for m in 5 6 7 8; do echo "$m 100"; done
  for m in 6 7 8; do echo "$m 90"; done
  for n in 60 70 80 90 100; do for m in 1 2 3 4; do echo "$m $n"; done; done
} | while read m n; do
  # skip cells already collected or already started
  [ -f results/regular/ppo_m${m}_n${n}.npy ] && continue
  [ -f /tmp/scale2/m${m}_n${n}.log ] && continue
  while [ "$(running)" -ge "$MAX_RUNNING" ]; do sleep 60; done
  echo "[$(date +%H:%M)] start m$m n$n"
  N_MACHINES=$m N_JOBS=$n JOB_USAGE_MODE=random TRAIN_SEED=3 WANDB_TAGS="scale2,m$m,n$n" \
    .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_scale.py > /tmp/scale2/m${m}_n${n}.log 2>&1 &
  sleep 90
done
wait
