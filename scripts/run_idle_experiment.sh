#!/usr/bin/env bash
# Masked PPO with forbid_idle_skip: (a) original reward (example_masked_idle.py), (b) cleaned+scaled
# reward (example_masked_clean_idle.py), on the same seeds as the earlier masked runs (paired).
#   SEEDS="1 2" scripts/run_idle_experiment.sh
cd "$(dirname "$0")/.."
unset JOB_USAGE_MODE
mkdir -p models
for s in ${SEEDS:-1 2}; do
  for cond in idle clean_idle; do
    (TRAIN_SEED=$s WANDB_TAGS="seedexp,idleexp,masked-$cond,seed$s" \
      .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_$cond.py > /tmp/seedexp_$cond\_s$s.log 2>&1
     echo "[$(date +%H:%M)] done $cond seed $s (exit $?)") &
  done
done
wait
