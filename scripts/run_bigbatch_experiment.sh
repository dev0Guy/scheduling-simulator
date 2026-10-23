#!/usr/bin/env bash
# Masked PPO with a bigger batch (example_masked_bigbatch.py) on the same training seeds as the
# masked runs of scripts/run_seed_experiment.sh, for a paired comparison.
#   SEEDS="1 2" scripts/run_bigbatch_experiment.sh
cd "$(dirname "$0")/.."
unset JOB_USAGE_MODE
mkdir -p models
for s in ${SEEDS:-1 2}; do
  (TRAIN_SEED=$s WANDB_TAGS="seedexp,bigbatch,masked,seed$s" \
    .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_bigbatch.py > /tmp/seedexp_bigbatch_s$s.log 2>&1
   echo "[$(date +%H:%M)] done bigbatch seed $s (exit $?)") &
done
wait
