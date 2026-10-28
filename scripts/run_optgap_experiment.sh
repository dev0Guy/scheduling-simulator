#!/usr/bin/env bash
# Masked PPO whose training reward is the gap to the optimal action (example_masked_optgap.py), on the
# same training seeds as the masked/unmasked runs of scripts/run_seed_experiment.sh (paired comparison).
#   SEEDS="1 2 3 4" scripts/run_optgap_experiment.sh
cd "$(dirname "$0")/.."
unset JOB_USAGE_MODE
mkdir -p models
for s in ${SEEDS:-1 2 3 4}; do
  (TRAIN_SEED=$s WANDB_TAGS="seedexp,optgap,masked-optgap,seed$s" \
    .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_optgap.py > /tmp/seedexp_optgap_s$s.log 2>&1
   echo "[$(date +%H:%M)] done optgap seed $s (exit $?)") &
done
wait
