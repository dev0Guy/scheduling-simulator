#!/usr/bin/env bash
# Masked PPO with a scaled (example_masked_scaled.py) and a cleaned+scaled (example_masked_clean.py)
# training reward, on the same training seeds as scripts/run_seed_experiment.sh (paired comparison).
#   SEEDS="1 2" scripts/run_reward_experiment.sh
cd "$(dirname "$0")/.."
unset JOB_USAGE_MODE
mkdir -p models
for s in ${SEEDS:-1 2}; do
  for cond in scaled clean; do
    (TRAIN_SEED=$s WANDB_TAGS="seedexp,rewardexp,masked-$cond,seed$s" \
      .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_$cond.py > /tmp/seedexp_reward${cond}_s$s.log 2>&1
     echo "[$(date +%H:%M)] done reward-$cond seed $s (exit $?)") &
  done
done
wait
