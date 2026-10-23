#!/usr/bin/env bash
# Seeded comparison: PPO from scratch WITHOUT and WITH invalid-action masking on the same training seeds.
# Same seed => same initial weights and same sequence of training job sets in both conditions.
#   scripts/run_seed_experiment.sh            # seeds 1-4, 4 runs at a time
#   SEEDS="1 2" PARALLEL=2 scripts/run_seed_experiment.sh
# Logs: /tmp/seedexp_<condition>_s<seed>.log ; W&B tags: seedexp, <condition>, seed<k>
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"1 2 3 4"}
PARALLEL=${PARALLEL:-4}
unset JOB_USAGE_MODE
mkdir -p models
for s in $SEEDS; do
  echo "unmasked $s example.py"
  echo "masked $s example_masked.py"
done | xargs -P "$PARALLEL" -L 1 bash -c '
  cond=$0; seed=$1; script=$2
  echo "[$(date +%H:%M)] start $cond seed $seed"
  TRAIN_SEED=$seed WANDB_TAGS="seedexp,$cond,seed$seed" \
    .venv/bin/python src/scheduling_simulator/expiremnt/$script > /tmp/seedexp_${cond}_s${seed}.log 2>&1
  echo "[$(date +%H:%M)] done  $cond seed $seed (exit $?)"
'
