#!/bin/bash
# Fine-tune from the 10-job model with critic warm-up (the "ft3" recipe), end-to-end tournament, 1 machine, for NS jobs;
# at most MAX_RUNNING at a time; then the tournament evaluation (1000 episodes) into results/tournament_loop_ft3/.
cd "$(dirname "$0")/.."
LOGDIR=${LOGDIR:-/tmp/tl_ft3}; mkdir -p $LOGDIR results/tournament_loop_ft3
MAX_RUNNING=${MAX_RUNNING:-3}
running() { ps -eo pid,ppid,command | awk '/example_masked_tournament_loop.py/ && !/awk/ {p[$1]=$2} END {c=0; for (i in p) if (!(p[i] in p)) c++; print c}'; }
for n in ${NS:-60 70 80 90 100}; do
  while [ "$(running)" -ge "$MAX_RUNNING" ]; do sleep 60; done
  N_MACHINES=1 N_JOBS=$n TRAIN_SEED=3 JOB_USAGE_MODE=random TRAIN_STEPS=1000000 CRITIC_WARMUP=300000 FT_LR=1e-5 \
    FT_ENT_START=0.01 FT_ENT_END=0.005 INIT_POLICY=models/init/m1_n10_ncpb2fyb_policy.pt \
    WANDB_TAGS="tournament-loop,tournament-loop-finetune,finetune-v3-critic-warmup,m1,n$n,masked-ppo,finetune-from-n10,final-slice-skip-mask,forbid-idle-skip" \
    nohup caffeinate -i .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_tournament_loop.py > $LOGDIR/m1_n$n.log 2>&1 &
  sleep 30
done
while pgrep -f example_masked_tournament_loop.py >/dev/null; do sleep 60; done
for n in ${NS:-60 70 80 90 100}; do
  id=$(grep -o "runs/[a-z0-9]\{8\}" $LOGDIR/m1_n$n.log | tail -1 | cut -d/ -f2)
  [ -f models/$id/final_model.zip ] || { echo "n=$n: no model ($id)" >> $LOGDIR/eval.log; continue; }
  MODEL=models/$id/final_model.zip N_MACHINES=1 N_JOBS=$n JOB_USAGE_MODE=random FORBID_IDLE_SKIP=1 \
    .venv/bin/python scripts/tournament/tournament_eval.py 1000 3 results/tournament_loop_ft3/ft3_m1_n$n.npz >> $LOGDIR/eval.log 2>&1
  echo "n=$n done ($id)" >> $LOGDIR/eval.log
done
echo ALL_DONE >> $LOGDIR/eval.log
