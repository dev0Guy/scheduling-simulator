#!/bin/bash
# Trains the end-to-end tournament models (1 machine) for N = 20..100 jobs, at most MAX_RUNNING at a time.
cd "$(dirname "$0")/.."
MAX_RUNNING=${MAX_RUNNING:-3}
running() {  # training processes (not their worker children)
  ps -eo pid,ppid,command | awk '/example_masked_tournament_loop.py/ && !/awk/ {p[$1]=$2} END {c=0; for (i in p) if (!(p[i] in p)) c++; print c}'
}
for n in ${NS:-20 30 40 50 60 70 80 90 100}; do
  while [ "$(running)" -ge "$MAX_RUNNING" ]; do sleep 60; done
  echo "$(date +%H:%M) start n=$n" >> ${LOGDIR:-/tmp/tl}/queue.log
  N_MACHINES=1 N_JOBS=$n TRAIN_SEED=3 JOB_USAGE_MODE=random WANDB_TAGS="tournament-loop,${EXTRA_TAGS}m1,n$n,masked-ppo,from-scratch" \
    nohup .venv/bin/python src/scheduling_simulator/expiremnt/example_masked_tournament_loop.py > ${LOGDIR:-/tmp/tl}/m1_n$n.log 2>&1 &
  sleep 30
done
wait
