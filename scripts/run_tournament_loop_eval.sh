#!/bin/bash
# Waits until the queue and all trainings of run_tournament_loop_queue.sh are finished, then evaluates every model
# with the tournament procedure (1000 episodes, same seeds as SJF) into results/tournament_loop/.
cd "$(dirname "$0")/.."
while pgrep -f run_tournament_loop_queue.sh >/dev/null || pgrep -f example_masked_tournament_loop.py >/dev/null; do sleep 120; done
for n in 20 30 40 50 60 70 80 90 100; do
  id=$(grep -o "runs/[a-z0-9]\{8\}" /tmp/tl/m1_n$n.log | tail -1 | cut -d/ -f2)
  [ -f models/$id/final_model.zip ] || { echo "n=$n: no model ($id)" >> /tmp/tl/eval.log; continue; }
  MODEL=models/$id/final_model.zip N_MACHINES=1 N_JOBS=$n JOB_USAGE_MODE=random \
    .venv/bin/python scripts/tournament/tournament_eval.py 1000 3 results/tournament_loop/loop_m1_n$n.npz >> /tmp/tl/eval.log 2>&1
  echo "n=$n done ($id)" >> /tmp/tl/eval.log
done
echo ALL_DONE >> /tmp/tl/eval.log
