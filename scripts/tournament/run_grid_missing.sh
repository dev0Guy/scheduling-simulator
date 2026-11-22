#!/bin/zsh
# Eval-only: tournament cells missing from the extended grid (1 machine 90-100 jobs, 2-8 machines 80-100 jobs).
# Results go to results/tournament_grid/m{M}_n{N}.npz (kept in the repo, not /tmp) and are uploaded to W&B at the end.
cd "$(dirname "$0")/../.."
mkdir -p results/tournament_grid /tmp/tg_logs
WORKERS=${WORKERS:-4}
for n in 80 90 100; do
  for m in 1 2 3 4 5 6 7 8; do
    { [ $m -eq 1 ] && [ $n -eq 80 ]; } && continue
    [ -f results/tournament_grid/m${m}_n${n}.npz ] && continue
    N_MACHINES=$m N_JOBS=$n JOB_USAGE_MODE=random .venv/bin/python scripts/tournament/tournament_eval.py 5000 $WORKERS results/tournament_grid/m${m}_n${n}.npz 2>&1 | grep -v objc > /tmp/tg_logs/m${m}_n${n}.log
  done
done
echo ALL_DONE > /tmp/tg_logs/done
