#!/bin/zsh
# Extended zoom + job tournament grid: 5-8 machines x 10..100 jobs and 3-4 machines x 60..100 jobs (5000 episodes per cell).
cd "$(dirname "$0")/../.."
mkdir -p /tmp/zt
WORKERS=${WORKERS:-4}
for n in 10 20 30 40 50 60 70 80 90 100; do
  for m in 3 4 5 6 7 8; do
    { [ $m -le 4 ] && [ $n -lt 60 ]; } && continue
    [ -f /tmp/zt/m${m}_n${n}.npz ] && continue
    N_MACHINES=$m N_JOBS=$n JOB_USAGE_MODE=random .venv/bin/python scripts/machine_ae/zoom_tournament_eval.py 5000 $WORKERS /tmp/zt/m${m}_n${n}.npz 2>&1 | grep -v -e objc -e pygame > /tmp/zt/m${m}_n${n}.log
  done
done
