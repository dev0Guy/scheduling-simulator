"""Collect machine states for the autoencoder: SJF rollouts (with some random legal actions for diversity).

    N_MACHINES=4 N_JOBS=30 JOB_USAGE_MODE=random .venv/bin/python scripts/machine_ae/collect.py 600 8 /tmp/ae_data_k4.npz
Saves free[K machines grids] / jobs_usage / status per decision step.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
M, N = int(os.environ.get("N_MACHINES", "4")), int(os.environ.get("N_JOBS", "30"))
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}


def run(ep):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.abc_scheduler import Scheduler
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    rng = np.random.default_rng(10_000 + ep)                  # job sets disjoint from the evaluation seeds (42 + ep)
    env = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)
    obs, _ = env.reset(seed=10_000 + ep)
    sj, done = ShortestJobScheduler(), False
    free, jobs, status = [], [], []
    while not done:
        free.append(((obs["machines_capacity"] - obs["machines_usage"]) / 255.0).astype(np.float32))
        jobs.append(obs["jobs_usage"].astype(np.uint8)); status.append(obs["status"].astype(np.uint8))
        opts = Scheduler.options(obs)
        if opts and rng.random() < 0.3:                       # random legal start (or skip) for diversity
            j, m = opts[rng.integers(len(opts))]
            a = 0 if rng.random() < 0.2 else 1 + m * N + j
        else:
            skip, j, m = sj.select(obs)
            a = 0 if skip else 1 + m * N + j
        obs, _, te, tr, _ = env.step(a)
        done = te or tr
    return np.stack(free), np.stack(jobs), np.stack(status)


if __name__ == "__main__":
    n, workers, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(workers) as p:
        res = p.map(run, range(n), chunksize=5)
    np.savez_compressed(out, free=np.concatenate([r[0] for r in res]), jobs=np.concatenate([r[1] for r in res]),
                        status=np.concatenate([r[2] for r in res]))
    print("states:", sum(len(r[0]) for r in res))
