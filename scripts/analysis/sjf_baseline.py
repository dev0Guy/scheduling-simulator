"""SJF baseline on any cluster size, same job sets as the PPO evaluation (seed 42 + episode).

    N_MACHINES=2 N_JOBS=10 JOB_USAGE_MODE=random .venv/bin/python scripts/analysis/sjf_baseline.py 5000 3 /tmp/sjf_m2_n10.npz
Saves per-episode reward (original reward), average wait time and average completion time.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
M, N = int(os.environ.get("N_MACHINES", "1")), int(os.environ.get("N_JOBS", "10"))
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}


def run(ep):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    sj = ShortestJobScheduler()
    env = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)
    obs, _ = env.reset(seed=42 + ep)
    tot, done, steps = 0.0, False, 0
    while not done:
        skip, j, m = sj.select(obs)
        obs, r, te, tr, _ = env.step(0 if skip else 1 + m * N + j)
        tot += float(r); done = te or tr; steps += 1
    w = np.asarray(obs["wait_time"], float)
    return ep, tot, w.mean(), (w + np.asarray(obs["size"], float)).mean(), steps


if __name__ == "__main__":
    n, workers, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(workers) as p:
        res = sorted(p.map(run, range(n), chunksize=25))
    a = np.array(res)
    np.savez(out, episode=a[:, 0].astype(int), reward=a[:, 1], wait=a[:, 2], completion=a[:, 3], steps=a[:, 4].astype(int))
    print(f"M={M} N={N}: SJF reward {a[:, 1].mean():.2f}, wait {a[:, 2].mean():.3f}, completion {a[:, 3].mean():.3f}, mean steps {a[:, 4].mean():.1f}, timeouts {(a[:, 4] >= 11 * N).sum()}")
