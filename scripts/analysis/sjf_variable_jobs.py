"""SJF on the same episodes as the variable-jobs model's final evaluation (k from default_rng(42+ep), seed 42+ep)."""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
CFG = {"n_machines": 1, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}

def run(ep):
    from scheduling_simulator.envioremnt.wrappers.variable_jobs_env import VariableJobsEnv
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    env = VariableJobsEnv(CFG)
    obs, info = env.reset(seed=42 + ep)
    k = info["n_real_jobs"]; sj = ShortestJobScheduler(); tot = 0.0; done = False
    while not done:
        skip, j, m = sj.select(obs)
        obs, r, te, tr, _ = env.step(0 if skip else 1 + m * 10 + j)
        tot += float(r); done = te or tr
    return ep, tot, k

if __name__ == "__main__":
    with Pool(3) as p:
        a = np.array(sorted(p.map(run, range(5000), chunksize=50)))
    np.savez(sys.argv[1], reward=a[:, 1], k=a[:, 2].astype(int))
    print(a[:, 1].mean())
