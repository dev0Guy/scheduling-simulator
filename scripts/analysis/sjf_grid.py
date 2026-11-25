"""SJF only, same episodes (seeds 42+ep) as the model evaluations. usage: sjf_grid.py <out_dir>; env MS, NS, EPS_SMALL, EPS_BIG, WORKERS, N_TIME"""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
T = int(os.environ.get("N_TIME", "15")); M = int(os.environ.get("MS", "2"))
def run(a):
    n, ep = a
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    env = FailureSkipTimeWrapper(SchedulingEnviorment({"n_machines": M, "n_jobs": n, "n_resource": 3, "n_time": T, "max_capacity": 255}, render_mode="none"), max_time=11 * n)
    o, _ = env.reset(seed=42 + ep); sch = ShortestJobScheduler(); tot, done = 0.0, False
    while not done:
        sk, j, m = sch.select(o); o, r, done, _, _ = env.step(0 if sk else 1 + m * n + j); tot += float(r)
    w = np.asarray(o["wait_time"], float); sz = np.asarray(o["size"], float)
    return tot, (w + sz).mean(), w.mean()
if __name__ == "__main__":
    outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
    for n in map(int, os.environ["NS"].split()):
        f = f"{outd}/m{M}_n{n}.npz"
        if os.path.exists(f): continue
        eps = int(os.environ.get("EPS_SMALL", 1000)) if n <= 100 else int(os.environ.get("EPS_BIG", 300))
        with Pool(int(os.environ.get("WORKERS", 6))) as p: a = np.array(p.map(run, [(n, e) for e in range(eps)], chunksize=10))
        np.savez(f, s_reward=a[:, 0], s_completion=a[:, 1], s_wait=a[:, 2]); print(n, a.mean(0), flush=True)
