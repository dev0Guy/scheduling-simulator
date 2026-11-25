"""SJF+BestFit, FIFO+BestFit and Tetris with several eps; same episodes (seeds 42+ep) as the model evaluation.
usage: heur_multi_grid.py <out_dir>; env MS, NS, EPS_SMALL (n<=100), EPS_BIG, WORKERS, N_TIME"""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
T = int(os.environ.get("N_TIME", "15")); M = int(os.environ.get("MS", "2"))
TET = [0.5, 2.0, 5.0, 20.0]
NAMES = ["sjf_bf", "fifo_bf"] + [f"tetris_eps{e:g}" for e in TET]
def run(a):
    n, ep = a
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.heuristic_schedulers import FifoBestFitScheduler, SjfBestFitScheduler, TetrisScheduler
    out = []
    for sch in [SjfBestFitScheduler(), FifoBestFitScheduler()] + [TetrisScheduler(e) for e in TET]:
        env = FailureSkipTimeWrapper(SchedulingEnviorment({"n_machines": M, "n_jobs": n, "n_resource": 3, "n_time": T, "max_capacity": 255}, render_mode="none"), max_time=11 * n)
        o, _ = env.reset(seed=42 + ep); tot, done = 0.0, False
        while not done:
            sk, j, m = sch.select(o); o, r, done, _, _ = env.step(0 if sk else 1 + m * n + j); tot += float(r)
        w = np.asarray(o["wait_time"], float); sz = np.asarray(o["size"], float)
        out += [tot, (w + sz).mean(), w.mean()]
    return out
if __name__ == "__main__":
    outd = sys.argv[1]; os.makedirs(outd, exist_ok=True)
    for n in map(int, os.environ["NS"].split()):
        f = f"{outd}/m{M}_n{n}.npz"
        if os.path.exists(f): continue
        eps = int(os.environ.get("EPS_SMALL", 1000)) if n <= 100 else int(os.environ.get("EPS_BIG", 300))
        with Pool(int(os.environ.get("WORKERS", 6))) as p: a = np.array(p.map(run, [(n, e) for e in range(eps)], chunksize=5))
        d = {}
        for i, k in enumerate(NAMES): d[k + "_reward"], d[k + "_completion"], d[k + "_wait"] = a[:, 3*i], a[:, 3*i+1], a[:, 3*i+2]
        np.savez(f, **d); print(n, {k: round(float(d[k + "_completion"].mean()), 1) for k in NAMES}, flush=True)
