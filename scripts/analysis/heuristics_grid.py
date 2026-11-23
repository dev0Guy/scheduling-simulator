"""Heuristic schedulers on the DeepRM environment, seeds 42+ep (identical episodes to the model grids).
usage: heuristics_grid.py <episodes> <workers> <out_dir>   env: MS, NS (space separated), N_TIME, job-gen vars"""
import os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
T = int(os.environ.get("N_TIME", "15"))
NAMES = ["sjf", "fifo_bf", "sjf_bf", "tetris"]


def run(args):
    m, n, ep = args
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    from scheduling_simulator.scheduler.heuristic_schedulers import FifoBestFitScheduler, SjfBestFitScheduler, TetrisScheduler
    cfg = {"n_machines": m, "n_jobs": n, "n_resource": 3, "n_time": T, "max_capacity": 255}
    out = []
    for sch in (ShortestJobScheduler(), FifoBestFitScheduler(), SjfBestFitScheduler(), TetrisScheduler()):
        env = FailureSkipTimeWrapper(SchedulingEnviorment(cfg, render_mode="none"), max_time=11 * n)
        o, _ = env.reset(seed=42 + ep); tot, done = 0.0, False
        while not done:
            sk, j, mm = sch.select(o); o, r, done, _, _ = env.step(0 if sk else 1 + mm * n + j); tot += float(r)
        w = np.asarray(o["wait_time"], float); sz = np.asarray(o["size"], float)
        out += [tot, (w + sz).mean()]          # reward, average completion time (= average wait + average size)
    return out + [float(sz.mean())]


if __name__ == "__main__":
    eps, workers, outd = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(outd, exist_ok=True)
    for m in map(int, os.environ.get("MS", "2 3 4 5 6 7 8").split()):
        for n in map(int, os.environ.get("NS", "10 20 30 40 50 60 70 80 90 100").split()):
            f = f"{outd}/m{m}_n{n}.npz"
            if os.path.exists(f): continue
            with Pool(workers) as p:
                a = np.array(p.map(run, [(m, n, e) for e in range(eps)], chunksize=20))
            d = {}
            for i, k in enumerate(NAMES):
                d[k] = a[:, 2 * i]; d[k + "_completion"] = a[:, 2 * i + 1]; d[k + "_wait"] = a[:, 2 * i + 1] - a[:, -1]
            d["mean_size"] = a[:, -1]
            np.savez(f, **d)
            print(m, n, {k: (round(float(d[k].mean()), 2), round(float(d[k + "_wait"].mean()), 2), round(float(d[k + "_completion"].mean()), 2)) for k in NAMES}, flush=True)
