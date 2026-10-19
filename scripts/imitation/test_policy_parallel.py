"""Parallel version of test_policy.py (CPU workers): policy vs SJF on seeds 42..42+n-1.

    .venv/bin/python scripts/imitation/test_policy_parallel.py models/dagger_it5.pt 5000 /tmp/dagger_test_5000.npz 6
"""
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dagger  # noqa: E402  (sets up sys.path for src/ and the solvers)
import wandb_log as wl  # noqa: E402

_n = dagger.CONFIG["n_jobs"]


def run_chunk(seeds):
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    sjf = ShortestJobScheduler()
    out = []
    for seed in seeds:
        res = []
        for who in ("policy", "sjf"):
            env = dagger.make_env()
            obs, _ = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                if who == "policy":
                    a, _ = dagger._policy.predict({k: np.asarray(v) for k, v in obs.items()}, deterministic=True)
                    a = int(a)
                else:
                    skip, j, k = sjf.select(obs)
                    a = 0 if skip else 1 + k * _n + j
                obs, r, te, tr, _ = env.step(a)
                total += float(r); done = te or tr
            res.append(total)
        out.append(res)
    return out


if __name__ == "__main__":
    import torch as th
    weights, n, path, workers = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    sd = th.load(weights, map_location="cpu")
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    seeds = list(range(start, start + n))
    chunks = [seeds[i:i + 25] for i in range(0, n, 25)]
    with mp.get_context("spawn").Pool(workers, initializer=dagger._init_worker, initargs=(sd,)) as pool:
        rows = [r for c in pool.imap(run_chunk, chunks) for r in c]
    arr = np.array(rows)
    pol, sj = arr[:, 0], arr[:, 1]
    d = pol - sj
    print(f"test on {n} job sets: policy {pol.mean():.2f}  SJF {sj.mean():.2f}  paired diff {d.mean():+.2f} "
          f"(95% CI ±{1.96 * d.std(ddof=1) / np.sqrt(n):.2f})", flush=True)
    np.savez(path, policy=pol, sjf=sj)
    opt_path = os.environ.get("OPTIMAL_NPZ")
    opt = np.load(opt_path)["reward"][:n] if opt_path and os.path.exists(opt_path) else None
    run = wl.start(os.environ.get("WANDB_NAME", f"test-{os.path.basename(weights)}-{dagger.N_MACHINES}M-3R"),
                   {"weights": weights, "n_machines": dagger.N_MACHINES, "episodes": n, "first_seed": start}, ["imitation", "test"])
    wl.log_episodes(run, pol, sj, opt)
    wl.upload(run, path, "result")
    wl.upload(run, weights, "model")
    wl.finish(run)
