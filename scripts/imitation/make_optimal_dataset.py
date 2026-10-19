"""Build an imitation dataset from the exact optimal solver.

For every job set (seed 100000 + i, never one of the evaluation seeds 42..5041) the
optimal start times are computed with scripts/solvers/optimal_solver_multires.py and
replayed through the real environment. Each step stores the observation, the optimal
action (0 = skip, k = start job k - 1) and the discounted return-to-go of the wrapper
reward along that optimal trajectory (used to pretrain the critic).

Run from the repo root:
    .venv/bin/python scripts/imitation/make_optimal_dataset.py 10000 6 /tmp/optimal_dataset.npz
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "solvers"))

CONFIG = {"n_machines": 1, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}
GAMMA = 0.995
BASE_SEED = 100_000
KEYS = ("machines_usage", "machines_capacity", "jobs_usage", "status", "ttl", "arrival",
        "wait_time", "scheduled_at", "finished_at", "size", "time", "action_success")


def one_episode(i):
    from scheduling_simulator.core.job import JobStatus
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from optimal_solver_multires import solve

    env = FailureSkipTimeWrapper(SchedulingEnviorment(CONFIG, render_mode="none"), max_time=110)
    obs, _ = env.reset(seed=BASE_SEED + i)
    arrivals = np.asarray(obs["arrival"]).tolist()
    sizes = np.asarray(obs["size"]).tolist()
    usages = np.asarray(obs["jobs_usage"])[:, :, 0].tolist()
    _, starts = solve(arrivals, sizes, usages, CONFIG["max_capacity"])

    rows, actions, rewards = [], [], []
    done = False
    while not done and len(actions) < 400:
        status = np.asarray(obs["status"])
        t = int(round(float(np.asarray(obs["time"]).reshape(-1)[0])))
        action = 0
        for j in range(CONFIG["n_jobs"]):
            if starts[j] <= t and status[j] == JobStatus.PENDING:
                action = 1 + j
                break
        rows.append({k: np.array(obs[k]) for k in KEYS})
        actions.append(action)
        obs, r, term, trunc, _ = env.step(action)
        rewards.append(float(r))
        done = term or trunc
    ret, g = [], 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
        ret.append(g)
    ret.reverse()
    out = {k: np.stack([row[k] for row in rows]) for k in KEYS}
    out["action"] = np.asarray(actions, dtype=np.int64)
    out["ret"] = np.asarray(ret, dtype=np.float32)
    out["episode_return"] = float(sum(rewards))
    return out


def main(n_episodes, workers, path):
    with Pool(workers) as pool:
        results = []
        for k, r in enumerate(pool.imap(one_episode, range(n_episodes), chunksize=4)):
            results.append(r)
            if (k + 1) % 1000 == 0:
                print(f"...{k + 1}/{n_episodes}", flush=True)
    data = {}
    for k in KEYS:
        arr = np.concatenate([r[k] for r in results])
        if k in ("machines_usage", "machines_capacity", "jobs_usage"):
            arr = arr.astype(np.int16)
        elif k in ("status", "action_success"):
            arr = arr.astype(np.int8)
        else:
            arr = arr.astype(np.float32)
        data[k] = arr
    data["action"] = np.concatenate([r["action"] for r in results])
    data["ret"] = np.concatenate([r["ret"] for r in results])
    np.savez_compressed(path, **data)
    ep = np.array([r["episode_return"] for r in results])
    print(f"episodes {n_episodes}, samples {len(data['action'])}, mean optimal episode return {ep.mean():.2f}")
    print("action counts (0 = skip):", np.bincount(data["action"], minlength=11).tolist())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/optimal_dataset.npz"
    main(n, w, out)
