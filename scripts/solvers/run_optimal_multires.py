"""
Replay the exact-optimal packing schedule with per-resource usage (optimal_solver_multires.py)
through the real environment, one episode per worker process, using the
same seeds as the SJF / PPO runs (seed = 42 + episode).

Prints mean reward / wait / completion plus two sanity counters:
  * failed allocations (should be 0: the plan must be feasible in the env)
  * episodes where the solver's predicted mean flow time differs from the
    measured one (should be 0: model of the env must be exact)
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

CONFIG = {"n_machines": 1, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}


def _unbatch(obs):
    return {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v) for k, v in obs.items()}


def run_episode(args):
    ep, seed = args
    from scheduling_simulator.core.job import JobStatus
    from scheduling_simulator.expiremnt.runners.env_factory import generate_scheduling_env
    from optimal_solver_multires import solve

    n_jobs = CONFIG["n_jobs"]
    env = generate_scheduling_env(
        config=CONFIG, max_time=110, path=f"/tmp/opt_pack/ep_{ep}", with_video=False, n_env=1,
    )
    env.seed(seed + ep)
    obs = env.reset()
    o = _unbatch(obs)

    arrivals = np.asarray(o["arrival"]).tolist()
    sizes = np.asarray(o["size"]).tolist()
    usages = np.asarray(o["jobs_usage"])[:, :, 0].tolist()   # one value per resource, per job
    cost, starts = solve(arrivals, sizes, usages, CONFIG["max_capacity"])
    predicted_flow = (cost - sum(int(a) for a in arrivals)) / n_jobs

    total_reward, failed, done, final = 0.0, 0, False, None
    guard = 0
    while not done and guard < 2000:
        guard += 1
        o = _unbatch(obs)
        status = np.asarray(o["status"])
        t = int(round(float(np.asarray(o["time"]).squeeze())))

        action = 0
        for j in range(n_jobs):
            if starts[j] <= t and status[j] == JobStatus.PENDING:
                action = 1 + j
                break

        obs, reward, done_arr, infos = env.step(np.array([action]))
        done = bool(done_arr[0])
        total_reward += float(reward[0])
        real = infos[0].get("terminal_observation", _unbatch(obs)) if done else _unbatch(obs)
        if action != 0:
            if not bool(np.asarray(real["action_success"])):
                failed += 1
        if done:
            final = real
    env.close()

    flow = np.asarray(final["wait_time"]) + np.asarray(final["size"])
    # env arrays are float32, so allow ~1e-4 of float noise
    mismatch = abs(float(np.mean(flow)) - predicted_flow) > 1e-4
    if mismatch:
        print(f"MISMATCH ep={ep} predicted_flow={predicted_flow:.3f} measured={float(np.mean(flow)):.3f} "
              f"arr={arrivals} size={sizes} usage={usages} starts={starts} "
              f"wait={np.asarray(final['wait_time']).tolist()} status={np.asarray(final['status']).tolist()}", flush=True)
    return (
        ep,
        total_reward,
        float(np.mean(final["wait_time"])),
        float(np.mean(flow)),
        failed,
        mismatch,
    )


def main(n_episodes=5000, seed=42, workers=6, out_path="/tmp/opt_newenv.npz"):
    with Pool(workers) as pool:
        results = []
        for i, r in enumerate(pool.imap(run_episode, [(ep, seed) for ep in range(n_episodes)], chunksize=5)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"...{i + 1}/{n_episodes}", flush=True)

    eps, rew, wait, comp, failed, mismatch = (np.array(x) for x in zip(*results))
    np.savez(out_path, episode=eps, reward=rew, wait=wait, completion=comp, failed=failed)
    print(f"\nEpisodes: {n_episodes}")
    print(f"Mean reward:              {rew.mean():.4f}")
    print(f"Mean avg_wait_time:       {wait.mean():.4f}")
    print(f"Mean avg_completion_time: {comp.mean():.4f}")
    print(f"Failed allocations (total): {int(failed.sum())}")
    print(f"Episodes where predicted != measured flow time: {int(mismatch.sum())}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/opt_newenv.npz"
    main(n_episodes=n, workers=w, out_path=out)
