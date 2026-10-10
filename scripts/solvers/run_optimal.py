"""
Replay the exact-optimal job order (scripts/solvers/optimal_solver.py) through the
real environment for each of N episodes, using the same seed convention
(seed = 42 + episode) as random_runner.py / train_runner.py, so results
are directly comparable to the existing SJF and PPO runs episode-for-episode.

No W&B logging, no video -- just prints the summary stats.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.expiremnt.runners.env_factory import generate_scheduling_env
from optimal_solver import solve_optimal_order


def _unbatch(obs: dict) -> dict:
    return {
        k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
        for k, v in obs.items()
    }


def main(n_episodes: int = 5000, seed: int = 42):
    config = {
        "n_machines": 1,
        "n_jobs": 10,
        "n_resource": 3,
        "n_time": 10,
        "max_capacity": 255,
    }

    n_jobs = config["n_jobs"]

    rewards = []
    wait_times = []
    completion_times = []
    not_proven_count = 0

    for ep in range(n_episodes):
        env = generate_scheduling_env(
            config=config, max_time=110, path=f"/tmp/optimal_scratch/ep_{ep}",
            with_video=False, n_env=1,
        )
        env.seed(seed + ep)
        obs = env.reset()  # batched (vec-env) observation throughout

        obs_single = _unbatch(obs)
        arrivals = np.asarray(obs_single["arrival"]).copy()
        sizes = np.asarray(obs_single["size"]).copy()

        order, _ = solve_optimal_order(arrivals.tolist(), sizes.tolist())
        order_ptr = [0]  # which position in `order` we're waiting to start next

        total_reward = 0.0
        done = False
        final_obs = None

        while not done:
            obs_single = _unbatch(obs)

            status = np.asarray(obs_single["status"])
            action = 0  # skip by default

            if order_ptr[0] < n_jobs:
                next_job = order[order_ptr[0]]
                if status[next_job] == JobStatus.PENDING:
                    # machine free (no job RUNNING) -- a PENDING job only
                    # becomes a valid allocation target once the machine
                    # has capacity, and with full-capacity exclusive jobs
                    # that's exactly "no job currently RUNNING"
                    running = np.any(status == JobStatus.RUNNING)
                    if not running:
                        action = 1 + 0 * n_jobs + next_job  # machine 0

            step_obs, reward, done_arr, infos = env.step(np.array([action]))
            done = bool(done_arr[0])
            total_reward += float(reward[0])

            # DummyVecEnv auto-resets on done: `step_obs` on the terminal
            # step is already the NEXT episode's reset observation, not
            # the true final state. infos[0]['terminal_observation'] is
            # already a single (unbatched) observation -- do NOT run it
            # through _unbatch again (that would index element 0 of each
            # already-scalar-per-job array, silently truncating every
            # field to a single number instead of all n_jobs values).
            if done:
                real_obs = infos[0].get("terminal_observation", _unbatch(step_obs))
            else:
                real_obs = _unbatch(step_obs)

            if action != 0:
                succeeded = bool(np.asarray(real_obs["action_success"]))
                if succeeded:
                    order_ptr[0] += 1
                # else: allocation failed this tick (shouldn't normally
                # happen given our PENDING+machine-free gating) -- leave
                # order_ptr unchanged and retry next tick.

            if done:
                final_obs = real_obs
            else:
                obs = step_obs

        env.close()

        rewards.append(total_reward)
        wait_times.append(float(np.mean(final_obs["wait_time"])))
        completion_times.append(float(np.mean(final_obs["wait_time"] + final_obs["size"])))

        if (ep + 1) % 500 == 0:
            print(f"...{ep + 1}/{n_episodes} episodes done", flush=True)

    print()
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward:            {np.mean(rewards):.4f}")
    print(f"Mean avg_wait_time:     {np.mean(wait_times):.4f}")
    print(f"Mean avg_completion_time: {np.mean(completion_times):.4f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(n_episodes=n)
