"""Run the exact optimal solver as the policy (any number of machines, N_MACHINES env var).

    N_MACHINES=2 .venv/bin/python scripts/imitation/expert_data.py eval    5000 6 /tmp/opt_2m.npz      # seeds 42..: optimal reward per episode
    N_MACHINES=2 .venv/bin/python scripts/imitation/expert_data.py dataset 3000 6 /tmp/optimal_dataset_2m.npz  # seeds 100000..: demonstrations
"""
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dagger  # noqa: E402


def main(mode, n, workers, path):
    import torch as th
    from scheduling_simulator.expiremnt.policy.new_sched import NewSchedulingPolicy
    env = dagger.make_env()
    pol = NewSchedulingPolicy(env.observation_space, env.action_space, lambda _: 1e-4, **dagger.POLICY_KWARGS)
    sd = {k: v.cpu() for k, v in pol.state_dict().items()}       # the policy is never used (beta = 1)
    start = 42 if mode == "eval" else 100_000
    seeds = [(start + i, 1.0) for i in range(n)]
    with mp.get_context("spawn").Pool(workers, initializer=dagger._init_worker, initargs=(sd,)) as pool:
        res = []
        for k, r in enumerate(pool.imap(dagger.rollout, seeds, chunksize=4)):
            res.append(r)
            if (k + 1) % 250 == 0:
                print(f"...{k + 1}/{n}", flush=True)
    ep = np.array([r["episode_return"] for r in res])
    if mode == "eval":
        np.savez(path, episode=np.arange(n), reward=ep)
        print(f"optimal on {n} job sets (seeds {start}..): mean reward {ep.mean():.4f}")
        return
    data = {}
    for k in dagger.KEYS:
        arr = np.concatenate([r[k] for r in res])
        data[k] = arr.astype(np.int16 if k in ("machines_usage", "machines_capacity", "jobs_usage")
                             else np.int8 if k in ("status", "action_success") else np.float32)
    data["action"] = np.concatenate([r["action"] for r in res])
    data["ret"] = np.concatenate([r["ret"] for r in res])
    np.savez_compressed(path, **data)
    print(f"{n} episodes, {len(data['action'])} samples, mean optimal return {ep.mean():.2f}, "
          f"action counts {np.bincount(data['action'], minlength=1 + dagger.N_MACHINES * 10).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
