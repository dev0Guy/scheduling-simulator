"""DAgger: relabel the states the POLICY visits with the exact optimal solver.

Behaviour cloning only sees states on the optimal trajectory. Here the current policy
is rolled out (with probability beta per step the expert's action is executed instead),
and at every visited state the exact solver is re-run FROM THAT STATE (running jobs with
their remaining time, pending jobs, future arrivals) to get the optimal action. The new
(state, optimal action) pairs are added to the data and the actor is retrained.
The critic is trained only on the original demonstrations (their returns are known).

Run from the repo root:
    .venv/bin/python scripts/imitation/dagger.py /tmp/optimal_dataset.npz models/bc_policy.pt models/dagger 6 1500 6
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "solvers"))
sys.path.insert(0, HERE)

N_MACHINES = int(os.environ.get("N_MACHINES", "1"))
CONFIG = {"n_machines": N_MACHINES, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}
POLICY_KWARGS = {
    "embedding_dim": 256, "n_machines": N_MACHINES, "max_time": 110.0,
    "per_resource_fit": True, "pair_features": True, "dynamic_machine": True,
}
KEYS = ("machines_usage", "machines_capacity", "jobs_usage", "status", "ttl", "arrival",
        "wait_time", "scheduled_at", "finished_at", "size", "time", "action_success")
SEED_BASE = 200_000


def make_env():
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    return FailureSkipTimeWrapper(SchedulingEnviorment(CONFIG, render_mode="none"), max_time=110)


class StateSolver:
    """Exact optimal action from any state of one episode (memo shared across calls)."""

    def __init__(self, sizes, usages, capacity, arrivals):
        from optimal_solver_multires import _as_vec, _add, _feasible_subsets
        self._add, self._subsets = _add, _feasible_subsets
        self.n = len(sizes)
        self.p = [int(round(s)) for s in sizes]
        self.u = [_as_vec(x) for x in usages]
        self.arr = [int(round(a)) for a in arrivals]
        self.cap = capacity
        self.zero = tuple(0 for _ in self.u[0])
        self.full = (1 << self.n) - 1
        self.memo = {}

    def dfs(self, t, done, active):
        if done == self.full:
            return 0, ()
        key = (t, done, active)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        n, p, u, arr = self.n, self.p, self.u, self.arr
        busy = {j for j, _ in active}
        pending = [j for j in range(n) if not (done >> j) & 1 and j not in busy and arr[j] <= t]
        used0 = self.zero
        for j, _ in active:
            used0 = self._add(used0, u[j])
        future = [arr[j] for j in range(n) if not (done >> j) & 1 and j not in busy and arr[j] > t]
        best_cost, best_choice = None, None
        for S in self._subsets(pending, u, used0, self.cap):
            new_active = tuple(sorted(active + tuple((j, t + p[j]) for j in S)))
            events = [f for _, f in new_active] + future
            if not events:
                continue
            next_t = min(events)
            finishing = [j for j, f in new_active if f == next_t]
            remaining = tuple((j, f) for j, f in new_active if f != next_t)
            new_done = done
            for j in finishing:
                new_done |= 1 << j
            sub, _ = self.dfs(next_t, new_done, remaining)
            if sub is None:
                continue
            cost = next_t * len(finishing) + sub
            if best_cost is None or cost < best_cost:
                best_cost, best_choice = cost, (S, next_t, new_done, remaining)
        self.memo[key] = (best_cost, best_choice)
        return self.memo[key]

    def executed(self, action, success):
        pass

    def action(self, obs):
        from scheduling_simulator.core.job import JobStatus
        status = [int(x) for x in obs["status"]]
        t = int(round(float(np.asarray(obs["time"]).reshape(-1)[0])))
        ttl = [int(round(float(x))) for x in obs["ttl"]]
        done = 0
        active = []
        for j, s in enumerate(status):
            if s == int(JobStatus.COMPLETED):
                done |= 1 << j
            elif s == int(JobStatus.RUNNING):
                active.append((j, t + ttl[j]))
        _, choice = self.dfs(t, done, tuple(sorted(active)))
        if choice is None or not choice[0]:
            return 0
        return 1 + choice[0][0]


_policy = None


def _init_worker(state_dict):
    global _policy
    import torch as th
    th.set_num_threads(1)
    from scheduling_simulator.expiremnt.policy.new_sched import NewSchedulingPolicy
    env = make_env()
    _policy = NewSchedulingPolicy(env.observation_space, env.action_space, lambda _: 1e-4, **POLICY_KWARGS)
    _policy.load_state_dict(state_dict)
    _policy.set_training_mode(False)


def make_solver(obs):
    sizes = np.asarray(obs["size"]).tolist()
    usages = np.asarray(obs["jobs_usage"])[:, :, 0].tolist()
    arrivals = np.asarray(obs["arrival"]).tolist()
    if N_MACHINES == 1:
        return StateSolver(sizes, usages, CONFIG["max_capacity"], arrivals)
    from optimal_solver_multi_machine import MultiMachineSolver
    return MultiMachineSolver(sizes, usages, CONFIG["max_capacity"], arrivals, N_MACHINES)


def rollout(args):
    seed, beta = args
    rng = np.random.default_rng(seed)
    env = make_env()
    obs, _ = env.reset(seed=seed)
    solver = make_solver(obs)
    rows, labels, rewards, mismatches = [], [], [], 0
    done = False
    while not done and len(labels) < 400:
        expert = solver.action(obs)
        pol, _ = _policy.predict({k: np.asarray(v) for k, v in obs.items()}, deterministic=True)
        pol = int(pol)
        rows.append({k: np.array(obs[k]) for k in KEYS})
        labels.append(expert)
        mismatches += int(pol != expert)
        chosen = expert if rng.random() < beta else pol
        obs, r, term, trunc, _ = env.step(chosen)
        solver.executed(chosen, bool(obs["action_success"]))
        rewards.append(float(r))
        done = term or trunc
    out = {k: np.stack([r[k] for r in rows]) for k in KEYS}
    out["action"] = np.asarray(labels, dtype=np.int64)
    ret, g = [], 0.0
    for r in reversed(rewards):
        g = r + 0.995 * g
        ret.append(g)
    out["ret"] = np.asarray(ret[::-1], dtype=np.float32)
    out["mismatch"] = mismatches
    out["episode_return"] = float(sum(rewards))
    return out


def collect(state_dict, it, n_episodes, beta, workers):
    seeds = [(SEED_BASE + it * 100_000 + i, beta) for i in range(n_episodes)]
    with Pool(workers, initializer=_init_worker, initargs=(state_dict,)) as pool:
        res = pool.map(rollout, seeds, chunksize=8)
    data = {}
    for k in KEYS:
        arr = np.concatenate([r[k] for r in res])
        data[k] = arr.astype(np.int16 if k in ("machines_usage", "machines_capacity", "jobs_usage")
                             else np.int8 if k in ("status", "action_success") else np.float32)
    data["action"] = np.concatenate([r["action"] for r in res])
    data["ret"] = np.concatenate([r["ret"] for r in res])
    if beta < 1.0:      # returns are only valid along the expert's own trajectory
        data["ret"][:] = np.nan
    mism = sum(r["mismatch"] for r in res)
    return data, mism / len(data["action"])


def main(base_path, init_path, out_prefix, n_iter, n_episodes, workers, epochs=4, lr=1e-4, batch=512):
    import torch as th
    import torch.nn.functional as F
    from pretrain_policy import build_policy, batch_obs, test_policy
    import wandb_log as wl

    run = wl.start(f"dagger-{N_MACHINES}M-3R", {"n_machines": N_MACHINES, "iterations": n_iter, "episodes_per_iteration": n_episodes, "epochs_per_iteration": epochs, "lr": lr, "batch": batch, "init": init_path, "base": base_path}, ["imitation", "dagger"])
    data = dict(np.load(base_path))
    model, _, device = build_policy()
    policy = model.policy
    policy.load_state_dict(th.load(init_path, map_location=device))
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    print(f"start: {len(data['action'])} samples", flush=True)
    for it in range(n_iter):
        t0 = time.time()
        sd = {k: v.cpu() for k, v in policy.state_dict().items()}
        beta = 0.3 if it == 0 else 0.0
        new, mism = collect(sd, it, n_episodes, beta, workers)
        print(f"iter {it + 1}: collected {len(new['action'])} states, policy disagrees with the optimum on {mism * 100:.1f}% ({time.time() - t0:.0f}s)", flush=True)
        for k in data:
            data[k] = np.concatenate([data[k], new[k].astype(data[k].dtype)])
        N = len(data["action"])
        actions = th.as_tensor(data["action"], device=device)
        returns = th.as_tensor(np.nan_to_num(data["ret"] / 100.0), device=device)
        has_ret = th.as_tensor(np.isfinite(data["ret"]), device=device)
        for epoch in range(epochs):
            policy.set_training_mode(True)
            perm = rng.permutation(N); tot = 0.0; k = 0
            for s in range(0, N, batch):
                b = perm[s:s + batch]
                obs = batch_obs(policy, data, b)
                feats = policy.extract_features(obs)
                logits = policy.action_net(policy.mlp_extractor.forward_actor(feats))
                v = policy.value_net(policy.mlp_extractor.forward_critic(feats)).squeeze(-1)
                m = has_ret[b]
                vloss = F.mse_loss(v[m], returns[b][m]) if m.any() else 0.0
                loss = F.cross_entropy(logits, actions[b]) + 0.5 * vloss
                opt.zero_grad(); loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step(); tot += float(loss); k += 1
            print(f"  epoch {epoch + 1}/{epochs}: loss {tot / k:.4f}", flush=True)
            wl.log(run, {"epoch": it * epochs + epoch + 1, "train/loss": tot / k})
        policy.set_training_mode(False)
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        th.save(policy.state_dict(), f"{out_prefix}_it{it + 1}.pt")
        pol, sjf = test_policy(policy, n_episodes=300)
        st = wl.summarize(pol, sjf)
        wl.log(run, {"iteration": it + 1, "dagger/disagreement_pct": 100 * mism, "dagger/new_states": len(new["action"]),
                     "dagger/total_states": N, "dagger/seconds": time.time() - t0, **{"dagger/test300_" + a: b for a, b in st.items()}})
        wl.upload(run, f"{out_prefix}_it{it + 1}.pt", "model")
        print(f"iter {it + 1} done in {time.time() - t0:.0f}s", flush=True)
    wl.finish(run)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]), int(a[5]), int(a[6]))
