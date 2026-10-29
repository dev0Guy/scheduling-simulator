"""Tournament evaluation of a model trained on a small cluster, applied to a larger job pool.

The model (M machines x 10 jobs) can only look at 10 job slots. On an environment with
more jobs, every decision is made by a knock-out tournament instead of one forward pass:

  1. the candidates are all unfinished jobs; only legal starts (pending and fitting) can win a slice;
  2. they are cut into slices of 10 in job order; a short last slice is padded with empty slots
     (status COMPLETED, usage / size / wait / ttl / arrival 0 - the policy masks them as not startable
     and the self-attention ignores them as keys);
  3. the model sees each slice (the real machine state, 10 job slots) and its best legal job wins the slice;
     the argmax is taken over the real slots only, never over padding and never over skip;
  4. the winners are sliced and padded again, round after round, until one job is left. That job is
     started in the real environment. If no pending job fits, the environment gets a skip.

With M machines the model is the one trained on M machines x 10 jobs (RUN_BY_MACHINES); the slice carries
all M machines and the 10 job slots, and the winner of a slice is its best legal (machine, job) pair.

    N_MACHINES=1 N_JOBS=30 JOB_USAGE_MODE=random .venv/bin/python scripts/tournament/tournament_eval.py 500 4 /tmp/tournament_m1_n30.npz

args: n_episodes workers out.npz   (job sets seeded 42 + episode, same as the other evaluations)
env : MODEL (default models/<run for M machines>/final_model.zip), SLICE (default 10, the model's job slots),
      ALLOW_SKIP=1 lets the model also pick skip in the final slice (with N_JOBS=10 this is
      exactly the model's normal deterministic policy, a sanity check)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
M, N = int(os.environ.get("N_MACHINES", "1")), int(os.environ.get("N_JOBS", "30"))
RUN_BY_MACHINES = {1: "ncpb2fyb", 2: "bsp1klua", 3: "perlec9c", 4: "t3lhn7a9"}   # masked PPO, M machines x 10 jobs
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", RUN_BY_MACHINES[M], "final_model.zip"))
ALLOW_SKIP = os.environ.get("ALLOW_SKIP", "1") == "1"
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}
COMPLETED = 3
_NEG = -1e8

_policy = None
_slots = None
_calls = 0


def _load():
    global _policy, _slots
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    _policy = PPO.load(MODEL, device="cpu").policy
    _policy.set_training_mode(False)
    _slots = int(_policy.observation_space["status"].shape[0])
    assert _policy.observation_space["machines_usage"].shape[0] == M, "the model must be trained on the same number of machines"
    assert int(os.environ.get("SLICE", _slots)) == _slots, "SLICE must equal the model's job slots"


def _slice(obs, jobs):
    """Observation for the model: all machines, the given jobs in the first slots, empty padding after."""
    k = len(jobs)
    out = {"machines_usage": obs["machines_usage"], "machines_capacity": obs["machines_capacity"],
           "time": obs["time"], "action_success": int(obs["action_success"])}
    ju = np.zeros((_slots,) + obs["jobs_usage"].shape[1:], obs["jobs_usage"].dtype)
    ju[:k] = obs["jobs_usage"][jobs]
    out["jobs_usage"] = ju
    st = np.full(_slots, COMPLETED, obs["status"].dtype)
    st[:k] = obs["status"][jobs]
    out["status"] = st
    for key in ("ttl", "arrival", "wait_time", "scheduled_at", "finished_at", "size"):
        v = np.zeros(_slots, obs[key].dtype)
        v[:k] = obs[key][jobs]
        out[key] = v
    return out


def _decide(obs, chunks, final):
    """One batched forward over the slices of a round. Returns [(job, machine, probability)]; job None = skip."""
    global _calls
    import torch as th
    batch = [_slice(obs, c) for c in chunks]
    stacked = {k: np.stack([np.asarray(b[k]) for b in batch]) for k in batch[0]}
    with th.no_grad():
        dist = _policy.get_distribution(_policy.obs_to_tensor(stacked)[0])   # applies the policy's own legality mask
        logits = dist.distribution.logits.clone()
    _calls += len(chunks)
    res = []
    for i, c in enumerate(chunks):
        lg = logits[i].clone()
        keep = th.zeros_like(lg, dtype=th.bool)
        for m in range(M):                                 # real slots only, never padding
            keep[1 + m * _slots:1 + m * _slots + len(c)] = True
        if final and ALLOW_SKIP:
            keep[0] = True
        lg = lg.masked_fill(~keep, _NEG)
        a = int(lg.argmax())
        if float(lg[a]) <= _NEG / 10:                       # nothing legal in this slice
            res.append((-1, 0, 0.0))
            continue
        p = float(th.softmax(lg, 0)[a])
        res.append((None, 0, p) if a == 0 else (c[(a - 1) % _slots], (a - 1) // _slots, p))
    return res


def choose(obs):
    """Knock-out over the jobs. Round 1 slices ALL unfinished jobs (pending, running, not yet arrived), so the
    model sees what it saw in training; only legal starts (pending and fitting) can win a slice, because the
    policy masks the rest. With N <= 10 jobs this is one slice holding the whole observation, i.e. exactly the
    model's normal decision. Later rounds slice the winners. Returns the environment action
    (0 = skip, 1 + m * N + j = start job j on machine m)."""
    from scheduling_simulator.scheduler.abc_scheduler import Scheduler
    if not Scheduler.options(obs):
        return 0
    if N <= _slots:
        cands = list(range(N))
    else:
        cands = [j for j in range(N) if obs["status"][j] != COMPLETED]
    while True:
        chunks = [cands[i:i + _slots] for i in range(0, len(cands), _slots)]
        final = len(chunks) == 1
        res = _decide(obs, chunks, final)
        if final:
            job, m, _ = res[0]
            return 0 if job is None else 1 + m * N + job
        cands = [w for w, _, _ in res if w != -1 and w is not None]


def run(ep):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    if _policy is None:
        _load()
    out = []
    for who in ("tournament", "sjf"):
        env = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)
        obs, _ = env.reset(seed=42 + ep)
        sj = ShortestJobScheduler()
        tot, done, steps = 0.0, False, 0
        while not done:
            if who == "tournament":
                a = choose(obs)
            else:
                skip, j, m = sj.select(obs)
                a = 0 if skip else 1 + m * N + j
            obs, r, te, tr, _ = env.step(a)
            tot += float(r); done = te or tr; steps += 1
        w = np.asarray(obs["wait_time"], float)
        out += [tot, (w + np.asarray(obs["size"], float)).mean(), steps]
    return (ep, *out)


if __name__ == "__main__":
    n, workers, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(workers, initializer=_load) as p:
        a = np.array(sorted(p.map(run, range(n), chunksize=5)))
    np.savez(path, episode=a[:, 0].astype(int), t_reward=a[:, 1], t_completion=a[:, 2], t_steps=a[:, 3],
             s_reward=a[:, 4], s_completion=a[:, 5], s_steps=a[:, 6])
    d = a[:, 1] - a[:, 4]
    tol = 1e-3
    w, l = int((d > tol).sum()), int((d < -tol).sum())
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"{MODEL.split('/')[-2]} M={M} N={N} episodes={n} allow_skip={ALLOW_SKIP}")
    print(f"tournament: reward {a[:, 1].mean():.2f}  completion {a[:, 2].mean():.2f}  steps {a[:, 3].mean():.1f}")
    print(f"SJF       : reward {a[:, 4].mean():.2f}  completion {a[:, 5].mean():.2f}  steps {a[:, 6].mean():.1f}")
    print(f"gain {d.mean():+.2f} +- {1.96 * se:.2f} (95% CI)   W/T/L {w}/{n - w - l}/{l}  ({100 * w / n:.1f}% / {100 * (n - w - l) / n:.1f}% / {100 * l / n:.1f}%)")
