"""PPO trained on ONE machine x 10 jobs, run on a K-machine cluster through the machine autoencoder.

Every decision:
  1. fold: the K machines' free grids -> the autoencoder -> ONE virtual-machine view (order of machines irrelevant);
  2. choose the job: the 1-machine model sees the view (not any real machine) and the jobs through the
     knock-out tournament (slices of 10, padded; only jobs that fit some real machine can win; skip may win the
     final slice). The legality mask is built from the real machines, not from the model's own mask;
  3. zoom in: the chosen job is offered to each REAL machine it fits (the model sees that machine and the same
     final slice); the machine with the highest logit for the job takes it.

    N_MACHINES=4 N_JOBS=30 JOB_USAGE_MODE=random .venv/bin/python scripts/machine_ae/eval_pipeline.py 1000 8 /tmp/ae_pipe_m4_n30.npz
env: SKIP=1 (let the model skip), VIEW=ae|max, ZOOM=model|first (ablations); AE (default models/machine_ae_k<M>.pt), MODEL (default models/ncpb2fyb/final_model.zip, the 1 x 10 model)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M, N = int(os.environ.get("N_MACHINES", "4")), int(os.environ.get("N_JOBS", "30"))
AE = os.environ.get("AE", os.path.join(ROOT, "models", f"machine_ae_k{M}.pt"))
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "ncpb2fyb", "final_model.zip"))
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}
COMPLETED, _NEG = 3, -1e8
VIEW = os.environ.get("VIEW", "ae")       # ae | max (ablation)
ZOOM = os.environ.get("ZOOM", "model")    # model | first (ablation: lowest-index machine that fits)
SKIP = os.environ.get("SKIP", "0") == "1"  # 0: never wait while a job fits some real machine (the 1-machine skip head means "my one machine is busy", which does not carry over to K machines)
_policy = _ae = _slots = None


def _load():
    global _policy, _ae, _slots
    import torch
    from stable_baselines3 import PPO
    from model import SetMachineAE
    torch.set_num_threads(1)
    _policy = PPO.load(MODEL, device="cpu").policy
    _policy.set_training_mode(False)
    _slots = int(_policy.observation_space["status"].shape[0])
    ck = torch.load(AE, map_location="cpu")
    _ae = SetMachineAE(ck["k"], at_least_max=ck.get("at_least_max", False))
    _ae.load_state_dict(ck["state"])
    _ae.eval()


def _view(obs):
    """K machines -> one virtual machine (usage grid in the env's integer units, capacity unchanged)."""
    import torch as th
    free = th.tensor((obs["machines_capacity"] - obs["machines_usage"]) / 255.0, dtype=th.float32)[None]
    if VIEW == "max":                                       # ablation: the plain elementwise max, no autoencoder
        v = free[0].max(0).values.numpy()
    else:
        with th.no_grad():
            v = _ae.encode(free)[0].numpy()
    usage = (obs["machines_capacity"][0] - np.round(v * 255.0)).clip(0, 255).astype(obs["machines_usage"].dtype)
    return usage[None]


def _slice(obs, usage, cap, jobs):
    k = len(jobs)
    out = {"machines_usage": usage, "machines_capacity": cap, "time": obs["time"], "action_success": int(obs["action_success"])}
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


def _logits(batch):
    """Raw action logits [skip, job slot 0..9] of the 1-machine model, no policy mask."""
    import torch as th
    stacked = {k: np.stack([np.asarray(b[k]) for b in batch]) for k in batch[0]}
    with th.no_grad():
        feats = _policy.extract_features(_policy.obs_to_tensor(stacked)[0])
        return _policy.action_net(_policy.mlp_extractor.forward_actor(feats))


def choose(obs):
    import torch as th
    from scheduling_simulator.scheduler.abc_scheduler import Scheduler
    options = Scheduler.options(obs)                         # (job, machine) pairs that are legal on the real machines
    if not options:
        return 0
    legal = {j for j, _ in options}
    cap1 = obs["machines_capacity"][:1]
    view = _view(obs)
    cands = list(range(N)) if N <= _slots else [j for j in range(N) if obs["status"][j] != COMPLETED]
    while True:
        chunks = [cands[i:i + _slots] for i in range(0, len(cands), _slots)]
        final = len(chunks) == 1
        lg = _logits([_slice(obs, view, cap1, c) for c in chunks])
        winners = []
        for i, c in enumerate(chunks):
            keep = th.zeros(lg.shape[1], dtype=th.bool)
            for s, j in enumerate(c):
                keep[1 + s] = j in legal
            keep[0] = final and SKIP
            row = lg[i].masked_fill(~keep, _NEG)
            a = int(row.argmax())
            winners.append(None if (a == 0 and final) else (c[a - 1] if row[a] > _NEG / 10 else -1))
        if final:
            job = winners[0]
            break
        cands = [w for w in winners if w is not None and w != -1]
    if job is None or job == -1:
        return 0                                            # the model prefers to wait
    ms = sorted(m for j, m in options if j == job)
    if len(ms) == 1 or ZOOM == "first":
        return 1 + ms[0] * N + job
    slot = chunks[0].index(job)
    lg = _logits([_slice(obs, obs["machines_usage"][m:m + 1], obs["machines_capacity"][m:m + 1], chunks[0]) for m in ms])
    return 1 + ms[int(lg[:, 1 + slot].argmax())] * N + job


def run(ep):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    if _policy is None:
        _load()
    out = []
    for who in ("ae", "sjf"):
        env = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)
        obs, _ = env.reset(seed=42 + ep)
        sj = ShortestJobScheduler()
        tot, done, steps = 0.0, False, 0
        while not done:
            if who == "ae":
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
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    se = d.std(ddof=1) / np.sqrt(n)
    print(f"autoencoder pipeline  M={M} N={N} episodes={n}")
    print(f"pipeline: reward {a[:, 1].mean():.2f}  completion {a[:, 2].mean():.2f}\nSJF     : reward {a[:, 4].mean():.2f}  completion {a[:, 5].mean():.2f}")
    print(f"gain {d.mean():+.2f} +- {1.96 * se:.2f}   W/T/L {w}/{n - w - l}/{l}  ({100 * w / n:.1f}% / {100 * (n - w - l) / n:.1f}% / {100 * l / n:.1f}%)")
