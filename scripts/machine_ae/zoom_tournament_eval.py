"""Machine zoom + job tournament: a 2-machine x 10-job policy schedules M machines x N jobs.

Machines: ZoomMachineWrapper (AE folds the machines into a tree, the agent sees 2 machines at every zoom level).
Jobs: at every zoom level the pick is a knock-out tournament over slices of 10 jobs (padded; round 1 slices all
unfinished jobs, only legal (machine, job) pairs under the wrapper's exact masks can win, skip may win the final
slice and is then a real skip). The winner (machine, job) is played in the wrapper: a virtual machine zooms into
its children and the tournament runs again on the new view, a real machine starts the job in the original env.

    N_MACHINES=4 N_JOBS=30 JOB_USAGE_MODE=random .venv/bin/python scripts/machine_ae/zoom_tournament_eval.py 5000 8 /tmp/zt/m4_n30.npz
env: MODEL (default models/zoom_ppo/final_model.zip), AE (default models/machine_ae_k2.pt). npz layout = tournament_eval.py.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M, N = int(os.environ.get("N_MACHINES", "4")), int(os.environ.get("N_JOBS", "30"))
AE = os.environ.get("AE", os.path.join(ROOT, "models", "machine_ae_k2.pt"))
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "zoom_ppo", "final_model.zip"))
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}
COMPLETED, _NEG = 3, -1e8
_model = _ae = _slots = None


def _load():
    global _model, _ae, _slots
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    from model import SetMachineAE
    torch.set_num_threads(1)
    _model = (MaskablePPO if "zoom_ppo" in MODEL else PPO).load(MODEL, device="cpu")
    _slots = int(_model.observation_space["status"].shape[0])
    ck = torch.load(AE, map_location="cpu")
    _ae = SetMachineAE(ck["k"], at_least_max=ck.get("at_least_max", False))
    _ae.load_state_dict(ck["state"])


def _slice(obs, jobs):
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


def choose(obs, env):
    """Action in the wrapper's space (0 skip, 1 + slot * N + job) from the job tournament on the machines on screen."""
    import torch as th
    legal = env.action_masks()[1:].reshape(env.g, N)              # exact: legal[slot, job]
    if not legal.any():
        return 0
    pol = _model.policy
    cands = list(range(N)) if N <= _slots else [j for j in range(N) if obs["status"][j] != COMPLETED]
    while True:
        chunks = [cands[i:i + _slots] for i in range(0, len(cands), _slots)]
        final = len(chunks) == 1
        batch = [_slice(obs, c) for c in chunks]
        stacked = {k: np.stack([np.asarray(b[k]) for b in batch]) for k in batch[0]}
        with th.no_grad():
            lg = pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(pol.obs_to_tensor(stacked)[0])))
        winners = []
        for i, c in enumerate(chunks):
            keep = th.zeros(lg.shape[1], dtype=th.bool)
            for slot in range(env.g):
                for s, j in enumerate(c):
                    keep[1 + slot * _slots + s] = bool(legal[slot, j])
            keep[0] = final
            row = lg[i].masked_fill(~keep, _NEG)
            a = int(row.argmax())
            if float(row[a]) <= _NEG / 10:
                winners.append(-1)                                 # nothing legal in this slice
            elif a == 0:
                winners.append(None)                               # skip (final slice only)
            else:
                winners.append((c[(a - 1) % _slots], (a - 1) // _slots))
        if final:
            w = winners[0]
            return 0 if w in (None, -1) else 1 + w[1] * N + w[0]
        cands = [w[0] for w in winners if w not in (None, -1)]


def run(ep):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    from zoom_wrapper import ZoomMachineWrapper
    if _model is None:
        _load()
    out = []
    for who in ("zoom", "sjf"):
        base = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)
        env = ZoomMachineWrapper(base, _ae, exact_masks=True) if who == "zoom" else base
        obs, _ = env.reset(seed=42 + ep)
        sj = ShortestJobScheduler()
        tot, done, real = 0.0, False, 0
        while not done:
            if who == "zoom":
                a = choose(obs, env)
            else:
                skip, j, m = sj.select(obs)
                a = 0 if skip else 1 + m * N + j
            obs, r, te, tr, info = env.step(a)
            tot += float(r); done = te or tr; real += int(info.get("real_step", True))
        w = np.asarray(env._obs["wait_time"] if who == "zoom" else obs["wait_time"], float)
        sz = np.asarray(env._obs["size"] if who == "zoom" else obs["size"], float)
        out += [tot, (w + sz).mean(), real]
    return (ep, *out)


if __name__ == "__main__":
    n, workers, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(workers, initializer=_load) as p:
        a = np.array(sorted(p.map(run, range(n), chunksize=5)))
    np.savez(path, episode=a[:, 0].astype(int), t_reward=a[:, 1], t_completion=a[:, 2], t_steps=a[:, 3],
             s_reward=a[:, 4], s_completion=a[:, 5], s_steps=a[:, 6])
    d = a[:, 1] - a[:, 4]
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    print(f"zoom + tournament  M={M} N={N} episodes={n}")
    print(f"zoom: reward {a[:, 1].mean():.2f}  completion {a[:, 2].mean():.2f}")
    print(f"SJF : reward {a[:, 4].mean():.2f}  completion {a[:, 5].mean():.2f}")
    print(f"gain {d.mean():+.2f} +- {1.96 * d.std(ddof=1) / np.sqrt(n):.2f}   W/T/L {w}/{n - w - l}/{l}  ({100 * w / n:.1f}% / {100 * (n - w - l) / n:.1f}% / {100 * l / n:.1f}%)")
