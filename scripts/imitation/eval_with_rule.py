"""Evaluate saved masked-PPO checkpoints WITH the forbid_idle_skip rule switched on at inference time
(no retraining) on seeds 42.., recording reward and episode length.

    .venv/bin/python scripts/imitation/eval_with_rule.py models/dc1okvl3/best_model.zip 5000 /tmp/rule_dc1okvl3.npz 4 [on|off|saved]
  rule = on (force forbid_idle_skip), off (force it off) or saved (as the checkpoint was trained).
"""
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
CFG = {"n_machines": 1, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}
_model = None
_RULE = "on"


def _init(path, rule):
    global _model, _RULE
    _RULE = rule
    import torch as th
    th.set_num_threads(1)
    from stable_baselines3 import PPO
    _model = PPO.load(path, device="cpu")
    if _RULE == "on":
        _model.policy.forbid_idle_skip = True
    elif _RULE == "off":
        _model.policy.forbid_idle_skip = False
    # "saved": keep whatever the checkpoint was trained with


def run(eps):
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    out = []
    for ep in eps:
        env = FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=110)
        obs, _ = env.reset(seed=42 + ep)
        tot, n, done = 0.0, 0, False
        while not done:
            a, _ = _model.predict({k: np.asarray(v) for k, v in obs.items()}, deterministic=True)
            obs, r, te, tr, _ = env.step(int(a))
            tot += float(r); n += 1; done = te or tr
        out.append((ep, tot, n))
    return out


if __name__ == "__main__":
    path, n, out, workers = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    rule = sys.argv[5] if len(sys.argv) > 5 else "on"          # on | off | saved
    chunks = [list(range(i, min(i + 25, n))) for i in range(0, n, 25)]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(path, rule)) as pool:
        rows = sorted(r for c in pool.imap(run, chunks) for r in c)
    a = np.array(rows)
    np.savez(out, episode=a[:, 0].astype(int), reward=a[:, 1], length=a[:, 2].astype(int))
    print(f"{path}: mean reward {a[:, 1].mean():.2f}, timeouts {(a[:, 2] >= 110).sum()}, mean length {a[:, 2].mean():.1f}")
