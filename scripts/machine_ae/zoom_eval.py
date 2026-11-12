"""PPO trained on `group` machines x 10 jobs, run on M machines x 10 jobs through ZoomMachineWrapper, vs SJF.

    N_MACHINES=8 .venv/bin/python scripts/machine_ae/zoom_eval.py 1000 8 /tmp/zoom_m8.npz
env: MODEL=models/zoom_ppo/final_model.zip evaluates the fine-tuned model (use EXACT=1); EXACT=1 (exact subtree masks, no dead ends), AE (default models/machine_ae_k2.pt), MODEL (default models/bsp1klua/final_model.zip, the 2 x 10 model)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M, N = int(os.environ.get("N_MACHINES", "8")), 10
AE = os.environ.get("AE", os.path.join(ROOT, "models", "machine_ae_k2.pt"))
MODEL = os.environ.get("MODEL", os.path.join(ROOT, "models", "bsp1klua", "final_model.zip"))
CFG = {"n_machines": M, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}
EXACT = os.environ.get("EXACT") == "1"
_model = _ae = None


def _load():
    global _model, _ae
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    from model import SetMachineAE
    torch.set_num_threads(1)
    _model = (MaskablePPO if "zoom_ppo" in MODEL else PPO).load(MODEL, device="cpu")
    ck = torch.load(AE, map_location="cpu")
    _ae = SetMachineAE(ck["k"], at_least_max=ck.get("at_least_max", False))
    _ae.load_state_dict(ck["state"])


def _act(obs, env):
    """Greedy action of the policy with the wrapper's exact masks instead of the policy's own mask."""
    import torch as th
    pol = _model.policy
    with th.no_grad():
        feats = pol.extract_features(pol.obs_to_tensor({k: np.asarray(v)[None] for k, v in obs.items()})[0])
        logits = pol.action_net(pol.mlp_extractor.forward_actor(feats))[0]
    return int(logits.masked_fill(th.tensor(~env.action_masks()), -1e8).argmax())


def _env():
    from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
    from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
    return FailureSkipTimeWrapper(SchedulingEnviorment(CFG, render_mode="none"), max_time=11 * N)


def run(ep):
    from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler
    from zoom_wrapper import ZoomMachineWrapper
    if _model is None:
        _load()
    out = []
    for who in ("zoom", "sjf"):
        env = ZoomMachineWrapper(_env(), _ae, exact_masks=EXACT) if who == "zoom" else _env()
        obs, _ = env.reset(seed=42 + ep)
        sj = ShortestJobScheduler()
        tot, done, real, agent = 0.0, False, 0, 0
        while not done:
            if who == "zoom":
                a = _act(obs, env) if EXACT else int(_model.predict(obs, deterministic=True)[0])
            else:
                skip, j, m = sj.select(obs)
                a = 0 if skip else 1 + m * N + j
            obs, r, te, tr, info = env.step(a)
            tot += float(r); done = te or tr; agent += 1; real += int(info.get("real_step", True))
        w = np.asarray(obs["wait_time"], float)
        out += [tot, (w + np.asarray(obs["size"], float)).mean(), real, agent]
    return (ep, *out)


if __name__ == "__main__":
    n, workers, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(workers, initializer=_load) as p:
        a = np.array(sorted(p.map(run, range(n), chunksize=5)))
    np.savez(path, z_reward=a[:, 1], z_completion=a[:, 2], z_real=a[:, 3], z_agent=a[:, 4], s_reward=a[:, 5], s_completion=a[:, 6])
    d = a[:, 1] - a[:, 5]
    w, l = int((d > 1e-3).sum()), int((d < -1e-3).sum())
    print(f"zoom wrapper  M={M} N={N} episodes={n}   agent steps per real step {a[:, 4].sum() / a[:, 3].sum():.2f}")
    print(f"zoom: reward {a[:, 1].mean():.2f}  completion {a[:, 2].mean():.2f}\nSJF : reward {a[:, 5].mean():.2f}  completion {a[:, 6].mean():.2f}")
    print(f"gain {d.mean():+.2f} +- {1.96 * d.std(ddof=1) / np.sqrt(n):.2f}   W/T/L {w}/{n - w - l}/{l}  ({100 * w / n:.1f}% / {100 * (n - w - l) / n:.1f}% / {100 * l / n:.1f}%)")
