"""Fine-tune the 2-machine x 10-job MaskablePPO on ZoomMachineWrapper (clusters of 3-8 machines), exact subtree masks.

    JOB_USAGE_MODE=random .venv/bin/python scripts/machine_ae/train_zoom_ppo.py [steps=1500000] [out=models/zoom_ppo]

The policy sees `group` (2) machines at every zoom level (real or virtual), starts from the weights of the plain
2-machine model (run bsp1klua) and keeps the original reward: zoom steps give reward 0, a real step gives the usual
environment reward. 8 envs with 3, 4, 5, 6, 7, 8, 8, 8 real machines. Checkpoints every 250k agent steps.
env: AE (default models/machine_ae_k2.pt), INIT (default models/bsp1klua/final_model.zip), LR (default 2e-5)
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

N = 10
AE = os.environ.get("AE", os.path.join(ROOT, "models", "machine_ae_k2.pt"))
INIT = os.environ.get("INIT", os.path.join(ROOT, "models", "bsp1klua", "final_model.zip"))
MACHINES = [3, 4, 5, 6, 7, 8, 8, 8]


def make_env(m):
    def _make():
        from model import SetMachineAE
        from zoom_wrapper import ZoomMachineWrapper
        from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
        from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
        torch.set_num_threads(1)
        ck = torch.load(AE, map_location="cpu")
        ae = SetMachineAE(ck["k"], at_least_max=ck.get("at_least_max", False))
        ae.load_state_dict(ck["state"])
        cfg = {"n_machines": m, "n_jobs": N, "n_resource": 3, "n_time": 10, "max_capacity": 255}
        env = Monitor(FailureSkipTimeWrapper(SchedulingEnviorment(cfg, render_mode="none"), max_time=11 * N))
        return ZoomMachineWrapper(env, ae, exact_masks=True)          # outermost: exposes action_masks()
    return _make


if __name__ == "__main__":
    from scheduling_simulator.expiremnt.policy.maskable_sched import MaskableSchedulingPolicy
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "models", "zoom_ppo")
    os.makedirs(out, exist_ok=True)
    envs = SubprocVecEnv([make_env(m) for m in MACHINES])
    lr0 = float(os.environ.get("LR", "2e-5"))
    model = MaskablePPO(
        MaskableSchedulingPolicy, envs, learning_rate=lambda p: lr0 * p, n_steps=512, batch_size=256, n_epochs=5,
        gamma=0.995, ent_coef=0.01, target_kl=0.03, verbose=1, seed=3,
        policy_kwargs={"embedding_dim": 256, "n_machines": 2, "max_time": 110.0, "per_resource_fit": True,
                       "pair_features": True, "dynamic_machine": True})
    from stable_baselines3 import PPO
    state = PPO.load(INIT, device="cpu").policy.state_dict()
    print("init from", INIT, model.policy.load_state_dict(state, strict=False))
    model.learn(steps, callback=CheckpointCallback(250_000 // len(MACHINES), out, name_prefix="zoom_ppo"), progress_bar=False)
    model.save(os.path.join(out, "final_model"))
    envs.close()
