"""Pretrain the actor-critic on the optimal-solver demonstrations, then test it.

Actor: cross-entropy between the policy's 11 logits and the optimal action.
Critic: mean squared error against the discounted return-to-go along the optimal
trajectory (scaled by 1/100). Saves the policy weights for PPO fine-tuning
(ExperimentRunner(init_policy_path=...)).

Run from the repo root:
    .venv/bin/python scripts/imitation/pretrain_policy.py /tmp/optimal_dataset.npz models/bc_policy.pt 12
"""
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import wandb_log as wl
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.expiremnt.policy.new_sched import NewSchedulingPolicy
from scheduling_simulator.expiremnt.runners.train_runner import ExperimentRunner, get_auto_device
from scheduling_simulator.scheduler.sjf_scheduler import ShortestJobScheduler

N_MACHINES = int(os.environ.get("N_MACHINES", "1"))
CONFIG = {"n_machines": N_MACHINES, "n_jobs": 10, "n_resource": 3, "n_time": 10, "max_capacity": 255}
POLICY_KWARGS = {
    "embedding_dim": 256, "n_machines": N_MACHINES, "max_time": 110.0,
    "per_resource_fit": True, "pair_features": True, "dynamic_machine": True,
}
KEYS = ("machines_usage", "machines_capacity", "jobs_usage", "status", "ttl", "arrival",
        "wait_time", "scheduled_at", "finished_at", "size", "time", "action_success")


def build_policy():
    runner = ExperimentRunner(CONFIG, train_steps=1, evalution_steps=1, policy_kwargs={}, max_time=110, run_with_wandb=False)
    env = runner.generate_enviroemnt("/tmp/bc_env", n_env=1)
    device = get_auto_device()
    model = PPO(NewSchedulingPolicy, env, policy_kwargs=POLICY_KWARGS, device=device, n_steps=8, batch_size=8)
    return model, env, device


def batch_obs(policy, data, idx):
    obs = {k: data[k][idx].astype(np.float32) if k not in ("status", "action_success") else data[k][idx].astype(np.int64) for k in KEYS}
    t, _ = policy.obs_to_tensor(obs)
    return t


def test_policy(policy, n_episodes=300, seed=42):
    """Deterministic policy vs SJF on the first evaluation job sets (original reward)."""
    sjf, n = ShortestJobScheduler(), CONFIG["n_jobs"]
    pol_r, sjf_r, pol_fail = [], [], 0
    for ep in range(n_episodes):
        for who in ("policy", "sjf"):
            env = FailureSkipTimeWrapper(SchedulingEnviorment(CONFIG, render_mode="none"), max_time=110)
            obs, _ = env.reset(seed=seed + ep)
            total, done = 0.0, False
            while not done:
                if who == "policy":
                    action, _ = policy.predict({k: np.asarray(v) for k, v in obs.items()}, deterministic=True)
                    action = int(action)
                else:
                    skip, j, k = sjf.select(obs)
                    action = 0 if skip else 1 + k * n + j
                obs, r, term, trunc, _ = env.step(action)
                total += float(r); done = term or trunc
                if who == "policy" and action != 0 and not bool(obs["action_success"]):
                    pol_fail += 1
            (pol_r if who == "policy" else sjf_r).append(total)
    pol_r, sjf_r = np.array(pol_r), np.array(sjf_r)
    d = pol_r - sjf_r
    print(f"test on {n_episodes} evaluation job sets: policy {pol_r.mean():.2f}  SJF {sjf_r.mean():.2f}  "
          f"margin {(pol_r.mean() - sjf_r.mean()) / abs(sjf_r.mean()) * 100:+.2f}%  paired diff {d.mean():+.2f} "
          f"(95% CI {d.mean() - 1.96 * d.std() / np.sqrt(n_episodes):+.2f} to {d.mean() + 1.96 * d.std() / np.sqrt(n_episodes):+.2f})  "
          f"wins/losses/ties {(d > 1e-9).sum()}/{(d < -1e-9).sum()}/{(abs(d) <= 1e-9).sum()}  failed starts/episode {pol_fail / n_episodes:.3f}", flush=True)
    return pol_r, sjf_r


def main(path, out_path, epochs=12, batch=512, lr=3e-4):
    data = dict(np.load(path))
    N = len(data["action"])
    # hold out the last 5% of the episodes' samples for a validation accuracy
    n_val = N // 20
    train_idx, val_idx = np.arange(N - n_val), np.arange(N - n_val, N)
    run = wl.start(f"bc-pretrain-{N_MACHINES}M-3R", {"n_machines": N_MACHINES, "samples": N, "epochs": epochs, "batch": batch, "lr": lr, "dataset": path}, ["imitation", "behaviour-cloning"])
    model, env, device = build_policy()
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=lr)
    actions = th.as_tensor(data["action"], device=device)
    returns = th.as_tensor(data["ret"] / 100.0, device=device)
    print(f"samples {N} (train {len(train_idx)}, validation {n_val}); device {device}", flush=True)

    def evaluate(idx):
        policy.set_training_mode(False)
        correct, vloss, n, start_n, start_ok = 0, 0.0, 0, 0, 0
        with th.no_grad():
            for s in range(0, len(idx), 2048):
                b = idx[s:s + 2048]
                obs = batch_obs(policy, data, b)
                feats = policy.extract_features(obs)
                logits = policy.action_net(policy.mlp_extractor.forward_actor(feats))
                v = policy.value_net(policy.mlp_extractor.forward_critic(feats)).squeeze(-1)
                pred = logits.argmax(-1)
                correct += int((pred == actions[b]).sum()); n += len(b)
                is_start = actions[b] > 0
                start_n += int(is_start.sum()); start_ok += int(((pred == actions[b]) & is_start).sum())
                vloss += float(F.mse_loss(v, returns[b], reduction="sum"))
        return correct / n, vloss / n, start_ok / max(start_n, 1)

    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        policy.set_training_mode(True)
        perm = rng.permutation(train_idx); t0 = time.time(); tot = 0.0; k = 0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            obs = batch_obs(policy, data, b)
            feats = policy.extract_features(obs)
            logits = policy.action_net(policy.mlp_extractor.forward_actor(feats))
            v = policy.value_net(policy.mlp_extractor.forward_critic(feats)).squeeze(-1)
            loss = F.cross_entropy(logits, actions[b]) + 0.5 * F.mse_loss(v, returns[b])
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); tot += float(loss); k += 1
        acc, vl, start_acc = evaluate(val_idx)
        print(f"epoch {epoch + 1}/{epochs}: train loss {tot / k:.4f}  validation accuracy {acc * 100:.2f}% (start actions only: {start_acc * 100:.2f}%)  critic MSE {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        wl.log(run, {"epoch": epoch + 1, "train/loss": tot / k, "train/val_accuracy": acc, "train/val_start_accuracy": start_acc, "train/critic_mse": vl, "train/epoch_seconds": time.time() - t0})
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        th.save(policy.state_dict(), out_path)
    policy.set_training_mode(False)
    pol, sjf = test_policy(policy)
    if run is not None:
        for k, v in wl.summarize(pol, sjf).items():
            run.summary["test300_" + k] = v
    wl.upload(run, out_path, "model")
    wl.finish(run)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
