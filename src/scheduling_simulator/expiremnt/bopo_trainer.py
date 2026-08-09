"""BOPO-style preference training as an SB3 callback.

Collects SJF/random/learned trajectories, constructs preference pairs,
and adds a differentiable Bradley-Terry preference loss after PPO updates.
The preference loss shares the policy optimizer and acts as a regularizer.
"""
from __future__ import annotations

import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.utils import get_action_masks



def collect_trajectory(
    policy_name: str,
    model,
    env_fn,
    seed: int,
    n_jobs: int,
    max_n_jobs: int,
    rng,
):
    env = env_fn(n_jobs)
    obs, _ = env.reset(seed=seed)
    traj = []
    while True:
        mask = get_action_masks(env)
        alloc = np.flatnonzero(mask[1:]) + 1
        if not alloc.size:
            action = 0
        elif policy_name == 'sjf':
            jobs = (alloc - 1) % n_jobs
            action = int(alloc[np.argmin(obs['size'][jobs])])
        elif policy_name == 'random':
            action = int(rng.choice(alloc))
        elif policy_name == 'learned':
            action = int(model.predict(obs, deterministic=True, action_masks=mask)[0])
        else:
            action = 0
        traj.append((copy.deepcopy(obs), action, mask.copy()))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    flow = float((obs['finished_at'] - obs['arrival']).sum())
    env.close()
    return traj, flow


def preference_loss_batch(
    model,
    pairs: list[tuple[tuple, tuple, float, float]],
    device: th.device,
    temperature: float = 100.0,
) -> th.Tensor:
    """Compute differentiable Bradley-Terry loss over matching steps.

    pair = (better_traj, worse_traj, better_flow, worse_flow)
    For each pair, sample the same step index from both trajectories,
    compute log_prob(better) - log_prob(worse), and weight by flow diff.
    """
    total = th.zeros((), device=device)
    n = 0
    for better, worse, bf, wf in pairs:
        idx = np.random.randint(min(len(better), len(worse)))
        obs_b, act_b, mask_b = better[idx]
        obs_w, act_w, mask_w = worse[idx]

        obs_b_t, _ = model.policy.obs_to_tensor(obs_b)
        feat_b = model.policy.extract_features(obs_b_t)
        latent_b = model.policy.mlp_extractor.forward_actor(feat_b)
        dist_b = model.policy._get_action_dist_from_latent(latent_b)
        mask_b_t = th.as_tensor(mask_b, device=device, dtype=th.bool).unsqueeze(0)
        dist_b.apply_masking(mask_b_t)
        lp_b = dist_b.log_prob(th.as_tensor([act_b], device=device, dtype=th.long))

        obs_w_t, _ = model.policy.obs_to_tensor(obs_w)
        feat_w = model.policy.extract_features(obs_w_t)
        latent_w = model.policy.mlp_extractor.forward_actor(feat_w)
        dist_w = model.policy._get_action_dist_from_latent(latent_w)
        mask_w_t = th.as_tensor(mask_w, device=device, dtype=th.bool).unsqueeze(0)
        dist_w.apply_masking(mask_w_t)
        lp_w = dist_w.log_prob(th.as_tensor([act_w], device=device, dtype=th.long))

        margin = (lp_b - lp_w) * (wf - bf) / temperature
        total = total - F.logsigmoid(margin)
        n += 1
    return total / max(n, 1)


class BOPOCallback(BaseCallback):
    """Periodically inject preference loss into PPO training."""

    def __init__(
        self,
        env_fn,
        n_jobs_train: list[int],
        max_n_jobs: int,
        n_pairs: int = 60,
        pref_freq: int = 4,
        pref_lr: float = 1e-4,
        temperature: float = 100.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.n_jobs_train = n_jobs_train
        self.max_n_jobs = max_n_jobs
        self.n_pairs = n_pairs
        self.pref_freq = pref_freq
        self.pref_lr = pref_lr
        self.temperature = temperature
        self.pref_optimizer = None
        self._pref_count = 0

    def _on_step(self) -> bool:
        # Preference injection happens at rollout boundaries.
        return True

    def _on_training_start(self) -> None:
        self.pref_optimizer = th.optim.AdamW(
            self.model.policy.parameters(), lr=self.pref_lr, weight_decay=0.01
        )

    def _on_rollout_end(self) -> None:
        self._pref_count += 1
        if self._pref_count % self.pref_freq != 0:
            return

        device = self.model.device
        pairs = []
        for seed in range(self.n_pairs):
            n_jobs = int(np.random.choice(self.n_jobs_train))
            sjf_traj, sf = collect_trajectory(
                'sjf', self.model, self.env_fn, seed, n_jobs, self.max_n_jobs,
                np.random.default_rng(seed)
            )
            rand_traj, rf = collect_trajectory(
                'random', self.model, self.env_fn, seed, n_jobs, self.max_n_jobs,
                np.random.default_rng(seed + 1000)
            )
            learned_traj, lf = collect_trajectory(
                'learned', self.model, self.env_fn, seed, n_jobs, self.max_n_jobs,
                np.random.default_rng(seed + 2000)
            )
            if sf < rf:
                pairs.append((sjf_traj, rand_traj, sf, rf))
            if sf < lf:
                pairs.append((sjf_traj, learned_traj, sf, lf))
            if lf < rf:
                pairs.append((learned_traj, rand_traj, lf, rf))

        if not pairs:
            return

        self.pref_optimizer.zero_grad()
        loss = preference_loss_batch(self.model, pairs, device, self.temperature)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
        self.pref_optimizer.step()
        if self.verbose:
            print(f"BOPO loss: {float(loss.detach()):.4f} pairs={len(pairs)}")
