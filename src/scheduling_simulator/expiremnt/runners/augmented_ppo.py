"""PPO with job-position augmentation.

The action is `0 = skip`, `a >= 1 = pick job a - 1`. Shuffling the order of the
jobs in an observation and moving the chosen action to the job's new position
gives the same decision in a different layout. Every rollout sample is trained
on `n_aug` cyclic shifts of the job order, each with the original action's
advantage, return and old log-prob, so the policy is pushed to give the same
answer whatever slot a job sits in.
"""
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

# observation entries with one row per job (axis 1 after the batch axis)
JOB_KEYS = (
    "arrival",
    "finished_at",
    "jobs_usage",
    "scheduled_at",
    "size",
    "status",
    "ttl",
    "wait_time",
)


class PermutationAugmentedPPO(PPO):
    def __init__(self, *args, n_aug: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_aug = n_aug

    def _augment(self, obs: dict, actions: th.Tensor, n_jobs: int):
        """Return `n_aug` cyclic shifts of the job order, concatenated."""
        obs_out = {k: [] for k in obs}
        act_out = []
        for shift in range(self.n_aug):
            perm = th.roll(th.arange(n_jobs, device=actions.device), shift)
            inv = th.argsort(perm)  # old job j now sits at position inv[j]
            for k, v in obs.items():
                obs_out[k].append(v[:, perm] if k in JOB_KEYS else v)
            act_out.append(
                th.where(actions == 0, actions, inv[(actions - 1).clamp(min=0)] + 1)
            )
        return (
            {k: th.cat(v, dim=0) for k, v in obs_out.items()},
            th.cat(act_out, dim=0),
        )

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                n_jobs = rollout_data.observations["status"].shape[1]
                obs, actions = self._augment(rollout_data.observations, actions, n_jobs)
                rep = self.n_aug
                advantages = advantages.repeat(rep)
                returns = rollout_data.returns.repeat(rep)
                old_log_prob = rollout_data.old_log_prob.repeat(rep)

                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                values = values.flatten()

                ratio = th.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                value_loss = F.mse_loss(returns, values)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_divs.append(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
