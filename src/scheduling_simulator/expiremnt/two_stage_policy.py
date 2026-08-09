"""Two-stage pointer policy for variable-size job scheduling.

Unlike a flat Discrete action over all (machine, job) pairs, this policy
samples a job (or skip) from the contextualized job embeddings, then
samples a machine for that job from machine embeddings conditioned on
the selected job. This is the structure used by Decima and scales to
variable candidate sets without padding the action space.
"""

from __future__ import annotations


import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment



class TwoStagePointerNet(nn.Module):
    """Actor network for the two-stage pointer policy.

    Stage 1: skip logit plus one logit per job.
    Stage 2: one logit per machine, conditioned on the selected job.
    Invalid slots are masked. The same parameters work for any job count.
    """

    def __init__(
        self,
        n_machines: int,
        n_resources: int,
        n_time: int,
        max_n_jobs: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_machines = n_machines
        self.n_resources = n_resources
        self.n_time = n_time
        self.max_n_jobs = max_n_jobs
        self.embedding_dim = embedding_dim

        job_input_dim = n_resources * n_time + 4 + 7
        machine_input_dim = 2 * n_resources * n_time

        self.job_encoder = nn.Sequential(
            nn.Linear(job_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        self.job_attention = nn.MultiheadAttention(
            embedding_dim, 4, batch_first=True
        )
        self.job_norm = nn.LayerNorm(embedding_dim)
        self.job_attention2 = nn.MultiheadAttention(
            embedding_dim, 4, batch_first=True
        )
        self.job_norm2 = nn.LayerNorm(embedding_dim)

        # Job selection: attended job embedding -> scalar
        self.job_scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.skip_scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        # Machine selection: concat(selected job emb, machine emb, dot)
        self.machine_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(3 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.time_scale = float(max(n_time, 1))

    def _features(self, obs: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        device = next(self.parameters()).device

        machine_usage = obs['machines_usage'].float().to(device) / 255.0
        machine_capacity = obs['machines_capacity'].float().to(device) / 255.0
        jobs_usage = obs['jobs_usage'].float().to(device) / 255.0
        status = obs['status'].to(device).long()

        is_real = (
            (status != 3)
            | (obs['ttl'].float().to(device) != 0)
            | (obs['size'].float().to(device) != 0)
        )
        key_padding = ~is_real

        status_oh = F.one_hot(status.clamp(0, 3), num_classes=4).float().to(device)
        relative_arrival = (
            obs['arrival'].float().to(device)
            - obs['time'].float().to(device).squeeze(-1).unsqueeze(-1)
        ) / self.time_scale

        job_meta = th.stack(
            [
                obs['ttl'].float().to(device) / self.time_scale,
                relative_arrival,
                obs['wait_time'].float().to(device) / self.time_scale,
                obs['scheduled_at'].float().to(device) / self.time_scale,
                obs['finished_at'].float().to(device) / self.time_scale,
                obs['size'].float().to(device) / self.time_scale,
                is_real.float().to(device),
            ],
            dim=-1,
        )
        job_features = th.cat(
            [jobs_usage.flatten(start_dim=2), status_oh, job_meta], dim=-1
        )
        machine_features = th.cat(
            [machine_usage.flatten(start_dim=2), machine_capacity.flatten(start_dim=2)],
            dim=-1,
        )
        return job_features, machine_features, key_padding, is_real

    def _encode(
        self, obs: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        job_features, machine_features, key_padding, is_real = self._features(obs)
        job_emb = self.job_encoder(job_features)
        machine_emb = self.machine_encoder(machine_features)

        attended, _ = self.job_attention(
            job_emb, job_emb, job_emb, key_padding_mask=key_padding
        )
        attended = self.job_norm(job_emb + attended)
        attended2, _ = self.job_attention2(
            attended, attended, attended, key_padding_mask=key_padding
        )
        attended = self.job_norm2(attended + attended2)

        return attended, machine_emb, is_real

    # --- Log-prob of selected actions ---
    def log_probs(
        self,
        obs: dict[str, th.Tensor],
        job_actions: th.Tensor,
        machine_actions: th.Tensor,
        job_masks: th.Tensor,
        machine_masks: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        attended, machine_emb, is_real = self._encode(obs)
        batch, max_j, _ = attended.shape
        n_m = machine_emb.shape[1]

        # Job logits
        job_logits = self.job_scorer(attended).squeeze(-1)  # (B, max_j)
        job_logits = job_logits.masked_fill(~is_real, -1e9)
        # Skip logit from pooled context
        real_mask = is_real.unsqueeze(-1).float()
        sum_emb = (attended * real_mask).sum(dim=1)
        count = real_mask.sum(dim=1).clamp(min=1)
        pooled = sum_emb / count
        skip_logit = self.skip_scorer(pooled).squeeze(-1)  # (B,)
        job_logits = th.cat([skip_logit.unsqueeze(-1), job_logits], dim=-1)

        # Job mask: first column = skip, remaining = job slots
        job_logits = job_logits.masked_fill(~job_masks, -1e9)
        job_log_probs = F.log_softmax(job_logits, dim=-1)

        # Gather selected job log-prob
        gathered = job_log_probs.gather(1, job_actions.unsqueeze(-1)).squeeze(-1)

        # Machine logits: condition on selected job
        selected_job_emb = attended.gather(
            1,
            (job_actions - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, attended.shape[-1]),
        ).squeeze(1)
        # For skip actions, use pooled context
        selected_job_emb = th.where(
            (job_actions == 0).view(-1, 1, 1),
            pooled.unsqueeze(1),
            selected_job_emb.unsqueeze(1),
        ).squeeze(1)

        m_exp = machine_emb
        j_exp = selected_job_emb.unsqueeze(1).expand(-1, n_m, -1)
        machine_logits = self.machine_scorer(
            th.cat([m_exp, j_exp, m_exp * j_exp], dim=-1)
        ).squeeze(-1)
        machine_logits = machine_logits.masked_fill(~machine_masks, -1e9)
        machine_log_probs = F.log_softmax(machine_logits, dim=-1)
        gathered_machine = machine_log_probs.gather(
            1, machine_actions.unsqueeze(-1)
        ).squeeze(-1)

        # Entropy
        entropy = -(job_log_probs * job_log_probs.exp()).sum(-1).mean()
        return gathered, gathered_machine, entropy

    # --- Sampling for rollouts ---
    def sample_actions(
        self,
        obs: dict[str, th.Tensor],
        job_masks: th.Tensor,
        machine_masks: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        attended, machine_emb, is_real = self._encode(obs)
        n_m = machine_emb.shape[1]

        real_mask = is_real.unsqueeze(-1).float()
        sum_emb = (attended * real_mask).sum(dim=1)
        count = real_mask.sum(dim=1).clamp(min=1)
        pooled = sum_emb / count

        job_logits = self.job_scorer(attended).squeeze(-1).masked_fill(~is_real, -1e9)
        skip_logit = self.skip_scorer(pooled).squeeze(-1)
        job_logits = th.cat([skip_logit.unsqueeze(-1), job_logits], dim=-1)
        job_logits = job_logits.masked_fill(~job_masks, -1e9)

        if deterministic:
            job_actions = job_logits.argmax(dim=-1)
        else:
            job_probs = F.softmax(job_logits, dim=-1).clamp(min=1e-8)
            job_actions = th.multinomial(job_probs.reshape(-1, job_probs.shape[-1]), 1).squeeze(-1)

        # Select machine
        selected_job_emb = attended.gather(
            1,
            (job_actions - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, attended.shape[-1]),
        ).squeeze(1)
        selected_job_emb = th.where(
            (job_actions == 0).view(-1, 1, 1),
            pooled.unsqueeze(1),
            selected_job_emb.unsqueeze(1),
        ).squeeze(1)

        m_exp = machine_emb
        j_exp = selected_job_emb.unsqueeze(1).expand(-1, n_m, -1)
        machine_logits = self.machine_scorer(
            th.cat([m_exp, j_exp, m_exp * j_exp], dim=-1)
        ).squeeze(-1)
        machine_logits = machine_logits.masked_fill(~machine_masks, -1e9)

        if deterministic:
            machine_actions = machine_logits.argmax(dim=-1)
        else:
            machine_probs = F.softmax(machine_logits, dim=-1).clamp(min=1e-8)
            machine_actions = th.multinomial(machine_probs.reshape(-1, machine_probs.shape[-1]), 1).squeeze(-1)

        return job_actions, machine_actions, job_logits, machine_logits

    def value(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        attended, machine_emb, is_real = self._encode(obs)
        real_mask = is_real.unsqueeze(-1).float()
        sum_emb = (attended * real_mask).sum(dim=1)
        count = real_mask.sum(dim=1).clamp(min=1)
        pooled_jobs = sum_emb / count
        pooled_machines = machine_emb.mean(dim=1)
        state = th.cat([pooled_jobs, pooled_machines, pooled_jobs * pooled_machines], dim=-1)
        return self.value_net(state)


class TwoStagePPOAgent:
    """Lightweight PPO training loop for the two-stage pointer policy.

    Kept intentionally compact: collects rollouts, computes generalized
    advantage estimates, and applies clipped policy/value updates.
    """

    def __init__(
        self,
        env: SchedulingEnviorment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        seed: int = 0,
    ) -> None:
        config = env._config
        self.env = env
        self.n_machines = config['n_machines']
        self.n_jobs_max = config['n_jobs'] if env._max_n_jobs is None else env._max_n_jobs
        self.n_resources = config['n_resource']
        self.n_time = config['n_time']
        self.device = th.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)

        self.net = TwoStagePointerNet(
            self.n_machines, self.n_resources, self.n_time, self.n_jobs_max
        ).to(self.device)
        self.optimizer = th.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=0.01
        )
        self.num_steps = 0

    def _obs_to_tensor(self, obs: dict) -> dict[str, th.Tensor]:
        return {
            k: th.as_tensor(np.asarray(v), dtype=th.float32).unsqueeze(0).to(self.device)
            for k, v in obs.items()
        }

    def _masks_to_tensor(self, mask: np.ndarray) -> th.Tensor:
        return th.as_tensor(mask, dtype=th.bool).unsqueeze(0).to(self.device)

    def rollout(
        self, n_steps: int
    ) -> tuple[list[dict], float]:
        """Run n_steps of the env and return transitions + total reward."""
        transitions = []
        obs, _ = self.env.reset()
        obs_t = self._obs_to_tensor(obs)
        total_reward = 0.0

        for _ in range(n_steps):
            mask = self.env.action_masks()
            # mask[1:] is (n_machines * max_n_jobs). Reshape to (n_machines, max_n_jobs)
            pair_valid = np.asarray(mask[1:]).reshape(self.n_machines, self.n_jobs_max)
            job_valid = pair_valid.any(axis=0)  # (max_n_jobs,)
            job_mask = np.concatenate([[True], job_valid]).astype(bool).reshape(1, -1)
            job_mask = th.as_tensor(job_mask, dtype=th.bool, device=self.device)
            # machine mask: all available (all true); env action encoding is one job per action
            machine_mask = th.ones(1, self.n_machines, dtype=th.bool, device=self.device)

            job_act, mach_act, job_logits, mach_logits = self.net.sample_actions(
                obs_t, job_mask, machine_mask
            )
            job_action = int(job_act.cpu().item())
            machine_action = int(mach_act.cpu().item())

            # Convert to env action: job_action-1 is job index; if skip -> 0
            if job_action == 0:
                env_action = 0
            else:
                job_idx = job_action - 1
                env_action = 1 + machine_action * self.n_jobs_max + job_idx

            next_obs, reward, term, trunc, _ = self.env.step(env_action)
            total_reward += float(reward)
            transitions.append({
                'obs': obs_t,
                'job_actions': job_act,
                'machine_actions': mach_act,
                'job_logits': job_logits,
                'machine_logits': mach_logits,
                'rewards': th.as_tensor([reward], dtype=th.float32, device=self.device),
                'dones': th.as_tensor([term or trunc], dtype=th.bool, device=self.device),
                'job_masks': job_mask,
                'machine_masks': machine_mask,
            })
            obs_t = self._obs_to_tensor(next_obs)
            if term or trunc:
                obs, _ = self.env.reset()
                obs_t = self._obs_to_tensor(obs)

        return transitions, total_reward

    def train_step(self, n_steps: int = 512) -> dict[str, float]:
        transitions, total_reward = self.rollout(n_steps)
        self.num_steps += n_steps

        # Store old log-probs during rollout (the rollout currently only stores logits)
        # We'll compute old log-probs from the stored logits in this compact trainer.
        old_logp_j = []
        old_logp_m = []
        for t in transitions:
            job_logits = t['job_logits'].squeeze(0)  # (max_j+1)
            job_mask = t['job_masks'].squeeze(0)
            job_logits = job_logits.masked_fill(~job_mask, -1e9)
            old_logp_j.append(F.log_softmax(job_logits, -1)[t['job_actions'].squeeze(0)])
            m_logits = t['machine_logits'].squeeze(0)
            m_mask = t['machine_masks'].squeeze(0)
            m_logits = m_logits.masked_fill(~m_mask, -1e9)
            old_logp_m.append(F.log_softmax(m_logits, -1)[t['machine_actions'].squeeze(0)])
        old_logp_j = th.stack(old_logp_j).detach().to(self.device)
        old_logp_m = th.stack(old_logp_m).detach().to(self.device)

        # Compute values
        values = []
        for t in transitions:
            with th.no_grad():
                values.append(self.net.value(t['obs']).squeeze(-1))
        values = th.cat(values).to(self.device)

        # Returns and GAE
        returns = []
        gae = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            rew = transitions[t]['rewards']
            done = transitions[t]['dones']
            if t == len(transitions) - 1:
                next_value = th.zeros_like(rew, device=self.device)
            else:
                next_value = values[t + 1]
            delta = rew + self.gamma * next_value * (~done).float() - values[t]
            gae = delta + self.gamma * self.gae_lambda * (~done).float() * gae
            returns.insert(0, values[t] + gae)
        returns = th.cat(returns).to(self.device)
        advs = (returns - values).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Stack transitions
        obs_all = {}
        keys = list(transitions[0]['obs'].keys())
        for k in keys:
            obs_all[k] = th.cat([t['obs'][k] for t in transitions], dim=0).to(self.device)
        job_actions = th.cat([t['job_actions'] for t in transitions]).to(self.device)
        machine_actions = th.cat([t['machine_actions'] for t in transitions]).to(self.device)
        job_masks = th.cat([t['job_masks'] for t in transitions], dim=0).to(self.device)
        machine_masks = th.cat([t['machine_masks'] for t in transitions], dim=0).to(self.device)

        # PPO updates with ratio clipping
        n = obs_all[keys[0]].shape[0]
        clip_eps = self.clip_eps
        for _ in range(self.n_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                obs_b = {k: obs_all[k][idx] for k in keys}
                job_act_b = job_actions[idx]
                mach_act_b = machine_actions[idx]
                job_mask_b = job_masks[idx]
                mach_mask_b = machine_masks[idx]

                logp_j, logp_m, _ = self.net.log_probs(
                    obs_b, job_act_b, mach_act_b, job_mask_b, mach_mask_b
                )
                old_j = old_logp_j[idx]
                old_m = old_logp_m[idx]
                ratio_j = th.exp(logp_j - old_j)
                ratio_m = th.exp(logp_m - old_m)
                ratio = ratio_j + ratio_m  # sum for staged selection

                adv_b = advs[idx]
                loss_pi = -th.minimum(ratio * adv_b, th.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b).mean()

                v = self.net.value(obs_b).squeeze(-1)
                loss_v = F.mse_loss(v, returns[idx])

                loss = loss_pi + 0.5 * loss_v
                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {'reward': total_reward, 'steps': n_steps}

