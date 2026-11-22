"""Masked DQN over (job, machine) options, with a job-vs-machine embedding difference.

Input: jobs_usage (J, R, T) and the machines' free capacity (M, R, T) = capacity - usage (both / 255).
  1. job CNN:     every (job, resource) time series is one sample; Conv1d with a kernel of the whole time axis (1 x T)
                  -> (J, R, E).   machine CNN: the same on every (machine, resource) -> (M, R, E)  (own weights).
  2. option embedding = machine embedding - job embedding for every (machine, job) pair -> (M*J, R, E); LayerNorm.
  3. option CNN:  a Conv1d over the resource axis (channels E) turns every option into one vector (d_model).
  4. self-attention between all options (plus one learned 'skip' token); options that are not startable
     (not pending / does not fit) are not attended to. A linear head gives one Q value per token: Q(skip), Q(option).
Action = 0 skip, 1 + machine * J + job (as in the environment). The action mask (pending & fits) is computed from the
observation and used for exploration, the greedy action and the target max (MaskedDQN).
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy

from scheduling_simulator.core.job import JobStatus


class _NoExtractor(BaseFeaturesExtractor):
    """The Q network reads the dict observation itself; SB3 only needs some extractor object."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

    def forward(self, obs):
        return next(iter(obs.values()))


class _Conv3(nn.Module):
    """Conv1d(kernel 3, padding 1) over the resource axis of (N, R, C_in), written as a linear layer on the 3 neighbouring
    resources (identical maths; torch's CPU Conv1d is ~10x slower for these tiny sequences)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(3 * c_in, c_out)

    def forward(self, x):
        R = x.shape[1]
        p = F.pad(x, (0, 0, 1, 1))
        return self.lin(th.cat([p[:, 0:R], p[:, 1:R + 1], p[:, 2:R + 2]], dim=-1))


class OptionQNetwork(BasePolicy):
    def __init__(self, observation_space, action_space, embed=32, d_model=64, n_heads=4, n_layers=2, scale=255.0,
                 use_extra=False, max_time=110.0):
        super().__init__(observation_space, action_space, features_extractor=_NoExtractor(observation_space),
                         normalize_images=False)
        self.embed, self.d_model, self.scale = embed, d_model, scale
        self.use_extra, self.max_time = use_extra, float(max_time)
        self.J, R, self.T = observation_space["jobs_usage"].shape
        self.M = observation_space["machines_usage"].shape[0]
        self.R = R
        E, H = embed, d_model
        # a Conv1d with kernel T over a length-T series (1 x Time kernel) is a linear map of the whole time axis
        self.job_cnn = nn.Sequential(nn.Linear(self.T, E), nn.ReLU(), nn.Linear(E, E))
        self.machine_cnn = nn.Sequential(nn.Linear(self.T, E), nn.ReLU(), nn.Linear(E, E))
        self.norm = nn.LayerNorm(E)
        self.option_cnn = nn.Sequential(_Conv3(E, H), nn.ReLU(), _Conv3(H, H), nn.ReLU())
        extra_dim = 0
        if use_extra:
            # job scalars: status one-hot (4), wait, size, time left (running), time to arrival; machine scalars: utilisation and
            # time to the first change of the usage profile per resource; pair: ticks until the job fits the machine
            self.job_scalar = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            self.machine_scalar = nn.Sequential(nn.Linear(2 * R, 16), nn.ReLU())
            extra_dim = 16 + 16 + 1
        self.option_proj = nn.Linear(H * R + extra_dim, d_model)
        self.skip_token = nn.Parameter(th.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.attention = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.q_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def _prep(self, obs):
        return preprocess_obs(obs, self.observation_space, normalize_images=False)

    @staticmethod
    def _mask(o) -> th.Tensor:
        """(B, 1 + M*J) bool: skip always valid; option valid = job pending and fits the machine on every resource / tick."""
        free = o["machines_capacity"].float() - o["machines_usage"].float()          # (B, M, R, T)
        fits = (free[:, :, None] - o["jobs_usage"].float()[:, None]).amin(dim=(3, 4)) >= 0   # (B, M, J)
        pending = o["status"].round().long() == int(JobStatus.PENDING)               # (B, J)
        valid = (fits & pending[:, None, :]).reshape(free.shape[0], -1)
        return th.cat([th.ones(valid.shape[0], 1, dtype=th.bool, device=valid.device), valid], dim=1)

    def compute_mask(self, obs) -> th.Tensor:
        return self._mask(self._prep(obs))

    def _q(self, o) -> th.Tensor:
        return self.q_head(self._tokens(o)[0]).squeeze(-1)                           # (B, 1 + M*J)

    def _tokens(self, o):
        B, J, R, T = o["jobs_usage"].shape
        M = o["machines_usage"].shape[1]
        jobs = o["jobs_usage"].float() / self.scale
        free = (o["machines_capacity"].float() - o["machines_usage"].float()) / self.scale
        ej = self.job_cnn(jobs.reshape(B * J * R, T)).reshape(B, J, R, self.embed)
        em = self.machine_cnn(free.reshape(B * M * R, T)).reshape(B, M, R, self.embed)
        diff = em[:, :, None] - ej[:, None]                                          # (B, M, J, R, E)
        diff = self.norm(diff.reshape(B * M * J, R, self.embed))
        h = self.option_cnn(diff).flatten(1)                         # (B*M*J, H*R)
        if self.use_extra:
            h = th.cat([h, self._extra(o)], dim=1)
        h = self.option_proj(h).reshape(B, M * J, self.d_model)
        tokens = th.cat([self.skip_token.expand(B, 1, -1), h], dim=1)                # (B, 1 + M*J, d)
        mask = self._mask(o)
        out = self.attention(tokens, src_key_padding_mask=~mask)
        return out, mask                                                             # (B, 1 + M*J, d), (B, 1 + M*J)

    def _extra(self, o) -> th.Tensor:
        """(B*M*J, 33) job scalars, machine scalars and the earliest-start pair feature."""
        B, J, R, T = o["jobs_usage"].shape
        M = o["machines_usage"].shape[1]
        mt = self.max_time
        status = o["status"].round().long().clamp(0, 3)
        now = o["time"].float().reshape(B, 1)
        run = status == int(JobStatus.RUNNING)
        fin = th.nan_to_num(o["finished_at"].float(), posinf=0.0)
        left = th.where(run, (fin - now).clamp(0, mt), th.zeros_like(fin)) / mt
        wait = th.nan_to_num(o["wait_time"].float(), posinf=0.0).clamp(0, 2 * mt) / mt
        size = th.nan_to_num(o["size"].float(), posinf=0.0).clamp(0, mt) / mt
        to_arr = (th.nan_to_num(o["arrival"].float(), posinf=0.0) - now).clamp(0, mt) / mt
        jf = th.cat([F.one_hot(status, 4).float(), wait[..., None], size[..., None], left[..., None], to_arr[..., None]], dim=-1)
        jemb = self.job_scalar(jf)                                                   # (B, J, 16)
        usage = o["machines_usage"].float()
        cap = o["machines_capacity"].float()
        util = usage.mean(dim=3) / cap[..., 0]                                       # (B, M, R)
        changed = (usage != usage[..., :1]).float()                                  # (B, M, R, T)
        first = th.where(changed.any(dim=3), changed.argmax(dim=3).float(), th.full_like(util, float(T))) / T
        memb = self.machine_scalar(th.cat([util, first], dim=-1))                    # (B, M, 16)
        free = cap - usage
        ext = th.cat([free, cap[..., :1].expand(-1, -1, -1, T)], dim=3)             # beyond the profile the machine is free
        jobs = o["jobs_usage"].float()
        ok = th.stack([((ext[..., s:s + T][:, :, None] - jobs[:, None]).amin(dim=(3, 4)) >= 0) for s in range(T + 1)], dim=-1)
        start = ok.float().argmax(dim=-1).float() / T                                # (B, M, J) ticks until it fits
        feats = th.cat([jemb[:, None].expand(-1, M, -1, -1), memb[:, :, None].expand(-1, -1, J, -1), start[..., None]], dim=-1)
        return feats.reshape(B * M * J, -1)

    def forward(self, obs):
        return self._q(self._prep(obs))

    def masked_q(self, obs):
        o = self._prep(obs)
        return self._q(o).masked_fill(~self._mask(o), -1e9)

    def _predict(self, observation, deterministic: bool = True):
        return self.masked_q(observation).argmax(dim=1)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d.update(embed=self.embed, d_model=self.d_model, scale=self.scale, use_extra=self.use_extra, max_time=self.max_time)
        return d


class OptionDQNPolicy(DQNPolicy):
    def __init__(self, *args, embed=32, d_model=64, n_heads=4, n_layers=2, use_extra=False, max_time=110.0, **kwargs):
        self.opt_kwargs = dict(embed=embed, d_model=d_model, n_heads=n_heads, n_layers=n_layers, use_extra=use_extra,
                               max_time=max_time)
        kwargs.setdefault("features_extractor_class", _NoExtractor)
        super().__init__(*args, **kwargs)

    def make_q_net(self):
        return OptionQNetwork(self.observation_space, self.action_space, **self.opt_kwargs).to(self.device)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d.update(self.opt_kwargs)
        return d


class MaskedDQN(DQN):
    """DQN where exploration, the greedy action and the target max only consider valid actions (double DQN)."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            obs_t, _ = self.policy.obs_to_tensor(observation)
            mask = self.policy.q_net.compute_mask(obs_t).cpu().numpy()
            action = np.array([np.random.choice(np.flatnonzero(m)) for m in mask])
            if not self.policy.is_vectorized_observation(observation):
                action = action[0]
            return action, state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = rd.discounts if rd.discounts is not None else self.gamma
            with th.no_grad():
                nxt_mask = self.q_net.compute_mask(rd.next_observations)
                online = self.q_net(rd.next_observations).masked_fill(~nxt_mask, -1e9)
                best = online.argmax(dim=1, keepdim=True)                      # double DQN: online picks, target values
                next_q = self.q_net_target(rd.next_observations).gather(1, best)
                target = rd.rewards + (1 - rd.dones) * discounts * next_q
            cur = self.q_net(rd.observations).gather(1, rd.actions.long())
            loss = F.smooth_l1_loss(cur, target)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class OptionPPOPolicy(ActorCriticPolicy):
    """PPO actor-critic on the option network: per-token logits (skip, options) with the pending&fits mask, value from the
    skip token plus the mean of the valid option tokens. use_extra adds the scalar / earliest-start features."""

    def __init__(self, observation_space, action_space, lr_schedule, embed=32, d_model=64, n_heads=4, n_layers=2,
                 use_extra=False, max_time=110.0, forbid_idle_skip=False, skip_head=False, **kwargs):
        self.skip_head = skip_head
        self.opt_kwargs = dict(embed=embed, d_model=d_model, n_heads=n_heads, n_layers=n_layers, use_extra=use_extra,
                               max_time=max_time)
        self.forbid_idle_skip = forbid_idle_skip
        kwargs.setdefault("features_extractor_class", _NoExtractor)
        kwargs["net_arch"] = []
        kwargs["share_features_extractor"] = True
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.net = OptionQNetwork(self.observation_space, self.action_space, **self.opt_kwargs)
        d = self.opt_kwargs["d_model"]
        self.logit_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 1))
        if self.skip_head:
            # separate skip head: skip token + mean and max of the valid option tokens + global state features
            self.skip_mlp = nn.Sequential(nn.Linear(3 * d + self._G, 64), nn.ReLU(), nn.Linear(64, 1))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def _G(self):
        return 5 + 2 * self.net.R

    def _global(self, o, mask):
        """(B, 5 + 2R): fractions pending / running / completed, fraction of startable options, time, machine
        utilisation and time-to-first-change per resource (mean over machines)."""
        B, J = o["status"].shape
        st = o["status"].round().long()
        T = self.net.T
        usage, cap = o["machines_usage"].float(), o["machines_capacity"].float()
        util = (usage.mean(dim=3) / cap[..., 0]).mean(dim=1)
        changed = (usage != usage[..., :1]).float()
        first = th.where(changed.any(dim=3), changed.argmax(dim=3).float(), th.full(usage.shape[:3], float(T), device=usage.device)) / T
        pend, run, comp = [(st == int(x)).float().mean(dim=1, keepdim=True) for x in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED)]
        opt = mask[:, 1:].float().mean(dim=1, keepdim=True)
        now = o["time"].float().reshape(B, 1) / self.net.max_time
        return th.cat([pend, run, comp, opt, now, util, first.mean(dim=1)], dim=1)

    def _heads(self, obs):
        o = self.net._prep(obs)
        out, mask = self.net._tokens(o)
        if self.forbid_idle_skip:
            nothing_running = ~(o["status"].round().long() == int(JobStatus.RUNNING)).any(dim=1)
            mask = mask.clone()
            mask[:, 0] = ~(nothing_running & mask[:, 1:].any(dim=1))
        logits = self.logit_head(out).squeeze(-1)
        w = mask[:, 1:].float().unsqueeze(-1)
        mean = (out[:, 1:] * w).sum(1) / w.sum(1).clamp(min=1.0)
        if self.skip_head:
            mx = out[:, 1:].masked_fill(w == 0, -1e4).amax(dim=1)
            mx = th.where(w.sum(1) > 0, mx, th.zeros_like(mx))
            skip = self.skip_mlp(th.cat([out[:, 0], mean, mx, self._global(o, mask)], dim=1))
            logits = th.cat([skip, logits[:, 1:]], dim=1)
        logits = logits.masked_fill(~mask, -1e8)
        value = self.value_head(th.cat([out[:, 0], mean], dim=1)).squeeze(-1)
        return logits, value

    def forward(self, obs, deterministic=False):
        logits, value = self._heads(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, value.unsqueeze(-1), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        logits, value = self._heads(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        return value.unsqueeze(-1), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self.action_dist.proba_distribution(action_logits=self._heads(obs)[0])

    def predict_values(self, obs):
        return self._heads(obs)[1].unsqueeze(-1)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d.update(self.opt_kwargs, forbid_idle_skip=self.forbid_idle_skip, skip_head=self.skip_head)
        return d
