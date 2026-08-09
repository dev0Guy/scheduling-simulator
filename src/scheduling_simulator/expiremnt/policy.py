from typing import Any

import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.nn import functional as F


def get_auto_device() -> str:
    """Pick the best available device: mps > cuda > cpu."""
    if th.backends.mps.is_available():
        return 'mps'
    if th.cuda.is_available():
        return 'cuda'
    return 'cpu'


class PointerFeaturesExtractor(BaseFeaturesExtractor):
    """Variable-cardinality pointer policy for job scheduling.

    Encodes jobs and machines with self-attention, then scores each
    (machine, job) pair with a shared function. The same parameters
    process any number of jobs — the output dimension is fixed by
    max_n_jobs, but the scoring function is applied per-candidate.

    Padded job slots are masked in attention and in the output via
    key_padding_mask. The action mask from the environment ensures
    only valid allocations are sampled.
    """

    def __init__(self, observation_space: spaces.Dict, embedding_dim: int = 64):
        n_machines, n_resources, n_time = observation_space['machines_usage'].shape
        max_n_jobs = observation_space['jobs_usage'].shape[0]
        self.n_actions = 1 + n_machines * max_n_jobs
        self.n_machines = n_machines
        self.max_n_jobs = max_n_jobs
        self.embedding_dim = embedding_dim
        self.max_capacity = float(observation_space['machines_capacity'].high.max())
        self.time_scale = float(n_time)

        job_input_dim = n_resources * n_time + 4 + 6
        machine_input_dim = 2 * n_resources * n_time

        super().__init__(
            observation_space,
            self.n_actions * embedding_dim + embedding_dim,
        )

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
            embed_dim=embedding_dim, num_heads=4, batch_first=True,
        )
        self.job_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embedding_dim)
        self.pair_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.skip_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim + 1, embedding_dim),
            nn.ReLU(),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(4 * embedding_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

    def _build_job_mask(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        """Return (batch, max_n_jobs) bool tensor: True for real jobs."""
        return ~(
            (observations['status'] == 3)
            & (observations['ttl'] == 0)
            & (observations['size'] == 0)
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        machine_usage = observations['machines_usage'].float() / self.max_capacity
        machine_capacity = observations['machines_capacity'].float() / self.max_capacity
        jobs_usage = observations['jobs_usage'].float() / self.max_capacity
        status = F.one_hot(observations['status'].long(), num_classes=4).float()
        current_time = observations['time'].float() / self.time_scale

        is_real_job = self._build_job_mask(observations)
        key_padding_mask = ~is_real_job

        job_metadata = th.stack(
            [
                observations['ttl'].float() / self.time_scale,
                observations['arrival'].float() / self.time_scale,
                observations['wait_time'].float() / self.time_scale,
                observations['scheduled_at'].float() / self.time_scale,
                observations['finished_at'].float() / self.time_scale,
                observations['size'].float() / self.time_scale,
            ],
            dim=-1,
        )
        job_features = th.cat(
            [jobs_usage.flatten(start_dim=2), status, job_metadata], dim=-1
        )
        machine_features = th.cat(
            [machine_usage.flatten(start_dim=2), machine_capacity.flatten(start_dim=2)],
            dim=-1,
        )

        job_emb = self.job_encoder(job_features)
        machine_emb = self.machine_encoder(machine_features)

        attended, _ = self.job_attention(
            job_emb, job_emb, job_emb, key_padding_mask=key_padding_mask,
        )
        attended = self.job_norm(job_emb + attended)

        cross, _ = self.cross_attention(attended, machine_emb, machine_emb)
        cross = self.cross_norm(attended + cross)

        n_m = machine_emb.shape[1]
        n_j = cross.shape[1]
        m_exp = machine_emb[:, :, None, :].expand(-1, -1, n_j, -1)
        j_exp = cross[:, None, :, :].expand(-1, n_m, -1, -1)
        pair_emb = self.pair_scorer(
            th.cat([m_exp, j_exp, m_exp * j_exp], dim=-1)
        )
        real_mask = is_real_job[:, None, :, None].float()
        pair_emb = pair_emb * real_mask

        job_mask_f = is_real_job.unsqueeze(-1).float()
        job_sum = (attended * job_mask_f).sum(dim=1)
        job_count = job_mask_f.sum(dim=1).clamp(min=1)
        job_mean = job_sum / job_count
        job_max = attended.masked_fill(
            ~is_real_job.unsqueeze(-1), float('-inf')
        ).max(dim=1).values.nan_to_num(0.0)

        skip_emb = self.skip_encoder(
            th.cat([job_mean, job_max, current_time], dim=-1)
        )

        action_emb = th.cat(
            [skip_emb[:, None, :], pair_emb.flatten(start_dim=1, end_dim=2)], dim=1,
        )

        m_mean = machine_emb.mean(dim=1)
        m_max = machine_emb.amax(dim=1)
        value_state = self.value_encoder(
            th.cat([job_mean, job_max, m_mean, m_max, current_time], dim=-1)
        )

        return th.cat([action_emb.flatten(start_dim=1), value_state], dim=-1)


class SchedulingMlpExtractor(nn.Module):
    """Splits features into action and value parts with no extra layers."""

    def __init__(self, n_actions: int, embedding_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.latent_dim_pi = n_actions * embedding_dim
        self.latent_dim_vf = embedding_dim
        self.policy_net = nn.Identity()
        self.value_net = nn.Identity()
        self.feature_dim = self.latent_dim_pi + self.latent_dim_vf

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        pi_features = features[:, : self.latent_dim_pi]
        vf_features = features[:, self.latent_dim_pi :]
        return pi_features, vf_features

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features[:, : self.latent_dim_pi]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return features[:, self.latent_dim_pi :]


class SharedActionHead(nn.Module):
    """Scores each action embedding with a shared MLP.

    Applied identically to every (machine, job) pair embedding,
    so the same parameters score 8 jobs or 32 jobs.
    """

    def __init__(self, n_actions: int, embedding_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        action_embeddings = features.reshape(-1, self.n_actions, self.embedding_dim)
        return self.scorer(action_embeddings).squeeze(-1)


class SchedulingValueNet(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.net(features)


class SchedulingPolicy(MaskableMultiInputActorCriticPolicy):
    """MaskablePPO policy with variable-cardinality pointer architecture.

    Supports training on one job count and evaluating on another via
    padding + masking. The environment must be constructed with
    max_n_jobs >= any n_jobs used during training or evaluation.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        embedding_dim: int = 64,
        **kwargs: Any,
    ):
        self.scheduling_n_actions = action_space.n
        self.scheduling_embedding_dim = embedding_dim
        kwargs.setdefault('optimizer_class', th.optim.AdamW)
        kwargs.setdefault('optimizer_kwargs', {'weight_decay': 0.01})
        kwargs.pop('net_arch', None)
        kwargs.pop('ortho_init', None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch={'pi': [], 'vf': []},
            ortho_init=False,
            features_extractor_class=PointerFeaturesExtractor,
            features_extractor_kwargs={'embedding_dim': embedding_dim},
            **kwargs,
        )
        self.action_net = SharedActionHead(action_space.n, embedding_dim)
        self.value_net = SchedulingValueNet(embedding_dim)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SchedulingMlpExtractor(
            self.scheduling_n_actions,
            self.scheduling_embedding_dim,
        )
