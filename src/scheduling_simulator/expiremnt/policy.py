from typing import Any

import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.nn import functional as F


def get_auto_device() -> str:
    """Pick the best available device: cuda > cpu.

    MPS is slower than CPU for this model size due to data-transfer
    overhead. Only use CUDA GPUs where the compute benefit outweighs
    the transfer cost.
    """
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

        # Detect padded machines: all-zero capacity
        is_real_machine = machine_capacity.sum(dim=(2, 3)) > 0
        machine_key_padding_mask = ~is_real_machine

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

        cross, _ = self.cross_attention(
            attended, machine_emb, machine_emb,
            key_padding_mask=machine_key_padding_mask,
        )
        cross = self.cross_norm(attended + cross)

        n_m = machine_emb.shape[1]
        n_j = cross.shape[1]
        m_exp = machine_emb[:, :, None, :].expand(-1, -1, n_j, -1)
        j_exp = cross[:, None, :, :].expand(-1, n_m, -1, -1)
        pair_emb = self.pair_scorer(
            th.cat([m_exp, j_exp, m_exp * j_exp], dim=-1)
        )
        real_mask = is_real_job[:, None, :, None].float() * is_real_machine[:, :, None, None].float()
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

        m_mask_f = is_real_machine.unsqueeze(-1).float()
        m_mean = (machine_emb * m_mask_f).sum(dim=1) / m_mask_f.sum(dim=1).clamp(min=1)
        m_max = machine_emb.masked_fill(~m_mask_f.bool(), float('-inf')).amax(dim=1).nan_to_num(0.0)
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


class PointerActionHead(nn.Module):
    """Attention-based pointer scoring for variable-cardinality actions.

    A learned query vector attends over candidate embeddings to produce
    action logits. Unlike independent MLP scoring, the query encodes
    global scheduling intent and candidates compete via dot-product
    attention. Same parameters score any number of candidates.

    Reference: Bello et al. (2016), Kool et al. (2019) for pointer
    networks in combinatorial optimization; Decima (Mao et al. 2018)
    for scheduling-specific pointer policies.
    """

    def __init__(self, n_actions: int, embedding_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.scale = embedding_dim ** -0.5

    def forward(self, features: th.Tensor) -> th.Tensor:
        action_embeddings = features.reshape(-1, self.n_actions, self.embedding_dim)
        # Global query: mean pool over all candidates (skip + allocations)
        query = self.query(action_embeddings.mean(dim=1, keepdim=True))
        keys = self.key(action_embeddings)
        # Pointer scores: dot-product attention (batch, n_actions)
        logits = (query * keys).sum(dim=-1) * self.scale
        return logits


class TwoStagePointerHead(nn.Module):
    """Two-stage pointer scoring: job selection then machine selection.

    Decomposes the flat action distribution into:
      p(action) = p(job) * p(machine | job)

    Stage 1: global query attends over job embeddings -> job logits
    Stage 2: each job embedding attends over machine embeddings -> machine logits

    The flat action logit for (machine m, job j) is:
      log p(job=j) + log p(machine=m | job=j)

    This reduces the effective action space from O(n_m * n_j) to
    O(n_j) + O(n_m), which is easier to learn and more natural for
    scheduling. Same parameters handle any job/machine count.

    Reference: Decima (Mao et al. 2018) for two-stage scheduling pointers.
    """

    def __init__(self, n_actions: int, embedding_dim: int, n_machines: int, max_n_jobs: int):
        super().__init__()
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.n_machines = n_machines
        self.max_n_jobs = max_n_jobs

        # Stage 1: job selection pointer
        self.job_query = nn.Linear(embedding_dim, embedding_dim)
        self.job_key = nn.Linear(embedding_dim, embedding_dim)

        # Stage 2: machine selection pointer (conditioned on job)
        self.machine_query = nn.Linear(embedding_dim, embedding_dim)
        self.machine_key = nn.Linear(embedding_dim, embedding_dim)

        # Skip scoring
        self.skip_score = nn.Linear(embedding_dim, 1)

        self.scale = embedding_dim ** -0.5

    def forward(self, features: th.Tensor) -> th.Tensor:
        batch = features.shape[0]
        embedding_dim = self.embedding_dim
        n_m = self.n_machines
        n_j = self.max_n_jobs

        # Reshape: skip embedding + pair embeddings
        action_embeddings = features.reshape(batch, self.n_actions, embedding_dim)
        skip_emb = action_embeddings[:, 0, :]  # (batch, dim)

        # Pair embeddings: (batch, n_m * n_j, dim) -> (batch, n_m, n_j, dim)
        pair_emb = action_embeddings[:, 1:, :].reshape(batch, n_m, n_j, embedding_dim)

        # Job embeddings: mean over machines for each job
        job_emb = pair_emb.mean(dim=1)  # (batch, n_j, dim)

        # Machine embeddings: mean over jobs for each machine
        machine_emb = pair_emb.mean(dim=2)  # (batch, n_m, dim)

        # Stage 1: job selection
        # Global context from mean-pooled jobs
        job_context = job_emb.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        job_q = self.job_query(job_context)  # (batch, 1, dim)
        job_k = self.job_key(job_emb)  # (batch, n_j, dim)
        job_logits = (job_q * job_k).sum(dim=-1) * self.scale  # (batch, n_j)
        job_log_probs = F.log_softmax(job_logits, dim=-1)  # (batch, n_j)

        # Stage 2: machine selection for each job
        # Query: job embedding, Key: machine embedding
        machine_q = self.machine_query(job_emb)  # (batch, n_j, dim)
        machine_k = self.machine_key(machine_emb)  # (batch, n_m, dim)
        # (batch, n_j, n_m)
        machine_logits = th.einsum('bjd,bmd->bjm', machine_q, machine_k) * self.scale
        machine_log_probs = F.log_softmax(machine_logits, dim=-1)  # (batch, n_j, n_m)

        # Skip logit
        skip_logit = self.skip_score(skip_emb).squeeze(-1)  # (batch,)

        # Combine into flat action logits
        # action 0 = skip
        # action 1 + m * n_j + j = (machine m, job j)
        # logit(m, j) = log p(job=j) + log p(machine=m | job=j)
        # Flat order: (m=0,j=0), (m=0,j=1), ..., (m=0,j=nj-1), (m=1,j=0), ...
        joint_log_probs = job_log_probs.unsqueeze(2) + machine_log_probs  # (batch, n_j, n_m)
        joint_log_probs = joint_log_probs.transpose(1, 2)  # (batch, n_m, n_j)
        joint_flat = joint_log_probs.reshape(batch, n_m * n_j)  # (batch, n_m * n_j)

        # Final: skip logit + joint logits
        return th.cat([skip_logit.unsqueeze(-1), joint_flat], dim=-1)


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
        self.action_net = PointerActionHead(action_space.n, embedding_dim)
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


class TwoStageSchedulingPolicy(MaskableMultiInputActorCriticPolicy):
    """Two-stage pointer policy: job selection then machine selection.

    Uses TwoStagePointerHead which decomposes p(action) = p(job) * p(machine|job).
    Same encoder and value network as SchedulingPolicy.
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
        n_machines = observation_space['machines_usage'].shape[0]
        max_n_jobs = observation_space['jobs_usage'].shape[0]
        self.action_net = TwoStagePointerHead(
            action_space.n, embedding_dim, n_machines, max_n_jobs,
        )
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
