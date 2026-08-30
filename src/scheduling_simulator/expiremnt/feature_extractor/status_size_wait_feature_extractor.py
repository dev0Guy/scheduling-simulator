# scheduling_simulator/expiremnt/models/status_size_wait_capacity_feature_extractor.py
from typing import TYPE_CHECKING
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from scheduling_simulator.core.job import JobStatus
if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


class StatusSizeWaitTtlCapacityFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 features extractor using:
    status, size, wait_time, and ttl (per job).

    Per-job features are embedded and then pooled over the job axis
    with a learned attention mechanism.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        status_embed_dim: int = 16,
        job_embed_dim: int = 32,
        attn_hidden_dim: int = 32,
        out_dim: int = 128,
        max_time: float = 250.0,
    ):
        super().__init__(observation_space, features_dim=out_dim)

        self.n_jobs = int(observation_space["status"].shape[0])
        self._max_time = float(max_time)
        self.n_statuses = len(JobStatus) + 1

        self.status_embed = nn.Embedding(
            self.n_statuses,
            status_embed_dim,
        )

        # Per-job raw features:
        # [status_embedding, size, wait_time, ttl]
        self.per_job_dim = status_embed_dim + 3
        self.job_embed_dim = job_embed_dim

        self.job_head = nn.Sequential(
            nn.Linear(self.per_job_dim, job_embed_dim),
            nn.LayerNorm(job_embed_dim),
            nn.ReLU(),

            nn.Linear(job_embed_dim, job_embed_dim),
            nn.LayerNorm(job_embed_dim),
            nn.ReLU(),
        )

        # Per-job attention score
        self.attn_scorer = nn.Sequential(
            nn.Linear(job_embed_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

        self.output_head = nn.Sequential(
            nn.Linear(job_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),

            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: 'ObservationDict',
    ) -> torch.Tensor:

        status = observations["status"]
        size = observations["size"] / self._max_time
        wait_time = observations["wait_time"] / self._max_time
        ttl = observations["ttl"] / self._max_time

        batch = status.shape[0]

        # ------------------------------------------------------------
        # Status embedding
        # ------------------------------------------------------------
        status_embed = self.status_embed(
            status.long()
        )
        # (batch, n_jobs, status_embed_dim)

        # ------------------------------------------------------------
        # Per-job features
        # ------------------------------------------------------------
        per_job_features = torch.cat(
            [
                status_embed,
                size.unsqueeze(-1),
                wait_time.unsqueeze(-1),
                ttl.unsqueeze(-1),
            ],
            dim=-1,
        )
        # (batch, n_jobs, status_embed_dim + 3)

        # ------------------------------------------------------------
        # Apply the same job network independently to every job
        # ------------------------------------------------------------
        flat_features = per_job_features.reshape(
            batch * self.n_jobs,
            self.per_job_dim,
        )

        flat_embed = self.job_head(flat_features)
        # (batch * n_jobs, job_embed_dim)

        job_embed = flat_embed.reshape(
            batch,
            self.n_jobs,
            self.job_embed_dim,
        )
        # (batch, n_jobs, job_embed_dim)

        # ------------------------------------------------------------
        # Attention pooling
        # ------------------------------------------------------------
        attn_scores = self.attn_scorer(job_embed)
        # (batch, n_jobs, 1)

        attn_weights = torch.softmax(
            attn_scores,
            dim=1,
        )
        # (batch, n_jobs, 1)

        pooled = (
            attn_weights * job_embed
        ).sum(dim=1)
        # (batch, job_embed_dim)

        # ------------------------------------------------------------
        # Final representation
        # ------------------------------------------------------------
        return self.output_head(pooled)
        # (batch, out_dim)
