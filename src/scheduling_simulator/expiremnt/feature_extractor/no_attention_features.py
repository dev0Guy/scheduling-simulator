"""Feature extractor with NO attention anywhere (a DeepSets-style model).

Compared with `StatusSizeWaitTtlCapacityFeaturesExtractor` it removes the fit
attention, the job self-attention and the learned attention pooling. What
replaces them:

  * every job's input already contains what the attention used to compute:
    its per-resource usage, the machine's per-resource room, and the 9
    exact pair features (fits now, ticks until it fits, finish time, ...);
  * jobs are summarised by plain mean and max pooling over the jobs that are
    not finished, followed by a Linear layer, instead of learned weights.

The output layout is identical to the attention model
([global | job embeddings | pair features | dynamic machine embedding]), so
the same actor head and critic are used.
"""
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.expiremnt.feature_extractor.new_features import (
    StatusSizeWaitTtlCapacityFeaturesExtractor,
)


class NoAttentionFeaturesExtractor(BaseFeaturesExtractor):

    # The exact pair features are reused from the attention model (same code).
    _pair_features = StatusSizeWaitTtlCapacityFeaturesExtractor._pair_features

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        status_embed_dim: int = 16,
        embedding_dim: int = 256,
        max_time: float = 250.0,
    ):
        self.n_jobs = int(observation_space["status"].shape[0])
        self.n_machines = int(observation_space["machines_usage"].shape[0])
        self.n_resource = int(observation_space["jobs_usage"].shape[1])
        self.n_time = int(observation_space["machines_usage"].shape[2])
        self.max_time = float(max_time)
        self.max_capacity = float(observation_space["machines_capacity"].high.flat[0])
        self.embedding_dim = embedding_dim
        self.n_statuses = len(JobStatus) + 1
        self.pair_feat_dim = 6 + self.n_resource

        features_dim = (
            embedding_dim * (self.n_jobs + 1)
            + self.n_machines * self.n_jobs * self.pair_feat_dim
            + self.n_machines * embedding_dim
        )
        super().__init__(observation_space, features_dim=features_dim)

        self.status_embed = nn.Embedding(self.n_statuses, status_embed_dim)

        # status embedding + size, wait, ttl, arrival
        # + this job's usage per resource
        # + every machine's room per resource
        # + the pair features of this job with every machine
        self.job_input_dim = (
            status_embed_dim + 4
            + self.n_resource
            + self.n_machines * self.n_resource
            + self.n_machines * self.pair_feat_dim
        )
        self.job_encoder = nn.Sequential(
            nn.Linear(self.job_input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # mean pool and max pool of the job embeddings -> one summary
        self.global_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # dynamic machine embedding from the free-capacity grid (as in the
        # attention model; the last layer starts at zero)
        self.machine_row_encoder = nn.Sequential(
            nn.Linear(self.n_time, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.machine_dyn_head = nn.Sequential(
            nn.Linear(self.n_resource * 64, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.machine_dyn_head[-1].weight)
        nn.init.zeros_(self.machine_dyn_head[-1].bias)

    def forward(self, observations) -> torch.Tensor:
        status = observations["status"].long()
        B = status.shape[0]

        size = observations["size"] / self.max_time
        wait_time = observations["wait_time"] / self.max_time
        ttl = torch.clamp(observations["ttl"] / self.max_time, 0.0, 1.0)
        arrival = torch.clamp(
            (observations["arrival"] - observations["time"]) / self.max_time, 0.0, 1.0
        )

        # usage per job and resource (max over time), room per machine and
        # resource (tightest tick), both divided by the capacity
        usage = torch.clamp(
            observations["jobs_usage"].float().amax(dim=3) / self.max_capacity, 0.0, 1.0
        )                                                           # (B, n, R)
        free_grid = torch.clamp(
            (observations["machines_capacity"].float() - observations["machines_usage"].float())
            / self.max_capacity,
            0.0,
            1.0,
        )                                                           # (B, m, R, T)
        room = free_grid.amin(dim=3)                                # (B, m, R)
        room_ctx = room.reshape(B, 1, -1).expand(B, self.n_jobs, -1)

        pair = self._pair_features(observations, status)            # (B, m, n, F)
        pair_per_job = pair.permute(0, 2, 1, 3).reshape(B, self.n_jobs, -1)

        job_features = torch.cat(
            [
                self.status_embed(status),
                size.unsqueeze(-1),
                wait_time.unsqueeze(-1),
                ttl.unsqueeze(-1),
                arrival.unsqueeze(-1),
                usage,
                room_ctx,
                pair_per_job,
            ],
            dim=-1,
        )
        job_embeddings = self.job_encoder(job_features)             # (B, n, E)

        # mean and max over the jobs that are not finished (guard: if every
        # job is finished, use all of them)
        live = status != JobStatus.COMPLETED
        live = live | ~live.any(dim=1, keepdim=True)
        live_f = live.unsqueeze(-1).to(job_embeddings.dtype)
        mean_pool = (job_embeddings * live_f).sum(dim=1) / live_f.sum(dim=1)
        max_pool = job_embeddings.masked_fill(~live.unsqueeze(-1), float("-inf")).amax(dim=1)
        global_embedding = self.global_head(torch.cat([mean_pool, max_pool], dim=-1))

        rows = self.machine_row_encoder(free_grid)                  # (B, m, R, 64)
        machine_dyn = self.machine_dyn_head(rows.flatten(start_dim=2))

        return torch.cat(
            [
                global_embedding,
                job_embeddings.flatten(start_dim=1),
                pair.flatten(start_dim=1),
                machine_dyn.flatten(start_dim=1),
            ],
            dim=1,
        )
