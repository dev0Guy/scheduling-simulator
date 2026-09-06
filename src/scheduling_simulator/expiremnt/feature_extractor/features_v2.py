# scheduling_simulator/expiremnt/feature_extractor/features_v2.py
from typing import TYPE_CHECKING
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from scheduling_simulator.core.job import JobStatus
if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


class FeaturesExtractorV2(BaseFeaturesExtractor):
    """SB3 features extractor using: status, size, wait_time, ttl, and
    resource usage (per job), plus machine usage/capacity (per machine).

    The resource encoder's final layer is zero-initialized, so at the
    start of training the resource embeddings are exactly zero for
    every job/machine — mathematically equivalent to a model that only
    uses [status, size, wait_time, ttl]. This matters because with a
    job generator that gives every job uniform full-capacity usage
    across all resources and its whole duration, the usage arrays carry
    NO information beyond size/status — they're a redundant, higher-
    dimensional restatement of data the model already has in clean
    scalar form. Forcing the network to process that redundant, higher-
    dimensional signal from a random initialization slows optimization
    and adds noise for no benefit. Zero-initializing the resource
    branch's output means training starts exactly as good as the
    simpler model, and ordinary gradient descent will grow the resource
    embeddings away from zero only once the generator is changed to
    produce genuinely informative, non-redundant usage profiles — no
    code changes needed here when that happens.

    A single SHARED resource encoder embeds each individual (n_time,)
    resource row into a fixed-size vector, applied identically across
    jobs and machines and across every resource within them.

    Per-job and per-machine embeddings are each pooled over their own
    entity axis with a learned attention mechanism (order-invariant,
    unlike flattening), then the two pooled vectors are concatenated
    and passed through an output head.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        status_embed_dim: int = 16,
        resource_embed_dim: int = 16,
        job_embed_dim: int = 32,
        machine_embed_dim: int = 32,
        attn_hidden_dim: int = 32,
        out_dim: int = 128,
        max_time: float = 250.0,
        max_capacity: float = 255.0,
    ):
        super().__init__(observation_space, features_dim=out_dim)
        self.n_jobs = int(observation_space["status"].shape[0])
        self.n_machines = int(observation_space["machines_usage"].shape[0])
        self.n_resource = int(observation_space["jobs_usage"].shape[1])
        self.n_time = int(observation_space["jobs_usage"].shape[2])
        self._max_time = float(max_time)
        self._max_capacity = float(max_capacity)
        self.n_statuses = len(JobStatus) + 1  # +1 safety margin

        self.status_embed = nn.Embedding(self.n_statuses, status_embed_dim)

        # --- SHARED per-resource-row encoder ---
        self.resource_embed_dim = resource_embed_dim
        self.resource_encoder = nn.Sequential(
            nn.Linear(self.n_time, resource_embed_dim),
            nn.LayerNorm(resource_embed_dim),
            nn.ReLU(),
            nn.Linear(resource_embed_dim, resource_embed_dim),
            nn.LayerNorm(resource_embed_dim),
            nn.ReLU(),
        )
        # Zero-init the final Linear so resource embeddings start at
        # exactly 0 (LayerNorm(0) = 0, ReLU(0) = 0) — see class docstring.
        nn.init.zeros_(self.resource_encoder[-2].weight)
        nn.init.zeros_(self.resource_encoder[-2].bias)

        self.per_entity_resource_dim = self.n_resource * resource_embed_dim

        # per-job raw feature = [status_embed, size, wait_time, ttl, flattened_resource_embed]
        self.per_job_dim = status_embed_dim + 3 + self.per_entity_resource_dim
        self.job_embed_dim = job_embed_dim

        self.job_head = nn.Sequential(
            nn.Linear(self.per_job_dim, job_embed_dim),
            nn.LayerNorm(job_embed_dim),
            nn.ReLU(),
            nn.Linear(job_embed_dim, job_embed_dim),
            nn.LayerNorm(job_embed_dim),
            nn.ReLU(),
        )

        # per-machine raw feature = [flattened_resource_embed(free_capacity)]
        self.per_machine_dim = self.per_entity_resource_dim
        self.machine_embed_dim = machine_embed_dim

        self.machine_head = nn.Sequential(
            nn.Linear(self.per_machine_dim, machine_embed_dim),
            nn.LayerNorm(machine_embed_dim),
            nn.ReLU(),
            nn.Linear(machine_embed_dim, machine_embed_dim),
            nn.LayerNorm(machine_embed_dim),
            nn.ReLU(),
        )

        self.job_attn_scorer = nn.Sequential(
            nn.Linear(job_embed_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )
        self.machine_attn_scorer = nn.Sequential(
            nn.Linear(machine_embed_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

        self.output_head = nn.Sequential(
            nn.Linear(job_embed_dim + machine_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def _encode_resource_rows(self, profiles: torch.Tensor, n_entities: int) -> torch.Tensor:
        """Shared per-resource-row encoder, run identically for jobs or
        machines, and identically across every resource within them.

        profiles: (batch, n_entities, n_resource, n_time) -- raw usage values
        returns:  (batch, n_entities, n_resource * resource_embed_dim)
        """
        batch = profiles.shape[0]
        flat_rows = profiles.reshape(batch * n_entities * self.n_resource, self.n_time)
        embedded_rows = self.resource_encoder(flat_rows)
        embedded_rows = embedded_rows.reshape(batch, n_entities, self.n_resource, self.resource_embed_dim)
        return embedded_rows.reshape(batch, n_entities, self.per_entity_resource_dim)

    def forward(self, observations: 'ObservationDict') -> torch.Tensor:
        status = observations['status']
        size = observations['size'] / self._max_time
        wait_time = observations['wait_time'] / self._max_time
        ttl = observations['ttl'] / self._max_time
        jobs_usage = observations['jobs_usage'] / self._max_capacity

        machines_usage = observations['machines_usage'] / self._max_capacity
        machines_capacity = observations['machines_capacity'] / self._max_capacity
        machines_free = machines_capacity - machines_usage

        batch = status.shape[0]

        # --- job branch ---
        status_embed = self.status_embed(status.long())
        job_resource_embed = self._encode_resource_rows(jobs_usage, self.n_jobs)

        per_job_features = torch.cat(
            [status_embed, size.unsqueeze(-1), wait_time.unsqueeze(-1), ttl.unsqueeze(-1), job_resource_embed],
            dim=-1,
        )

        flat_job_features = per_job_features.reshape(batch * self.n_jobs, self.per_job_dim)
        flat_job_embed = self.job_head(flat_job_features)
        job_embed = flat_job_embed.reshape(batch, self.n_jobs, self.job_embed_dim)

        job_attn_scores = self.job_attn_scorer(job_embed)
        job_attn_weights = torch.softmax(job_attn_scores, dim=1)
        job_pooled = (job_attn_weights * job_embed).sum(dim=1)

        # --- machine branch (same resource_encoder, different head) ---
        machine_resource_embed = self._encode_resource_rows(machines_free, self.n_machines)

        flat_machine_features = machine_resource_embed.reshape(batch * self.n_machines, self.per_machine_dim)
        flat_machine_embed = self.machine_head(flat_machine_features)
        machine_embed = flat_machine_embed.reshape(batch, self.n_machines, self.machine_embed_dim)

        machine_attn_scores = self.machine_attn_scorer(machine_embed)
        machine_attn_weights = torch.softmax(machine_attn_scores, dim=1)
        machine_pooled = (machine_attn_weights * machine_embed).sum(dim=1)

        # --- combine ---
        combined = torch.cat([job_pooled, machine_pooled], dim=-1)
        return self.output_head(combined)
