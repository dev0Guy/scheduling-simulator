from typing import TYPE_CHECKING

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from scheduling_simulator.core.job import JobStatus

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


class StatusSizeWaitTtlCapacityFeaturesExtractor(BaseFeaturesExtractor):
    """
    Reconstructed to be exactly the `np7aenua` ("silver-butterfly-298")
    baseline architecture -- see docs/ppo_vs_sjf_results.md's "Baseline"
    row and docs/experiment_log.md entry #3. This predates queue_pressure
    (entry #5-6), job self-attention (entry #7-8), and everything from
    the dual-attention / bin-packing eras: base per-job features only,
    pure pooling attention, no cross-job comparison mechanism at all.

    Per-job features:
        - status
        - size
        - wait_time
        - ttl
        - arrival (time until arrival, see experiment log #3 -- the
          first genuine win recorded in this project, +1.03% margin
          over the pre-arrival baseline)

    Each job is independently embedded -- no self-attention, no
    cross-job interaction of any kind. The only place information from
    other jobs reaches a given job's decision is through the pooled
    global_embedding (a single shared vector, attention-weighted over
    all jobs).

    Output:
        [global_embedding, job_0_embedding, ..., job_N_embedding]

    Shape:
        (batch, embedding_dim * (n_jobs + 1))
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        status_embed_dim: int = 16,
        embedding_dim: int = 256,
        attention_hidden_dim: int = 64,
        max_time: float = 250.0,
    ):
        self.n_jobs = int(
            observation_space["status"].shape[0]
        )

        self.max_time = float(max_time)
        self.embedding_dim = embedding_dim

        # +1 safety margin
        self.n_statuses = len(JobStatus) + 1

        features_dim = embedding_dim * (
            self.n_jobs + 1
        )

        super().__init__(
            observation_space,
            features_dim=features_dim,
        )

        # ---------------------------------------------------------
        # Status embedding
        # ---------------------------------------------------------

        self.status_embed = nn.Embedding(
            self.n_statuses,
            status_embed_dim,
        )

        # status embedding + size + wait_time + ttl + arrival
        self.job_input_dim = (
            status_embed_dim + 4
        )

        # ---------------------------------------------------------
        # Job encoder
        # ---------------------------------------------------------

        self.job_encoder = nn.Sequential(
            nn.Linear(
                self.job_input_dim,
                embedding_dim,
            ),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),

            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Attention scorer
        # ---------------------------------------------------------

        self.attention = nn.Sequential(
            nn.Linear(
                embedding_dim,
                attention_hidden_dim,
            ),
            nn.Tanh(),

            nn.Linear(
                attention_hidden_dim,
                1,
            ),
        )

        # ---------------------------------------------------------
        # Global representation
        # ---------------------------------------------------------

        self.global_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: "ObservationDict",
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # Raw observations
        # ---------------------------------------------------------

        status = observations["status"].long()

        size = (
            observations["size"]
            / self.max_time
        )

        wait_time = (
            observations["wait_time"]
            / self.max_time
        )

        ttl = torch.clamp(
            observations["ttl"] / self.max_time,
            0.0,
            1.0,
        )

        arrival = torch.clamp(
            (observations["arrival"] - observations["time"]) / self.max_time,
            0.0,
            1.0,
        )

        batch_size = status.shape[0]

        # ---------------------------------------------------------
        # Status embedding
        #
        # (batch, n_jobs)
        #
        # ->
        #
        # (batch, n_jobs, status_embed_dim)
        # ---------------------------------------------------------

        status_embedding = self.status_embed(
            status
        )

        # ---------------------------------------------------------
        # Build per-job representation
        #
        # [status_embedding, size, wait_time, ttl, arrival]
        # ---------------------------------------------------------

        job_features = torch.cat(
            [
                status_embedding,
                size.unsqueeze(-1),
                wait_time.unsqueeze(-1),
                ttl.unsqueeze(-1),
                arrival.unsqueeze(-1),
            ],
            dim=-1,
        )

        # ---------------------------------------------------------
        # Flatten jobs so every job goes through
        # exactly the same encoder.
        # ---------------------------------------------------------

        flat_job_features = job_features.reshape(
            batch_size * self.n_jobs,
            self.job_input_dim,
        )

        flat_job_embeddings = self.job_encoder(
            flat_job_features
        )

        # ---------------------------------------------------------
        # Restore:
        #
        # (batch, n_jobs, embedding_dim)
        # ---------------------------------------------------------

        job_embeddings = flat_job_embeddings.reshape(
            batch_size,
            self.n_jobs,
            self.embedding_dim,
        )

        # ---------------------------------------------------------
        # Attention over jobs
        # ---------------------------------------------------------

        attention_scores = self.attention(
            job_embeddings
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        # ---------------------------------------------------------
        # Global job representation
        # ---------------------------------------------------------

        global_embedding = (
            attention_weights * job_embeddings
        ).sum(dim=1)

        global_embedding = self.global_head(
            global_embedding
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Return both:
        #
        # global embedding
        #
        # and
        #
        # every individual job embedding
        # ---------------------------------------------------------

        return torch.cat(
            [
                global_embedding,
                job_embeddings.flatten(
                    start_dim=1
                ),
            ],
            dim=1,
        )
