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

    The job encoder does NOT see usage. Usage enters later, through a
    job-to-machine "fit attention" applied to the job summaries:
        - job usage: max over the job's (resource, time) usage matrix,
          divided by max_capacity (one scalar per job)
        - machine room: min over the machine's (resource, time) cells of
          (capacity - usage), divided by max_capacity -- the room left
          at the tightest point, which is what decides whether a job fits
    Per (job, machine) pair, weight = sigmoid(Q(usage) . K(room) / sqrt(d))
    (a sigmoid instead of a softmax: with one machine a softmax is always
    1 and could never depend on the job). Pairs where the machine cannot
    run the job (job usage > machine room) are MASKED OUT: their weight
    is forced to 0, so a job only attends to machines it fits on (a job
    that fits nowhere gets a zero fit vector). The attention output for
    a job is the weighted sum over the remaining machines of V(room). It
    is concatenated with the job's summary and projected back to
    embedding_dim:
        summary = ReLU(LayerNorm(Linear([summary, fit_out])))

    Job self-attention (after the fit step, before pooling): every job
    attends to every other job (4 heads, residual + LayerNorm), so each
    job's summary can be compared against the specific other jobs
    (e.g. "I am smaller than the one that fits"). Finished jobs are
    masked out as attention keys; a job's own status still reaches it
    through its own summary.

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
        per_resource_fit: bool = False,
        pair_features: bool = False,
        dynamic_machine: bool = False,
    ):
        # pair_features   : append per-(machine, job) numbers (fits now,
        #                   delay until it fits, headroom, ...) for the actor.
        # dynamic_machine : append a state-dependent machine embedding built
        #                   from the machine's free-capacity grid.
        self.use_pair_features = pair_features
        self.use_dynamic_machine = dynamic_machine
        self.n_jobs = int(
            observation_space["status"].shape[0]
        )

        # False: the original M9 input, ONE scalar per job / machine.
        # True : one value PER RESOURCE per job / machine, each the max
        #        (jobs) or tightest room (machines) taken over time only.
        self.per_resource_fit = per_resource_fit
        self.n_resource = int(
            observation_space["jobs_usage"].shape[1]
        )
        self.fit_in_dim = self.n_resource if per_resource_fit else 1

        self.n_machines = int(
            observation_space["machines_usage"].shape[0]
        )

        self.max_time = float(max_time)
        self.embedding_dim = embedding_dim

        # true theoretical max usage / capacity of one cell
        self.max_capacity = float(
            observation_space["machines_capacity"].high.flat[0]
        )

        # +1 safety margin
        self.n_statuses = len(JobStatus) + 1

        self.n_time = int(
            observation_space["machines_usage"].shape[2]
        )
        # [pending, fits_now, delay, finish, headroom x n_resource,
        #  dominant_share, packing]
        self.pair_feat_dim = 6 + self.n_resource

        features_dim = embedding_dim * (
            self.n_jobs + 1
        )
        if pair_features:
            features_dim += (
                self.n_machines * self.n_jobs * self.pair_feat_dim
            )
        if dynamic_machine:
            features_dim += self.n_machines * embedding_dim

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
        # Job-to-machine fit attention (usage never enters the encoder)
        #
        # queries: from each job's usage scalar
        # keys, values: from each machine's room scalar
        # ---------------------------------------------------------

        self.fit_dim = 64

        def scalar_mlp():
            return nn.Sequential(
                nn.Linear(self.fit_in_dim, self.fit_dim),
                nn.ReLU(),
                nn.Linear(self.fit_dim, self.fit_dim),
            )

        self.fit_query = scalar_mlp()
        self.fit_key = scalar_mlp()
        self.fit_value = scalar_mlp()

        # concat(job summary, fit attention output) -> job summary
        self.fit_fuse = nn.Sequential(
            nn.Linear(
                embedding_dim + self.fit_dim,
                embedding_dim,
            ),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Job self-attention (jobs compare against each other)
        # ---------------------------------------------------------

        self.job_self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True,
        )

        self.job_self_attention_norm = nn.LayerNorm(embedding_dim)

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

        # ---------------------------------------------------------
        # Dynamic machine embedding: the machine's free-capacity grid
        # (n_resource x n_time) -> shared per-resource-row encoder ->
        # concat rows -> embedding. Final layer zero-initialised so the
        # model starts out identical to the static-embedding one.
        # ---------------------------------------------------------

        if dynamic_machine:
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

    def _pair_features(self, observations, status) -> torch.Tensor:
        """Per (machine, job) numbers, shape (batch, n_machines, n_jobs, F).

        For each start offset s in [0, n_time) check whether the job (one
        usage value per resource, held for `size` ticks) fits in the
        machine's free-capacity grid during ticks [s, s + size). Ticks past
        the observed window are treated as free.
        """
        T = self.n_time
        cap = self.max_capacity

        # (b, j, R): peak usage of each job per resource
        job_u = torch.clamp(
            observations["jobs_usage"].float().amax(dim=3) / cap, 0.0, 1.0
        )
        # (b, m, R, T): free capacity grid
        free_t = torch.clamp(
            (
                observations["machines_capacity"].float()
                - observations["machines_usage"].float()
            ) / cap,
            0.0,
            1.0,
        )
        free_min = free_t.amin(dim=3)                      # (b, m, R)
        size_i = observations["size"].float().round().long().clamp(1, T)  # (b, j)

        t_idx = torch.arange(T, device=free_t.device)
        fits = []
        for s in range(T):
            in_win = (t_idx >= s)[None, None, :] & (
                t_idx[None, None, :] < (s + size_i).unsqueeze(-1)
            )                                              # (b, j, T)
            masked = torch.where(
                in_win[:, None, :, None, :],               # (b,1,j,1,T)
                free_t[:, :, None, :, :],                  # (b,m,1,R,T)
                torch.ones((), device=free_t.device),
            )                                              # (b,m,j,R,T)
            window_min = masked.amin(dim=-1)               # (b,m,j,R)
            fits.append(
                (job_u[:, None, :, :] <= window_min + 1e-6).all(dim=-1)
            )                                              # (b,m,j)
        fits = torch.stack(fits, dim=-1)                   # (b,m,j,T)

        any_fit = fits.any(dim=-1)
        first_fit = fits.float().argmax(dim=-1).float()
        delay = torch.where(
            any_fit, first_fit, torch.full_like(first_fit, float(T))
        )                                                  # ticks, T = never
        fits_now = fits[..., 0].float()
        finish = (delay + size_i[:, None, :].float()) / self.max_time

        headroom = torch.clamp(
            free_min[:, :, None, :] - job_u[:, None, :, :], -1.0, 1.0
        )                                                  # (b,m,j,R)
        dominant = job_u.amax(dim=-1)[:, None, :].expand_as(delay)
        packing = (
            job_u[:, None, :, :] * free_min[:, :, None, :]
        ).mean(dim=-1)
        pending = (status == JobStatus.PENDING).float()[:, None, :].expand_as(delay)

        return torch.cat(
            [
                pending.unsqueeze(-1),
                fits_now.unsqueeze(-1),
                (delay / T).unsqueeze(-1),
                finish.unsqueeze(-1),
                headroom,
                dominant.unsqueeze(-1),
                packing.unsqueeze(-1),
            ],
            dim=-1,
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

        # Reduce over (resource, time) -> one scalar, or over time only
        # -> one value per resource (per_resource_fit).
        reduce_dims = 3 if self.per_resource_fit else (2, 3)

        # Per job: the max over its usage.
        #   scalar mode:       (batch, n_jobs)
        #   per-resource mode: (batch, n_jobs, n_resource)
        usage = torch.clamp(
            observations["jobs_usage"].float().amax(dim=reduce_dims)
            / self.max_capacity,
            0.0,
            1.0,
        )

        # Per machine: min over time of (capacity - usage), i.e. the room
        # at the tightest point of the window (= capacity - max usage).
        #   scalar mode:       (batch, n_machines)
        #   per-resource mode: (batch, n_machines, n_resource)
        free = torch.clamp(
            (
                observations["machines_capacity"].float()
                - observations["machines_usage"].float()
            ).amin(dim=reduce_dims)
            / self.max_capacity,
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
        # Fit attention between each job's usage and every machine's room
        #
        # score[b, j, m] = Q(usage_j) . K(room_m) / sqrt(d)
        # weight         = sigmoid(score)   (not softmax, see docstring)
        # out[b, j]      = sum_m weight[b, j, m] * V(room_m)
        # ---------------------------------------------------------

        usage_in = usage if self.per_resource_fit else usage.unsqueeze(-1)
        free_in = free if self.per_resource_fit else free.unsqueeze(-1)

        q = self.fit_query(usage_in)   # (b, n_jobs, d)
        k = self.fit_key(free_in)      # (b, n_machines, d)
        v = self.fit_value(free_in)    # (b, n_machines, d)

        fit_scores = torch.einsum("bjd,bmd->bjm", q, k) / (self.fit_dim ** 0.5)
        fit_weights = torch.sigmoid(fit_scores)

        # Mask out (job, machine) pairs where the machine cannot run the
        # job: the job needs more than the machine's room.
        if self.per_resource_fit:
            # fits only if EVERY resource fits
            can_fit = (
                usage.unsqueeze(2) <= free.unsqueeze(1) + 1e-6
            ).all(dim=-1)                                          # (b, j, m)
        else:
            can_fit = usage.unsqueeze(-1) <= free.unsqueeze(1) + 1e-6  # (b, j, m)
        fit_weights = fit_weights * can_fit.to(fit_weights.dtype)
        fit_out = torch.einsum("bjm,bmd->bjd", fit_weights, v)

        job_embeddings = self.fit_fuse(
            torch.cat([job_embeddings, fit_out], dim=-1)
        )

        # ---------------------------------------------------------
        # Job self-attention. COMPLETED jobs are excluded as keys so
        # their stale data does not dilute the others' comparisons.
        # Guard the all-completed row (last step of an episode), which
        # would otherwise give an all-True mask and NaN the softmax.
        # ---------------------------------------------------------

        completed_mask = (status == JobStatus.COMPLETED)
        all_completed = completed_mask.all(dim=1, keepdim=True)
        key_padding_mask = completed_mask & ~all_completed

        attended_jobs, _ = self.job_self_attention(
            job_embeddings, job_embeddings, job_embeddings,
            key_padding_mask=key_padding_mask,
        )

        job_embeddings = self.job_self_attention_norm(
            job_embeddings + attended_jobs
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

        parts = [
            global_embedding,
            job_embeddings.flatten(
                start_dim=1
            ),
        ]

        # Tail 1: per-(machine, job) pair features for the actor.
        if self.use_pair_features:
            parts.append(
                self._pair_features(observations, status).flatten(start_dim=1)
            )

        # Tail 2: dynamic machine embedding from the free-capacity grid.
        if self.use_dynamic_machine:
            grid = torch.clamp(
                (
                    observations["machines_capacity"].float()
                    - observations["machines_usage"].float()
                ) / self.max_capacity,
                0.0,
                1.0,
            )                                               # (b, m, R, T)
            rows = self.machine_row_encoder(grid)           # (b, m, R, 64)
            machine_dyn = self.machine_dyn_head(
                rows.flatten(start_dim=2)
            )                                               # (b, m, E)
            parts.append(machine_dyn.flatten(start_dim=1))

        return torch.cat(parts, dim=1)
