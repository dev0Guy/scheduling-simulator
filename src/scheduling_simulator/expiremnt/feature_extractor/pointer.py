from gymnasium import spaces
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PointerFeaturesExtractor(BaseFeaturesExtractor):
    """Fixed-cardinality pointer encoder for job scheduling.

    Encodes jobs and machines with self/cross-attention, then scores
    every (machine, job) pair. No padding/masking needed since n_jobs
    and n_machines are fixed by the env config.
    """

    def __init__(self, observation_space: spaces.Dict, embedding_dim: int = 64):
        n_machines, n_resources, n_time = observation_space['machines_usage'].shape
        n_jobs = observation_space['jobs_usage'].shape[0]
        self.n_actions = 1 + n_machines * n_jobs
        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.embedding_dim = embedding_dim
        self.max_capacity = float(observation_space['machines_capacity'].high.max())
        self.time_scale = float(n_time)
        self.n_status = int(observation_space['status'].high[0]) + 1

        job_input_dim = n_resources * n_time + self.n_status + 6
        machine_input_dim = 2 * n_resources * n_time

        super().__init__(
            observation_space,
            self.n_actions * embedding_dim + embedding_dim,
        )
        # TODO: try to use cnnn inside the encoder
        self.job_encoder = nn.Sequential(
            nn.Linear(job_input_dim, 128), nn.ReLU(), nn.Linear(128, embedding_dim),
        )
        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_input_dim, 128), nn.ReLU(), nn.Linear(128, embedding_dim),
        )
        self.job_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        self.job_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        self.cross_norm = nn.LayerNorm(embedding_dim)
        self.pair_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim), nn.ReLU(),
        )
        self.skip_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim + 1, embedding_dim), nn.ReLU(),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(4 * embedding_dim + 1, 128), nn.ReLU(),
            nn.Linear(128, embedding_dim), nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        machine_usage = observations['machines_usage'].float() / self.max_capacity
        machine_capacity = observations['machines_capacity'].float() / self.max_capacity
        jobs_usage = observations['jobs_usage'].float() / self.max_capacity
        status = F.one_hot(observations['status'].long(), num_classes=self.n_status).float()
        current_time = observations['time'].float() / self.time_scale

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

        attended, _ = self.job_attention(job_emb, job_emb, job_emb)
        attended = self.job_norm(job_emb + attended)

        cross, _ = self.cross_attention(attended, machine_emb, machine_emb)
        cross = self.cross_norm(attended + cross)

        n_m, n_j = machine_emb.shape[1], cross.shape[1]
        m_exp = machine_emb[:, :, None, :].expand(-1, -1, n_j, -1)
        j_exp = cross[:, None, :, :].expand(-1, n_m, -1, -1)
        pair_emb = self.pair_scorer(th.cat([m_exp, j_exp, m_exp * j_exp], dim=-1))

        job_mean = attended.mean(dim=1)
        job_max = attended.max(dim=1).values
        # why this doesn't get the machine as well and take into acount
        skip_emb = self.skip_encoder(th.cat([job_mean, job_max, current_time], dim=-1))

        action_emb = th.cat(
            [skip_emb[:, None, :], pair_emb.flatten(start_dim=1, end_dim=2)], dim=1,
        )

        m_mean = machine_emb.mean(dim=1)
        m_max = machine_emb.max(dim=1).values
        value_state = self.value_encoder(
            th.cat([job_mean, job_max, m_mean, m_max, current_time], dim=-1)
        )

        return th.cat([action_emb.flatten(start_dim=1), value_state], dim=-1)
