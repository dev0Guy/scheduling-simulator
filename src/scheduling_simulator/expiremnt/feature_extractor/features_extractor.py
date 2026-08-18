# scheduling_simulator/expiremnt/models/feature_extractor.py
from typing import TYPE_CHECKING
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from gymnasium import spaces

from scheduling_simulator.core.job import JobStatus

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict


class SchedulingFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_out_dim: int = 128,
        mlp_out_dim: int = 128,
        max_time: int = 250,
    ):
        super().__init__(observation_space, features_dim=1)

        # NOTE: "wait_time" and "size" added -- previously computed by the
        # env and used in the reward function, but never fed to the network.
        self.vector_keys = ["ttl", "arrival", "wait_time", "size"]
        self.scalar_keys = ["time"]

        # normalization bounds, pulled directly from the declared spaces
        self._matrix_high = float(observation_space["machines_capacity"].high.flat[0])
        self._vector_highs = {}
        for key in self.vector_keys:
            high = observation_space[key].high
            # wait_time / ttl / size can have inf bounds in the declared
            # space -- fall back to max_time as a sane finite normalizer
            # so we never divide by / multiply by inf.
            high = np.where(np.isinf(high), float(max_time), high)
            self._vector_highs[key] = torch.as_tensor(high, dtype=torch.float32)

        # time is unbounded (Box(0, inf)) in the declared space, so it must
        # be normalized manually using the known episode horizon instead of
        # the (infinite) declared high -- previously this was fed in raw
        # and unbounded, which is a likely cause of growing loss over
        # training as episodes progress and "time" grows unboundedly.
        self._max_time = float(max_time)

        # Shared conv extractor, applied per-entity (per machine or per job)
        # with 1 input channel: (1, n_resources, n_time)
        self.entity_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            # Global average pool over spatial dims (done manually via
            # .mean() in _process_entities) -- MPS-safe replacement for
            # AdaptiveAvgPool2d, which errors on MPS when input spatial
            # size isn't evenly divisible by the output size.
        )
        flat_size = 32  # output channels after last conv, spatial dims collapsed by mean
        self.entity_head = nn.Sequential(
            nn.Linear(flat_size, cnn_out_dim),
            nn.LayerNorm(cnn_out_dim),
            nn.ReLU(),
        )

        total_concat_size = cnn_out_dim * 2  # remaining-capacity + jobs_usage

        extractors = {}
        for key in self.vector_keys:
            shape = observation_space[key].shape
            extractors[key] = nn.Sequential(
                nn.Linear(shape[0], mlp_out_dim),
                nn.LayerNorm(mlp_out_dim),
                nn.ReLU(),
            )
            total_concat_size += mlp_out_dim

        for key in self.scalar_keys:
            space = observation_space[key]
            if isinstance(space, spaces.Discrete):
                total_concat_size += int(space.n)
            else:
                total_concat_size += int(np.prod(space.shape))

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def _process_entities(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, n_entities, n_resources, n_time) -- either machines or jobs.
        mask: optional (batch, n_entities) bool/float tensor -- entities
            where mask is False/0 are excluded from the pooling average
            (e.g. jobs that are NOT_CREATED yet). Without this, entities
            that aren't "real" still get averaged in unweighted, adding
            noise that scales with however many slots happen to be active.
        """
        x = x / self._matrix_high  # normalize to [0, 1] using the declared space bound

        batch, n_entities, r, t = x.shape
        x = x.reshape(batch * n_entities, 1, r, t)
        feat = self.entity_extractor(x)                  # (batch * n_entities, 32, r', t')
        feat = feat.mean(dim=(-2, -1))                    # global avg pool -> (batch*n_entities, 32)
        feat = self.entity_head(feat)                     # (batch * n_entities, cnn_out_dim)
        feat = feat.reshape(batch, n_entities, -1)         # (batch, n_entities, cnn_out_dim)

        if mask is not None:
            mask = mask.reshape(batch, n_entities, 1).float()
            summed = (feat * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1.0)
            return summed / count

        return feat.mean(dim=1)                            # (batch, cnn_out_dim)

    def forward(self, observations: 'ObservationDict') -> torch.Tensor:
        tensors = []

        pending_mask = observations['status'] == JobStatus.PENDING

        # Derived matrix: remaining machine capacity (still in raw units,
        # normalization happens inside _process_entities).
        # jobs_usage is cloned before the masked overwrite below -- mutating
        # observations in place is risky since this tensor may be a view
        # into SB3's replay buffer / rollout storage rather than an
        # independent copy.
        jobs_usage = observations["jobs_usage"].clone()
        jobs_usage[~pending_mask] = 256

        remaining_capacity = observations["machines_capacity"] - observations["machines_usage"]
        tensors.append(self._process_entities(remaining_capacity))

        # Only average over jobs that actually exist / are pending -- see
        # _process_entities docstring.
        tensors.append(self._process_entities(jobs_usage, mask=pending_mask))

        for key in self.vector_keys:
            high = self._vector_highs[key].to(observations[key].device)
            normalized = observations[key] / torch.clamp(high, min=1.0)  # avoid div-by-zero
            tensors.append(self.extractors[key](normalized))

        for key in self.scalar_keys:
            val = observations[key]
            if key == "time":
                val = val / self._max_time  # normalize unbounded time by episode horizon
            tensors.append(val.reshape(val.shape[0], -1))

        return torch.cat(tensors, dim=1)
