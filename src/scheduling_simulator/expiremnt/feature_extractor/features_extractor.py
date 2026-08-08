# scheduling_simulator/expiremnt/models/feature_extractor.py
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from gymnasium import spaces


class SchedulingFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_out_dim: int = 128,
        mlp_out_dim: int = 128,
        adaptive_pool_size: int = 4,
    ):
        super().__init__(observation_space, features_dim=1)

        self.vector_keys = ["ttl", "arrival"]
        self.scalar_keys = ["time"]

        # normalization bounds, pulled directly from the declared spaces
        self._matrix_high = float(observation_space["machines_capacity"].high.flat[0])
        self._vector_highs = {
            key: torch.as_tensor(observation_space[key].high, dtype=torch.float32)
            for key in self.vector_keys
        }

        # Shared conv extractor, applied per-entity (per machine or per job)
        # with 1 input channel: (1, n_resources, n_time)
        self.entity_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            # AdaptiveAvgPool2d fixes the spatial output to a known size
            # regardless of n_resources/n_time, so the flatten size is
            # always predictable without a dummy forward pass.
            nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size)),
            nn.Flatten(),
        )
        flat_size = 32 * adaptive_pool_size * adaptive_pool_size
        self.entity_head = nn.Sequential(nn.Linear(flat_size, cnn_out_dim), nn.ReLU())

        total_concat_size = cnn_out_dim * 2  # remaining-capacity + jobs_usage

        extractors = {}
        for key in self.vector_keys:
            shape = observation_space[key].shape
            extractors[key] = nn.Sequential(
                nn.Linear(shape[0], mlp_out_dim),
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

    def _process_entities(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_entities, n_resources, n_time) -- either machines or jobs.
        Normalizes, runs the shared per-entity conv extractor, mean-pools
        over the entity dimension.
        """
        x = x / self._matrix_high  # normalize to [0, 1] using the declared space bound

        batch, n_entities, r, t = x.shape
        x = x.reshape(batch * n_entities, 1, r, t)
        feat = self.entity_extractor(x)                # (batch * n_entities, flat_size)
        feat = self.entity_head(feat)                   # (batch * n_entities, cnn_out_dim)
        feat = feat.reshape(batch, n_entities, -1)       # (batch, n_entities, cnn_out_dim)
        return feat.mean(dim=1)                          # (batch, cnn_out_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        tensors = []

        # Derived matrix: remaining machine capacity (still in raw units,
        # normalization happens inside _process_entities)
        remaining_capacity = observations["machines_capacity"] - observations["machines_usage"]
        tensors.append(self._process_entities(remaining_capacity))

        tensors.append(self._process_entities(observations["jobs_usage"]))

        for key in self.vector_keys:
            high = self._vector_highs[key].to(observations[key].device)
            normalized = observations[key] / torch.clamp(high, min=1.0)  # avoid div-by-zero
            tensors.append(self.extractors[key](normalized))

        for key in self.scalar_keys:
            val = observations[key]
            tensors.append(val.reshape(val.shape[0], -1))

        return torch.cat(tensors, dim=1)
