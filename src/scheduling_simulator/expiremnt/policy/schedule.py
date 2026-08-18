from typing import Any
import torch as th
from torch import nn

from scheduling_simulator.expiremnt.feature_extractor.pointer import PointerFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class SchedulingMlpExtractor(nn.Module):
    """Splits the extractor's output into action/value latents (no extra layers)."""

    def __init__(self, n_actions: int, embedding_dim: int):
        super().__init__()
        self.latent_dim_pi = n_actions * embedding_dim
        self.latent_dim_vf = embedding_dim
        self.policy_net = nn.Identity()
        self.value_net = nn.Identity()

    def forward(self, features: th.Tensor):
        return features[:, :self.latent_dim_pi], features[:, self.latent_dim_pi:]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features[:, :self.latent_dim_pi]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return features[:, self.latent_dim_pi:]


class PointerActionHead(nn.Module):
    """Attention-based scoring over the flat action set (skip + every machine,job pair)."""

    def __init__(self, n_actions: int, embedding_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.scale = embedding_dim ** -0.5

    def forward(self, features: th.Tensor) -> th.Tensor:
        action_embeddings = features.reshape(-1, self.n_actions, self.embedding_dim)
        query = self.query(action_embeddings.mean(dim=1, keepdim=True))
        keys = self.key(action_embeddings)
        return (query * keys).sum(dim=-1) * self.scale


class SchedulingValueNet(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.net(features)


class SchedulingPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, embedding_dim=64, **kwargs):
        self.scheduling_n_actions = action_space.n
        self.scheduling_embedding_dim = embedding_dim
        kwargs.setdefault('optimizer_class', th.optim.AdamW)
        kwargs.setdefault('optimizer_kwargs', {'weight_decay': 0.01})
        kwargs.pop('net_arch', None)
        kwargs.pop('ortho_init', None)
        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch={'pi': [], 'vf': []}, ortho_init=False,
            features_extractor_class=PointerFeaturesExtractor,
            features_extractor_kwargs={'embedding_dim': embedding_dim},
            **kwargs,
        )
        self.action_net = PointerActionHead(action_space.n, embedding_dim)
        self.value_net = SchedulingValueNet(embedding_dim)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SchedulingMlpExtractor(self.scheduling_n_actions, self.scheduling_embedding_dim)
