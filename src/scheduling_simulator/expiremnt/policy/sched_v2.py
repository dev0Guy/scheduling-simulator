# scheduling_simulator/expiremnt/policies/scheduling_policy.py
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from scheduling_simulator.expiremnt.feature_extractor.features_v2 import FeaturesExtractorV2


class SchedulingPolicyV2(ActorCriticPolicy):
    """Standard actor-critic policy over FeaturesExtractorV2's pooled
    (batch, out_dim) output. No pointer/per-action structure — ordinary
    SB3 MlpExtractor + action_net/value_net handle the pi/vf split and
    final projections to n_actions logits / a scalar value.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        status_embed_dim: int = 16,
        resource_embed_dim: int = 16,
        job_embed_dim: int = 32,
        machine_embed_dim: int = 32,
        attn_hidden_dim: int = 32,
        out_dim: int = 128,
        max_time: float = 250.0,
        max_capacity: float = 255.0,
        **kwargs,
    ):
        kwargs.setdefault('optimizer_class', th.optim.AdamW)
        kwargs.setdefault('optimizer_kwargs', {'weight_decay': 0.01})
        kwargs.pop('net_arch', None)

        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            features_extractor_class=FeaturesExtractorV2,
            features_extractor_kwargs=dict(
                status_embed_dim=status_embed_dim,
                resource_embed_dim=resource_embed_dim,
                job_embed_dim=job_embed_dim,
                machine_embed_dim=machine_embed_dim,
                attn_hidden_dim=attn_hidden_dim,
                out_dim=out_dim,
                max_time=max_time,
                max_capacity=max_capacity,
            ),
            **kwargs,
        )
