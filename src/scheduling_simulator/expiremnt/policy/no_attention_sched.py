"""Actor-critic policy with no attention: `NoAttentionFeaturesExtractor` plus the
same actor head and critic as `NewSchedulingPolicy`."""
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from scheduling_simulator.expiremnt.feature_extractor.no_attention_features import (
    NoAttentionFeaturesExtractor,
)
from scheduling_simulator.expiremnt.policy.new_sched import (
    SchedulingActionHead,
    SchedulingMlpExtractor,
    SchedulingValueNet,
)


class NoAttentionSchedulingPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        embedding_dim=256,
        max_time=250.0,
        n_machines=1,
        **kwargs,
    ):
        self.scheduling_embedding_dim = embedding_dim
        self.scheduling_n_jobs = int(observation_space["status"].shape[0])
        self.scheduling_n_machines = n_machines
        self.scheduling_pair_feat_dim = 6 + int(observation_space["jobs_usage"].shape[1])

        kwargs.setdefault("optimizer_class", th.optim.Adam)
        kwargs.setdefault("optimizer_kwargs", {})
        kwargs.pop("net_arch", None)
        kwargs.pop("ortho_init", None)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch={"pi": [], "vf": []},
            ortho_init=False,
            features_extractor_class=NoAttentionFeaturesExtractor,
            features_extractor_kwargs={
                "status_embed_dim": 16,
                "embedding_dim": embedding_dim,
                "max_time": max_time,
            },
            **kwargs,
        )

        self.action_net = SchedulingActionHead(
            n_jobs=self.scheduling_n_jobs,
            n_machines=self.scheduling_n_machines,
            embedding_dim=embedding_dim,
            pair_feat_dim=self.scheduling_pair_feat_dim,
            dynamic_machine=True,
        )
        self.value_net = SchedulingValueNet(embedding_dim=embedding_dim)

        # action_net and value_net replaced the ones created by the parent, so
        # the optimizer has to be rebuilt over the final parameters
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SchedulingMlpExtractor(
            n_jobs=self.scheduling_n_jobs,
            n_machines=self.scheduling_n_machines,
            embedding_dim=self.scheduling_embedding_dim,
            features_dim=self.features_dim,
        )
