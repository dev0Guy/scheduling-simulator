import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from scheduling_simulator.expiremnt.feature_extractor.new_features import (
    StatusSizeWaitTtlCapacityFeaturesExtractor,
)


class SchedulingMlpExtractor(nn.Module):
    """
    Splits the feature extractor output into:

        Actor:
            global embedding + all job embeddings

        Critic:
            global embedding only
    """

    def __init__(
        self,
        n_jobs: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.n_jobs = n_jobs
        self.embedding_dim = embedding_dim

        # Actor receives:
        #
        # global
        # + n_jobs * job_embedding
        #
        self.latent_dim_pi = (
            embedding_dim * (n_jobs + 1)
        )

        # Critic receives global only
        self.latent_dim_vf = embedding_dim

        self.policy_net = nn.Identity()
        self.value_net = nn.Identity()

    def forward(
        self,
        features: th.Tensor,
    ):
        return (
            self.forward_actor(features),
            self.forward_critic(features),
        )

    def forward_actor(
        self,
        features: th.Tensor,
    ):
        return self.policy_net(features)

    def forward_critic(
        self,
        features: th.Tensor,
    ):
        global_embedding = features[
            :,
            :self.embedding_dim,
        ]

        return self.value_net(
            global_embedding
        )


class SchedulingActionHead(nn.Module):
    """
    Produces one logit for every action.

    Action ordering:

        0
            skip

        1
            machine 0, job 0

        2
            machine 0, job 1

        ...

        n
            machine 0, job n

        ...

    The score for a machine/job pair is based on:

        global state
        +
        machine embedding
        +
        job embedding
    """

    def __init__(
        self,
        n_jobs: int,
        n_machines: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.embedding_dim = embedding_dim

        # ---------------------------------------------------------
        # Machine identity embedding
        # ---------------------------------------------------------

        self.machine_embedding = nn.Embedding(
            n_machines,
            embedding_dim,
        )

        # ---------------------------------------------------------
        # Machine + job scoring network
        #
        # Input:
        #
        # global
        # machine
        # job
        #
        # => 3 * embedding_dim
        # ---------------------------------------------------------

        self.score_net = nn.Sequential(
            nn.Linear(
                embedding_dim * 3,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                1,
            ),
        )

        # ---------------------------------------------------------
        # Skip action
        # ---------------------------------------------------------

        self.skip_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        features: th.Tensor,
    ) -> th.Tensor:

        batch_size = features.shape[0]

        # ---------------------------------------------------------
        # Global embedding
        #
        # (batch, embedding_dim)
        # ---------------------------------------------------------

        global_embedding = features[
            :,
            :self.embedding_dim,
        ]

        # ---------------------------------------------------------
        # Job embeddings
        #
        # (batch, n_jobs * embedding_dim)
        #
        # ->
        #
        # (batch, n_jobs, embedding_dim)
        # ---------------------------------------------------------

        job_embeddings = features[
            :,
            self.embedding_dim:,
        ]

        job_embeddings = job_embeddings.reshape(
            batch_size,
            self.n_jobs,
            self.embedding_dim,
        )

        # ---------------------------------------------------------
        # Machine embeddings
        #
        # (n_machines, embedding_dim)
        # ---------------------------------------------------------

        machine_ids = th.arange(
            self.n_machines,
            device=features.device,
        )

        machine_embeddings = (
            self.machine_embedding(
                machine_ids
            )
        )

        # ---------------------------------------------------------
        # Expand global
        #
        # (batch, n_machines, n_jobs, embedding_dim)
        # ---------------------------------------------------------

        global_expanded = (
            global_embedding
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(
                batch_size,
                self.n_machines,
                self.n_jobs,
                self.embedding_dim,
            )
        )

        # ---------------------------------------------------------
        # Expand jobs
        # ---------------------------------------------------------

        job_expanded = (
            job_embeddings
            .unsqueeze(1)
            .expand(
                batch_size,
                self.n_machines,
                self.n_jobs,
                self.embedding_dim,
            )
        )

        # ---------------------------------------------------------
        # Expand machines
        # ---------------------------------------------------------

        machine_expanded = (
            machine_embeddings
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(
                batch_size,
                self.n_machines,
                self.n_jobs,
                self.embedding_dim,
            )
        )

        # ---------------------------------------------------------
        # Combine
        #
        # [global, machine, job]
        # ---------------------------------------------------------

        pair_features = th.cat(
            [
                global_expanded,
                machine_expanded,
                job_expanded,
            ],
            dim=-1,
        )

        # ---------------------------------------------------------
        # Flatten machine/job pairs
        # ---------------------------------------------------------

        pair_features = pair_features.reshape(
            batch_size,
            self.n_machines * self.n_jobs,
            self.embedding_dim * 3,
        )

        # ---------------------------------------------------------
        # Score every machine/job pair
        # ---------------------------------------------------------

        pair_scores = self.score_net(
            pair_features
        ).squeeze(-1)

        # ---------------------------------------------------------
        # Skip
        # ---------------------------------------------------------

        skip_score = self.skip_head(
            global_embedding
        )

        # ---------------------------------------------------------
        # Final action logits
        # ---------------------------------------------------------

        return th.cat(
            [
                skip_score,
                pair_scores,
            ],
            dim=1,
        )


class SchedulingValueNet(nn.Module):
    """
    Critic.

    Receives only the global state representation.
    """

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                embedding_dim,
                256,
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                256,
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                1,
            ),
        )

    def forward(
        self,
        features: th.Tensor,
    ):
        return self.net(features)


class NewSchedulingPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        embedding_dim=128,
        max_time=250.0,
        n_machines=1,
        **kwargs,
    ):

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.scheduling_embedding_dim = embedding_dim

        self.scheduling_n_jobs = int(
            observation_space["status"].shape[0]
        )

        self.scheduling_n_machines = (
            n_machines
        )

        self.scheduling_n_actions = (
            action_space.n
        )

        # ---------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------

        kwargs.setdefault(
            "optimizer_class",
            th.optim.Adam,
        )

        kwargs.setdefault(
            "optimizer_kwargs",
            {
                # "weight_decay": 0.01,
            },
        )

        # We define these ourselves.
        kwargs.pop("net_arch", None)
        kwargs.pop("ortho_init", None)

        # ---------------------------------------------------------
        # SB3 ActorCriticPolicy
        # ---------------------------------------------------------

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,

            net_arch={
                "pi": [],
                "vf": [],
            },

            ortho_init=False,

            features_extractor_class=(
                StatusSizeWaitTtlCapacityFeaturesExtractor
            ),

            features_extractor_kwargs={
                "status_embed_dim": 16,
                "embedding_dim": embedding_dim,
                "attention_hidden_dim": 32,
                "max_time": max_time,
            },

            **kwargs,
        )

        # ---------------------------------------------------------
        # Actor
        # ---------------------------------------------------------

        self.action_net = SchedulingActionHead(
            n_jobs=self.scheduling_n_jobs,
            n_machines=self.scheduling_n_machines,
            embedding_dim=embedding_dim,
        )

        # ---------------------------------------------------------
        # Critic
        # ---------------------------------------------------------

        self.value_net = SchedulingValueNet(
            embedding_dim=embedding_dim,
        )

        # ---------------------------------------------------------
        # Optimizer
        #
        # We replaced action_net and value_net after
        # ActorCriticPolicy created the optimizer.
        # Therefore recreate it.
        # ---------------------------------------------------------

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:

        self.mlp_extractor = SchedulingMlpExtractor(
            n_jobs=self.scheduling_n_jobs,
            embedding_dim=self.scheduling_embedding_dim,
        )
