import typing as tp

from scheduling_simulator.expiremnt.policy.no_attention_sched import (
    NoAttentionSchedulingPolicy,
)
from scheduling_simulator.expiremnt.runners.train_runner import (
    ExperimentRunner,
)

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import (
        ClusterGenerationConfig,
    )


config: "ClusterGenerationConfig" = {
    "n_machines": 1,
    "n_jobs": 10,
    "n_resource": 3,
    "n_time": 10,
    "max_capacity": 255,
}

# Model without any attention (DeepSets style): see feature_extractor/no_attention_features.py
runner = ExperimentRunner(
    config,
    policy_class=NoAttentionSchedulingPolicy,
    train_steps=2_000_000,
    evalution_steps=5000,
    policy_kwargs={
        "embedding_dim": 256,
        "n_machines": config["n_machines"],
        "max_time": 110.0,
        # "attention_hidden_dim":64
    },
    max_time=110,
    run_with_wandb=True,
    eval_every_steps=50_000,
    n_envs=8,
    n_steps=512,
    learning_rate=2e-5,
    lr_decay=True,
    target_kl=0.03,
    n_eval_episodes=50,
)

runner.run()
