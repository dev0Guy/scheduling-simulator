import typing as tp

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

# Exploration through a long, strong entropy phase: the bonus stays at 0.2 for the first
# half of training (the policy cannot collapse to a deterministic rule), then decays to 0.01.
# Same rollout and update sizes as the earlier runs (8 envs x 512, batch 64, 5 epochs).
runner = ExperimentRunner(
    config,
    train_steps=2_000_000,
    evalution_steps=5000,
    policy_kwargs={
        "embedding_dim": 256,
        "n_machines": config["n_machines"],
        "max_time": 110.0,
        "per_resource_fit": True,
        "pair_features": True,
        "dynamic_machine": True,
    },
    max_time=110,
    run_with_wandb=True,
    n_eval_episodes=50,
    eval_every_steps=50_000,     # x 8 envs = every 400k steps
    n_envs=8,
    n_steps=512,
    learning_rate=2e-5,
    lr_decay=True,
    target_kl=0.03,
    ent_coef_start=0.2,
    ent_coef_end=0.01,
    ent_hold_fraction=0.5,
)
runner.run()
