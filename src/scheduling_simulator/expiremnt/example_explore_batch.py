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

# Exploration through a bigger, more diverse rollout and gentler policy updates:
# 16 envs x 1024 steps = 16,384 transitions per update (was 4,096), minibatch 512,
# 4 epochs, KL limit 0.02, and a strong entropy bonus that decays slowly.
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
    eval_every_steps=25_000,     # x 16 envs = every 400k steps
    n_envs=16,
    n_steps=1024,
    batch_size=512,
    n_epochs=4,
    learning_rate=3e-5,
    lr_decay=True,
    target_kl=0.02,
    ent_coef_start=0.2,
    ent_coef_end=0.02,
)
runner.run()
