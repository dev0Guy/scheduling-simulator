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

runner = ExperimentRunner(
    config,
    train_steps=750_000,
    evalution_steps=100,
    policy_kwargs={
        "embedding_dim": 128,
        "n_machines": config["n_machines"],
        "max_time": 100.0,
    },
    max_time=100,
    run_with_wandb=True,
    eval_every_steps=50_000,
)

runner.run()
