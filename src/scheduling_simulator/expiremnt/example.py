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
    train_steps=2_000_000,
    evalution_steps=100,
    policy_kwargs={
        "embedding_dim": 256,
        "n_machines": config["n_machines"],
        "max_time": 110.0,
        # "attention_hidden_dim":64
    },
    max_time=110,
    run_with_wandb=True,
    eval_every_steps=50_000,
)

runner.run()
