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
    evalution_steps=5000,
    policy_kwargs={
        "embedding_dim": 256,
        "n_machines": config["n_machines"],
        "max_time": 110.0,
        # M9 input with one value PER RESOURCE (max over time) for each
        # job and machine, instead of one scalar. False = original M9.
        "per_resource_fit": True,
        # Actor extras: per-(machine, job) pair numbers (fits now, delay until
        # it fits, headroom, ...) and a state-dependent machine embedding.
        "pair_features": True,
        "dynamic_machine": True,
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
    # Reward = r + gamma*V_SJF(s') - V_SJF(s) on the training env only (evaluation keeps the original reward)
    sjf_relative_reward=True,
)

runner.run()
