import typing as tp

from scheduling_simulator.expiremnt.policy.masked_sched import MaskedSchedulingPolicy
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

# Masked PPO as example_masked.py, but with the reward changed (in training AND evaluation):
#   skip time = -(waiting + running jobs) / max_ticks      (max_ticks = max_time = 110 steps)
# the other terms are unchanged: +1 per successful start, +100 when all jobs finished, -100 at the limit.
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
    policy_class=MaskedSchedulingPolicy,   # invalid actions get probability 0
    max_time=110,
    run_with_wandb=True,
    eval_every_steps=50_000,
    n_envs=8,
    n_steps=512,
    learning_rate=2e-5,
    lr_decay=True,
    target_kl=0.03,
    n_eval_episodes=50,
    ent_coef_start=0.05,
    ent_coef_end=0.01,
    normalize_skip_penalty=True,
)

runner.run()
