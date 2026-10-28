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

# Masked PPO whose TRAINING reward is the gap to the optimal action from the same state:
#   r' = [r(s, a) + V*(s')] - V*(s)  (model action + rest of the optimal trace, minus the optimal action +
# rest of the optimal trace; V* from the exact solver). 0 for an optimal action, negative otherwise.
# No reward normalization, no idle rule; every other setting is as in example_masked.py. Evaluation
# keeps the original reward.
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
    optimal_gap_reward=True,
)

runner.run()
