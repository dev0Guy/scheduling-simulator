import typing as tp

from scheduling_simulator.expiremnt.policy.maskable_sched import MaskableSchedulingPolicy
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

# sb3-contrib MaskablePPO (action masks from the environment) with the same network and settings as
# example_masked.py (which computes the same mask inside the policy): 8 envs x 512, lr 2e-5, entropy 0.05 -> 0.01.
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
    policy_class=MaskableSchedulingPolicy,
    maskable_ppo=True,                     # sb3-contrib MaskablePPO with the environment's own action masks
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
)

runner.run()
