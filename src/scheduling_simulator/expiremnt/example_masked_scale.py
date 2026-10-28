"""Masked PPO from scratch (original reward) on a cluster of a configurable size.

    N_MACHINES=2 N_JOBS=10 JOB_USAGE_MODE=random TRAIN_SEED=3 python .../example_masked_scale.py

Everything else is as in example_masked.py (8 envs x 512, lr 2e-5, entropy 0.05 -> 0.01, 2M steps, original
reward). The step limit scales with the number of jobs (11 steps per job, 110 for 10 jobs) so that the
episode can still finish; the policy normalises sizes and waits by the same number.
"""
import os

from scheduling_simulator.expiremnt.policy.masked_sched import MaskedSchedulingPolicy
from scheduling_simulator.expiremnt.runners.train_runner import ExperimentRunner

n_machines = int(os.environ.get("N_MACHINES", "1"))
n_jobs = int(os.environ.get("N_JOBS", "10"))
max_time = 11 * n_jobs

config = {
    "n_machines": n_machines,
    "n_jobs": n_jobs,
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
        "n_machines": n_machines,
        "max_time": float(max_time),
        "per_resource_fit": True,
        "pair_features": True,
        "dynamic_machine": True,
    },
    policy_class=MaskedSchedulingPolicy,
    max_time=max_time,
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
