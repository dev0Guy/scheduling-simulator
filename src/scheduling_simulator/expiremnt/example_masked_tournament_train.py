"""Masked PPO from scratch for the tournament form: one machine, a window of 10 job slots holding 1..10 real jobs.

    N_MACHINES=1 TRAIN_SEED=3 JOB_USAGE_MODE=random python .../example_masked_tournament_train.py

Same recipe as example_masked_scale.py (8 envs x 512, lr 2e-5, entropy 0.05 -> 0.01, 2M steps, original reward,
random job usage); the only difference is the environment: every episode draws k ~ U{MIN_JOBS..10} real jobs and
pads the window to 10 slots exactly like the tournament does (see envioremnt/wrappers/variable_jobs_env.py).
The model is meant to be applied with scripts/tournament/tournament_eval.py (MODEL=models/<run>/final_model.zip).
The built-in final evaluation (5000 episodes, seed 42 + episode) uses the same random job counts.
"""
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scheduling_simulator.envioremnt.wrappers.variable_jobs_env import VariableJobsEnv
from scheduling_simulator.expiremnt.policy.masked_sched import MaskedSchedulingPolicy
from scheduling_simulator.expiremnt.runners.train_runner import ExperimentRunner

n_machines = int(os.environ.get("N_MACHINES", "1"))
slots = int(os.environ.get("SLOTS", "10"))
min_jobs = int(os.environ.get("MIN_JOBS", "1"))
max_time = 11 * slots          # the policy normalises sizes and waits by the full window, as the 10-job models do

config = {"n_machines": n_machines, "n_jobs": slots, "n_resource": 3, "n_time": 10, "max_capacity": 255}


class VariableJobsRunner(ExperimentRunner):
    """The runner of the other models, with the variable-job-count environment instead of the fixed one."""

    def generate_enviroemnt(self, path, with_video=False, n_env=4, sjf_relative=False, reward_scale=1.0,
                            drop_constant_rewards=False, optimal_gap_reward=False):
        return DummyVecEnv([lambda: Monitor(VariableJobsEnv(self.config, min_jobs=min_jobs)) for _ in range(n_env)])


runner = VariableJobsRunner(
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
