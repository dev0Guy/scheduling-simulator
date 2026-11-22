"""Fine-tune a masked-PPO model on a window of SLOTS job slots holding 1..SLOTS real jobs (padding like the tournament).

    N_MACHINES=2 SLOTS=20 N_TIME=15 JOB_USAGE_MODE=random JOB_GEN_MODE=deeprm DEEPRM_LOAD=10 DEEPRM_DEMAND_SCALE=2 \
    INIT_POLICY=models/init/deeprm_m2_n20_q9050mka_policy.pt python .../example_masked_variable_ft.py
"""
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from scheduling_simulator.envioremnt.wrappers.variable_jobs_env import VariableJobsEnv
from scheduling_simulator.expiremnt.policy.masked_sched import MaskedSchedulingPolicy
from scheduling_simulator.expiremnt.runners.train_runner import ExperimentRunner

n_machines = int(os.environ.get("N_MACHINES", "2"))
slots = int(os.environ.get("SLOTS", "20"))
min_jobs = int(os.environ.get("MIN_JOBS", "1"))
max_time = 11 * slots
config = {"n_machines": n_machines, "n_jobs": slots, "n_resource": 3, "n_time": int(os.environ.get("N_TIME", "10")), "max_capacity": 255}


class VariableJobsRunner(ExperimentRunner):
    def generate_enviroemnt(self, path, with_video=False, n_env=4, sjf_relative=False, reward_scale=1.0,
                            drop_constant_rewards=False, optimal_gap_reward=False):
        return DummyVecEnv([lambda: Monitor(VariableJobsEnv(self.config, min_jobs=min_jobs)) for _ in range(n_env)])


runner = VariableJobsRunner(
    config,
    train_steps=int(os.environ.get("TRAIN_STEPS", "1000000")),
    evalution_steps=5000,
    policy_kwargs={"embedding_dim": 256, "n_machines": n_machines, "max_time": float(max_time),
                   "per_resource_fit": True, "pair_features": True, "dynamic_machine": True},
    policy_class=MaskedSchedulingPolicy,
    max_time=max_time,
    run_with_wandb=True,
    eval_every_steps=50_000,
    n_envs=8,
    n_steps=512,
    init_policy_path=os.environ.get("INIT_POLICY"),
    learning_rate=float(os.environ.get("FT_LR", 1e-5)),
    lr_decay=True,
    target_kl=0.03,
    n_eval_episodes=50,
    ent_coef_start=float(os.environ.get("FT_ENT_START", 0.01)),
    ent_coef_end=float(os.environ.get("FT_ENT_END", 0.005)),
)
runner.run()
