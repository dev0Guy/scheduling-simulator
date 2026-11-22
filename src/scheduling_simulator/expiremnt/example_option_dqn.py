"""Masked double-DQN with the (job, machine) option network (policy/option_dqn.py), 1 machine x 10 jobs from scratch.

    N_MACHINES=1 N_JOBS=10 JOB_USAGE_MODE=random TRAIN_SEED=3 python .../example_option_dqn.py

Same environment, reward and final evaluation (5000 episodes, seed 42 + episode, eval/reward per episode to W&B) as
the PPO models, so the result is directly comparable with them and with SJF.
"""
import os

import wandb
from stable_baselines3.common.callbacks import EvalCallback

from scheduling_simulator.expiremnt.policy.option_dqn import MaskedDQN, OptionDQNPolicy
from scheduling_simulator.expiremnt.runners.train_runner import ExperimentRunner

n_machines = int(os.environ.get("N_MACHINES", "1"))
n_jobs = int(os.environ.get("N_JOBS", "10"))
steps = int(os.environ.get("TRAIN_STEPS", "2000000"))
max_time = 11 * n_jobs
config = {"n_machines": n_machines, "n_jobs": n_jobs, "n_resource": 3, "n_time": 10, "max_capacity": 255}


def lr_schedule(progress_remaining: float) -> float:       # 1e-4 -> 1e-5
    return 1e-5 + (float(os.environ.get("DQN_LR", "1e-4")) - 1e-5) * progress_remaining


class DQNRunner(ExperimentRunner):
    def _train(self, env):
        model = MaskedDQN(
            OptionDQNPolicy, env,
            policy_kwargs=dict(embed=32, d_model=64, n_heads=4, n_layers=2, use_extra=os.environ.get("OPT_EXTRA") == "1",
                               max_time=float(max_time)),
            learning_rate=lr_schedule,
            buffer_size=500_000,
            learning_starts=20_000,
            batch_size=256,
            tau=1.0,
            gamma=0.995,
            train_freq=(4, "step"),
            gradient_steps=1,
            target_update_interval=2_000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            max_grad_norm=10.0,
            tensorboard_log=f"runs/{self.run_id}",
            seed=self.seed,
            device="cpu",
            verbose=1,
        )
        eval_env = self.generate_enviroemnt(f"videos/{self.run_id}/eval", with_video=False)
        if self.seed is not None:
            eval_env.seed(self.seed + 10_000)
        cb = EvalCallback(eval_env, best_model_save_path=f"models/{self.run_id}", log_path=f"models/{self.run_id}",
                          eval_freq=max(self.eval_every_steps // self.n_envs, 1), n_eval_episodes=self.n_eval_episodes)
        model.learn(self.train_steps, callback=cb)
        model.save(f"models/{self.run_id}/final_model")
        if self.run_with_wandb:
            wandb.save(f"models/{self.run_id}/final_model.zip")
        eval_env.close()
        return model


runner = DQNRunner(
    config,
    train_steps=steps,
    evalution_steps=int(os.environ.get("EVAL_EPISODES", "5000")),
    policy_kwargs={},
    max_time=max_time,
    run_with_wandb=True,
    eval_every_steps=50_000,
    n_envs=8,
    n_eval_episodes=50,
)
runner.run()
