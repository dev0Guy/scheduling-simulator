from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList
import wandb
import typing as tp
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from wandb.integration.sb3 import WandbCallback

from scheduling_simulator.expiremnt.callbacks.scheduler_callbacks import CustomMetricsCallback
from scheduling_simulator.expiremnt.policy import SchedulingPolicy, ValidityPPO, get_auto_device

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.creator import ClusterGenerationConfig


class TrainExperimentRunner:

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
    ) -> None:
        self.config = config
        self._run = wandb.init(
            project="cluster-scheduling-simulator",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    def run(self) -> None:
        env = self.generate_enviroemnt()
        eval_config = {**self.config, 'n_jobs': 32}
        eval_env = DummyVecEnv([lambda: Monitor(
            gym.wrappers.TimeLimit(
                SchedulingEnviorment(eval_config, render_mode='rgb_array', max_n_jobs=48),
                max_episode_steps=500,
            )
        )])
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        model = ValidityPPO(
            SchedulingPolicy,
            env,
            learning_rate=3e-4,
n_steps=512,
            batch_size=128,
            gamma=1.0,
            gae_lambda=0.95,
            ent_coef=0.02,
            n_epochs=6,
            max_grad_norm=0.5,
            verbose=1,
            device=get_auto_device(),
            validity_coef=lambda progress: 0.0 if progress > 0.5 else (0.5 * (0.5 - progress) / 0.25 if progress > 0.25 else 0.5),
            tensorboard_log=f"runs/{self._run.id}"
        )
        model.learn(100_000, callback=CallbackList([
                    MaskableEvalCallback(
                        eval_env,
                        best_model_save_path=f"models/{self._run.id}",
                        log_path=f"models/{self._run.id}",
                        eval_freq=5_000,
                        n_eval_episodes=32,
                        deterministic=False,
                    ),
                    WandbCallback(
                        gradient_save_freq=1_000,
                        model_save_path=f"models/{self._run.id}",
                        verbose=2,
                    ),
                    CustomMetricsCallback()
                ]))
        model = ValidityPPO.load(f"models/{self._run.id}/best_model", env=env)
        model.save(f"models/{self._run.id}/final_model")
        model_path = f"models/{self._run.id}/final_model.zip"
        wandb.save(model_path)
        env.close()
        eval_env.close()
        wandb.finish()

    def generate_enviroemnt(self):
        max_n_jobs = 48
        job_counts = [20, 24, 28, 32, 36, 40]

        def _make_env():
            n_jobs = int(np.random.choice(job_counts))
            train_config = {**self.config, 'n_jobs': n_jobs}
            return Monitor(
                gym.wrappers.TimeLimit(
                    SchedulingEnviorment(
                        train_config,
                        render_mode='rgb_array',
                        max_n_jobs=max_n_jobs,
                    ),
                    max_episode_steps=500,
                )
            )

        envs = DummyVecEnv([_make_env for _ in range(8)])
        return envs
