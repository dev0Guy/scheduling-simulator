from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList
import wandb
import typing as tp
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from wandb.integration.sb3 import WandbCallback

from scheduling_simulator.expiremnt.callbacks.scheduler_callbacks import CustomMetricsCallback
from scheduling_simulator.expiremnt.policy import SchedulingPolicy, get_auto_device

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
        eval_env = self.generate_enviroemnt()
        eval_env.seed(10_000)
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        model = MaskablePPO(
            SchedulingPolicy,
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=1.0,
            gae_lambda=0.95,
            ent_coef=0.02,
            n_epochs=4,
            max_grad_norm=0.5,
            verbose=1,
            device=get_auto_device(),
            tensorboard_log=f"runs/{self._run.id}"
        )
        model.learn(50_000, callback=CallbackList([
                    MaskableEvalCallback(
                        eval_env,
                        best_model_save_path=f"models/{self._run.id}",
                        log_path=f"models/{self._run.id}",
                        eval_freq=5_000,
                        n_eval_episodes=32,
                        deterministic=True,
                    ),
                    WandbCallback(
                        gradient_save_freq=1_000,
                        model_save_path=f"models/{self._run.id}",
                        verbose=2,
                    ),
                    CustomMetricsCallback()
                ]))
        model = MaskablePPO.load(f"models/{self._run.id}/best_model", env=env)
        model.save(f"models/{self._run.id}/final_model")
        model_path = f"models/{self._run.id}/final_model.zip"
        wandb.save(model_path)
        env.close()
        eval_env.close()
        wandb.finish()

    def generate_enviroemnt(self):
        envs = DummyVecEnv([lambda: Monitor(
            gym.wrappers.TimeLimit(
                SchedulingEnviorment(self.config, render_mode='rgb_array'),
                max_episode_steps=500,
            )
        )])
        return envs
