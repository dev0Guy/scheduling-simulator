from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
import wandb
import typing as tp
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from wandb.integration.sb3 import WandbCallback

import glob

from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.expiremnt.callbacks.scheduler_callbacks import CustomMetricsCallback

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
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=10_000,
            learning_starts=100,
            batch_size=32,
            gamma=0.99,
            train_freq=1,
            target_update_interval=250,
            verbose=1,
            tensorboard_log=f"runs/{self._run.id}"
        )
        model.learn(50_000, callback=CallbackList([
                    WandbCallback(
                        gradient_save_freq=1_000,
                        model_save_path=f"models/{self._run.id}",
                        verbose=2,
                    ),
                    CustomMetricsCallback()
                ]))
        model.save(f"models/{self._run.id}/final_model")
        model_path = f"models/{self._run.id}/final_model.zip"
        wandb.save(model_path)
        for f in glob.glob(f"videos/{self._run.id}/*.mp4"):
            wandb.log({"video": wandb.Video(f, fps=30, format="mp4")})
        env.close()
        wandb.finish()

    def generate_enviroemnt(self):
        envs = DummyVecEnv([lambda: Monitor(
            FailureSkipTimeWrapper(
                SchedulingEnviorment(self.config, render_mode='rgb_array'), max_time=500)
            )
        ])
        envs = VecVideoRecorder(
            envs,
            f"videos/{self._run.id}",
            record_video_trigger=lambda x: x % 2_000 == 0,
            video_length=200,
        )
        return envs
