from collections import defaultdict
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb
import typing as tp
import numpy as np
from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment

import glob

from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.scheduler.random_scheduler import RandomScheduler

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict
    from scheduling_simulator.core.creator import ClusterGenerationConfig


class RandomBaselineRunner:

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
        evalution_steps: int,
        max_time: int = 250,
        seed: int = 42,
    ) -> None:
        self.config = config
        self._run = wandb.init(
            project="cluster-scheduling-simulator",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        self.evalution_steps = evalution_steps
        self.max_time = max_time
        self.seed = seed
        self.scheduler = RandomScheduler(rng=np.random.default_rng(seed))

    def run(self) -> None:
        env = self.generate_enviroemnt(f"videos/evaluation/{self._run.id}")
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        print("Evaluating random baseline")
        self._evaluate(env, n_episodes=self.evalution_steps)
        env.close()
        wandb.finish()

    def _encode_action(self, skip: bool, dim1: int, dim2: int, n_dim2: int) -> int:
        if skip:
            return 0
        return 1 + dim1 * n_dim2 + dim2

    def _evaluate(self, envs, *, n_episodes: int) -> None:
        n_dim2 = envs.get_attr("n_machines")[0]

        for ep in range(n_episodes):
            envs.seed(self.seed + ep)
            obs = envs.reset()
            obs: 'ObservationDict'
            total_reward, steps, done = 0.0, 0, False
            allocations = 0
            while not done:
                steps += 1
                skip, dim1, dim2 = self.scheduler.select(obs)
                action = self._encode_action(skip, dim1, dim2, n_dim2)
                obs, reward, done, infos = envs.step(np.array([action]))
                total_reward += reward
                allocations += int(not skip and obs['action_success'])

            completed_count = np.sum(obs['status'] == JobStatus.COMPLETED)
            running_count = np.sum(obs['status'] == JobStatus.RUNNING)
            pending_count = np.sum(obs['status'] == JobStatus.PENDING)
            not_created_count = np.sum(obs['status'] == JobStatus.NOT_CREATED)

            wandb.log({
                "evaluation/episode": ep,
                "eval/length": steps,
                "eval/avg_wait_time": np.mean(obs['wait_time']),
                "eval/max_wait_time": np.max(obs['wait_time']),
                "eval/allocations": allocations,
                "eval/time": obs['time'],
                "eval/scheduled": completed_count + running_count,
                "eval/pending": pending_count,
                "eval/not_created": not_created_count,
                "eval/reward": total_reward,
                "eval/avg_completion_time": (obs['wait_time'] + obs['ttl']).mean()
            })

        for f in glob.glob(f"videos/evaluation/{self._run.id}/*.mp4"):
            wandb.log({"video/evaluation": wandb.Video(f, fps=30, format="mp4")})

    def generate_enviroemnt(self, path: str):
        envs = DummyVecEnv([lambda: Monitor(
            FailureSkipTimeWrapper(
                SchedulingEnviorment(self.config, render_mode='rgb_array'), max_time=self.max_time)
            )
        ])
        envs = VecVideoRecorder(
            envs,
            path,
            record_video_trigger=lambda x: x % 10_000 == 0,
            video_length=200,
        )
        return envs
