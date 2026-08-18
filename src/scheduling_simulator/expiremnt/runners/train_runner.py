import logging
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
import typing as tp
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym

import glob

from scheduling_simulator.envioremnt.wrappers.failure_skip_time_wrapper import FailureSkipTimeWrapper
from scheduling_simulator.expiremnt.runners.env_factory import generate_scheduling_env
from scheduling_simulator.expiremnt.callbacks.entconf_callback import EntCoefScheduler
from scheduling_simulator.expiremnt.callbacks.scheduler_callbacks import CustomMetricsCallback
from scheduling_simulator.expiremnt.policy.schedule import SchedulingPolicy

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict
    from scheduling_simulator.core.creator import ClusterGenerationConfig


def _unbatch(obs: dict) -> dict:
    """Strip the leading vec-env batch dimension."""
    return {
        k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
        for k, v in obs.items()
    }


def get_auto_device() -> str:
    import torch as th
    """Pick the best available device: cuda > cpu.

    MPS is slower than CPU for this model size due to data-transfer
    overhead. Only use CUDA GPUs where the compute benefit outweighs
    the transfer cost.
    """
    if th.cuda.is_available():
        return 'cuda'
    elif th.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class ExperimentRunner:

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
        train_steps: int,
        evalution_steps: int,
        policy_kwargs: dict,
        max_time: int = 250,
        run_with_wandb: bool = True,
        eval_every_steps: int = 10_000,
    ) -> None:
        self.eval_every_steps = eval_every_steps
        self.config = config
        self.run_with_wandb = run_with_wandb
        if self.run_with_wandb:
            self._run = wandb.init(
                project="cluster-scheduling-simulator",
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
        self.train_steps = train_steps
        self.evalution_steps = evalution_steps
        self.policy_kwargs = policy_kwargs
        self.max_time = max_time
        self.run_id = "defualt" if not self.run_with_wandb else self._run.id

    def run(self) -> None:
        env = self.generate_enviroemnt(f"videos/{self.run_id}", with_video=False)
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        model = self._train(env)
        self._evaluate(model, n_episodes=self.evalution_steps)
        env.close()
        if self.run_with_wandb:
            wandb.finish()

    def _train(self, env) -> BaseAlgorithm:
        model = PPO(
            SchedulingPolicy,
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            n_epochs=6,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"runs/{self.run_id}",
            policy_kwargs=self.policy_kwargs,
            device=get_auto_device(),
        )
        callbacks = []
        eval_env = self.generate_enviroemnt(f"videos/{self.run_id}/eval", with_video=False)
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=f"models/{self.run_id}",
            log_path=f"models/{self.run_id}",
            eval_freq=self.eval_every_steps,
            n_eval_episodes=10,
        ))
        callbacks.append(EntCoefScheduler(start=0.02, end=0.002, total_timesteps=self.train_steps))
        if self.run_with_wandb:
            callbacks.append(WandbCallback(
                gradient_save_freq=50_000,
                model_save_path=f"models/{self.run_id}",
                verbose=2,
            ))
            callbacks.append(CustomMetricsCallback())

        model.learn(self.train_steps, callback=CallbackList(callbacks))
        model.save(f"models/{self.run_id}/final_model")
        model_path = f"models/{self.run_id}/final_model.zip"
        if self.run_with_wandb:
            wandb.save(model_path)
        eval_env.close()
        return model

    def _evaluate(self, model: BaseAlgorithm, *, n_episodes: int, seed: int = 42, video_every: int = 5) -> None:
        print("Evaluation")
        n_jobs = self.config['n_jobs']

        for ep in range(n_episodes):
            record_this_ep = (ep % video_every == 0)
            envs = self.generate_enviroemnt(
                f"videos/evaluation/{self.run_id}/ep_{ep}",
                with_video=record_this_ep,
                n_env=1,
            )
            envs.seed(seed + ep)
            obs = envs.reset()
            obs: 'ObservationDict'
            total_reward, steps, done = 0.0, 0, False
            allocations = 0
            final_obs = None

            while not done:
                steps += 1
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done_arr, infos = envs.step(action)
                done = bool(done_arr[0])
                total_reward += float(reward[0])

                # DummyVecEnv auto-resets on done: `obs` on the terminal
                # step is already the NEXT episode's reset observation.
                # The true final observation lives in infos[0]['terminal_observation'].
                if done:
                    final_obs = _unbatch(infos[0].get('terminal_observation', obs))
                else:
                    final_obs = _unbatch(obs)

                allocations += int(action[0] != 0 and final_obs['action_success'])

            # Sanity check: each of the n_jobs jobs can only be
            # successfully allocated once per episode (job status moves
            # strictly NOT_CREATED -> PENDING -> RUNNING -> COMPLETED,
            # never back to PENDING, and jobs aren't replenished mid-episode
            # — see job.pyx/creator.pyx). If this ever fires, it means the
            # observation being used for the success/failure check is
            # stale (e.g. an auto-reset obs) rather than the real one.
            if allocations > n_jobs:
                logging.warning(
                    "eval episode %d: counted %d allocations but only %d jobs exist "
                    "— likely a stale/auto-reset observation bug, not real behavior.",
                    ep, allocations, n_jobs,
                )

            completed_count = np.sum(final_obs['status'] == JobStatus.COMPLETED)
            running_count = np.sum(final_obs['status'] == JobStatus.RUNNING)
            pending_count = np.sum(final_obs['status'] == JobStatus.PENDING)
            not_created_count = np.sum(final_obs['status'] == JobStatus.NOT_CREATED)

            information = {
                "evaluation/episode": ep,
                "eval/length": steps,
                "eval/avg_wait_time": np.mean(final_obs['wait_time']),
                "eval/max_wait_time": np.max(final_obs['wait_time']),
                "eval/allocations": allocations,
                "eval/time": float(np.asarray(final_obs['time']).squeeze()),
                "eval/scheduled": completed_count + running_count,
                "eval/pending": pending_count,
                "eval/not_created": not_created_count,
                "eval/reward": total_reward,
                "eval/avg_completion_time": (final_obs['wait_time'] + final_obs['ttl']).mean(),
            }
            envs.close()
            if self.run_with_wandb:
                wandb.log(information)
                if record_this_ep:
                    for f in glob.glob(f"videos/evaluation/{self.run_id}/ep_{ep}/*.mp4"):
                        wandb.log({"video/evaluation": wandb.Video(f, fps=30, format="mp4")})
            else:
                print(information)

    def generate_enviroemnt(self, path: str, with_video: bool = False, n_env: int = 4):
        return generate_scheduling_env(
            config=self.config,
            max_time=self.max_time,
            path=path,
            with_video=with_video,
            n_env=n_env,
        )
