import logging
import torch as th
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
import os
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
from scheduling_simulator.expiremnt.policy.new_sched import NewSchedulingPolicy
from scheduling_simulator.expiremnt.runners.env_factory import generate_scheduling_env
from scheduling_simulator.expiremnt.callbacks.entconf_callback import EntCoefScheduler
from scheduling_simulator.expiremnt.callbacks.scheduler_callbacks import CustomMetricsCallback
from scheduling_simulator.expiremnt.policy.schedule import SchedulingPolicy

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict
    from scheduling_simulator.core.creator import ClusterGenerationConfig
    from stable_baselines3.common.base_class import BaseAlgorithm

def _unbatch(obs: dict) -> dict:
    """Strip the leading vec-env batch dimension."""
    return {
        k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
        for k, v in obs.items()
    }


def _linear_schedule(initial_lr: float, floor: float = 0.05):
    """Linear decay of the learning rate to `floor * initial_lr`.

    Closes over plain floats only (NOT the runner), because SB3 pickles the
    schedule inside model.save().
    """
    def schedule(progress_remaining: float) -> float:
        return initial_lr * max(progress_remaining, floor)

    return schedule


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
        n_envs: int = 4,
        n_steps: int = 2048,
        learning_rate: float = 7e-6,
        lr_decay: bool = True,
        target_kl: float | None = 0.03,
        n_eval_episodes: int = 50,
        use_best_model: bool = False,
        ent_coef_start: float = 0.05,
        ent_coef_end: float = 0.01,
        policy_class=None,
        sjf_relative_reward: bool = False,
        init_policy_path: str | None = None,
        batch_size: int = 64,
        n_epochs: int = 5,
        gamma: float = 0.995,
        gae_lambda: float = 0.995,
        clip_range: float = 0.2,
        ent_hold_fraction: float = 0.0,
        maskable_ppo: bool = False,
        seed: int | None = None,
        reward_scale: float = 1.0,
        drop_constant_rewards: bool = False,
        optimal_gap_reward: bool = False,
        normalize_skip_penalty: bool = False,
        skip_reward_mode: str = "count",
    ) -> None:
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay          # linear decay to 5% of learning_rate
        self.target_kl = target_kl        # stop an epoch early when a policy update is too large
        self.n_eval_episodes = n_eval_episodes
        # True: the final test uses best_model.zip, the checkpoint with the highest 50-episode check. That
        # score is noisy (SE ~8 vs differences of 1-2), so the pick is mostly luck and often an early
        # checkpoint; False (default) tests the model at the end of training.
        self.use_best_model = use_best_model
        self.ent_coef_start = ent_coef_start  # entropy bonus is annealed linearly from start to end
        self.ent_coef_end = ent_coef_end
        self.policy_class = policy_class or NewSchedulingPolicy  # e.g. NoAttentionSchedulingPolicy
        self.sjf_relative_reward = sjf_relative_reward  # shaped reward on the TRAINING env only
        self.init_policy_path = init_policy_path  # policy weights to start PPO from (e.g. imitation pretraining)
        self.batch_size, self.n_epochs = batch_size, n_epochs
        self.gamma, self.gae_lambda, self.clip_range = gamma, gae_lambda, clip_range
        # training seed: network init, action sampling and the training job sets. None -> the
        # TRAIN_SEED environment variable if set, otherwise unseeded (every run differs).
        self.seed = seed if seed is not None else (int(os.environ["TRAIN_SEED"]) if os.environ.get("TRAIN_SEED") else None)
        self.reward_scale, self.drop_constant_rewards = reward_scale, drop_constant_rewards  # training env only
        # skip reward divided by max_time (training AND evaluation: it is a change of the reward itself)
        self.normalize_skip_penalty = normalize_skip_penalty
        self.skip_reward_mode = skip_reward_mode        # "count" (original) or "wait_sum"
        self.optimal_gap_reward = optimal_gap_reward  # training env only: reward = gap to the optimal action
        self.maskable_ppo = maskable_ppo  # sb3-contrib MaskablePPO with the environment's action masks
        self.ent_hold_fraction = ent_hold_fraction  # entropy bonus stays at its start value for this share of training
        self.n_envs = n_envs
        self.n_steps = n_steps  # per env; PPO batch per update = n_envs * n_steps
        self.eval_every_steps = eval_every_steps
        self.config = config
        self.run_with_wandb = run_with_wandb
        if self.run_with_wandb:
            self._run = wandb.init(
                project="cluster-scheduling-simulator",
                config={"train_seed": self.seed},
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
            wandb.define_metric("evaluation/episode")
            wandb.define_metric("eval/*", step_metric="evaluation/episode")
            wandb.define_metric("video/evaluation", step_metric="evaluation/episode")

        self.train_steps = train_steps
        self.evalution_steps = evalution_steps
        self.policy_kwargs = policy_kwargs
        self.max_time = max_time
        self.run_id = "defualt" if not self.run_with_wandb else self._run.id

    def run(self) -> None:
        env = self.generate_enviroemnt(f"videos/{self.run_id}", with_video=False, n_env=self.n_envs, sjf_relative=self.sjf_relative_reward,
                                       reward_scale=self.reward_scale, drop_constant_rewards=self.drop_constant_rewards,
                                       optimal_gap_reward=self.optimal_gap_reward)
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        model = self._train(env)
        self._evaluate(model, n_episodes=self.evalution_steps)
        env.close()
        if self.run_with_wandb:
            wandb.finish()

    def _train(self, env) -> BaseAlgorithm:
        algo = MaskablePPO if self.maskable_ppo else PPO
        model = algo(
            # "MultiInputPolicy",
            self.policy_class,
            env,
            learning_rate=(
                _linear_schedule(self.learning_rate)
                if self.lr_decay
                else self.learning_rate
            ),
            target_kl=self.target_kl,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            # gamma=0.999/gae_lambda=0.99 was tried and measured as a
            # mild regression (run r23681yd, see docs/experiment_log.md)
            # vs 0.995/0.995. gamma=1.0 (undiscounted) was tried next, on
            # top of the no-arrival ablation, and was a much larger
            # regression (run r84hnw8u, see docs/experiment_log.md #17).
            # Reverted back to the validated-best 0.995/0.995 here.
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"runs/{self.run_id}",
            policy_kwargs=self.policy_kwargs,
            seed=self.seed,
            device=get_auto_device(),
        )
        if self.init_policy_path:
            model.policy.load_state_dict(th.load(self.init_policy_path, map_location=model.device))
            print(f"PPO starts from the pretrained policy: {self.init_policy_path}")
        callbacks = []
        eval_env = self.generate_enviroemnt(f"videos/{self.run_id}/eval", with_video=False)
        if self.seed is not None:
            eval_env.seed(self.seed + 10_000)   # the 50-episode checks used to pick the best checkpoint
        callbacks.append((MaskableEvalCallback if self.maskable_ppo else EvalCallback)(
            eval_env,
            best_model_save_path=f"models/{self.run_id}",
            log_path=f"models/{self.run_id}",
            eval_freq=self.eval_every_steps,
            n_eval_episodes=self.n_eval_episodes,
        ))
        callbacks.append(EntCoefScheduler(start=self.ent_coef_start, end=self.ent_coef_end, total_timesteps=self.train_steps, hold_fraction=self.ent_hold_fraction))
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

        # The last model can be much worse than the best one seen during
        # training (policy collapse late in training), so evaluate the best
        # checkpoint saved by EvalCallback when available.
        best_path = f"models/{self.run_id}/best_model.zip"
        if self.use_best_model and os.path.exists(best_path):
            print(f"Final evaluation uses best checkpoint: {best_path}")
            model = (MaskablePPO if self.maskable_ppo else PPO).load(best_path, device=get_auto_device())
        return model

    def _evaluate(self, model: BaseAlgorithm, *, n_episodes: int, seed: int = 42, video_every: int = 100) -> None:
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
            failed_allocations = 0
            final_obs = None

            while not done:
                steps += 1
                if self.maskable_ppo:
                    action, _states = model.predict(obs, deterministic=True, action_masks=get_action_masks(envs))
                else:
                    action, _states = model.predict(obs, deterministic=True)
                obs, reward, done_arr, infos = envs.step(action)
                done = bool(done_arr[0])
                total_reward += float(reward[0])

                # DummyVecEnv auto-resets on done: `obs` on the terminal
                # step is already the NEXT episode's reset observation.
                # The true final observation lives in infos[0]['terminal_observation'].
                if done:
                    final_obs = infos[0].get('terminal_observation', obs)
                else:
                    final_obs = _unbatch(obs)

                attempted_allocation = action[0] != 0
                succeeded = bool(final_obs['action_success'])
                allocations += int(attempted_allocation and succeeded)
                failed_allocations += int(attempted_allocation and not succeeded)

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

            envs.close()

            completed_count = np.sum(final_obs['status'] == JobStatus.COMPLETED)
            running_count = np.sum(final_obs['status'] == JobStatus.RUNNING)
            pending_count = np.sum(final_obs['status'] == JobStatus.PENDING)
            not_created_count = np.sum(final_obs['status'] == JobStatus.NOT_CREATED)
            information = {
                "evaluation/episode": ep,
                "eval/length": steps,
                "eval/avg_wait_time": np.mean(final_obs['wait_time']),
                "eval/max_wait_time": np.max(final_obs['wait_time']),
                "eval/max_completion_time": (final_obs['wait_time'] + final_obs['size']).max(),
                "eval/avg_completion_time": (final_obs['wait_time'] + final_obs['size']).mean(),
                "eval/status-completed": completed_count,
                "eval/status-running": running_count,
                "eval/status-pending": pending_count,
                "eval/allocations": allocations,
                "eval/failed_allocations": failed_allocations,
                "eval/time": float(np.asarray(final_obs['time']).squeeze()),
                "eval/scheduled": completed_count + running_count,
                "eval/pending": pending_count,
                "eval/not_created": not_created_count,
                "eval/reward": total_reward,
            }
            if self.run_with_wandb:
                if record_this_ep:
                    for f in glob.glob(f"videos/evaluation/{self.run_id}/ep_{ep}/*.mp4"):
                        information["video/evaluation"] = wandb.Video(f, fps=30, format="mp4")
                wandb.log(information)
            else:
                print(information)




    def generate_enviroemnt(self, path: str, with_video: bool = False, n_env: int = 4, sjf_relative: bool = False,
                            reward_scale: float = 1.0, drop_constant_rewards: bool = False,
                            optimal_gap_reward: bool = False):
        return generate_scheduling_env(
            config=self.config,
            max_time=self.max_time,
            path=path,
            with_video=with_video,
            n_env=n_env,
            sjf_relative=sjf_relative,
            maskable=self.maskable_ppo,
            reward_scale=reward_scale,
            drop_constant_rewards=drop_constant_rewards,
            optimal_gap_reward=optimal_gap_reward,
            normalize_skip_penalty=self.normalize_skip_penalty,
            skip_reward_mode=self.skip_reward_mode,
        )
