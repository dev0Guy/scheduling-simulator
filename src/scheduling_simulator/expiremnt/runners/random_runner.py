import glob
import typing as tp

import numpy as np
import wandb

from scheduling_simulator.core.job import JobStatus
from scheduling_simulator.expiremnt.runners.env_factory import generate_scheduling_env
from scheduling_simulator.scheduler.random_scheduler import RandomScheduler

if tp.TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict
    from scheduling_simulator.core.creator import ClusterGenerationConfig


def _unbatch(obs: dict) -> dict:
    """Strip the leading vec-env batch dimension so per-observation
    consumers (like Scheduler.options) see the shapes they expect."""
    return {
        k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
        for k, v in obs.items()
    }


class RandomBaselineRunner:

    def __init__(
        self,
        config: 'ClusterGenerationConfig',
        evalution_steps: int,
        max_time: int = 250,
        seed: int = 42,
        run_with_wandb: bool = True,
    ) -> None:
        self.config = config
        self.run_with_wandb = run_with_wandb
        if self.run_with_wandb:
            self._run = wandb.init(
                project="cluster-scheduling-simulator",
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
        self.evalution_steps = evalution_steps
        self.max_time = max_time
        self.seed = seed
        self.run_id = "defualt" if not self.run_with_wandb else self._run.id
        self.scheduler = RandomScheduler(rng=np.random.default_rng(seed))

    def run(self) -> None:
        env = self.generate_enviroemnt(f"videos/evaluation/{self.run_id}", with_video=False)
        print("Env:")
        print("\t Action space: ", env.action_space)
        print("\t Observation space: ", env.observation_space)
        print("Evaluating random baseline")
        env.close()
        self._evaluate(n_episodes=self.evalution_steps)
        if self.run_with_wandb:
            wandb.finish()

    def _action_from_selection(self, skip: bool, job_idx: int, machine_idx: int) -> int:
        """Flatten (job_idx, machine_idx) into the env's Discrete action.

        Scheduler.options()/RandomScheduler.select() return
        (skip, job_idx, machine_idx) — job first, machine second (see
        abc_scheduler.py: `job_idx, machine_idx = np.nonzero(possible)`).
        This must match Cluster.allocation_to_action exactly:
            1 + machine_idx * n_jobs + job_idx
        """
        if skip:
            return 0
        n_jobs = self.config['n_jobs']
        return 1 + machine_idx * n_jobs + job_idx

    def _evaluate(self, *, n_episodes: int, video_every: int = 5) -> None:
        for ep in range(n_episodes):
            record_this_ep = (ep % video_every == 0)
            envs = self.generate_enviroemnt(
                f"videos/evaluation/{self.run_id}/ep_{ep}",
                with_video=record_this_ep,
            )
            envs.seed(self.seed + ep)
            obs = envs.reset()
            obs: 'ObservationDict'
            total_reward, steps, done = 0.0, 0, False
            allocations = 0
            final_obs = None

            while not done:
                steps += 1
                obs_single = _unbatch(obs)
                skip, job_idx, machine_idx = self.scheduler.select(obs_single)
                action = self._action_from_selection(skip, job_idx, machine_idx)
                obs, reward, done_arr, infos = envs.step(np.array([action]))
                done = bool(done_arr[0])
                total_reward += float(reward[0])

                # DummyVecEnv auto-resets on done: `obs` on the terminal
                # step is already the NEXT episode's reset observation.
                # The true final observation lives in infos[0]['terminal_observation'].
                if done:
                    final_obs = _unbatch(infos[0].get('terminal_observation', obs))
                else:
                    final_obs = _unbatch(obs)

                allocations += int(not skip and final_obs['action_success'])

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
            if self.run_with_wandb:
                wandb.log(information)
            else:
                print(information)

            envs.close()
            if record_this_ep and self.run_with_wandb:
                for f in glob.glob(f"videos/evaluation/{self.run_id}/ep_{ep}/*.mp4"):
                    wandb.log({"video/evaluation": wandb.Video(f, fps=30, format="mp4")})

    def generate_enviroemnt(self, path: str, with_video: bool = False, n_env: int = 1):
        return generate_scheduling_env(
            config=self.config,
            max_time=self.max_time,
            path=path,
            with_video=with_video,
            n_env=n_env,
        )
