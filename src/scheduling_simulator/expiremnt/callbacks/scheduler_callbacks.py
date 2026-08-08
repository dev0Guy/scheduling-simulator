from typing import TypedDict, List
from scheduling_simulator.core.job import JobStatus

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class EpisodesMetrics(TypedDict):
    count: int
    length: List[int]
    reward: List[float]
    avg_wait_time: List[float]
    max_wait_time: List[float]
    current_time: List[float]
    allocations: List[int]
    failed_allocations: List[int]
    completion_time: List[int]


class CustomMetricsCallback(BaseCallback):
    """
    Pulls custom metrics from the observation dict (not info) and logs
    them to W&B at every rollout end. Episode reward/length still come
    from Monitor's info["episode"].
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = EpisodesMetrics(
            count=0,
            allocations=[],
            length=[],
            reward=[],
            avg_wait_time=[],
            max_wait_time=[],
            current_time=[],
            failed_allocations=[],
            completion_time = []
        )

    def _update_episode_metrics(self):
        for key in self.episode_metrics:
            if key == "count":
                continue
            self.episode_metrics[key].clear()

    def _on_step(self) -> bool:
        # VecEnv returns arrays; new_obs is a dict of batched arrays,
        # shape (n_envs, *field_shape)
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        new_obs = self.locals["new_obs"]

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            action_success = int(new_obs["action_success"][env_idx])

            if not done:
                self.episode_metrics["failed_allocations"].append(action_success == 0)
                continue

            if "episode" in info:
                self.episode_metrics["reward"].append(info["episode"]["r"])
                self.episode_metrics["length"].append(info["episode"]["l"])

            # NOTE: on the terminal step, VecEnv's new_obs for this env index
            # is already the RESET observation, not the final one (SB3/Gymnasium
            # auto-resets under the hood). If you need the true final-step
            # values, use info["terminal_observation"] instead (Gymnasium/SB3
            # populates this automatically when an episode ends).
            final_obs = info.get("terminal_observation", None)
            obs_source = final_obs if final_obs is not None else {
                k: v[env_idx] for k, v in new_obs.items()
            }

            wait_time = np.asarray(obs_source["wait_time"], dtype=np.float32)
            size = np.asarray(obs_source["size"], dtype=np.float32)
            status = np.asarray(obs_source["status"], dtype=np.int32)
            current_time = float(np.asarray(obs_source["time"]).squeeze())

            self.episode_metrics["avg_wait_time"].append(np.average(wait_time))
            self.episode_metrics["max_wait_time"].append(np.max(wait_time))
            self.episode_metrics["current_time"].append(current_time)
            self.episode_metrics["completion_time"].append(np.average(wait_time + size))

            allocations = int(np.sum(status != JobStatus.PENDING))
            self.episode_metrics["allocations"].append(allocations)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return

        for idx in range(len(self.episode_metrics["length"])):
            self.episode_metrics["count"] += 1
            wandb.log({
                "episode/count": self.episode_metrics["count"],
                "episode/length": self.episode_metrics["length"][idx],
                "episode/reward": self.episode_metrics["reward"][idx],
                "episode/allocations": self.episode_metrics["allocations"][idx],
                "episode/max_wait_time": self.episode_metrics["max_wait_time"][idx],
                "episode/avg_wait_time": self.episode_metrics["avg_wait_time"][idx],
                "episode/time": self.episode_metrics["current_time"][idx],
                "episode/failed_allocations": np.sum(self.episode_metrics["failed_allocations"]),
                "episode/avg_completion_time": self.episode_metrics["completion_time"][idx]

            })

        self._update_episode_metrics()
