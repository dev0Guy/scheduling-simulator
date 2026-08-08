from typing import Any, SupportsFloat,TYPE_CHECKING
from gymnasium import Wrapper
from gymnasium.core import RenderFrame
from scheduling_simulator.core.job import JobStatus
import numpy as np

from scheduling_simulator.envioremnt.envioremnt import SchedulingEnviorment

if TYPE_CHECKING:
    from scheduling_simulator.core.cluster import ObservationDict

class FailureSkipTimeWrapper(Wrapper['ObservationDict', int, 'ObservationDict', int]):

    def __init__(self, env: SchedulingEnviorment, max_time: int = 500):
        super().__init__(env)
        self._max_time = max_time
        self._time_counter = 0
        self._selected_jobs = set()

    def foward_time_untill_max_tick(self, observation: 'ObservationDict') -> 'ObservationDict':
        for _ in range(self._max_time - self._time_counter):
            observation, *_ = super().step(0)
        return observation

    def completion_time_reward(self, observation: 'ObservationDict') -> float:
        observation['wait_time'][np.where(observation['wait_time'] == -1)] = self._max_time
        return sum(
            - 10 * (observation['wait_time'][j_idx] + observation['ttl'][j_idx])
            for j_idx in range(observation['ttl'].shape[0])
            if j_idx not in self._selected_jobs
        )

    def step(self, action: int) -> tuple['ObservationDict', SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0
        observation, _, terminated, trunced, info = super().step(action)
        observation['jobs_usage'][np.where(observation['status'] != JobStatus.PENDING)] = 256
        is_last_step = self._time_counter >= self._max_time
        has_allocation_failed = not observation['action_success']

        skip_time, m_idx, j_idx = self.env.unwrapped._cluster.action_to_value(action)

        if np.all(observation['status'] == JobStatus.COMPLETED):
            return observation, 10_000, True, trunced, info

        if is_last_step or has_allocation_failed:
            terminated = True
            # TODO: maybe don't have to
            new_observation = self.foward_time_untill_max_tick(observation)
            reward = self.completion_time_reward(new_observation)
        elif terminated:
            reward = self.completion_time_reward(observation)
        elif not skip_time:
            self._selected_jobs.add(j_idx)
            reward = -(observation['wait_time'][j_idx] + observation['ttl'][j_idx])
        else:
            self._time_counter += 1

        return observation, reward, terminated, trunced, info


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', dict[str, Any]]:
        observation, _extra  = super().reset(seed=seed, options=options)
        self._time_counter = 0
        self._selected_jobs = set()
        return observation, _extra
