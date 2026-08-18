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
        self._selected_jobs = set()


    def foward_time_untill_max_tick(self, observation: 'ObservationDict') -> 'ObservationDict':
        for _ in range(self._max_time - int(observation['time'][0])):
            observation, *_ = super().step(0)
        return observation

    def completion_time_reward(self, observation: 'ObservationDict') -> float:
        assert int(observation['time'][0]) == self._max_time, f"{observation['time']}, {self._max_time}"
        observation['wait_time'][np.where(observation['wait_time'] == -1)] = self._max_time
        return sum(
            -1 * (observation['wait_time'][j_idx] + observation['ttl'][j_idx])
            for j_idx in range(observation['ttl'].shape[0])
        )

    @staticmethod
    def _skip_time_reward(observation: 'ObservationDict') -> float:
        return sum(
            -1 / observation['size'][idx]
            for idx, status in enumerate(observation["status"])
            if status in (JobStatus.PENDING, JobStatus.RUNNING)
        )

    def step(self, action: int) -> tuple['ObservationDict', SupportsFloat, bool, bool, dict[str, Any]]:
        observation, _, terminated, trunced, info = super().step(action)
        observation['jobs_usage'][np.where(observation['status'] != JobStatus.PENDING)] = 256
        has_allocation_failed = not observation['action_success']
        self._time_counter += 1

        skip_time, m_idx, j_idx = self.env.unwrapped._cluster.action_to_value(action)

        if np.all(observation['status'] == JobStatus.COMPLETED):
            return observation, 100, True, False, info

        if self._time_counter > self._max_time:
            return observation, -100, True, False, info

        elif skip_time:
            return observation, self._skip_time_reward(observation), terminated, False, info

        elif has_allocation_failed:
            # TODO: maybe change
            observation,_, terminated, trunced, info =  self.env.step(0)
            self._time_counter += 1
            return observation, 1.2 * self._skip_time_reward(observation), terminated, False, info

        return observation, 0, terminated, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple['ObservationDict', dict[str, Any]]:
        observation, _extra  = super().reset(seed=seed, options=options)
        self._time_counter = 0
        self._selected_jobs = set()
        return observation, _extra
